import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import math
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import NestedLoRAVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbone.nested_lora_block import NestedLoRAModules
import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = NestedLoRAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.checkpoint_dir = args.get("checkpoint_dir", "checkpoints")
        self.save_checkpoints = args.get("save_checkpoints", True)

    def after_task(self):
        self._known_classes = self._total_classes
        if not self.save_checkpoints:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"{self.args['prefix']}_seed{self.args['seed']}_task{self._cur_task}.pth"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self._network.state_dict(),
                "task": self._cur_task,
                "known_classes": self._known_classes,
                "total_classes": self._total_classes,
                "args": self.args,
            },
            path,
        )
        logging.info(f"[NestedLoRA] Saved checkpoint to {path}")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0:
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

        # Setup optimizer with different LRs for fast/slow
        self.update_optimizer_and_scheduler(num_epoch=self.args['epochs'], lr=self.init_lr)
        
        # Training loop
        self._init_train(self.args['epochs'], train_loader, test_loader, self.optimizer, self.scheduler)
        
        # Consolidation at the end of task
        self._log_nested_lora_norms(tag=f"task{self._cur_task}_pre_consolidation")
        
        consolidation_interval = self.args.get("nested_lora_consolidation_interval", 1)
        do_consolidate = ((self._cur_task + 1) % consolidation_interval == 0)
        
        logging.info(f"[NestedLoRA] Task {self._cur_task} finished. Consolidation interval={consolidation_interval}. Do consolidate? {do_consolidate}")

        for module in self._network.backbone.modules():
            if isinstance(module, NestedLoRAModules):
                module.end_of_task_training(do_consolidate=do_consolidate, task_id=self._cur_task)
                
        self._log_nested_lora_norms(tag=f"task{self._cur_task}_post_consolidation")

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(total_epoch))
        global_step = 0
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                global_step += 1
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Forward
                outcome = self._network(inputs, task_id=self._cur_task)
                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]
                
                # Masking old classes to prevent forgetting in fc layer (no replay setting)
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float('inf')

                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                # Nested LoRA: control slow update frequency (optional)
                update_interval = self.args.get('nested_lora_update_interval_slow', 1)
                if update_interval > 1 and global_step % update_interval != 0:
                    for n, p in self._network.named_parameters():
                        if 'functional.slow' in n and p.grad is not None:
                            p.grad.zero_()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)


            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                total_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        
        slow_params = []
        fast_params = []
        other_params = []
        
        for n, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            
            if 'functional.slow' in n:
                slow_params.append(p)
            elif 'functional.fast' in n:
                fast_params.append(p)
            elif 'functional.fast_adapters' in n:
                fast_params.append(p)
            else:
                other_params.append(p)
        
        # Default LRs if not specified
        lr_slow = self.args.get('nested_lora_lr_slow', lr)
        lr_fast = self.args.get('nested_lora_lr_fast', lr * 5) # Default fast is 5x base
        
        param_groups = [
            {'params': slow_params, 'lr': lr_slow},
            {'params': fast_params, 'lr': lr_fast},
            {'params': other_params, 'lr': lr}
        ]
        
        if self.args['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.args["weight_decay"])
        elif self.args['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.args["weight_decay"])
            
        logging.info(f"[NestedLoRA] optimizer groups => slow: lr={lr_slow}, params={len(slow_params)}; fast: lr={lr_fast}, params={len(fast_params)}; other: lr={lr}, params={len(other_params)}")

        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=min_lr)

    def _log_nested_lora_norms(self, tag: str):
        fast_norm, slow_norm = 0.0, 0.0
        for name, param in self._network.named_parameters():
            if not param.requires_grad:
                continue
            if "functional.fast" in name:
                fast_norm += torch.norm(param.detach()).item() ** 2
            elif "functional.slow" in name:
                slow_norm += torch.norm(param.detach()).item() ** 2
            elif "functional.fast_adapters" in name:
                # Sum norms of all fast adapters
                fast_norm += torch.norm(param.detach()).item() ** 2
        fast_norm = math.sqrt(fast_norm)
        slow_norm = math.sqrt(slow_norm)
        logging.info(f"[NestedLoRA][{tag}] fast_norm={fast_norm:.4f}, slow_norm={slow_norm:.4f}")
