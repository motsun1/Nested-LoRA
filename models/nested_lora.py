import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import math
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
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
        self.selection_history = {} # Key: expert_idx, Value: count
        self.selection_records = [] # Per-task expert accuracies and picked expert
        self.num_workers = args.get("num_workers", 0)

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

        # Split current task train into train/val for expert selection (CIL-safe)
        val_ratio = self.args.get("moe_val_ratio", 0.1)
        val_len = int(len(train_dataset) * val_ratio)
        if val_len < 1:
            val_len = 1 if len(train_dataset) > 1 else 0
        train_len = len(train_dataset) - val_len
        if val_len > 0:
            g = torch.Generator().manual_seed(self.args.get("seed", 0) + self._cur_task)
            train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=g)
            self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.val_loader = None

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader, val_loader=self.val_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, val_loader=None):
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

        # Selection Logic for Parallel MoE
        moe_mode = self.args.get("moe_mode", "standard")
        selected_expert_idx = self._cur_task # Default to current task ID for standard mode
        
        logging.info(f"[NestedLoRA] moe_mode={moe_mode}, do_consolidate={do_consolidate}")
        
        if moe_mode == "parallel" and do_consolidate:
            logging.info("[NestedLoRA] Parallel MoE: Selecting best expert based on validation accuracy...")
            best_acc = -1.0
            best_idx = 0
            expert_accs = []
            nb_tasks = self.args.get("nb_tasks", 1)
            
            # Temporarily switch to eval mode
            self._network.eval()

            eval_loader = val_loader if val_loader is not None else test_loader
            for k in range(nb_tasks):
                # Evaluate expert k on current-task val set (no cross-task leakage)
                acc = self._compute_accuracy(self._network, eval_loader, task_id=k)
                expert_accs.append((k, acc))
                logging.info(f"[NestedLoRA] Expert {k} Validation Accuracy: {acc:.2f}% (loader={'val' if val_loader is not None else 'test'})")
                if acc > best_acc:
                    best_acc = acc
                    best_idx = k

            logging.info(f"[NestedLoRA] Selected Expert {best_idx} with Accuracy {best_acc:.2f}%")
            selected_expert_idx = best_idx
            
            # Update history
            self.selection_history[best_idx] = self.selection_history.get(best_idx, 0) + 1
            logging.info(f"[NestedLoRA] Expert Selection History: {self.selection_history}")

            # Persist per-task record for later inspection
            self.selection_records.append(
                {
                    "task": self._cur_task,
                    "expert_accs": expert_accs,
                    "selected": best_idx,
                    "selected_acc": best_acc,
                }
            )

        for module in self._network.backbone.modules():
            if isinstance(module, NestedLoRAModules):
                module.end_of_task_training(do_consolidate=do_consolidate, task_id=selected_expert_idx)
                
        self._log_nested_lora_norms(tag=f"task{self._cur_task}_post_consolidation")

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(total_epoch))
        global_step = 0
        
        moe_mode = self.args.get("moe_mode", "standard")
        nb_tasks = self.args.get("nb_tasks", 1)
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                global_step += 1
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Batch Expansion for Parallel MoE
                if moe_mode == "parallel":
                    # Repeat inputs and targets K times
                    # inputs: (B, C, H, W) -> (K*B, C, H, W)
                    # targets: (B) -> (K*B)
                    inputs = inputs.repeat(nb_tasks, 1, 1, 1)
                    targets = targets.repeat(nb_tasks)
                
                # Forward
                outcome = self._network(inputs, task_id=self._cur_task)
                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]
                
                # Masking old classes to prevent forgetting in fc layer (no replay setting)
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float('inf')

                # Use mean reduction to average loss over the expanded batch (B*K).
                # This ensures gradients for shared parameters are not scaled up by K.
                loss = F.cross_entropy(logits, targets, reduction='mean')

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
        other_params = []
        fast_params_groups = {} # Key: expert_idx, Value: list of params
        
        nb_tasks = self.args.get("nb_tasks", 1)
        for i in range(nb_tasks):
            fast_params_groups[i] = []

        for n, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            
            if 'functional.slow' in n:
                slow_params.append(p)
            elif 'functional.fast_adapters' in n:
                # Identify which expert this param belongs to
                # Name format: ...functional.fast_adapters.0.down_proj.weight
                parts = n.split('.')
                try:
                    idx = int(parts[parts.index('fast_adapters') + 1])
                    fast_params_groups[idx].append(p)
                except (ValueError, IndexError):
                    # Fallback
                    logging.warning(f"Could not identify expert index for {n}, adding to default group 0")
                    fast_params_groups[0].append(p)
            elif 'functional.fast' in n:
                # Legacy single fast
                fast_params_groups[0].append(p)
            else:
                other_params.append(p)
        
        # Default LRs if not specified
        lr_slow = self.args.get('nested_lora_lr_slow', lr)
        base_lr_fast = self.args.get('nested_lora_lr_fast', lr * 5)
        
        param_groups = [
            {'params': slow_params, 'lr': lr_slow},
            {'params': other_params, 'lr': lr}
        ]
        
        # Add fast groups with diverse LRs
        moe_mode = self.args.get("moe_mode", "standard")
        if moe_mode == "parallel":
            # Diverse LRs: base_lr * [1.0, 0.5, 0.1, ...]
            # Or user defined multipliers
            multipliers = self.args.get("fast_lr_multipliers", None)
            if multipliers is None:
                # Auto generate: 1.0, 0.5, 0.25, ...
                multipliers = [1.0 / (2**i) for i in range(nb_tasks)]
            
            for i in range(nb_tasks):
                mult = multipliers[i] if i < len(multipliers) else multipliers[-1]
                lr_fast_i = base_lr_fast * mult
                param_groups.append({'params': fast_params_groups[i], 'lr': lr_fast_i})
                logging.info(f"Fast Expert {i} LR: {lr_fast_i}")
        else:
            # Standard: all fast params share same LR
            all_fast = []
            for group in fast_params_groups.values():
                all_fast.extend(group)
            param_groups.append({'params': all_fast, 'lr': base_lr_fast})
        
        if self.args['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.args["weight_decay"])
        elif self.args['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.args["weight_decay"])
            
        logging.info(f"[NestedLoRA] optimizer groups => slow: lr={lr_slow}, params={len(slow_params)}; other: lr={lr}, params={len(other_params)}")

        min_lr = self.args.get('min_lr', 1e-8) or 1e-8
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
