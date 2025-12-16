import torch
from torch import nn
from typing import List
import copy
import logging
from backbone.sema_components import NestedLoRAAdapter

def _select_device():
    if torch.cuda.is_available():
        try:
            _ = torch.cuda.device_count()
            return 'cuda'
        except Exception as e:
            logging.warning(f"CUDA appears unavailable, fallback to CPU. reason={e}")
    return 'cpu'

device = _select_device()

class AdapterModule(nn.Module):    
    def __init__(self, config, adapter_id):
        super().__init__()
        self.config = config
        # Always use NestedLoRAAdapter for this minimal implementation
        self.functional = NestedLoRAAdapter(
            config=config,
            adapter_id=adapter_id,
            dropout=0.1
        )
        self.adapter_id = adapter_id

    def forward(self, x, task_id=None):
        func_out = self.functional(x, task_id=task_id)
        return func_out

class NestedLoRAModules(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer
        
        # In minimal v2, we only have ONE shared adapter per block.
        # It is added at initialization and never grows.
        self.adapter = self._create_adapter()
        
        logging.info(f"NestedLoRAModules initialized at layer {layer_id}")

    def _create_adapter(self):
        adapter_id = f"{self.layer_id}.0"
        return AdapterModule(self.config, adapter_id).to(device)

    def forward(self, x, task_id=None):
        # Check if we should apply adapter at this layer
        if self.layer_id < self.adapt_start_layer or self.layer_id > self.adapt_end_layer:
            # No adapter for this layer
            return {"func_out": torch.zeros_like(x).to(device)}
        
        # Apply the single shared adapter
        func_out = self.adapter(x, task_id=task_id)
        return {"func_out": func_out}

    def end_of_task_training(self, do_consolidate=True, task_id=None):
        """
        Perform consolidation: slow += alpha * fast_k; fast_k = 0
        """
        use_consolidation = getattr(self.config, "nested_lora_use_consolidation", False)
        alpha = getattr(self.config, "nested_lora_consolidation_alpha", 0.1)
        
        if use_consolidation and do_consolidate:
            self.consolidate_nested_lora(alpha, task_id=task_id)
            
    def consolidate_nested_lora(self, alpha: float, task_id=None):
        method = getattr(self.config, "nested_lora_consolidation_method", "task_arithmetic")
        
        with torch.no_grad():
            adapter = self.adapter
            
            # Determine which fast adapter to consolidate
            if hasattr(adapter.functional, "slow") and hasattr(adapter.functional, "fast_adapters"):
                if task_id is None:
                    # Default to 0 if not specified (should be specified)
                    logging.warning("Consolidating Nested LoRA but task_id is None. Defaulting to fast_adapters[0].")
                    fast_adapter = adapter.functional.fast_adapters[0]
                else:
                    nb_tasks = len(adapter.functional.fast_adapters)
                    idx = task_id % nb_tasks
                    fast_adapter = adapter.functional.fast_adapters[idx]
                
                # slow += alpha * fast
                suppressed_elems = 0
                total_elems = 0
                fast_nonzero = 0
                for p_s, p_f in zip(adapter.functional.slow.parameters(), fast_adapter.parameters()):
                    if method == "pam":
                        # PAM-lite: Only merge if signs match (or one is zero)
                        # mask = 1 if p_s * p_f >= 0 else 0
                        mask = (p_s * p_f >= 0).float()
                        total_elems += mask.numel()
                        suppressed_elems += (mask.numel() - mask.sum()).item()
                        fast_nonzero += (p_f != 0).sum().item()
                        p_f_aligned = p_f * mask
                        p_s.add_(alpha * p_f_aligned)
                    else:
                        # Default Task Arithmetic
                        p_s.add_(alpha * p_f)

                # fast = 0
                for p_f in fast_adapter.parameters():
                    p_f.zero_()

                if method == "pam":
                    allowed = total_elems - suppressed_elems
                    blocked_ratio = (suppressed_elems / total_elems * 100) if total_elems > 0 else 0.0
                    logging.info(
                        f"[NestedLoRA][Layer {self.layer_id}][Fast {idx}] PAM-lite merge: alpha={alpha}, allowed={allowed}, "
                        f"suppressed={suppressed_elems} ({blocked_ratio:.2f}% blocked), fast_nonzero={fast_nonzero}"
                    )
                else:
                    logging.info(
                        f"[NestedLoRA][Layer {self.layer_id}] Consolidated fast_{idx} -> slow with alpha={alpha}, method={method}"
                    )
            elif hasattr(adapter.functional, "slow") and hasattr(adapter.functional, "fast"):
                 # Legacy support for single fast adapter
                suppressed_elems = 0
                total_elems = 0
                fast_nonzero = 0
                for p_s, p_f in zip(adapter.functional.slow.parameters(), adapter.functional.fast.parameters()):
                    if method == "pam":
                        mask = (p_s * p_f >= 0).float()
                        total_elems += mask.numel()
                        suppressed_elems += (mask.numel() - mask.sum()).item()
                        fast_nonzero += (p_f != 0).sum().item()
                        p_f_aligned = p_f * mask
                        p_s.add_(alpha * p_f_aligned)
                    else:
                        p_s.add_(alpha * p_f)
                        
                for p_f in adapter.functional.fast.parameters():
                    p_f.zero_()

                if method == "pam":
                    allowed = total_elems - suppressed_elems
                    blocked_ratio = (suppressed_elems / total_elems * 100) if total_elems > 0 else 0.0
                    logging.info(
                        f"[NestedLoRA][Layer {self.layer_id}] PAM-lite merge (single fast): alpha={alpha}, allowed={allowed}, "
                        f"suppressed={suppressed_elems} ({blocked_ratio:.2f}% blocked), fast_nonzero={fast_nonzero}"
                    )
                else:
                    logging.info(
                        f"[NestedLoRA][Layer {self.layer_id}] Consolidated single fast -> slow with alpha={alpha}, method={method}"
                    )
