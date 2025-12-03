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

    def forward(self, x):
        func_out = self.functional(x)
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

    def forward(self, x):
        # Check if we should apply adapter at this layer
        if self.layer_id < self.adapt_start_layer or self.layer_id > self.adapt_end_layer:
            # No adapter for this layer
            return {"func_out": torch.zeros_like(x).to(device)}
        
        # Apply the single shared adapter
        func_out = self.adapter(x)
        return {"func_out": func_out}

    def end_of_task_training(self, do_consolidate=True):
        """
        Perform consolidation: slow += alpha * fast; fast = 0
        """
        use_consolidation = getattr(self.config, "nested_lora_use_consolidation", False)
        alpha = getattr(self.config, "nested_lora_consolidation_alpha", 0.1)
        
        if use_consolidation and do_consolidate:
            self.consolidate_nested_lora(alpha)
            
    def consolidate_nested_lora(self, alpha: float):
        with torch.no_grad():
            adapter = self.adapter
            if hasattr(adapter.functional, "slow") and hasattr(adapter.functional, "fast"):
                # slow += alpha * fast
                for p_s, p_f in zip(adapter.functional.slow.parameters(), adapter.functional.fast.parameters()):
                    p_s.add_(alpha * p_f)
                # fast = 0
                for p_f in adapter.functional.fast.parameters():
                    p_f.zero_()
        logging.info(f"Consolidated Nested LoRA at layer {self.layer_id} with alpha={alpha}")
