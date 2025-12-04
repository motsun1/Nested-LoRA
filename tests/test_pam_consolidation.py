import torch
import torch.nn as nn
from easydict import EasyDict
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backbone.sema_block import SEMAModules
from backbone.sema_components import NestedLoRAAdapter

def test_pam_consolidation_logic():
    print("\n=== Testing PAM-lite Consolidation Logic ===")
    config = EasyDict(
        nested_lora_rank=4,
        d_model=32,
        attn_bn=16,
        nb_tasks=1,
        nested_lora_eval_ablation="none",
        nested_lora_use_consolidation=True,
        nested_lora_consolidation_alpha=1.0, # Use 1.0 for easy calculation
        adapt_start_layer=0,
        adapt_end_layer=1,
        ffn_adapter_type="nested_lora",
        ffn_num=16,
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_adapter_layernorm_option="none",
        exp_threshold=0.5,
        buffer_size=10,
        nested_lora_consolidation_method="pam", # Enable PAM
        rd_dim=16,
        rd_beta=0.5,
        rd_init_option="orthogonal"
    )
    
    # Mock SEMAModules
    # We need to mock the writer as well since SEMAModules requires it
    class MockWriter:
        def add_scalar(self, *args, **kwargs): pass
        
    module = SEMAModules(config, layer_id=0, writer=MockWriter())
    
    # Get the adapter
    # SEMAModules initializes with one adapter at index 0
    adapter = module.adapters[0]
    
    # Ensure it's a NestedLoRAAdapter
    assert isinstance(adapter.functional, NestedLoRAAdapter)
    
    # Initialize weights for testing PAM logic
    # We will test 4 cases in a single tensor:
    # 0: slow > 0, fast > 0 (Same sign) -> Merge
    # 1: slow > 0, fast < 0 (Opposite sign) -> No Merge
    # 2: slow < 0, fast < 0 (Same sign) -> Merge
    # 3: slow < 0, fast > 0 (Opposite sign) -> No Merge
    
    with torch.no_grad():
        # Reset all weights to 0 first
        adapter.functional.slow.down_proj.weight.zero_()
        adapter.functional.slow.up_proj.weight.zero_()
        adapter.functional.fast_adapters[0].down_proj.weight.zero_()
        adapter.functional.fast_adapters[0].up_proj.weight.zero_()
        
        # Set up values
        # We'll use the first 4 elements of down_proj.weight
        # slow: [1.0, 1.0, -1.0, -1.0]
        # fast: [0.5, -0.5, -0.5, 0.5]
        
        adapter.functional.slow.down_proj.weight[0, 0] = 1.0
        adapter.functional.slow.down_proj.weight[0, 1] = 1.0
        adapter.functional.slow.down_proj.weight[0, 2] = -1.0
        adapter.functional.slow.down_proj.weight[0, 3] = -1.0
        
        adapter.functional.fast_adapters[0].down_proj.weight[0, 0] = 0.5
        adapter.functional.fast_adapters[0].down_proj.weight[0, 1] = -0.5
        adapter.functional.fast_adapters[0].down_proj.weight[0, 2] = -0.5
        adapter.functional.fast_adapters[0].down_proj.weight[0, 3] = 0.5
        
    print("Before consolidation:")
    print(f"Slow: {adapter.functional.slow.down_proj.weight[0, :4]}")
    print(f"Fast: {adapter.functional.fast_adapters[0].down_proj.weight[0, :4]}")
    
    # Run consolidation
    module.end_of_task_training(do_consolidate=True)
    
    print("After consolidation:")
    print(f"Slow: {adapter.functional.slow.down_proj.weight[0, :4]}")
    print(f"Fast: {adapter.functional.fast_adapters[0].down_proj.weight[0, :4]}")
    
    # Verification
    slow_weights = adapter.functional.slow.down_proj.weight[0, :4]
    
    # Case 0: 1.0 + 0.5 = 1.5
    assert abs(slow_weights[0].item() - 1.5) < 1e-5, f"Case 0 failed: expected 1.5, got {slow_weights[0].item()}"
    
    # Case 1: 1.0 + 0.0 (masked) = 1.0
    assert abs(slow_weights[1].item() - 1.0) < 1e-5, f"Case 1 failed: expected 1.0, got {slow_weights[1].item()}"
    
    # Case 2: -1.0 + (-0.5) = -1.5
    assert abs(slow_weights[2].item() - -1.5) < 1e-5, f"Case 2 failed: expected -1.5, got {slow_weights[2].item()}"
    
    # Case 3: -1.0 + 0.0 (masked) = -1.0
    assert abs(slow_weights[3].item() - -1.0) < 1e-5, f"Case 3 failed: expected -1.0, got {slow_weights[3].item()}"
    
    # Check fast is reset
    fast_weights = adapter.functional.fast_adapters[0].down_proj.weight[0, :4]
    assert torch.all(fast_weights == 0), "Fast weights should be reset to 0"
    
    print("PAM-lite consolidation logic check passed.")

if __name__ == "__main__":
    test_pam_consolidation_logic()
