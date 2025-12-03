import torch
import torch.nn as nn
from easydict import EasyDict
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backbone.sema_components import NestedLoRAAdapter
from backbone.nested_lora_block import NestedLoRAModules
from backbone.vit_nested_lora import vit_base_patch16_224_nested_lora

def test_nested_lora_adapter_structure():
    print("\n=== Testing NestedLoRAAdapter Structure ===")
    config = EasyDict(
        nested_lora_rank=4,
        d_model=32,
        attn_bn=16, # bottleneck
        nb_tasks=3,
        nested_lora_eval_ablation="none"
    )
    adapter = NestedLoRAAdapter(config, "test_adapter")
    
    print(f"Slow adapter: {adapter.slow}")
    print(f"Fast adapters: {len(adapter.fast_adapters)}")
    
    assert hasattr(adapter, "slow")
    assert hasattr(adapter, "fast_adapters")
    assert len(adapter.fast_adapters) == 3
    print("Structure check passed.")

def test_nested_lora_forward_training():
    print("\n=== Testing NestedLoRAAdapter Forward (Training) ===")
    config = EasyDict(
        nested_lora_rank=4,
        d_model=32,
        attn_bn=16,
        nb_tasks=3,
        nested_lora_eval_ablation="none"
    )
    adapter = NestedLoRAAdapter(config, "test_adapter")
    adapter.train()
    
    x = torch.randn(2, 32)
    
    # Task 0
    out0 = adapter(x, task_id=0)
    # Task 1
    out1 = adapter(x, task_id=1)
    
    # Check if different fast adapters are used
    # We can check gradients or just output if weights are different
    # Let's manually modify weights to be sure
    with torch.no_grad():
        adapter.fast_adapters[0].down_proj.weight.fill_(1.0)
        adapter.fast_adapters[0].up_proj.weight.fill_(1.0)
        adapter.fast_adapters[1].down_proj.weight.fill_(2.0)
        adapter.fast_adapters[1].up_proj.weight.fill_(1.0)
        adapter.slow.down_proj.weight.fill_(0.0) # Ignore slow for now
        adapter.slow.up_proj.weight.fill_(0.0)
    
    out0 = adapter(x, task_id=0)
    out1 = adapter(x, task_id=1)
    
    print(f"Output Task 0 mean: {out0.mean().item()}")
    print(f"Output Task 1 mean: {out1.mean().item()}")
    
    assert not torch.allclose(out0, out1), "Outputs should be different for different tasks"
    print("Forward training check passed.")

def test_nested_lora_forward_inference():
    print("\n=== Testing NestedLoRAAdapter Forward (Inference) ===")
    config = EasyDict(
        nested_lora_rank=4,
        d_model=32,
        attn_bn=16,
        nb_tasks=3,
        nested_lora_eval_ablation="none"
    )
    adapter = NestedLoRAAdapter(config, "test_adapter")
    adapter.eval()
    
    x = torch.randn(2, 32)
    
    with torch.no_grad():
        adapter.fast_adapters[0].down_proj.weight.fill_(1.0)
        adapter.fast_adapters[0].up_proj.weight.fill_(1.0)
        adapter.fast_adapters[1].down_proj.weight.fill_(2.0)
        adapter.fast_adapters[1].up_proj.weight.fill_(1.0)
        adapter.fast_adapters[2].down_proj.weight.fill_(3.0)
        adapter.fast_adapters[2].up_proj.weight.fill_(1.0)
        adapter.slow.down_proj.weight.fill_(0.0)
        adapter.slow.up_proj.weight.fill_(0.0)
        
    out = adapter(x) # No task_id needed for inference
    
    # Should be sum of all fast adapters
    # fast0 + fast1 + fast2
    # Since we filled weights, let's just check it runs and returns something
    print(f"Output Inference mean: {out.mean().item()}")
    assert out.shape == x.shape
    print("Forward inference check passed.")

def test_consolidation():
    print("\n=== Testing Consolidation ===")
    config = EasyDict(
        nested_lora_rank=4,
        d_model=32,
        attn_bn=16,
        nb_tasks=3,
        nested_lora_eval_ablation="none",
        nested_lora_use_consolidation=True,
        nested_lora_consolidation_alpha=0.5,
        adapt_start_layer=0,
        adapt_end_layer=1
    )
    
    # Mock NestedLoRAModules
    module = NestedLoRAModules(config, layer_id=0)
    
    # Initialize weights
    with torch.no_grad():
        module.adapter.functional.slow.down_proj.weight.fill_(1.0)
        module.adapter.functional.slow.up_proj.weight.fill_(1.0)
        module.adapter.functional.fast_adapters[0].down_proj.weight.fill_(2.0)
        module.adapter.functional.fast_adapters[0].up_proj.weight.fill_(1.0)
        module.adapter.functional.fast_adapters[1].down_proj.weight.fill_(3.0)
        module.adapter.functional.fast_adapters[1].up_proj.weight.fill_(1.0)
    
    print("Before consolidation task 0:")
    print(f"Slow: {module.adapter.functional.slow.down_proj.weight[0,0].item()}")
    print(f"Fast 0: {module.adapter.functional.fast_adapters[0].down_proj.weight[0,0].item()}")
    
    # Consolidate Task 0
    module.end_of_task_training(do_consolidate=True, task_id=0)
    
    print("After consolidation task 0:")
    slow_val = module.adapter.functional.slow.down_proj.weight[0,0].item()
    fast0_val = module.adapter.functional.fast_adapters[0].down_proj.weight[0,0].item()
    fast1_val = module.adapter.functional.fast_adapters[1].down_proj.weight[0,0].item()
    
    print(f"Slow: {slow_val}")
    print(f"Fast 0: {fast0_val}")
    print(f"Fast 1: {fast1_val}")
    
    # Expected: Slow = 1.0 + 0.5 * 2.0 = 2.0
    # Fast 0 = 0.0
    # Fast 1 = 3.0 (unchanged)
    
    assert abs(slow_val - 2.0) < 1e-5
    assert abs(fast0_val - 0.0) < 1e-5
    assert abs(fast1_val - 3.0) < 1e-5
    
    print("Consolidation check passed.")

if __name__ == "__main__":
    test_nested_lora_adapter_structure()
    test_nested_lora_forward_training()
    test_nested_lora_forward_inference()
    test_consolidation()
    print("\nAll tests passed!")
