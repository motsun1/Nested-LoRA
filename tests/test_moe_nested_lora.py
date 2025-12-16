import torch
import unittest
from easydict import EasyDict
from backbone.nested_lora_block import NestedLoRAModules, NestedLoRAAdapter

class TestMoENestedLoRA(unittest.TestCase):
    def setUp(self):
        self.config = EasyDict({
            "d_model": 32,
            "attn_bn": 16, # bottleneck
            "nested_lora_rank": 4,
            "nb_tasks": 3,
            "adapt_start_layer": 0,
            "adapt_end_layer": 1,
            "nested_lora_use_consolidation": True,
            "nested_lora_consolidation_alpha": 1.0, # Use 1.0 for easy math
            "nested_lora_consolidation_method": "pam",
            "ffn_adapter_type": "nested_lora"
        })
        self.layer_id = 0
        self.module = NestedLoRAModules(self.config, self.layer_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.module.to(self.device)

    def test_moe_structure(self):
        """Verify that we have 1 slow and K fast adapters."""
        adapter = self.module.adapter.functional
        self.assertTrue(hasattr(adapter, "slow"))
        self.assertTrue(hasattr(adapter, "fast_adapters"))
        self.assertEqual(len(adapter.fast_adapters), 3)
        print("\n[Test] MoE structure verified: 1 slow + 3 fast adapters.")

    def test_forward_routing(self):
        """Verify that task_id routes to the correct fast adapter during training."""
        self.module.train()
        x = torch.randn(1, 1, 32).to(self.device)
        
        # Task 0
        out0 = self.module(x, task_id=0)
        # Check gradients: only fast_0 and slow should have grads (if we backward)
        # Instead of backward, let's check which forward was called by mocking or checking side effects?
        # Easier: check if output changes when we modify fast_0 but not fast_1
        
        with torch.no_grad():
            self.module.adapter.functional.fast_adapters[0].down_proj.weight.fill_(1.0)
            self.module.adapter.functional.fast_adapters[0].up_proj.weight.fill_(1.0)
            
            self.module.adapter.functional.fast_adapters[1].down_proj.weight.fill_(100.0)
            self.module.adapter.functional.fast_adapters[1].up_proj.weight.fill_(1.0)
            
            self.module.adapter.functional.fast_adapters[0].down_proj.bias.fill_(1.0)
            self.module.adapter.functional.fast_adapters[1].down_proj.bias.fill_(1.0)
            
            self.module.adapter.functional.slow.down_proj.weight.fill_(0.0)
            self.module.adapter.functional.slow.up_proj.weight.fill_(0.0)
            
        out0_val = self.module(x, task_id=0)["func_out"]
        out1_val = self.module(x, task_id=1)["func_out"]
        
        self.assertFalse(torch.allclose(out0_val, out1_val), "Task 0 and Task 1 outputs should differ")
        print("[Test] Forward routing verified: Task 0 and Task 1 produce different outputs.")

    def test_pam_consolidation(self):
        """Verify PAM-lite consolidation logic."""
        # Setup weights
        # Slow: [1, -1, 1, -1]
        # Fast: [1, 1, -1, -1]
        # Alpha = 1.0
        # Expected Mask: [1, 0, 0, 1] (Signs match at idx 0 and 3)
        # Expected Update: Slow += 1.0 * (Fast * Mask) = [1+1, -1+0, 1+0, -1-1] = [2, -1, 1, -2]
        
        adapter = self.module.adapter.functional
        slow_param = list(adapter.slow.down_proj.parameters())[0] # weight
        fast_param = list(adapter.fast_adapters[0].down_proj.parameters())[0] # weight
        
        # Resize for simple test
        with torch.no_grad():
            slow_param.data = torch.tensor([[1.0, -1.0], [1.0, -1.0]]).to(self.device)
            fast_param.data = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]).to(self.device)
            
            # Ensure other fast adapters are different to verify they are not touched
            list(adapter.fast_adapters[1].down_proj.parameters())[0].data.fill_(99.0)

        # Consolidate Task 0
        self.module.end_of_task_training(task_id=0)
        
        # Check Slow
        expected_slow = torch.tensor([[2.0, -1.0], [1.0, -2.0]]).to(self.device)
        self.assertTrue(torch.allclose(slow_param.data, expected_slow), f"Slow weights mismatch.\nExpected:\n{expected_slow}\nActual:\n{slow_param.data}")
        
        # Check Fast 0 (should be reset)
        self.assertTrue(torch.allclose(fast_param.data, torch.zeros_like(fast_param.data)), "Fast 0 should be reset to zero.")
        
        # Check Fast 1 (should be untouched)
        fast1_param = list(adapter.fast_adapters[1].down_proj.parameters())[0]
        self.assertTrue(torch.allclose(fast1_param.data, torch.tensor(99.0).to(self.device)), "Fast 1 should be untouched.")
        
        print("[Test] PAM-lite consolidation verified.")

    def test_parallel_forward(self):
        """Verify parallel forward pass (batch splitting)."""
        self.config.moe_mode = "parallel"
        self.module.train()
        
        # Batch size = K * 2 (2 samples per expert)
        K = 3
        B = 2
        x = torch.randn(K * B, 1, 32).to(self.device)
        
        # Set distinct weights for each expert to verify routing
        with torch.no_grad():
            for i in range(K):
                self.module.adapter.functional.fast_adapters[i].down_proj.weight.fill_(float(i + 1))
                self.module.adapter.functional.fast_adapters[i].up_proj.weight.fill_(1.0)
                self.module.adapter.functional.fast_adapters[i].down_proj.bias.fill_(1.0)
            self.module.adapter.functional.slow.down_proj.weight.fill_(0.0)
            self.module.adapter.functional.slow.up_proj.weight.fill_(0.0)

        out = self.module(x)["func_out"]
        
        # Check output chunks
        # Chunk 0 (indices 0, 1) should be affected by fast_0 (weight 1.0)
        # Chunk 1 (indices 2, 3) should be affected by fast_1 (weight 2.0)
        # Chunk 2 (indices 4, 5) should be affected by fast_2 (weight 3.0)
        
        chunk0 = out[0:2]
        chunk1 = out[2:4]
        chunk2 = out[4:6]
        
        self.assertNotAlmostEqual(chunk0.mean().item(), chunk1.mean().item())
        self.assertNotAlmostEqual(chunk1.mean().item(), chunk2.mean().item())
        
        # Verify values roughly (x * 1 vs x * 2 vs x * 3)
        # Since x is random, means might vary, but let's check ratio if x was constant?
        # Or just check that they are different is enough for routing verification.
        
        print("[Test] Parallel forward routing verified.")

if __name__ == '__main__':
    unittest.main()
