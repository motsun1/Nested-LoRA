#!/bin/bash
set -e

# Baseline (Single LoRA)
echo "Running Baseline (Single LoRA)..."
conda run -n sema_env python main.py --config exps/nested_lora_baseline.json

# Nested LoRA v2 (Alpha 0.1)
echo "Running Nested LoRA v2 (Alpha 0.1)..."
conda run -n sema_env python main.py --config exps/nested_lora_alpha0.1.json

# Nested LoRA v2 (Alpha 0.5)
echo "Running Nested LoRA v2 (Alpha 0.5)..."
conda run -n sema_env python main.py --config exps/nested_lora_alpha0.5.json

echo "All Nested LoRA v2 experiments completed."
