#!/bin/bash
set -e

# 1. Train to generate checkpoints (if not already done, but we assume we need to redo it or use existing if compatible)
# Since checkpoints dir was empty, we must train.
echo "Training Nested LoRA (Alpha 0.1) to generate checkpoints..."
conda run -n sema_env python main.py --config exps/nested_lora_alpha0.1.json

# # Checkpoint path (Task 9 is the last task, 0-indexed)
# CHECKPOINT="checkpoints/nested_lora_alpha0.1_seed1993_task9.pth"

# # 2. Eval Fast-Only
# echo "Running Fast-Only Ablation..."
# conda run -n sema_env python main.py --config exps/nested_lora_alpha0.1_fast_only.json \
#     --eval True --checkpt_path $CHECKPOINT

# # 3. Eval Slow-Only
# echo "Running Slow-Only Ablation..."
# conda run -n sema_env python main.py --config exps/nested_lora_alpha0.1_slow_only.json \
#     --eval True --checkpt_path $CHECKPOINT

# echo "Ablation experiments completed."
