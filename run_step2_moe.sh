#!/bin/bash
# Run Step 2 MoE Experiments

# 1. Task Arithmetic
echo "Running MoE + Task Arithmetic..."
conda run -n sema_env python main.py --config exps/step2_moe_ta.json

# 2. PAM-lite
echo "Running MoE + PAM-lite..."
conda run -n sema_env python main.py --config exps/step2_moe_pam.json
