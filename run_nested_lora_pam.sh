#!/bin/bash
set -e

# Step 1: Chain x Task Arithmetic (Baseline)
echo "Running Step 1: Chain x Task Arithmetic (Baseline)..."
conda run -n sema_env python main.py --config exps/step1_chain_ta_alpha0.1.json

# Step 1: Chain x PAM-lite
echo "Running Step 1: Chain x PAM-lite..."
conda run -n sema_env python main.py --config exps/step1_chain_pam_alpha0.1.json

echo "Step 1 experiments completed."
