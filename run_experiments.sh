#!/bin/bash
set -e

# Alpha 0 (Baseline)
echo "Running Alpha 0 (Baseline)..."
conda run -n sema_env python main.py --config exps/exp_alpha0.json

# Alpha 0.05, Interval 1
echo "Running Alpha 0.05, Interval 1..."
conda run -n sema_env python main.py --config exps/exp_alpha0.05_int1.json

# Alpha 0.1, Interval 1
echo "Running Alpha 0.1, Interval 1..."
conda run -n sema_env python main.py --config exps/exp_alpha0.1_int1.json

# Alpha 0.2, Interval 1
echo "Running Alpha 0.2, Interval 1..."
conda run -n sema_env python main.py --config exps/exp_alpha0.2_int1.json

# Alpha 0.05, Interval 5
echo "Running Alpha 0.05, Interval 5..."
conda run -n sema_env python main.py --config exps/exp_alpha0.05_int5.json

# Alpha 0.1, Interval 5
echo "Running Alpha 0.1, Interval 5..."
conda run -n sema_env python main.py --config exps/exp_alpha0.1_int5.json

# Alpha 0.2, Interval 5
echo "Running Alpha 0.2, Interval 5..."
conda run -n sema_env python main.py --config exps/exp_alpha0.2_int5.json

echo "All experiments completed."
