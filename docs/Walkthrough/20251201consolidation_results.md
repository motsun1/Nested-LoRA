# Nested LoRA Consolidation Experiment Results

## Overview
This document summarizes the results of experiments testing the "Fast->Slow" knowledge transfer (consolidation) mechanism in Nested LoRA. The goal was to reproduce the mechanism described in `docs/NL.pdf` where "fast" weights are absorbed into "slow" weights with a decay factor $\alpha$.

## Experimental Setup
- **Dataset**: CIFAR-100 (10 tasks, 10 classes per task)
- **Model**: SEMA with Nested LoRA (Rank=16)
- **Baseline**: Alpha=0 (No consolidation)
- **Variations**:
    - $\alpha \in \{0.05, 0.1, 0.2\}$
    - Consolidation Interval $\in \{1, 5\}$ (Every task vs. Every 5 tasks)

## Results

| Configuration | Alpha | Interval | Avg Accuracy (10 tasks) | Final Task Acc (New) | Old Tasks Acc (Old) | Final Total Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0 | - | 86.19% | 84.9% | 80.23% | 80.7% |
| **Exp 1** | 0.05 | 1 | 85.50% | 82.1% | 79.78% | 80.01% |
| **Exp 2** | 0.1 | 1 | 86.19% | 84.9% | 80.23% | 80.7% |
| **Exp 3** | 0.2 | 1 | **87.61%** | 83.9% | **81.69%** | **81.91%** |
| **Exp 4** | 0.05 | 5 | 86.17% | 84.6% | 76.10% | 76.95% |
| **Exp 5** | 0.1 | 5 | 86.76% | 84.6% | 77.33% | 78.06% |
| **Exp 6** | 0.2 | 5 | 87.81% | 85.1% | 79.50% | 80.06% |

### Key Observations

1.  **Effect of Alpha**:
    - $\alpha=0.2$ with Interval=1 provided the best overall performance, improving the final total accuracy to **81.91%** compared to the baseline's 80.7%.
    - $\alpha=0.1$ with Interval=1 performed identically to the baseline in this run.
    - $\alpha=0.05$ with Interval=1 slightly degraded performance.

2.  **Effect of Interval**:
    - **Interval=1 (Every task)**: Generally resulted in better stability and higher final accuracy (Old Tasks retention).
    - **Interval=5**: While achieving high *average* accuracy over the course of training (e.g., Exp 6 has 87.81% avg), the *final* accuracy dropped significantly (e.g., Exp 6 final is 80.06% vs Exp 3's 81.91%). This suggests that infrequent consolidation might lead to larger "shocks" or forgetting when it does happen, or fails to stabilize the "slow" weights effectively for long-term retention.

3.  **Forgetting**:
    - The "Old Tasks Acc" metric is a good proxy for retention.
    - **Exp 3 ($\alpha=0.2, Int=1$)** achieved the highest Old Tasks Accuracy (**81.69%**), indicating it was most effective at mitigating forgetting.
    - Interval 5 experiments consistently showed lower Old Tasks Accuracy (76-79%), confirming higher forgetting.

## Conclusion
The "Fast->Slow" consolidation mechanism with **$\alpha=0.2$ applied at every task (Interval=1)** successfully improves knowledge retention and final accuracy compared to the baseline. This supports the hypothesis that transferring knowledge from the plastic "fast" module to the stable "slow" module helps in continual learning.

## Next Steps
- Consider testing higher $\alpha$ values (e.g., 0.3, 0.5) to see if the trend continues.
- Investigate why $\alpha=0.1$ (Int=1) had no visible effect compared to baseline (check if weights are actually changing significantly).
- Proceed with multi-level (3-tier) Nested LoRA implementation as planned.
