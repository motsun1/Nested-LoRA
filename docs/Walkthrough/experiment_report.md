# Nested LoRA v2 (Minimal) Experiment Report

## Experiment Setup
- **Dataset**: CIFAR-100 (Split into 10 tasks, 10 classes each), resized to 224x224.
- **Backbone**: ViT-B/16 (ImageNet-21k pretrained).
- **Method**: Nested LoRA v2 (Minimal)
    - **Baseline**: Single LoRA (Fast LR=0.01, Slow LR=0.0, No consolidation).
    - **Nested LoRA**: Fast LR=0.05, Slow LR=0.005.
        - **Alpha 0.1**: Consolidation `slow += 0.1 * fast`.
        - **Alpha 0.5**: Consolidation `slow += 0.5 * fast`.

## Results Summary

| Task | Baseline (Acc %) | Alpha 0.1 (Run 1) | Alpha 0.5 (Run 1) | Alpha 0.1 (Run 2 - Re-run) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 99.20 | 96.00 | 97.80 | 96.00 |
| 2 | 86.30 | 81.10 | 84.85 | 66.20 |
| 3 | 71.81 | 70.88 | 72.82 | 50.43 |
| 4 | 65.49 | 64.17 | 64.72 | 44.05 |
| 5 | 60.63 | 60.48 | 59.99 | 45.74 |
| 6 | 57.45 | 57.45 | 56.29 | 42.25 |
| 7 | 53.55 | 53.38 | 51.82 | 29.00 |
| 8 | 50.69 | 51.13 | 48.81 | 35.39 |
| 9 | 47.86 | 48.12 | 45.95 | 24.06 |
| 10 | **45.92** | **46.11** | 43.96 | **28.01** |

> **Note**: Run 2 (performed for ablation) showed a significant performance degradation compared to Run 1. This suggests instability or an issue introduced with the ablation code changes. The analysis below focuses on the ablation results from Run 2.

## Ablation Study (from Run 2)

We evaluated the contribution of Fast and Slow components using the checkpoint from Run 2 (Task 10).

| Configuration | Accuracy (%) | Interpretation |
| :--- | :--- | :--- |
| **Fast-Only** | **39.57** | Effectively **Frozen Backbone** (since Fast weights are reset to 0). |
| **Slow-Only** | **28.01** | **Backbone + Slow Adapter**. |
| **Combined** | **28.01** | **Backbone + Slow + Fast** (Fast is 0). |

### Analysis

1.  **Destructive Consolidation**:
    - The **Slow-Only** performance (28.01%) is significantly **worse** than the **Fast-Only** performance (39.57%).
    - Since "Fast-Only" with zero weights represents the frozen backbone baseline, this indicates that the **Slow Adapter is actively degrading the model's performance**.
    - The consolidation mechanism (`slow += 0.1 * fast`) appears to be accumulating noise or destructive interference in this run.

2.  **Comparison with Baseline**:
    - The Single LoRA Baseline (45.92%) improves upon the Frozen Backbone (39.57%).
    - The successful Alpha 0.1 Run 1 (46.11%) also improved upon it.
    - The failed Alpha 0.1 Run 2 (28.01%) failed to improve and caused damage.

## Conclusion
- **Instability**: The consolidation mechanism shows instability. While Run 1 showed promise (slight improvement over baseline), Run 2 resulted in severe degradation.
- **Harmful Slow Weights**: The ablation confirms that in the failed run, the accumulated "Slow" weights hurt the representation compared to doing nothing (Frozen Backbone).
- **Next Steps**: Investigate the cause of the degradation in Run 2. Check for code regressions or seed sensitivity.
