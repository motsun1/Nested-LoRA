# Experiment Report: Step 1 (Chain) & Step 2 (MoE)

## 1. Overview
We conducted experiments to compare:
1.  **Structure**: Chain (Single Fast) vs. MoE (Parallel Multi-Fast).
2.  **Consolidation**: Task Arithmetic (TA) vs. PAM-lite.

## 2. Results Summary

| ID | Structure | Consolidation | Avg Accuracy (CNN) |
| :--- | :--- | :--- | :--- |
| Step 1 | Chain | Task Arithmetic | **86.88%** |
| Step 1 | Chain | PAM-lite | **86.88%** |
| Step 2 | MoE (4 Experts) | Task Arithmetic | 86.84% |
| Step 2 | MoE (4 Experts) | PAM-lite | 86.84% |

## 3. Analysis

### Observation 1: TA vs. PAM-lite yields identical results
In both Step 1 and Step 2, the average accuracy for Task Arithmetic and PAM-lite is **exactly identical** (down to the floating point precision).
-   **Step 1**: 86.87500000000001 (Both)
-   **Step 2**: 86.843 (Both)

**Implication**:
This strongly suggests that the **PAM-lite logic was not effectively applied**.
Possible causes:
1.  **Configuration Issue**: The `nested_lora_consolidation_method` parameter might not be correctly propagated to the `NestedLoRAModules`.
2.  **Logic Issue**: The condition `p_s * p_f >= 0` might be always true (highly unlikely) or the masking logic has a bug.
3.  **Implementation Bug**: The code path for "pam" might not be reachable or is functionally equivalent to TA due to a coding error.

### Observation 2: MoE vs. Chain difference is negligible (-0.03%)
The difference between Chain (86.88%) and MoE (86.84%) is insignificant and slightly negative.
-   **Step 1 (Chain)**: 86.88%
-   **Step 2 (MoE)**: 86.84%

**Implication**:
The current MoE implementation (Parallel Competition) is not providing a benefit over the simple Chain structure.
Possible causes:
1.  **Insufficient Diversity**: Even with diverse LRs, the experts might be converging to very similar solutions, leading to the "same" expert being selected or all experts being effectively the same.
2.  **Selection Instability**: The validation accuracy based selection might be noisy or not correlated well with long-term retention.
3.  **Consolidation Frequency**: Consolidating every task might be too frequent, washing out the benefits of the "expert" specialization.

## 4. Next Steps

### Immediate Actions (Debugging)
1.  **Verify PAM-lite**: Add debug prints in `consolidate_nested_lora` to confirm:
    -   Which method is being used (`method` variable).
    -   If "pam", how many elements are being masked (sparsity check).
2.  **Verify MoE Diversity**:
    -   Check the `selection_history` logs (already added) to see if different experts are actually winning.
    -   Calculate and log the cosine similarity between experts before selection to see if they are distinct.

### Strategic Adjustments
1.  **Force Diversity**: If experts are too similar, introduce stronger diversity mechanisms (e.g., different initialization seeds per expert, different data subsets if possible, or stronger regularization differences).
2.  **Tuning**: Adjust `alpha` or consolidation interval.
