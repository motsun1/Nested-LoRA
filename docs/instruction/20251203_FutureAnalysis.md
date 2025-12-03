# Future Analysis Plan: Nested LoRA v2

Based on the initial experiments and ablation study, we have identified a critical discrepancy and some interesting behaviors. Here is the plan for the next phase of analysis.

## 1. Investigate the "Combined vs Slow-Only" Discrepancy

**Observation**: Combined Accuracy (46.11%) >> Slow-Only Accuracy (28.01%).
**Theory**: Combined = Slow + Fast. If Fast is reset to 0, Combined should equal Slow.
**Hypothesis**:
1.  **Fast weights are NOT zero**: The consolidation mechanism might be failing to zero the weights, or they are being updated after consolidation.
2.  **Checkpoint Mismatch**: The model in memory (Combined) might differ from the saved checkpoint (Slow-Only source).

**Action Items**:
- **Debug Consolidation**: Add logs to print `fast_norm` immediately after `zero_()` and before `eval_task`.
- **Verify Checkpoint**: Load the checkpoint and print `fast_norm`.
- **Check Optimizer**: Ensure the optimizer is not stepping after consolidation.

## 2. Analyze "Slow-Only" Degradation

**Observation**: Slow-Only (28.01%) < Fast-Only/Frozen (39.57%).
**Meaning**: The accumulated "Slow" weights are hurting the backbone's representation.
**Hypothesis**:
1.  **Destructive Accumulation**: `slow += alpha * fast` might be adding noise or conflicting features over time.
2.  **Alpha too high**: `alpha=0.1` might be too aggressive.
3.  **Drift**: The slow weights might be drifting away from the pretrained manifold.

**Action Items**:
- **Alpha Sweep**: Test smaller alphas (e.g., 0.01, 0.05) to see if degradation is reduced.
- **Orthogonality Check**: Measure the cosine similarity between `slow` and `fast` updates.
- **Feature Visualization**: Visualize the feature space (t-SNE/PCA) of Frozen vs Slow-Only to see how classes are separated.

## 3. Understand "Fast-Only" Performance

**Observation**: Fast-Only (39.57%) is the Frozen Backbone performance.
**Context**: Single LoRA Baseline (45.92%) improves on this.
**Goal**: We want Nested LoRA to improve on this significantly by leveraging "Slow" knowledge.

**Action Items**:
- **Fast-Only Training**: Run an experiment where we *only* train Fast (reset every task) and never update Slow (Slow=0). This establishes the "Independent Task Learning" baseline.

## 4. Refine the Method

If "Slow" accumulation is indeed harmful with simple addition:
- **Explore Alternatives**:
    - **EMA**: `slow = (1-beta)*slow + beta*fast` (instead of addition).
    - **Selective Merge**: Only merge `fast` components that are "important" (Fisher Information, SVD).
    - **Masking**: Apply a mask to `slow` weights to prevent interference.

## Summary
The immediate priority is to **solve the mystery of why Combined is good while Slow-Only is bad**. This will reveal the true mechanism currently at play (likely unintended non-zero Fast weights).
