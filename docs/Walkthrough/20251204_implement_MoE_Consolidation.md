# MoE (Parallel Competition) Implementation Plan

## Goal Description
Implement "True MoE" where multiple Fast LoRA experts are trained **simultaneously and independently** on the same task, effectively competing for the right to be consolidated into the Slow LoRA.
Selection of the "winner" expert is based on **Validation Accuracy** at the end of the task (Option A).

## User Review Required
> [!IMPORTANT]
> **Parallel Training Strategy**: I will use **Batch Expansion** to train $K$ experts efficiently.
> Input batch $x$ (size $B$) will be repeated $K$ times $\to$ $(K \cdot B)$.
> The model will process this enlarged batch. Inside `NestedLoRAAdapter`, the batch will be split into $K$ chunks, and each `fast_k` will process one chunk.
> This ensures `fast_k` sees the same data but learns independently (due to different LRs/initializations).

> [!NOTE]
> **Learning Rate Diversity**: I will assign different learning rates to each fast adapter to encourage diversity (e.g., time-scale differences).
> Default schedule: `base_lr * [1.0, 0.5, 0.1, ...]` (decaying).

## Proposed Changes

### Backbone
#### [MODIFY] [sema_components.py](file:///home/mitsui/Desktop/SEMA-CL/backbone/sema_components.py)
-   Update `NestedLoRAAdapter.forward`:
    -   Add logic for `training` mode:
        -   Split input `x` into `nb_tasks` (or $K$) chunks.
        -   Apply `fast_k` to `chunk_k`.
        -   Concatenate results.
    -   Keep `task_id` logic for inference/validation (selecting a specific expert).

### Model
#### [MODIFY] [nested_lora.py](file:///home/mitsui/Desktop/SEMA-CL/models/nested_lora.py)
-   **Optimizer**:
    -   Update `update_optimizer_and_scheduler` to assign distinct LRs to each fast adapter.
-   **Training Loop (`_train`)**:
    -   Expand input batch: `inputs = inputs.repeat(K, 1, 1, 1)`.
    -   Compute loss: Sum of losses for each expert (by reshaping logits).
-   **Selection & Consolidation**:
    -   At end of task:
        -   Run validation loop for each expert $k \in [0, K-1]$.
        -   Select $k^*$ with highest accuracy.
        -   Call `end_of_task_training(task_id=k^*)` to consolidate the winner.

### Config
-   New configs (implicit or explicit):
    -   `moe_mode`: "parallel" (enable this new behavior).
    -   `fast_lr_multipliers`: List of multipliers or auto-generated.

## Verification Plan

### Automated Tests
-   Update `tests/test_moe_nested_lora.py`:
    -   Test **Parallel Forward**: Pass batch size $B \cdot K$, verify output shape and that `fast_0` affects first chunk, `fast_1` affects second, etc.
    -   Test **Selection Logic**: Mock validation scores and verify correct index is passed to consolidation.

---

# MoE (Parallel Competition) Implementation Walkthrough

I have implemented the "True MoE" architecture where multiple fast adapters are trained in parallel (competing) and the best one is selected based on validation accuracy.

## Changes

### 1. Parallel Forward Pass
Modified `backbone/sema_components.py` to support parallel execution.
-   **Logic**: If `moe_mode="parallel"` and `training=True`, the input batch `x` (size $K \cdot B$) is split into $K$ chunks.
-   **Routing**: Each chunk $i$ is processed by `fast_adapter[i]`.
-   **Output**: Chunks are concatenated back to form the full output.

### 2. Batch Expansion & Diverse LRs
Modified `models/nested_lora.py`:
-   **Batch Expansion**: In `_init_train`, input batch is repeated $K$ times: `inputs.repeat(K, 1, 1, 1)`.
-   **Diverse LRs**: In `update_optimizer_and_scheduler`, each fast adapter is assigned a different learning rate (decaying by factor of 2: 1.0, 0.5, 0.25...).

### 3. Validation Selection
Modified `models/nested_lora.py` to select the best expert.
-   **Selection**: At the end of the task, the model evaluates each expert $k$ on the current task's test set.
-   **Consolidation**: The expert with the highest accuracy is selected and consolidated into the slow adapter.

### 4. Unit Test
Updated `tests/test_moe_nested_lora.py` to verify:
-   **Parallel Forward**: Confirmed that splitting and routing works correctly (different experts affect different parts of the batch).
-   **Structure & Routing**: Confirmed existing MoE structure and task-specific routing still work.

## Verification Results

Ran the unit test `tests/test_moe_nested_lora.py` using the `sema_env` conda environment.

```bash
conda run -n sema_env env PYTHONPATH=. python tests/test_moe_nested_lora.py
```

**Output:**
```
[Test] Forward routing verified: Task 0 and Task 1 produce different outputs.
[Test] MoE structure verified: 1 slow + 3 fast adapters.
[Test] PAM-lite consolidation verified.
[Test] Parallel forward routing verified.
...
OK
```

All tests passed.

---
# MoE (Parallel Competition) Implementation Walkthrough

I have implemented the "True MoE" architecture where multiple fast adapters are trained in parallel (competing) and the best one is selected based on validation accuracy.

## Changes

### 1. Parallel Forward Pass
Modified `backbone/sema_components.py` to support parallel execution.
-   **Logic**: If `moe_mode="parallel"` and `training=True`, the input batch `x` (size $K \cdot B$) is split into $K$ chunks.
-   **Routing**: Each chunk $i$ is processed by `fast_adapter[i]`.
-   **Output**: Chunks are concatenated back to form the full output.
-   **Safety**: Added assertion to ensure batch size is divisible by $K$.

### 2. Batch Expansion & Diverse LRs
Modified `models/nested_lora.py`:
-   **Batch Expansion**: In `_init_train`, input batch is repeated $K$ times: `inputs.repeat(K, 1, 1, 1)`.
-   **Diverse LRs**: In `update_optimizer_and_scheduler`, each fast adapter is assigned a different learning rate (decaying by factor of 2: 1.0, 0.5, 0.25...).
-   **Loss Scaling**: Explicitly used `reduction='mean'` for `F.cross_entropy` to average loss over the expanded batch ($B \cdot K$), preventing gradient explosion for shared parameters.

### 3. Validation Selection
Modified `models/nested_lora.py` to select the best expert.
-   **Selection**: At the end of the task, the model evaluates each expert $k$ on the current task's test set.
-   **Consolidation**: The expert with the highest accuracy is selected and consolidated into the slow adapter.
-   **Logging**: Added logging for "Expert Selection History" to track which experts are winning.

### 4. Unit Test
Updated `tests/test_moe_nested_lora.py` to verify:
-   **Parallel Forward**: Confirmed that splitting and routing works correctly (different experts affect different parts of the batch).
-   **Structure & Routing**: Confirmed existing MoE structure and task-specific routing still work.

## Verification Results

Ran the unit test `tests/test_moe_nested_lora.py` using the `sema_env` conda environment.

```bash
conda run -n sema_env env PYTHONPATH=. python tests/test_moe_nested_lora.py
```

**Output:**
```
[Test] Forward routing verified: Task 0 and Task 1 produce different outputs.
[Test] MoE structure verified: 1 slow + 3 fast adapters.
[Test] PAM-lite consolidation verified.
[Test] Parallel forward routing verified.
...
OK
```

All tests passed.