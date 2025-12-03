# Multi-LoRA + MoE (v0 Design) Implementation Plan

## Goal Description
Implement the "v0 design" of Multi-LoRA + MoE as described in `docs/instruction/20251203ProjectDirection2.md`.
This involves:
1.  **Structure**: 1 shared Slow LoRA + K Fast LoRAs (where K = number of tasks).
2.  **Routing**:
    -   **Training**: Use the Fast LoRA corresponding to the current task (`task_id`).
    -   **Inference**: Use Slow LoRA + Sum/Average of all Fast LoRAs (since task ID is unknown).
3.  **Consolidation**:
    -   At the end of each task, consolidate the *current task's* Fast LoRA into the shared Slow LoRA.
    -   Reset (or partially reset) the current Fast LoRA.

## User Review Required
> [!IMPORTANT]
> I will assume `K = args["nb_tasks"]` for the number of Fast LoRA instances.
> I will implement the inference routing as `slow + sum(fast_k)` for now.

## Proposed Changes

### Backbone Components
#### [MODIFY] [sema_components.py](file:///home/mitsui/Desktop/SEMA-CL/backbone/sema_components.py)
-   Modify `NestedLoRAAdapter` (or create `MoENestedLoRAAdapter`) to support multiple Fast LoRAs.
-   Constructor will take `nb_tasks` from config.
-   `forward` will accept `task_id`.
    -   If `training`: `slow + fast[task_id]`.
    -   If `eval`: `slow + sum(fast)`.

#### [MODIFY] [nested_lora_block.py](file:///home/mitsui/Desktop/SEMA-CL/backbone/nested_lora_block.py)
-   Update `AdapterModule` to use the new adapter class.
-   Update `NestedLoRAModules` to accept `task_id` in `forward` and pass it down.
-   Update `end_of_task_training` to consolidate only the specific `fast_k` (using `cur_task` from somewhere, or passing it in).

### Vision Transformer
#### [MODIFY] [vit_nested_lora.py](file:///home/mitsui/Desktop/SEMA-CL/backbone/vit_nested_lora.py)
-   Update `Block.forward` to accept `task_id`.
-   Update `VisionTransformer.forward` and `forward_features` to accept `task_id`.

### Utils
#### [MODIFY] [inc_net.py](file:///home/mitsui/Desktop/SEMA-CL/utils/inc_net.py)
-   Update `NestedLoRAVitNet.forward` to accept `task_id`.

### Model
#### [MODIFY] [nested_lora.py](file:///home/mitsui/Desktop/SEMA-CL/models/nested_lora.py)
-   Update `_train` loop to pass `self._cur_task` as `task_id` to the model.
-   Update `update_optimizer_and_scheduler` to include parameters from all Fast LoRAs.
-   Update `end_of_task_training` call to pass the current task ID so consolidation happens on the correct Fast LoRA.

## Verification Plan

### Automated Tests
-   Create a test script `tests/test_moe_nested_lora.py` that:
    1.  Instantiates `NestedLoRAVitNet` with `nb_tasks=3`.
    2.  Checks if there are 3 Fast LoRAs and 1 Slow LoRA.
    3.  Runs a forward pass with `task_id=0` and checks if gradients flow only to `fast[0]` and `slow`.
    4.  Runs a forward pass with `task_id=1` and checks if gradients flow only to `fast[1]` and `slow`.
    5.  Runs a forward pass in eval mode and checks if output involves all fast adapters.
    6.  Simulates consolidation for task 0 and checks if `slow` is updated and `fast[0]` is reset.

### Manual Verification
-   Run a short experiment (e.g., 2 tasks, few epochs) using `run_experiments.sh` (modified for this test) to ensure the training loop runs without errors and loss decreases.
