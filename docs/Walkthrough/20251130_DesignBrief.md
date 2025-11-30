# Nested LoRA Design Brief (SEMA-CL)

Audience: lab seminar overview of the current Nested LoRA implementation on SEMA.  
Scope: design/engineering summary, not experimental results.

## 1. Goal & Context
- Base system: SEMA (Mixture-of-Adapters on ViT for class-incremental learning).
- Idea: add dual time-scale adapters (Fast/Slow LoRA) per layer to balance plasticity/stability without breaking SEMA routing.

## 2. Where the code lives
- Adapter definition: `backbone/sema_components.py` (`NestedLoRAAdapter`, wrapping two `Adapter` instances: `fast`/`slow`).
- Adapter module integration: `backbone/sema_block.py` (`AdapterModule` chooses Nested LoRA when `ffn_adapter_type == "nested_lora"`; consolidation hook).
- Backbone wiring: `utils/inc_net.py` (passes Nested LoRA hyperparams into ViT tuning config).
- Optimizer/scheduler + training loop: `models/sema.py` (param groups, slow-update interval, task loops, norm logging).
- Configs: `exps/nested_lora_cifar.json`, `exps/nested_lora_inr_10task.json` (and debug/baseline variants).

## 3. Module design
- `NestedLoRAAdapter`:
  - Two parallel low-rank adapters (`fast`, `slow`), both `Adapter` with LoRA-style init.
  - Forward: `out = slow(x) + fast(x)` (no router inside; SEMA router remains outside).
- `AdapterModule`:
  - If `ffn_adapter_type == "nested_lora"`, uses `NestedLoRAAdapter`; otherwise standard adapter.
  - Representation descriptor (AE) unchanged; only functional adapter swapped.
- `SEMAModules` (per layer):
  - Manages multiple `AdapterModule` plus router. Adds adapters on outlier detection as before.
  - Consolidation (optional, v1): at task end, `slow += alpha * fast; fast = 0` when `nested_lora_use_consolidation` is true.
  - CPU fallback for router allocation (in case CUDA init fails).

## 4. Training/Optimization behavior
- Param groups (when `ffn_adapter_type == "nested_lora"`):
  - Slow params: lr = `nested_lora_lr_slow`
  - Fast params: lr = `nested_lora_lr_fast`
  - Other functional/router/fc: lr = `init_lr`
- Slow update interval:
  - If `nested_lora_update_interval_slow > 1`, slow grads are zeroed on non-update steps → fast updates every step; slow updates sparsely.
- Logging:
  - Optimizer group lr/param-count.
  - Fast/slow L2 norms at task boundaries (pre/post consolidation).

## 5. Key hyperparameters (JSON)
- `ffn_adapter_type`: `"nested_lora"` to enable.
- `nested_lora_rank`
- `nested_lora_lr_fast`, `nested_lora_lr_slow`
- `nested_lora_update_interval_slow` (int; 1 = every step)
- `nested_lora_use_consolidation` (bool), `nested_lora_consolidation_alpha`
- Existing SEMA knobs reused: `ffn_num`, `exp_threshold`, `adapt_start_layer`, `adapt_end_layer`, etc.

## 6. How to run (examples)
- CIFAR-100 10-task: `python main.py --config exps/nested_lora_cifar.json`
- ImageNet-R 10-task: `python main.py --config exps/nested_lora_inr_10task.json`
- Debug short run: `python main.py --config exps/debug_nested_lora.json`

## 7. Limitations / next engineering steps
- Consolidation exists but needs experiments; logging shows norms but not per-step contributions.
- Slow-update interval and lr ratios need ablations; current defaults are conservative.
- Checkpoints are not auto-saved; add `save_checkpoint` calls if needed for ablations.

## 8. Talking points for the seminar
- Design intent: dual-scale updates (fast adapts quickly; slow is stable, optional consolidation to transfer knowledge).
- Integration choice: kept SEMA’s router and adapter-management intact; Nested LoRA only swaps the functional block.
- Engineering safeguards: param-group separation, optional sparse slow updates, consolidation hook, CPU fallback for router init.
