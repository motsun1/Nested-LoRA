# Walkthrough - 2025-12-06 MoE/PAM-lite Debug Notes

## Scope
- PAM-lite consolidation verification and logging.
- MoE expert selection sanity checks and CIL-safe validation split.
- Config updates for Step2 MoE experiments.

## Changes
- Propagate `nested_lora_consolidation_method` into tuning config so PAM-lite runs instead of silently falling back to task arithmetic (`utils/inc_net.py`).
- Add PAM-lite consolidation logs with allowed/suppressed element counts for each fast adapter to confirm it executes (`backbone/sema_block.py`, `backbone/nested_lora_block.py`).
- Fix MoE inference to honor `task_id` and use a single fast expert instead of summing all fasts; added per-task expert accuracy/selection records (`backbone/sema_components.py`, `models/nested_lora.py`).
- Insert validation-based expert selection using a task-local hold-out split (`moe_val_ratio`, default 0.1) to avoid cross-task leakage (`models/nested_lora.py`).
- Update MoE experiment configs to set `moe_val_ratio: 0.1` (`exps/step2_moe_pam.json`, `exps/step2_moe_ta.json`).

## How to run
- Step1: unchanged configs; rerun as before to see new PAM-lite logs.
- Step2 MoE: `bash run_step2_moe.sh` (uses updated configs). Check logs for `Expert k Validation Accuracy (loader=val)` and `Expert Selection History` to verify expert diversity/selection.

## Notes
- Winner-take-all merge: top-1 expert (by val acc) is merged into slow; others remain for future tasks.
- Softmax/weighted merge and gradient/output-delta based scoring are not implemented yet.
