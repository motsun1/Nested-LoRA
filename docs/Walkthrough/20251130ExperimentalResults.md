# Nested LoRA vs Baseline (CIFAR-100, 10-Task)

**Date:** 2025-11-30  
**Objective:** Nested LoRA (Fast/Slow) vs. parameter-matched standard SEMA adapter. Validate whether dual time-scales help beyond parameter count.

**Logs:**  
- Nested LoRA: `logs/sema/cifar224/0/10/nested_lora_1993_pretrained_vit_b16_224_adapter.log`  
- Baseline: `logs/sema/cifar224/0/10/baseline_sema_matched_1993_pretrained_vit_b16_224_adapter.log`

## 1. Experiment Setup
| Feature | Nested LoRA | Baseline SEMA (matched) |
| --- | --- | --- |
| Config | `exps/nested_lora_cifar.json` | `exps/baseline_sema_cifar.json` |
| Structure | Dual LoRA (rank 16 x 2 paths) | Single adapter (bottleneck 32) |
| Params / task | ~76.9k | ~76.9k |
| Learning rates | Fast 0.01, Slow 0.001 | Single 0.005 |
| Update rule | Slow updates every 5 steps | Every step |
| Consolidation | Off | N/A |

## 2. Results (Top-1 %)
| Metric | Nested LoRA | Baseline | Diff |
| --- | --- | --- | --- |
| Final Avg Acc | 91.39 | 91.42 | -0.03 |
| Final Task0 Acc | 86.4 | 87.1 | -0.7 |
| Final Task9 (new) | 91.0 | 88.2 | +2.8 |

### Accuracy Progression (per task)
| Task | Nested LoRA | Baseline |
| --- | --- | --- |
| T0 | 98.90 | 98.70 |
| T1 | 95.85 | 95.55 |
| T2 | 94.43 | 94.17 |
| T3 | 93.40 | 93.22 |
| T4 | 91.38 | 91.46 |
| T5 | 90.08 | 89.90 |
| T6 | 89.83 | 89.93 |
| T7 | 87.45 | 87.79 |
| T8 | 86.58 | 87.03 |
| T9 | 85.98 | 86.49 |

## 3. Analysis
- Overall parity: 91.39 vs 91.42 — essentially identical; dual-scale alone (v0) does not outperform matched adapter.
- Plasticity: Final task “new” accuracy favors Nested LoRA (91.0 vs 88.2), consistent with the higher fast LR.
- Stability: Baseline retains Task0 slightly better (87.1 vs 86.4); slow path + interval=5 did not reduce forgetting enough.
- Internal dynamics: fast/slow norms are non-zero (e.g., T0 fast_norm≈8.25, slow_norm≈7.97). “post_consolidation” norm=0.0 in logs is a logging artifact (params frozen → skipped by norm calc), not weight loss.

## 4. Conclusion & Next Steps
- Conclusion: Splitting into Fast/Slow (v0) reaches baseline parity at matched params; no stability gain yet.
- Next steps:
  1) Enable consolidation (`nested_lora_use_consolidation: true`, test α ∈ {0.05, 0.1}).  
  2) Try slow interval=1 (always update) vs 5 to balance stability.  
  3) Run rank/learning-rate ablations (rank=8; fast/slow LR ratio=5x).  
  4) Re-run baseline comparison with consolidation on to check stability improvement.
