Seminar Presentation Plan: Nested LoRA for Continual Learning
1. Presentation Structure (Draft)
Since this is a progress report ("ゼミ発表"), the goal is to share the current status, the "bug" discovery, and the roadmap. You don't need perfect results yet.

Slide 1: Motivation (The "Why")
Concept: Human-like learning has two time-scales:
Fast: Quick adaptation to new tasks (Short-term memory).
Slow: Gradual consolidation of general knowledge (Long-term memory).
Problem: Existing PEFT methods (LoRA, Adapters) typically use a single time-scale (one learning rate).
Proposal: Nested LoRA – A dual-path adapter (Fast + Slow) to better manage the plasticity-stability trade-off in Continual Learning.
Slide 2: Method (The "How")
Architecture:
Parallel Fast and Slow LoRA modules in each transformer block.
Output = $W x + \Delta_{fast}(x) + \Delta_{slow}(x)$.
Optimization:
Fast: High learning rate, updates every step.
Slow: Low learning rate, updates less frequently (e.g., every $k$ steps) or via consolidation.
Slide 3: Implementation Status
Completed:
Implemented 
NestedLoRAAdapter
 class.
Integrated into SEMA codebase (ViT backbone).
Configured separate parameter groups for Fast/Slow optimization.
Preliminary Experiment:
CIFAR-100 (10 tasks).
Result: Model learned (Avg Acc ~91%), but behavior was identical to a standard adapter.
Slide 4: Analysis & "The Bug"
Observation: The "Fast/Slow" distinction wasn't active in the first run.
Cause: A coding error (hasattr on a dictionary) caused the optimizer to treat both paths as a single group with the same learning rate.
Insight: This inadvertently served as a "control group" experiment (Monolithic Adapter with 2x rank).
Fix: Bug identified and fixed. Ready for the "real" dual-scale experiment.
Slide 5: Roadmap & Discussion
Immediate Next Step: Re-run the experiment with the fix (Fast LR $\gg$ Slow LR).
Hypothesis: The fixed model should show better stability (less forgetting) than the "bugged" (monolithic) version.
Questions for Seminar:
"Is the parallel addition the best way to combine them?"
"Should we consolidate 'Fast' into 'Slow' at task boundaries (v1)?"
2. Recommended Additional Experiments
To make the research convincing, you need to compare the "Structure" vs. just "More Parameters".

Experiment A: The "Real" Nested LoRA (Priority: High)
Config: Same as before, but with the bug fix.
Settings: lr_fast=0.01, lr_slow=0.001, update_interval=5.
Goal: Show that time-scale separation changes the learning dynamics (e.g., forgetting curve).
Experiment B: Parameter-Matched Baseline (Priority: Medium)
Config: Standard SEMA (adaptmlp).
Settings: Increase ffn_num (bottleneck) so the total parameters match Nested LoRA.
Nested LoRA (rank 16) $\approx$ Standard Adapter (bottleneck 32).
Goal: Prove that any gain comes from the Nested structure, not just having more parameters.
Experiment C: Consolidation (Priority: Low/Future)
Config: Enable nested_lora_use_consolidation.
Goal: Test if explicitly merging weights at task boundaries helps long-term retention.
3. Preparation for Seminar
You don't need to finish Exp A/B/C before the seminar.
Action: Start Exp A now. If it finishes in time, show the comparison. If not, present the "Analysis & Bug" story—it shows you understand the code and the mechanism deeply.