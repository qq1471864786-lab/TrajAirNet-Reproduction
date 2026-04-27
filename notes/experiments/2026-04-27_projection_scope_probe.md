# Projection Supervision Scope Probe

Date: 2026-04-27

## Question

The coefficient oracle shows large path-shape headroom, but previous projection and coefficient-correction variants did not close it. This probe checks whether the default projection supervision scope is misaligned with best-of-20 training.

Default projection supervision is `winner_gt_proto`, which supervises both the current best candidate and all candidates whose prototype matches the GT prototype. A possible risk is that GT-prototype supervision may collapse micro-candidate diversity inside the same prototype.

## Setup

Shared setup:

- Init checkpoint: `save_model_probe_control_shape32_e16_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`
- Training: 8 short epochs, 80 train batches/epoch
- Phase schedule: all epochs in `joint_refiner`
- Eval during training: 20 batches
- Official final eval: `test.py`, 50 batches, no AMP

Compared against the matched default continuation run from `2026-04-27_temporal_memory_decoder_probe.md`:

| variant | save_dir |
| --- | --- |
| default continuation, `winner_gt_proto` | `save_model_probe_control_continue_e8_111_short` |
| winner-only projection | `save_model_probe_projection_winner_e8_111_short` |
| projection off | `save_model_probe_projection_none_e8_111_short` |

## Results

Training-time best 20-batch metrics:

| variant | best ADE@20 | epoch |
| --- | ---: | ---: |
| default continuation | 0.21574 | 7 |
| winner-only projection | 0.21573 | 7 |
| projection off | 0.21597 | 7 |

Official 50-batch no-AMP eval:

| variant | ADE@20 | FDE@20 | rare_FDE@20 | Top1_ADE |
| --- | ---: | ---: | ---: | ---: |
| default continuation | 0.2137 | 0.3046 | 0.7581 | 0.6063 |
| winner-only projection | 0.2140 | 0.3046 | 0.7575 | 0.6076 |
| projection off | 0.2142 | 0.3039 | 0.7565 | 0.6070 |

## Decision

Do not change the default projection scope. Winner-only projection does not improve ADE over the matched default continuation, and turning projection off is worse for ADE.

This excludes projection-scope mismatch as the main remaining bottleneck. The larger bottleneck is still that the model cannot infer the LS-style future-shape code well enough from available inputs, not that the current projection mask is obviously wrong.
