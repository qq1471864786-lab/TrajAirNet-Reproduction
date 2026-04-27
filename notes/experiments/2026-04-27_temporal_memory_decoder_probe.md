# Temporal Memory Decoder Probe

Date: 2026-04-27

## Question

The remaining 111_days bottleneck is coefficient/future-shape inference. One plausible structural weakness is that the current decoder receives only pooled per-agent tokens from `TemporalEncoder`; the target aircraft's full observed time sequence is not directly visible to the coefficient decoder.

This probe tests a target-temporal-memory cross-attention branch inside the prototype-conditioned query decoder.

## Setup

Shared setup:

- Init checkpoint: `save_model_probe_control_shape32_e16_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`
- Training: 8 short epochs, 80 train batches/epoch
- Phase schedule: all epochs in `joint_refiner`
- Eval during training: 20 batches
- Official final eval: `test.py`, 50 batches, no AMP

Runs:

| variant | save_dir | GPU |
| --- | --- | --- |
| control continue | `save_model_probe_control_continue_e8_111_short` | 5 |
| temporal memory decoder | `save_model_probe_temporal_memory_e8_111_short` | 3 |

## Results

Training-time best 20-batch metrics:

| variant | best ADE@20 | epoch | FDE@20 at final epoch |
| --- | ---: | ---: | ---: |
| control continue | 0.21574 | 7 | 0.30788 |
| temporal memory decoder | 0.21610 | 8 | 0.30510 |

Official 50-batch no-AMP eval:

| variant | ADE@20 | FDE@20 | rare_FDE@20 | Top1_ADE |
| --- | ---: | ---: | ---: | ---: |
| control continue | 0.2137 | 0.3046 | 0.7581 | 0.6063 |
| temporal memory decoder | 0.2139 | 0.3034 | 0.7538 | 0.6068 |

## Decision

Do not promote the temporal-memory decoder branch. It does not improve ADE over a matched continuation baseline under the short protocol. The small FDE/rare-FDE improvement is not enough because the project objective is ADE-first.

The code branch was removed from the local default architecture after this probe to keep the model clean.
