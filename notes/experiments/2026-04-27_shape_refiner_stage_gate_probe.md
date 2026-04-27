# Shape Refiner Stage-Gate Probe

Date: 2026-04-27

## Question

The current default enables both endpoint and control-point shape refiners. During scheduled training phases, the local residual refiner is disabled in the warm-up / no-refiner stages, but the shape refiners still run. This probe tested whether shape refiners should also be disabled whenever `enable_refiner=False`, so the model first learns a cleaner coarse prototype/basis path before activating all refinement modules.

## Change Tested

Added an experimental `shape_refiner_stage_gate` option:

- off: default behavior; endpoint/control shape refiners stay active even when the local residual refiner is stage-disabled.
- on: endpoint/control shape refiners are stage-gated together with the local residual refiner.

Default behavior was not changed during the probe.

## Remote Short Setup

Both runs used `111_days`, seed `3407`, from-scratch short schedule:

- `phase_a_epochs=4`
- `phase_b_epochs=2`
- `phase_c_epochs=6`
- `extra_epochs=0`
- `limit_train_batches=80`
- `limit_eval_batches=20`
- `batch_size=512`
- `eval_batch_size=1024`

Runs:

- off: `save_model_probe_stage_gate_off_e12_111_short/111_days/seed3407`
- on: `save_model_probe_stage_gate_on_e12_111_short/111_days/seed3407`

## Results

Training-time 20-batch validation at epoch 12:

| variant | ADE@20 | FDE@20 | rare_FDE@20 | proto_top1_acc |
| --- | ---: | ---: | ---: | ---: |
| gate off | 0.27054 | 0.39118 | 1.03955 | 0.30962 |
| gate on | 0.27811 | 0.39838 | 1.05546 | 0.30371 |

Official 50-batch `test.py` evaluation:

| variant | ADE@20 | FDE@20 | rare_FDE@20 | Top1_ADE | proto_top1_acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate off | 0.2664 | 0.3860 | 0.9747 | 0.7345 | 0.3198 |
| gate on | 0.2736 | 0.3907 | 0.9860 | 0.7274 | 0.3188 |

## Decision

Rejected. Stage-gating the shape refiners makes ADE@20 worse in both the training-time validation and the 50-batch official evaluation. This does not support the hypothesis that early shape-refiner activation is the current bottleneck.

The experimental code path should not be promoted to default.

## Interpretation

The result is consistent with the stronger bottleneck evidence from coefficient/future-shape diagnostics: the model benefits from the existing shape refinement modules, but the limiting factor is not simply their training schedule. The remaining gap is more likely in observation-to-future-shape/intent inference, not in whether the refiners are active during warm-up.
