# Control-Point Shape Refiner Probe

Date: 2026-04-27

## Question

The coefficient/path-shape oracle shows large ADE headroom after fixing the predicted endpoint. The earlier endpoint-preserving convolutional shape refiner captured only a shallow part of this headroom, so this probe tests a stronger endpoint-preserving control-point decoder.

## Implementation

`EndpointPreservingControlPointRefiner` predicts a low-frequency residual path from each candidate query and coarse path statistics. The residual is linearly interpolated from control points and multiplied by `1 - t/T`, so it can correct mid-horizon shape while preserving the final endpoint.

This directly targets the verified coefficient/future-shape bottleneck and should not change endpoint/FDE by construction except through training interactions.

## Short Validation

Shared setup:

- Dataset: `111_days`
- Init/control checkpoint family: endpoint-shape-refiner short baseline
- Training: 16 short epochs, 80 train batches/epoch
- Eval: official `test.py`, 50 batches, no AMP

| variant | ADE@20 | FDE@20 | decision |
| --- | ---: | ---: | --- |
| default continue | 0.2170 | 0.3068 | control |
| control shape, 16 points | 0.2157 | 0.3073 | positive but not best |
| control shape, 32 points | 0.2152 | 0.3066 | promote as new short baseline |
| obs extra columns | 0.2195 | 0.3113 | reject |
| strong projection only | 0.2184 | worse than control | reject |
| control shape 32 + strong projection | 0.2149 | 0.3110 | reject for default because FDE regressed |

## Decision

Promote the clean `control_shape_refiner=True`, `control_shape_points=32` variant as the default unified architecture extension. Do not keep the observed-extra-column branch in the model path, and do not promote stronger projection guidance.

The gain is real but small (`ADE@20 0.2170 -> 0.2152` under the 50-batch short protocol). This does not solve the full `0.19` target by itself. It is a cleaner baseline for the next bottleneck round because it addresses the confirmed shape-code weakness without perturbing endpoint prediction.

## Next Bottleneck

The remaining gap is still coefficient/future-shape inference. Further work should focus on a stronger decoder or intent/context signal before full training; do not spend full runs on small post-hoc heads unless a short diagnostic shows they capture a meaningful part of the LS oracle gain.
