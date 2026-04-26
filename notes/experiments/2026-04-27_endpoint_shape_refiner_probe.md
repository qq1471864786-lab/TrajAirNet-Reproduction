# Endpoint-Preserving Shape Refiner Probe

Date: 2026-04-27

## Question

The coefficient-oracle blend showed that the current 111_days ADE ceiling is mostly path shape after the predicted endpoint anchor: a small oracle move in coefficient/path space lowered ADE while leaving FDE nearly unchanged. This probe tests whether a learnable endpoint-preserving shape head can capture part of that gain without GT at inference.

## Baseline

- Checkpoint: `save_model_probe_coupled_proj_fixed_111_e26/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`, test split
- Batch size: 1024
- AMP: disabled

Official 20-batch eval:

| model | ADE@20 | FDE@20 |
| --- | ---: | ---: |
| baseline | 0.22843 | 0.30798 |

## Implementation Tested

`EndpointPreservingShapeRefiner` runs after the existing temporal residual refiner. It predicts a local path-shape delta from the refined local path, local step deltas, and the candidate query feature. The update is multiplied by an envelope `1 - t/T`, so the final predicted endpoint is unchanged by construction.

This targets ADE/path shape while protecting FDE and endpoint behavior.

## Short Validation

Run: `save_model_probe_endpoint_shape_refiner_111_short`

- Initialized from the baseline checkpoint.
- Frozen backbone; only the new shape refiner head was trained.
- 8 short epochs.
- 80 train batches per epoch.
- 20 eval batches during training.

Official 20-batch no-AMP eval:

| model | ADE@20 | FDE@20 | delta ADE |
| --- | ---: | ---: | ---: |
| baseline | 0.22843 | 0.30798 | - |
| endpoint shape refiner | 0.22085 | 0.30798 | -0.00758 |

Follow-up 50-batch no-AMP fair eval on the same test split:

| model | ADE@20 | FDE@20 | rare_FDE@20 | Top1_ADE | delta ADE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.2296 | 0.3132 | 0.7773 | 0.6248 | - |
| endpoint shape refiner | 0.2219 | 0.3132 | 0.7773 | 0.6222 | -0.0077 |

The 50-batch result confirms the same mechanism: ADE improves while FDE and rare FDE are unchanged.

## Rejected Branches

Coefficient-correction head probes were run but are not kept as the default architecture:

| variant | short result | decision |
| --- | --- | --- |
| coeff correction only, frozen backbone | ADE@20 about 0.22410, FDE@20 0.30798 | positive but weak |
| coeff correction with all-candidate supervision | ADE@20 about 0.2366 | reject; collapses multimodal candidates toward one LS target |
| coeff correction end-to-end micro-tune | ADE@20 about 0.2309, FDE also worse | reject; perturbs endpoint/score/router |
| shape refiner + coeff correction, frozen backbone | ADE@20 about 0.2228 | reject; worse than shape-only |

## Decision

Keep `EndpointPreservingShapeRefiner` as the new default head for the validated unified protocol. Do not keep coefficient correction in the main architecture.

This is not enough evidence for full training by itself, but it is strong enough to update the short-validation baseline because it directly targets the confirmed path-shape bottleneck and has a protected endpoint.
