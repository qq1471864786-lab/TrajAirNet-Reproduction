# 2026-04-27 Path-to-Coeff Projection Probe

Dataset: `111_days`

Purpose: test whether moving projection from auxiliary supervision into the forward path improves the remaining bottleneck:

```text
predicted endpoint anchor -> predicted coeff/path inside basis space
```

Baseline:

- Current default short checkpoint: `save_model_probe_coupled_proj_fixed_111_e26`
- `ADE@20=0.2284536`, `FDE@20=0.3080577`

Shared setup:

- Seed: `3407`
- `phase_a=10`, `phase_b=4`, `phase_c=12`, `extra=0`
- `limit_train_batches=120`, `limit_eval_batches=20`
- `topk_proto=15`, `micro_per_proto=2`, `candidate_dense_topk=5`
- Path control points: `12`
- Path projection supervision: `winner_gt_proto`
- `lambda_path_projection_coeff=0.02`
- `lambda_path_projection_path=0.05`

## Tested Variants

| Variant | Description | ADE@20 | FDE@20 | Top1 ADE | Decision |
|---|---|---:|---:|---:|---|
| Current default | Coupled endpoint-coeff decoder + projection guidance as auxiliary loss | 0.2284536 | 0.3080577 | 0.6445004 | Keep |
| Absolute path-to-coeff | Predict an absolute residual path, project it to coeff, mix with old coeff | 0.2291778 | 0.3111395 | 0.6441602 | Reject |
| Delta path-to-coeff | Predict a correction path from current coeff toward LS coeff | 0.2295987 | 0.3100937 | 0.6491256 | Reject |

## Interpretation

The mechanism was implemented and tested, but neither forward-path projection variant improved the current default under the short fair comparison.

Observed behavior:

- Absolute path-to-coeff learned a reasonable path projection loss and slightly improved rare FDE (`0.80652 -> 0.80414`), but average ADE/FDE worsened.
- Delta path-to-coeff improved early convergence relative to the same-epoch current default, but ended worse at epoch 26 and also worsened Top1 ADE.
- Both variants indicate that directly inserting the projected path into the forward trajectory path can interfere with the already useful coupled decoder/refiner balance.

Decision:

- Do not promote path-to-coeff projection as the new default.
- Revert the experimental model-code changes to keep the current architecture clean.
- Keep the experiment as negative evidence: the remaining coefficient bottleneck is real, but a hard forward projection path is not the right implementation in this form.

Next implication:

- If revisiting this bottleneck, prefer a softer mechanism such as better coefficient-target scheduling or candidate-specific distillation after the main decoder has stabilized, rather than forcing projected coefficients directly into the forward path from early training.
