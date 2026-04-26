# 2026-04-27 Coupled Projection Probe

Dataset: `111_days`

Purpose: test whether the locked bottleneck is the endpoint-coefficient decoder coupling, not router recall or score ranking.

Validation setup:

- Seed: `3407`
- Training: `phase_a=10`, `phase_b=4`, `phase_c=12`, `extra=0`
- Short-run limits: `limit_train_batches=120`, `limit_eval_batches=20`
- Candidate budget: `topk_proto=15`, `micro_per_proto=2`, `candidate_dense_topk=5`, `K=20`
- Device: remote GPU3

Implementation sanity note:

- The first coupled implementation was stopped because it updated `active_query` before downstream local-basis/refiner heads, so zero initialization did not preserve the old path.
- The fixed implementation only uses the coupled state to update endpoint and coeff; zero-initialized heads produce exactly identical `query`, `endpoint`, `coeff`, and `coarse` outputs before training.

## Results

| Run | Best epoch | ADE@20 | FDE@20 | Top1 ADE | Proto top1 |
|---|---:|---:|---:|---:|---:|
| Control baseline | 25 | 0.2387627 | 0.3215676 | 0.6453225 | 0.3639648 |
| Coupled decoder only | 26 | 0.2318508 | 0.3078798 | 0.6471042 | 0.3625488 |
| Coupled decoder + projection guidance | 26 | 0.2284536 | 0.3080577 | 0.6445004 | 0.3618652 |

Delta vs control:

- Coupled only: `ADE@20 -0.0069119`, `FDE@20 -0.0136878`
- Coupled + projection: `ADE@20 -0.0103091`, `FDE@20 -0.0135099`

## Failure-Bucket Check

Same 20 eval batches, best checkpoints.

| Bucket | Control ADE | Coupled+projection ADE | Delta |
|---|---:|---:|---:|
| All | 0.2387627 | 0.2284536 | -0.0103091 |
| Router-hit | 0.2167323 | 0.2076522 | -0.0090801 |
| Router-miss | 0.5217915 | 0.5059801 | -0.0158114 |
| Top1-hit | 0.2002866 | 0.1903664 | -0.0099202 |
| Top1-miss | 0.2607803 | 0.2500516 | -0.0107287 |
| Rare-hit | 0.2477614 | 0.2370745 | -0.0106869 |
| Rare-miss | 0.5814397 | 0.5651147 | -0.0163250 |

Router/ranking rates did not meaningfully improve:

- Router top-k hit: `0.9277832 -> 0.9302734`
- Top1 hit: `0.3639648 -> 0.3618652`

Interpretation: the gain comes mainly from better candidate trajectory quality after endpoint-coeff coupling, not from an accidental router or score improvement.

## Decision

Promote the coupled decoder as the new default architecture for validated `trajair_40to120_best20` profiles.

For `111_days`, also promote the projection-guided auxiliary loss:

- `lambda_projection_coeff=0.02`
- `lambda_projection_path=0.05`
- `projection_supervision=winner_gt_proto`

For `7days*`, keep the same coupled decoder architecture by default, but keep projection weights at zero until a matching small-dataset validation is run. This keeps architecture unified while avoiding an unvalidated dataset-specific loss change.

Remote artifacts:

- Summary JSON: `/3250604003/ProtoBasis-Net/notes/experiments/2026-04-27_coupled_projection_probe_fixed.json`
- Diagnostics: `/3250604003/ProtoBasis-Net/notes/experiments/2026-04-27_coupled_probe_diagnostics/`
