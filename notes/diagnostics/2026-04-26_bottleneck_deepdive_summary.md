# 2026-04-26 Bottleneck Deep Dive Summary

Dataset: `111_days`

Primary checkpoint for conclusion:

- Remote: `/3250604003/ProtoBasis-Net/save_model_ft_softanchor_t035_from_topk10m2_e34/111_days/seed3407/best_best20.pt`
- Existing epoch-metric best: `ADE@20=0.21587`, `FDE@20=0.31842`

Diagnostics saved on remote:

- `/3250604003/ProtoBasis-Net/notes/diagnostics/2026-04-26_bottleneck_deepdive/ade_bottlenecks_softanchor_best.json`
- `/3250604003/ProtoBasis-Net/notes/diagnostics/2026-04-26_bottleneck_deepdive/router_score_softanchor_best.json`
- `/3250604003/ProtoBasis-Net/notes/diagnostics/2026-04-26_bottleneck_deepdive/failure_buckets_softanchor_best.json`

Sampled diagnostics:

- Representation/model oracle: 20,000 random test samples, 80,000 train-bank samples.
- Router/score/failure buckets: 80 eval batches, 40,960 test samples.

## Main Results

| Probe | ADE@20 | FDE@20 | Interpretation |
|---|---:|---:|---|
| Actual best checkpoint, sampled | 0.21782 | 0.32322 | Current best operating point. |
| Force GT prototype | 0.20505 | 0.29006 | Router matters, but router alone does not reach 0.19. |
| No refiner | 0.25007 | 0.33733 | Refiner is helping a lot; removing it is not viable. |
| Force GT prototype, no refiner | 0.23853 | 0.30537 | Router gain depends on refiner; basis path alone is weaker. |

Failure buckets on 40,960 samples:

| Bucket | Count | ADE | FDE |
|---|---:|---:|---:|
| All | 40,960 | 0.22599 | 0.44823 |
| Router-hit | 36,173 | 0.18932 | 0.35296 |
| Router-miss | 4,787 | 0.50310 | 1.16817 |
| Rare-hit | 16,318 | 0.21073 | 0.41089 |
| Rare-miss | 2,194 | 0.55505 | 1.24676 |
| Top1-hit | 14,124 | 0.17050 | 0.29151 |
| Top1-miss | 26,836 | 0.25520 | 0.53072 |

Router rates:

- `router_topk_hit=0.88313`
- `router_topk_miss=0.11687`
- `router_top1_acc=0.34482`
- `rare_router_topk_hit=0.88148`

## Oracle Conclusions

Representation is not the hard ceiling:

- `gt_endpoint_plus_global_basis_ls`: `ADE=0.01680`
- `gt_proto_anchor_plus_global_basis_ls`: `ADE=0.07431`
- `gt_proto_mean_path_plus_global_local_basis_ls`: `ADE=0.06917`

Endpoint/coeff coupling is the main ceiling:

- Current sampled actual: `ADE@20=0.21782`
- `gt_endpoint_plus_pred_coeff`: `ADE=0.16618`
- `pred_endpoint_plus_global_basis_ls`: `ADE=0.04191`
- `pred_endpoint_straight`: `ADE=0.57937`

This means:

- The basis subspace can represent the future very well when the endpoint/coeff are right.
- Current predicted endpoints and learned coeffs do not exploit that space.
- Perfecting only router would likely stop around `0.205`, still above 0.19.

Scoring/ranking is not the `ADE@20` bottleneck:

- Score oracle does not change `ADE@20`, because best-of-20 ignores ranking inside the available set.
- It would improve `ADE@5` and `Top1_ADE` strongly:
  - Actual score: `ADE@5=0.38888`, `Top1_ADE=0.62199`
  - Oracle by ADE: `ADE@5=0.22599`, `Top1_ADE=0.22599`

## Decision

The current architecture has not reached representation capacity, but its current decoder factorization is close to a practical ceiling around `0.21` unless endpoint/coeff coupling is redesigned.

## Locked Bottleneck

The bottleneck to target next is the **endpoint-coefficient coupling inside the trajectory decoder**, not the prototype basis representation itself.

Current forward path:

```text
prototype router -> endpoint prediction -> linear endpoint anchor -> basis coeff prediction -> global/local basis reconstruction -> refiner
```

The weak link is:

```text
endpoint prediction + basis coeff prediction
```

Evidence:

- The basis space is strong enough: with GT endpoint or LS coefficients, oracle ADE drops far below `0.19`.
- Router is important but insufficient: forcing GT prototype improves sampled ADE from `0.21782` to only `0.20505`.
- Ranking is not the `ADE@20` limiter: score oracle changes Top1/ADE@5, but not best-of-20 ADE.
- Refiner is useful but downstream: removing it worsens ADE to `0.25007`, so it should not be the first thing removed or redesigned.

Therefore, the next valid modification must directly improve how endpoint and coeff are learned jointly. Examples of in-scope changes:

- projection-guided coefficient supervision under the candidate's own endpoint anchor;
- endpoint update conditioned on basis reconstruction error or coeff state;
- a decoder path that predicts an endpoint-consistent basis trajectory, rather than independent endpoint and coeff heads.

Out-of-scope as the next main knife:

- only increasing `topk_proto`;
- only changing score/ranking losses;
- only increasing `basis_dim`;
- only enlarging/refactoring the refiner;
- adding unrelated modules that do not touch endpoint-coeff coupling.

Not enough:

- More router top-k alone.
- More score/ranking loss for `ADE@20`.
- Removing or lightly changing the refiner.
- Basis-dim-only expansion.

Most likely next structural target:

- Replace the current linear endpoint-anchor plus coefficient decoder coupling with a stronger endpoint-path / coefficient reconstruction mechanism, while preserving the prototype-basis story.

## 2026-04-27 Follow-Up

Short-run validation confirmed this target was productive.

- Control short baseline: `ADE@20=0.2387627`, `FDE@20=0.3215676`
- Coupled decoder only: `ADE@20=0.2318508`, `FDE@20=0.3078798`
- Coupled decoder + projection guidance: `ADE@20=0.2284536`, `FDE@20=0.3080577`

Failure-bucket checks showed router top-k hit stayed nearly unchanged (`0.9278 -> 0.9303`), while both router-hit and router-miss ADE improved. This supports the original diagnosis: improving endpoint-coeff coupling raises candidate trajectory quality without relying on a router/ranking artifact.

Decision: promote coupled endpoint-coeff decoding as the new architecture default for validated ProtoBasis runs. For `111_days`, also use projection-guided auxiliary supervision as the current ADE-best default.
