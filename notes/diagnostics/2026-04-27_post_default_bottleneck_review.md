# 2026-04-27 Post-Default Bottleneck Review

Checkpoint reviewed:

- Remote: `/3250604003/ProtoBasis-Net/save_model_probe_coupled_proj_fixed_111_e26/111_days/seed3407/best_best20.pt`
- Short-run setting: `111_days`, seed `3407`, `limit_train_batches=120`, `limit_eval_batches=20`
- New default under review: coupled endpoint-coeff decoder + projection guidance

## Key Metrics

| Probe | ADE@20 | FDE@20 | Meaning |
|---|---:|---:|---|
| Actual new default short checkpoint | 0.22792 | 0.31046 | Current reviewed operating point. |
| Force GT prototype | 0.22308 | 0.29620 | Router is still useful, but not the main remaining lever. |
| No refiner | 0.23128 | 0.31165 | Refiner now gives only a small gain in this short checkpoint. |
| Force GT prototype, no refiner | 0.22665 | 0.29691 | Router gain is modest even without refiner. |

Endpoint/coeff oracle probes:

| Probe | ADE | Meaning |
|---|---:|---|
| `gt_endpoint_plus_pred_coeff` | 0.14794 | If endpoint were perfect, current coeffs are already below 0.19. |
| `pred_endpoint_plus_global_basis_ls` | 0.03935 | Current endpoints plus optimal basis coeffs can fit very well. |
| `gt_endpoint_plus_global_basis_ls` | 0.01679 | Basis representation is far from the ceiling. |
| `pred_endpoint_straight` | 0.54938 | Endpoint-only linear anchor remains a poor path model. |

Failure-bucket recap from the paired short comparison:

| Bucket | Control ADE | New default ADE | Delta |
|---|---:|---:|---:|
| All | 0.23876 | 0.22845 | -0.01031 |
| Router-hit | 0.21673 | 0.20765 | -0.00908 |
| Router-miss | 0.52179 | 0.50598 | -0.01581 |
| Top1-hit | 0.20029 | 0.19037 | -0.00992 |
| Top1-miss | 0.26078 | 0.25005 | -0.01073 |

Router rates barely changed:

- Router top-k hit: `0.92778 -> 0.93027`
- Top1 hit: `0.36396 -> 0.36187`

## Review Conclusion

The coupled decoder fixed part of the previous endpoint-coeff mismatch, but the main remaining bottleneck is still not router recall or representation capacity. The largest remaining gap is that the learned candidate path coefficients do not exploit the basis space well enough under the predicted endpoint anchors.

The practical bottleneck has narrowed to:

```text
predicted endpoint anchor -> predicted coeff/path inside basis space
```

Evidence:

- Perfecting router only improves sampled ADE by about `0.0048`.
- Removing refiner only worsens ADE by about `0.0034`.
- Optimal LS coefficients under the current predicted endpoints would reduce ADE to `0.03935`, so the basis and endpoint anchors have enough representational headroom.
- The current learned coeff/path output is still far above that oracle, even after projection guidance.

## Next Valid Target

The next change should directly improve coefficient/path generation, not router-only, score-only, or basis-dim-only changes.

Most promising direction:

- Replace the plain coefficient residual head with a basis-aware path-to-coeff decoder:
  - predict a compact residual path or residual summary conditioned on endpoint/prototype/context;
  - project or regularize it through the fixed basis pseudo-inverse;
  - use the projected coeff/path as the candidate coarse trajectory before local basis/refiner.

This keeps the ProtoBasis innovation story intact: the model still predicts interpretable prototype-basis trajectories, but the decoder is forced to generate coefficients that are geometrically consistent with the endpoint-conditioned residual path.

Lower-priority directions:

- Router hardening: useful for tail cases, but not enough to reach `0.19` alone.
- Refiner redesign: current refiner gain is small after the coupled change, but it is downstream and should not be the next primary target.
- Score/ranking: important for Top1/ADE@5, but not for best-of-20 ADE.
