# Coefficient Decode Bottleneck Follow-up

Date: 2026-04-27

## Why This Was Run

The endpoint-preserving shape refiner improved 111_days ADE only modestly. This follow-up checks whether the earlier coefficient/path-shape bottleneck diagnosis was wrong, or whether the attempted fix was simply too weak.

## Key Evidence

### 1. Shape refiner improves too little to be the main fix

50-batch official no-AMP eval:

| model | ADE@20 | FDE@20 |
| --- | ---: | ---: |
| baseline before shape refiner | 0.2296 | 0.3132 |
| endpoint shape refiner | 0.2219 | 0.3132 |

Failure bucket delta from baseline to shape refiner:

| bucket | ADE before | ADE after | delta |
| --- | ---: | ---: | ---: |
| all | 0.22960 | 0.22186 | -0.00774 |
| router_hit | 0.20939 | 0.20157 | -0.00782 |
| router_miss | 0.52558 | 0.51907 | -0.00651 |
| rare | 0.25872 | 0.24964 | -0.00908 |
| top1_miss | 0.25259 | 0.24518 | -0.00741 |

Interpretation: the head makes a real but shallow correction. It does not remove the large residual buckets.

### 2. Router is no longer the main bottleneck

On a 4096-sample diagnostic subset using the shape-refiner checkpoint:

| variant | ADE@20 | FDE@20 |
| --- | ---: | ---: |
| actual | 0.22311 | 0.31343 |
| force_gt_proto | 0.21701 | 0.29902 |

Ground-truth routing improves ADE by only about `0.0061`, so routing is not the main remaining limiter.

### 3. Representation space is not the bottleneck

On the same subset:

| oracle | ADE |
| --- | ---: |
| actual | 0.22311 |
| pred_endpoint_plus_global_basis_ls | 0.03967 |
| gt_endpoint_plus_pred_coeff | 0.14909 |

Interpretation: with the same predicted endpoint, a least-squares coefficient under the current global basis almost solves the path. The basis space is expressive enough. The weak part is predicting the right coefficients from available inputs.

### 4. Wrong-anchor coefficient supervision was tested and is not the main culprit

Hypothesis: `gt_basis_coeff` is solved under the true endpoint anchor, while the model uses predicted endpoint anchors. A short controlled run tested removing this supervision and strengthening predicted-anchor projection guidance.

Shared setup:

- Init: `save_model_probe_endpoint_shape_refiner_111_short/111_days/seed3407/best_best20.pt`
- 8 short epochs, 80 train batches/epoch
- 50-batch official no-AMP eval

| variant | ADE@20 | FDE@20 |
| --- | ---: | ---: |
| default continue | 0.2189 | 0.3103 |
| no true-endpoint coeff loss + stronger predicted-anchor projection | 0.2186 | 0.3112 |

Decision: not a meaningful gain; do not promote this as the fix.

### 5. Simple observed-history retrieval is not enough

A 4096-test / 50000-train kNN oracle used only observed target history in a local frame:

| feature | 1-NN ADE | top20 minADE |
| --- | ---: | ---: |
| tail10 local path | 1.3674 | 0.3785 |
| tail20 local path | 1.3249 | 0.3637 |
| vel20 | 1.3909 | 0.3812 |

Interpretation: local observed-history similarity does not recover a good future. The missing signal is not a simple nearest-neighbor memory over the observed prefix.

## Updated Conclusion

## 2026-04-27 Default Module Re-Audit

Purpose: re-check the cleaned current default module-by-module before making any new architecture changes, so later experiments do not drift into directionless add-ons.

Setup:

- Checkpoint: `save_model_probe_control_continue_e8_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`, test split
- Eval: 50 batches, batch size 128, 6400 samples
- Current default: `topk_proto=15`, `micro_per_proto=2`, `candidate_dense_topk=5`, `local_basis_dim=2`, two-stage decoder, coupled decoder, endpoint/control shape refiners

Module ablation results:

| Variant | ADE@20 | FDE@20 | ADE@5 | Top1_ADE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| full default | 0.25884 | 0.37556 | 0.51484 | 0.75726 | baseline for this audit |
| force GT prototype | 0.23873 | 0.33108 | 0.51014 | 0.75512 | router helps, but is not enough |
| no social | 0.28310 | 0.40923 | 0.54680 | 0.78675 | social is useful |
| no router / generic modes | 0.57723 | 1.02784 | 0.71594 | 1.01530 | prototype router is essential |
| no temporal refiner | 0.26398 | 0.38281 | 0.52072 | 0.76154 | small positive module |
| no endpoint shape refiner | 0.28168 | 0.37556 | 0.52893 | 0.77579 | endpoint-preserving shape refiner helps ADE |
| no control shape refiner | 0.26794 | 0.37556 | 0.52103 | 0.75882 | control refiner helps modestly |
| no shape refiners | 0.28257 | 0.37556 | 0.52831 | 0.77159 | keep refiners, but they are not the main fix |
| no local basis | 0.26215 | 0.37555 | 0.51510 | 0.75602 | local basis is a small gain |
| no two-stage decoder | 0.43547 | 0.60915 | 0.67332 | 0.88496 | two-stage decoder is core |
| no coupled decoder | 0.36757 | 0.62132 | 0.63839 | 0.91140 | coupled endpoint-coeff decoder is core |

Failure buckets:

| Bucket | Count | ADE | FDE |
| --- | ---: | ---: | ---: |
| all | 6400 | 0.25884 | 0.50358 |
| router-hit | 5402 | 0.20975 | 0.39321 |
| router-miss | 998 | 0.52453 | 1.10099 |
| rare | 3250 | 0.27320 | 0.54763 |
| top1-hit | 1654 | 0.19192 | 0.32298 |
| top1-miss | 4746 | 0.28216 | 0.56652 |

Router rates:

- `topk_hit = 0.84406`
- `top1_hit = 0.25844`
- `rare_topk_hit = 0.84615`

Oracle checks:

| Probe | ADE | Interpretation |
| --- | ---: | --- |
| `gt_endpoint_plus_pred_coeff` | 0.17398 | better endpoints would be enough to pass 0.19 with current coeffs |
| `pred_endpoint_plus_global_basis_ls` | 0.04549 | current endpoint anchors plus optimal basis coeffs have large headroom |
| `gt_endpoint_plus_global_basis_ls` | 0.01586 | basis representation is not the ceiling |
| `pred_endpoint_straight` | 0.54878 | endpoint-only linear anchor is not a path model |
| `coeff_blend_alpha_0.10` | 0.25766 | small oracle move is weak on this harder sample window |
| `coeff_blend_alpha_0.25` | 0.21838 | larger coeff/path correction still gives large ADE gain |

Candidate allocation check:

- Winner from dense top-5 prototype ranks: `4406`
- Winner from tail ranks 5-14: `1994`
- Dense-rank-only min ADE: `0.34368`
- Tail-rank-only min ADE: `0.43210`

Audit conclusion:

- Keep the current cleaned default modules. Router, social, two-stage decoding, coupled decoding, and endpoint/control shape refiners are all useful.
- Do not remove the prototype-basis story. The basis space is strong enough; the issue is how the model predicts endpoint-conditioned future shape.
- Do not spend the next round on score-only, router-only, basis-dim-only, or another small post-hoc refiner unless a short oracle first shows a large gain.
- The next valid architecture work must directly target coefficient/future-shape inference after the predicted endpoint anchor.

The original coefficient/path-shape bottleneck is real, but the problem is not just a small refiner, router recall, or one conflicting coefficient loss. The remaining hard part is coefficient inference: predicting the correct future-shape code from the currently available observations is weak.

This means the next serious change must be larger than a tail correction head:

1. Add stronger future-intent/context information if available in the data pipeline.
2. Or replace the coefficient decoder with a stronger multi-modal dynamics decoder that predicts trajectory shape directly from query/prototype context while preserving the ProtoBasis interpretability path.

Do not keep adding small post-hoc correction heads unless a diagnostic first shows that the head can capture a large part of the LS coefficient oracle.

## 2026-04-27 Endpoint-Conditioned Basis Bridge Short Test

Purpose: test the planned bridge fix for the confirmed `predicted endpoint anchor -> predicted coeff/path` bottleneck. The bridge predicts endpoint-preserving residual control points, projects them through the existing basis pseudo-inverse, and reconstructs candidate paths from the fixed basis. It is off by default and zero-initialized to preserve old checkpoints.

Setup:

- Init: `/3250604003/ProtoBasis-Net/save_model_probe_control_continue_e8_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`
- Train: bridge-only freeze, 8 epochs, 80 train batches/epoch, batch size 512, no AMP
- Eval: first 50 test batches, eval batch size 1024, no AMP
- Run: `/3250604003/ProtoBasis-Net/save_model_bridge_frozen_e8_111_short/111_days/seed3407`

Fair 50-batch comparison:

| Variant | ADE@20 | FDE@20 | ADE@5 | Top1_ADE | GLeV@20 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current checkpoint baseline | 0.2137 | 0.3046 | 0.3848 | 0.6063 | 0.0898 | reference |
| basis bridge, best epoch 1 | 0.2137 | 0.3046 | n/a | n/a | n/a | no gain |
| basis bridge, epoch 8 | 0.2139 | 0.3046 | 0.3855 | 0.6057 | 0.0898 | no gain |

Training signal:

- `bridge_path` stayed around `0.0453` by epoch 8.
- `bridge_coeff` stayed around `0.4810` by epoch 8.
- Final winner train ADE was `0.2234`, close to the frozen baseline range.

Conclusion:

- This bridge implementation does not pass the short-test threshold (`>=0.012` ADE@20 gain).
- Do not make `basis_bridge_decoder` default.
- The failure suggests that merely predicting a low-resolution residual path and projecting it back to the fixed basis is not enough under frozen context/query features.
- The next backup direction should be a more direct basis-aware coefficient inference mechanism, e.g. basis-token cross-attention or a stronger coeff decoder, rather than another endpoint-preserving post-hoc path control head.
