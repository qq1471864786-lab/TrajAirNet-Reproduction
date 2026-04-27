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

The original coefficient/path-shape bottleneck is real, but the problem is not just a small refiner, router recall, or one conflicting coefficient loss. The remaining hard part is coefficient inference: predicting the correct future-shape code from the currently available observations is weak.

This means the next serious change must be larger than a tail correction head:

1. Add stronger future-intent/context information if available in the data pipeline.
2. Or replace the coefficient decoder with a stronger multi-modal dynamics decoder that predicts trajectory shape directly from query/prototype context while preserving the ProtoBasis interpretability path.

Do not keep adding small post-hoc correction heads unless a diagnostic first shows that the head can capture a large part of the LS coefficient oracle.
