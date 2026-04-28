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

## 2026-04-27 Basis-Aware Coeff Decoder Short Test

Purpose: test whether replacing the MLP-only coeff update with basis-token cross-attention can close the `predicted endpoint -> coeff` gap without changing the final basis trajectory formulation. This is off by default.

Setup:

- Init: `/3250604003/ProtoBasis-Net/save_model_probe_control_continue_e8_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`
- Train: coeff decoder stack, 8 epochs, 80 train batches/epoch, batch size 512, no AMP
- Eval: first 50 test batches, eval batch size 1024, no AMP
- Run: `/3250604003/ProtoBasis-Net/save_model_basis_coeff_stack_e8_111_short/111_days/seed3407`

Result:

| Variant | ADE@20 | FDE@20 | Decision |
| --- | ---: | ---: | --- |
| current checkpoint baseline | 0.2137 | 0.3046 | reference |
| basis-aware coeff, best epoch 6 | 0.2134 | 0.3046 | tiny gain only |
| basis-aware coeff, epoch 8 | 0.2135 | 0.3056 | no meaningful gain |

Conclusion:

- This does not pass the short-test threshold.
- Do not make `basis_coeff_decoder` default.
- Basis-token attention alone is still trapped by the old endpoint/basis trajectory formulation.

## 2026-04-27 Temporal Basis Dynamics Short Test

Purpose: make a larger replacement inside the basis path: future control tokens attend to the target 40-step temporal sequence and agent tokens, predict an endpoint-preserving residual path, then project back to the fixed basis. This is off by default.

Setup:

- Init: `/3250604003/ProtoBasis-Net/save_model_probe_control_continue_e8_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`
- Train: temporal dynamics stack, 8 epochs, 80 train batches/epoch, batch size 512, no AMP
- Eval: first 50 test batches, eval batch size 1024, no AMP
- Run: `/3250604003/ProtoBasis-Net/save_model_temporal_dynamics_stack_e8_111_short/111_days/seed3407`

Result:

| Variant | ADE@20 | FDE@20 | ADE@5 | GLeV@20 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| current checkpoint baseline | 0.2137 | 0.3046 | 0.3848 | 0.0898 | reference |
| temporal basis dynamics, best ADE epoch 8 | 0.2128 | 0.3046 | 0.3884 | 0.0890 | tiny gain only |

Conclusion:

- This also does not pass the short-test threshold.
- Do not make `temporal_dynamics_decoder` default.
- The important negative evidence is that even a stronger temporal/context decoder fails when its final output is projected back into the same fixed basis trajectory channel. The next valid test should let the dynamics decoder produce the final trajectory directly, with basis projection kept only as an auxiliary interpretation/regularization path.

## 2026-04-27 Direct Dynamics Rollout Plan

Next test: `direct_dynamics_decoder`.

Design:

- Keep router, candidate allocation, endpoint prediction, two-stage/coupled context, and current candidate scoring.
- Replace the final trajectory generation channel after local-basis mixing: predict velocity residual controls from candidate query + endpoint + coeff + target temporal memory + agent memory.
- Integrate velocities for 120 steps and correct the cumulative endpoint error analytically, so FDE is not destabilized.
- Do not project the final path back into fixed basis space. Project only into `direct_dynamics_coeff` for aux/diagnostics, preserving interpretability without making basis coeff the bottleneck.

Validation rule:

- Run the same 111_days 8-epoch/80-batch/50-eval-batch short test.
- Only keep as default if ADE@20 improves materially; tiny changes like `0.2137 -> 0.2128` are failures.

Implementation correction:

- First direct-dynamics run used zero-initialized velocity head and zero-initialized gate head with `gate * velocity_delta`.
- That made the direct rollout initially equivalent to the old path, but also blocked gradients to both new branches because both multiplicative terms were zero.
- Treat `/3250604003/ProtoBasis-Net/save_model_direct_dynamics_stack_e8_b256_111_short/111_days/seed3407` as an invalid implementation check, not a method result (`ADE@20=0.2129`, tiny change).
- Fix: initialize the effective gate to `0.25` while keeping velocity residual output zero, so the checkpoint still starts from the old path but gradients reach the velocity rollout head.

Corrected result:

| Variant | ADE@20 | FDE@20 | ADE@5 | GLeV@20 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| current checkpoint baseline | 0.2137 | 0.3046 | 0.3848 | 0.0898 | reference |
| direct dynamics rollout, best epoch 7 | 0.2139 | 0.3055 | n/a | n/a | no gain |
| direct dynamics rollout, epoch 8 | 0.2140 | 0.3056 | 0.3862 | 0.0887 | no gain |

Conclusion:

- Do not make `direct_dynamics_decoder` default.
- This is now valid negative evidence: after fixing the dead gate, `direct_path` was active (`~0.0467`) but did not improve ADE.
- The repeated failure of bridge, basis-aware coeff, temporal-basis dynamics, and direct rollout suggests the next intervention should move upstream from path generation to candidate generation/routing. A strong next test is a soft prototype-memory decoder that generates K candidates by attending over all prototypes instead of pruning to hard top-k first, so router-miss samples are not irrevocably discarded.

## 2026-04-27 Soft Prototype-Memory Decoder Plan

Purpose: test a larger upstream change against the router-miss bottleneck. Instead of letting hard top-k prototypes be the only information source after routing, let each evaluated candidate cross-attend to all prototype memories and learn endpoint/coeff/score residuals.

Design:

- Keep the temporal/social encoder and router logits for auxiliary prototype learning and reporting.
- Keep the current hard candidates as a strong initialization so the short test starts from the validated checkpoint behavior.
- Apply `soft_proto_decoder` after candidate pruning: each of the 20 candidates attends over all 64 prototype tokens plus prototype summary features.
- Generate residual updates to endpoint, basis coeff, score, and refiner gate for the 20 candidates.
- Continue through existing basis reconstruction, coupled decoder, local-basis/refiner stack.
- Use prototype attention argmax only as a diagnostic/alignment id; do not rely on GT-prototype-aligned losses for the first short run.

Validation:

- Init from current best checkpoint with partial load.
- Train soft proto stack only, 8 epochs, short 111_days validation.
- Disable GT-prototype coeff/shape/FDE losses for this run; use winner-based projection guidance so the soft candidates are not forced into hard-router labels.

Result:

| Variant | ADE@20 | FDE@20 | ADE@5 | GLeV@20 | endpoint_var | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current checkpoint baseline | 0.2137 | 0.3046 | 0.3848 | 0.0898 | n/a | reference |
| soft proto residual, best epoch 1 | 0.2160 | n/a | n/a | n/a | n/a | worse |
| soft proto residual, epoch 8 | 0.2160 | 0.3082 | 0.3815 | 0.0896 | 4.6622 | worse |

Conclusion:

- Do not make `soft_proto_decoder` default.
- This is valid negative evidence for a medium/large upstream change: giving each hard candidate soft access to all prototype memories increased endpoint spread but did not improve ADE/FDE.
- Together with the failed decoder replacements, the current short-test evidence says the remaining gap is not solved by adding stronger heads on top of the current observed-trajectory context. A future large redesign would need a different source of intent signal or a full candidate-generation retraining schedule, not another partial checkpoint add-on.

## 2026-04-27 Anchor-Set Trajectory Decoder Plan

Purpose: test a more radical candidate-generation chain that removes per-sample hard router selection from the evaluated candidates. This is a performance-first sacrificial branch: if it works, later work can decide how to reintroduce ProtoBasis interpretability.

Design:

- Select 20 global prototype anchors by frequency-aware farthest-point sampling over prototype endpoints.
- Generate 20 candidates directly from those anchors, target/social context, and agent attention.
- Initialize endpoint and coeff residual heads to zero so the run starts from meaningful global anchor trajectories, not collapsed random endpoints.
- Keep the router logits only for reporting/prototype auxiliary compatibility; disable router-aligned losses for the short test.

Validation:

- Init from current best checkpoint with partial load.
- Freeze encoder/social/router, train anchor-set decoder + coupled/refiner stack first.
- Use winner-based projection guidance and best-of-20 ADE/FDE to see whether removing hard router candidate pruning can open a path toward 0.19.

Short result:

| Variant | ADE@20 | FDE@20 | ADE@5 | Decision |
| --- | ---: | ---: | ---: | --- |
| anchor-set smoke | 0.5183 | 0.6415 | 0.8539 | trainable but far |
| anchor-set frozen stack, epoch 3 | 0.2924 | 0.4157 | 0.4498 | learns quickly |
| anchor-set frozen stack, epoch 8 | 0.2694 | 0.3828 | 0.4183 | still far from 0.19 |

Conclusion:

- The chain is not dead: removing hard router candidates can learn from 0.52 to 0.27 in a short frozen-stack run.
- It is not yet competitive with the default checkpoint. The next valid check is to unfreeze the full model from this anchor-set checkpoint and see whether joint adaptation can approach the baseline range.

Unfrozen follow-up:

| Variant | ADE@20 | FDE@20 | ADE@5 | GLeV@20 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| anchor-set full unfreeze, epoch 12 | 0.2233 | 0.3227 | 0.3691 | 0.0895 | worse than current default |

Conclusion:

- Do not make `anchor_set_decoder` default.
- Removing hard router selection alone did not create a path to 0.19. After full unfreeze it approached the default range but stayed clearly worse than the current checkpoint (`0.2137/0.3046` on the same 50-batch reference).
- The remaining 0.19 attempt should now bypass the whole prototype/basis coefficient generation path, not keep adapting it.

## 2026-04-27 Direct Intention Trajectory Decoder Plan

Purpose: test whether the 0.19 gap is caused by the ProtoBasis candidate-generation path itself. This branch bypasses hard prototype top-k, basis coeff, basis projection, linear anchor, and post-hoc refiner as the main generator. It keeps only the temporal/social encoder and protocol-compatible outputs.

Design:

- Add `intention_trajectory_decoder`, default off.
- Use 20 learned intention queries, cross-attention to target temporal history and all observed agents, and direct 120-step local trajectory output.
- Use constant-velocity and endpoint-conditioned path priors only as initialization structure; the final path is not projected back into the fixed basis.
- Keep router logits only for compatibility/reporting; set prototype/basis/router-aligned training losses to zero in validation runs.

Validation:

- Short 111_days run from current best checkpoint with partial load.
- First freeze encoder/social/router and train only the direct intention decoder to test whether the decoder can exploit the existing representation.
- If 50-batch ADE@20 does not approach or beat `0.2137` quickly, do not spend full training on it. If it beats baseline by at least `0.015`, unfreeze and expand.

Result:

| Variant | ADE@20 | FDE@20 | ADE@5 | endpoint_var | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| direct intention, frozen encoder, epoch 8 | 1.1798 | 2.0989 | 1.2054 | 0.0057 | failed |

Conclusion:

- Do not continue this chain.
- Fully bypassing prototype/basis without endpoint anchors collapses the 20 learned queries into almost identical trajectories. The problem is not only decoder capacity; the model needs a reliable intent/endpoint candidate source.

## 2026-04-27 Aggressive LS Projection Distillation

Purpose: test whether the strong coefficient oracle (`coeff_blend_alpha=0.25` reaching about `0.18~0.22`, depending on sample window) can be made learnable by directly supervising all candidates toward LS coefficients under their predicted endpoints.

Validation:

- Init from current best checkpoint.
- Freeze encoder/router, train coeff-generation stack.
- Use `projection_supervision=all`, `lambda_projection_coeff=0.5`, `lambda_projection_path=2.0`.

Result:

| Variant | ADE@20 | FDE@20 | ADE@5 | Decision |
| --- | ---: | ---: | ---: | --- |
| aggressive LS projection, epoch 8 | 0.2381 | 0.3244 | 0.4098 | worse than current default |

Conclusion:

- Do not make this default and do not expand to full training.
- The LS oracle is real, but forcing every candidate toward the per-sample LS target disturbs endpoint/candidate structure. The learnable fix must improve endpoint-specific candidate allocation, not just increase LS supervision weight.

## 2026-04-27 Micro Endpoint Offset Plan

Purpose: use the endpoint oracle evidence without adding unfair context. Current `micro_per_proto=2` spends multiple candidates inside high-rank prototypes, but those micro candidates share the same endpoint and only differ in coeff/shape. This underuses the K=20 budget for endpoint/intent coverage.

Design:

- Add `micro_endpoint_offsets`, default off.
- After the prototype-conditioned query decoder, each micro candidate predicts its own endpoint delta from its query feature.
- Zero-initialize the endpoint-offset head so old checkpoints start exactly from the current default behavior.
- Keep the ProtoBasis route, basis, coupled decoder, and refiners; this changes candidate endpoint allocation, not the full method story.

Validation:

- First short run: train only the micro endpoint head from the current best checkpoint.
- If FDE/ADE improves materially, unfreeze the endpoint/coupled/refiner stack for a second short run.

Results:

| Variant | Eval | ADE@20 | FDE@20 | ADE@5 | Top1_ADE | rare_FDE@20 | GLeV@20 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current default | 50 batch | 0.2137 | 0.3046 | 0.3848 | 0.6063 | n/a | 0.0898 | reference |
| micro endpoint head only | 50 batch | 0.2138 | 0.3055 | 0.3838 | 0.6106 | 0.7572 | 0.0905 | no material gain |
| micro endpoint stack | 50 batch | 0.2104 | 0.3004 | 0.3825 | 0.6012 | 0.7499 | 0.0875 | promote to short baseline |
| current default | 100 batch | 0.2173 | 0.3130 | 0.3937 | 0.6202 | 0.7111 | 0.0903 | reference |
| micro endpoint stack | 100 batch | 0.2130 | 0.3069 | 0.3915 | 0.6147 | 0.6991 | 0.0879 | confirmed gain |

Conclusion:

- Promote `micro_endpoint_offsets` as the unified default for the next baseline because the gain survives a 100-batch check and improves ADE, FDE, Top1, and rare FDE.
- This is not the 0.19 solution. It recovers about `0.004` ADE on the 100-batch check, so endpoint allocation is a real but insufficient bottleneck.
- GLeV drops slightly, so future runs should monitor diversity before claiming a pure win.

## 2026-04-27 Micro Endpoint Stack Continuation to 0.19

Purpose: test whether the validated micro-endpoint chain can actually reach the 111_days `0.19` ADE target when trained through, before adding another architecture module.

Setup:

- Start checkpoint: `save_model_micro_endpoint_stack_continue2_e12_b512_111_mid/111_days/seed3407/best_best20.pt`.
- Train only the coefficient / endpoint / coupled / shape-refiner stack (`--freeze_backbone_except_coeff_decoder_stack`).
- Keep protocol `trajair_40to120_best20`, `K=20`, no weather or extra context.
- Evaluate short/mid with 200 test batches, then confirm the best checkpoint with full 111_days test.

Results:

| Variant | Eval | ADE@20 | FDE@20 | ADE@5 | Top1_ADE | rare_FDE@20 | GLeV@20 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| continue3, standard stack | 200 batch | 0.1925 | 0.2781 | n/a | 0.5735 | 0.6570 | 0.0778 | strong, full-test |
| continue3, standard stack | full 111_days | 0.1908 | 0.2728 | 0.3594 | 0.5726 | 0.6394 | 0.0770 | near target |
| 30-candidate internal expansion | 200 batch | 0.2090 | 0.3165 | 0.3821 | 0.5951 | 0.7598 | 0.1043 | stop; diversity up but ADE worse |
| full unfreeze low LR | 200 batch | 0.2016 | 0.2934 | n/a | n/a | 0.6532 | 0.0802 | stop; not the short-term bottleneck |
| continue4, low LR | 200 batch | 0.1914 | 0.2775 | n/a | 0.5737 | 0.6567 | 0.0769 | useful but weaker |
| continue4, low diversity pressure | 200 batch | 0.1908 | 0.2749 | n/a | 0.5728 | 0.6539 | 0.0783 | full-test |
| continue4, low diversity pressure | full 111_days | 0.1892 | 0.2697 | 0.3611 | 0.5720 | 0.6363 | 0.0776 | new best baseline |

Conclusion:

- The effective chain is not another large decoder: it is endpoint allocation (`micro_endpoint_offsets`) plus sustained training of the endpoint/coeff/coupled/refiner stack.
- Lowering stage-C diversity pressure from `2.0` to `1.0` improves ADE and FDE in the continuation run while keeping GLeV close to the previous full result (`0.0776` vs `0.0770`).
- The 30-candidate branch shows that simply adding more internal candidates increases endpoint spread/GLeV but hurts ADE/FDE, so the next architecture work should not be raw candidate expansion.
- Full backbone/router unfreeze is not the immediate bottleneck; it stayed around `0.2016` on the same 200-batch check.
- Promote the low-diversity micro-endpoint continuation as the current 111_days baseline: full `ADE@20=0.1892`, `FDE@20=0.2697`.

## 2026-04-27 Endpoint Coverage / FDE Repair Attempts

Purpose: after the 100-batch diagnostic showed that forcing GT prototype only improved ADE by about `0.0045`, test whether the remaining FDE gap can be repaired by endpoint candidate allocation rather than more coeff/path decoding.

Same-window reference:

| Variant | Eval | ADE@20 | FDE@20 | rare_FDE@20 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| current best checkpoint | 50 batch | 0.2052 | 0.2964 | 0.7884 | reference |

Tested variants:

| Variant | Eval | ADE@20 | FDE@20 | rare_FDE@20 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| endpoint set refiner, head only | 50 batch | 0.2050 | 0.2965 | 0.7878 | noise-level ADE, no FDE gain |
| endpoint set refiner + endpoint/coeff stack, strong FDE-min loss | 50 batch | 0.2068 | 0.2965 | 0.7868 | worse ADE, no FDE gain |
| motion-aware endpoint set head | 50 batch | 0.2050 | 0.2963 | 0.7868 | noise-level only |
| existing endpoint/coeff stack + mild FDE-min loss | 50 batch | 0.2064 | 0.2988 | 0.7936 | worse |
| proto-conditioned endpoint head + hit-only residual supervision | 50 batch | 0.2074 | 0.3015 | 0.7975 | worse |

Conclusion:

- Do not keep `endpoint_set_refiner`, endpoint coverage losses, or proto-conditioned endpoint as default.
- Directly pulling the nearest candidate endpoint toward GT reduces training `endpoint_min_fde`, but it does not transfer to validation FDE in short checks. This suggests the remaining FDE gap is not solved by a late endpoint correction head or by a simple min-FDE loss.
- The default rank-conditioned endpoint head is not obviously weak in practice; replacing it with prototype-conditioned endpoint generation disrupted the learned micro/coupled/refiner stack.
- Next endpoint-side investigation should be diagnostic first: split FDE by route/turn/altitude-change regimes and compare constant-velocity, prototype endpoint, micro endpoint, and final refined endpoint errors before adding another module.

## 2026-04-27 FDE Tail Diagnosis

Purpose: identify why the current best checkpoint has good ADE but still relatively high FDE before adding another endpoint module.

Checkpoint:

`save_model_micro_endpoint_stack_continue4_lowdiv_e8_b512_111_mid/111_days/seed3407/best_best20.pt`

Validation:

- 111_days test, first 100 batches, 51,200 samples.
- Current default candidate budget: `topk_proto=15`, `micro_per_proto=2`, `candidate_dense_topk=5`; this means rank 1-5 prototypes get two micro candidates and rank 6-15 get one candidate.

Main results:

| Split | Share | ADE@20 | FDE@20 | endpoint minFDE | GT proto rank p50/p90 | top15 hit | top20 hit | rare rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 100.0% | 0.1953 | 0.2818 | 0.3018 | 2 / 11 | 0.9375 | 0.9599 | 0.4653 |
| worst 10% by FDE | 10.0% | 0.4540 | 0.9596 | 0.9660 | 12 / 33 | 0.5959 | 0.7191 | 0.6406 |
| worst 5% by FDE | 5.0% | 0.5691 | 1.2307 | 1.2472 | 18 / 37 | 0.4359 | 0.5809 | 0.6559 |
| router hit | 93.8% | 0.1767 | 0.2427 | 0.2621 | 2 / 8 | 1.0000 | 1.0000 | 0.4602 |
| router miss | 6.2% | 0.4743 | 0.8686 | 0.8964 | 23 / 40 | 0.0000 | 0.3577 | 0.5407 |

Rank-bucket results:

| GT proto rank | Share | FDE@20 | endpoint minFDE | force GT proto FDE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 36.1% | 0.1972 | 0.2206 | 0.1972 |
| 2-5 | 41.2% | 0.2236 | 0.2439 | 0.2236 |
| 6-15 | 16.5% | 0.3903 | 0.3989 | 0.3903 |
| 16-30 | 4.6% | 0.7772 | 0.8010 | 0.6584 |
| 31-64 | 1.6% | 1.1285 | 1.1675 | 0.8633 |

Candidate-budget check:

| Candidate set | ADE@20/window | FDE@20/window | Notes |
| --- | ---: | ---: | --- |
| full20 current | 0.1953 | 0.2818 | rank 1-5 have two micro endpoints |
| sparse15 first micro only | 0.2213 | 0.3377 | removing second micro from rank 1-5 costs +0.0559 FDE |
| dense top5 only | 0.2592 | 0.4476 | rank 6-15 candidates are needed |

Best-FDE candidate source:

- Overall, the best FDE candidate comes from rank 1-5 in 71.9% samples and from rank 6-15 in 28.1%.
- In the worst 10%, best FDE candidate comes from rank 6-15 in 59.7%.
- In the worst 5%, best FDE candidate comes from rank 6-15 in 66.1%.
- The second micro candidate in rank 1-5 is useful: removing it increases overall FDE by about `0.056`, so those slots cannot be globally reallocated without loss.

Conclusion:

- FDE is a tail problem, not a uniform endpoint-head weakness.
- The tail is mainly caused by endpoint candidate coverage failure: in the worst 10%, only `59.6%` of samples include the GT prototype in the current top-15 candidate pool; in the worst 5%, only `43.6%` do.
- The final FDE is almost the same as endpoint minFDE (`0.9596` vs `0.9660` in worst 10%), so coeff/path/refiner modules cannot repair these cases after the endpoint candidate pool misses the right intent.
- Simple global candidate reallocation is unsafe because the duplicated top-5 micro endpoints are materially useful.
- Next repair should be conditional tail rescue: add or train a low-confidence/rare-aware rescue candidate mechanism that introduces rank 16-20 or rare endpoint alternatives only when the router distribution indicates tail risk. Do not repeat generic endpoint refiner, FDE-min loss, coeff-only, or raw candidate expansion experiments.

## 2026-04-28 Tail Rescue Candidate Attempts

Purpose: test whether the FDE tail can be repaired by conditionally replacing redundant top-5 micro endpoint slots with rescue endpoint candidates while keeping the public output budget at `K=20`.

Setup:

- Init checkpoint: `save_model_micro_endpoint_stack_continue4_lowdiv_e8_b512_111_mid/111_days/seed3407/best_best20.pt`.
- Short train: 8 epochs, 80 train batches/epoch, 50 eval batches.
- Main comparison window: old 100-batch diagnostic `ADE@20=0.1953`, `FDE@20=0.2818`.
- All variants kept weather/extra context off and did not change the public `K=20` evaluation protocol.

Results:

| Variant | Eval | ADE@20 | FDE@20 | rare_FDE@20 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| rank16-20 rescue + learned gate | 50 batch best | 0.1987 | ~0.287 | ~0.694 | not enough |
| rank16-20 rescue + learned gate | 100 batch | 0.1987 | 0.2871 | 0.6940 | worse than old 100-batch baseline |
| rank16-20 rescue + oracle selection | 100 batch diagnostic | 0.1947 | 0.2766 | 0.6694 | mechanism has small oracle value |
| rank16-20 rescue + always on | 100 batch diagnostic | 0.2151 | 0.3159 | 0.7655 | too many false positives |
| rank16-20 rescue + score-pooled top20 | 50 batch best | 0.1988 | 0.2891 | 0.7146 | worse |
| rank16-20 rescue + strong gate loss | 50 batch best | 0.2031 | 0.2940 | 0.7288 | worse |
| endpoint-router rescue source | 50 batch best | 0.2035 | ~0.303 | ~0.748 | worse |

Conclusion:

- The oracle result confirms the diagnosis direction but the learnable conversion is too weak. Rank-tail rescue can repair some FDE-tail cases only when the model is told exactly which samples need rescue.
- The learned gate under-selects tail cases (`1.7%` active vs `6.3%` target), and lowering the threshold quickly over-selects (`39%` to nearly all samples), which hurts ADE/FDE.
- Score-pooled rescue and endpoint-router rescue both increase false-positive rescue candidates and degrade `ADE@20/FDE@20`.
- Do not keep these rescue modules as default. If the code is present only as an experiment switch, do not use it in paper tables.
- Next FDE work should not be another late candidate swap. The stronger direction is to redesign endpoint generation itself, likely by predicting endpoint distributions/anchors jointly with the main decoder rather than trying to patch missed modes after the router has already committed.

## 2026-04-28 Architecture Cleanup / Runtime Diagnosis

Purpose: respond to the training-time and architecture-bloat issue after the failed large-branch tests.

Findings:

- The current strong 111_days checkpoints use the validated heavy default: `d_model=128`, `encoder_layers=4`, `topk_proto=15`, `micro_per_proto=2`, `candidate_dense_topk=5`, `batch_size=512`.
- Parameter count is still small in absolute terms: current default is about `1.66M` parameters, while a 96/3-layer profile is about `0.98M`.
- The failed experimental branches were default-off, so they did not materially slow active forward passes, but they made the code and paper story look patched together.
- The manually launched full run used `limit_eval_batches=0`; the older best continuation used `limit_eval_batches=200`. Full evaluation every epoch adds overhead, but the larger active default and full 111_days train pass are the main reason epoch time is no longer close to the old short-run timing.

Cleanup applied:

- Removed failed default-off branches from code: `BasisBridgeDecoder`, `TemporalBasisDynamicsDecoder`, `PrototypeDynamicsRolloutDecoder`, `SoftPrototypeSetDecoder`, `AnchorSetTrajectoryDecoder`, `IntentionTrajectoryDecoder`, and `BasisAwareCoeffDecoder`.
- Removed their CLI flags, freeze policies, loss terms, logging stats, and test-time construction compatibility.
- Kept only the active validated chain: hard prototype router, micro endpoint offsets, query decoder, two-stage endpoint/coeff update, coupled endpoint-coeff update, local basis, temporal residual refiner, endpoint shape refiner, and control shape refiner.
- Removed the unused target-temporal-sequence return path from `TemporalEncoder`; no active module consumes it after cleanup.

Decision:

- Do not immediately make the 96/3-layer compact profile default, because the best available 111_days result (`ADE@20=0.1830` on the current full run) comes from the 128/4-layer profile.
- If training speed becomes the priority, the next fair short test should compare the current default against a compact command using `--d_model 96 --ff_dim 192 --encoder_layers 3 --topk_proto 10 --micro_per_proto 2 --candidate_dense_topk 0` under the same `limit_train_batches` and `limit_eval_batches`.

## 2026-04-28 Training Hyperparameter Refit

Purpose: update the defaults for the cleaned current architecture instead of carrying the old long-training schedule.

Evidence from the ongoing 111_days full run:

- Best validation point is `epoch 15`, the first `joint_refiner` epoch: `ADE@20=0.1830`, `FDE@20=0.2746`.
- Later `joint_refiner` epochs keep reducing training loss but validation ADE/FDE drift worse (`epoch 24`: `ADE@20=0.1868`, `FDE@20=0.2840`).
- Therefore the refiner stage is useful as a short polish stage, not as a 50+ epoch long-training stage.

Default update:

- 111_days schedule becomes `phase_a_epochs=10`, `phase_b_epochs=4`, `phase_c_epochs=8` (`22` epochs total).
- 111_days early stopping becomes `early_stop_min_epoch=15`, `early_stop_patience=5`.
- 111_days training-time validation defaults to `limit_eval_batches=200`; pass `--limit_eval_batches 0` explicitly when a full validation sweep is needed.
- 7days defaults are unchanged for now because the evidence above is from 111_days.

## 2026-04-28 FDE Repair Follow-up

Purpose: test FDE-focused fixes after the 100-batch diagnosis showed that the remaining error is mainly endpoint candidate coverage in tail samples, not coeff/path reconstruction.

Reference:

- Checkpoint: `save_model_111days_current_default_full/111_days/seed3407/best_best20.pt`
- Eval window: 111_days first 100 test batches
- Baseline: `ADE@20=0.1872`, `FDE@20=0.2840`

Results:

| Variant | Eval | ADE@20 | FDE@20 | rare_FDE@20 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| endpoint set refiner, coverage 0.15 | 100 batch | 0.1872 | 0.2842 | ~0.650 | no gain |
| endpoint set refiner, coverage 1.0 | 100 batch | 0.1873 | 0.2840 | 0.6489 | no gain |
| top20 internal pool, random new-rank init | 100 batch | 0.1947 | 0.3028 | 0.6934 | worse |
| top20 internal pool, tail-rank init | 100 batch | 0.1942 | 0.3009 | 0.6909 | worse |
| router top-k margin, strong | 100 batch | 0.1881 | 0.2857 | 0.6568 | near but not better |
| router top-k margin, light | 100 batch | 0.1878 | 0.2852 | 0.6570 | near but not better |
| tail-balanced candidate keep | 100 batch | 0.1885 | 0.2872 | 0.6602 | worse |

Findings:

- Expanding the internal candidate pool increases theoretical tail coverage but disrupts the learned endpoint/score structure enough to hurt both ADE and FDE.
- Router top-k margin raises train-time top-k hit rate, but the gain does not convert into validation FDE improvement. Stronger routing pressure also hurts candidate quality.
- Reallocating duplicate micro endpoint slots from rank 4-5 to rank 6-7 is unsafe; the head-rank duplicate candidates are still needed.
- The remaining FDE gap is not fixed by late endpoint correction, raw candidate expansion, router-only loss, or simple static candidate reallocation.

Decision:

- Do not promote these FDE repair attempts to default.
- Revert the failed router-margin and tail-balanced code paths after recording the results.
- Keep the current best default checkpoint/config as baseline. The next credible FDE direction needs a new endpoint/intention generator trained from scratch or a new external intent signal; partial continuation patches are not showing enough headroom.

## 2026-04-28 Module and Loss Contribution Audit

Purpose: re-check each active structure and each active loss under the current best default, using short but matched comparisons instead of old mixed-window evidence.

Reference:

- Checkpoint: `save_model_111days_current_default_full/111_days/seed3407/best_best20.pt`
- Structure audit: same checkpoint, 111_days first 50 test batches, batch size 1024, one inference module disabled at a time.
- Loss audit: same checkpoint init, 111_days, 3 joint-refiner epochs, 80 train batches/epoch, 50 eval batches/epoch; compare best short-run ADE against the all-loss continuation.
- `Delta ADE` is variant minus baseline, so positive means the removed structure/loss was helping.

Structure contribution:

| Variant | ADE@20 | Delta ADE | FDE@20 | Delta FDE | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| default eval | 0.1839 | +0.0000 | 0.2774 | +0.0000 | baseline |
| no social | 0.2004 | +0.0165 | 0.3047 | +0.0273 | useful |
| no router / generic modes | 0.4900 | +0.3061 | 1.1226 | +0.8452 | essential |
| no temporal refiner | 0.1845 | +0.0006 | 0.2770 | -0.0004 | negligible at current checkpoint |
| no endpoint shape refiner | 0.3154 | +0.1315 | 0.2774 | +0.0000 | essential for mid-path ADE |
| no control shape refiner | 0.3534 | +0.1695 | 0.2774 | +0.0000 | essential for mid-path ADE |
| no shape refiners | 0.2229 | +0.0390 | 0.2774 | +0.0000 | keep; endpoint-preserving |
| no local basis | 0.1932 | +0.0093 | 0.2774 | +0.0000 | small useful ADE term |
| no support-aware local basis | 0.1901 | +0.0062 | 0.2774 | +0.0000 | small useful stabilizer |
| no two-stage decoder | 0.3939 | +0.2100 | 0.6327 | +0.3553 | essential |
| no two-stage endpoint update | 0.2041 | +0.0202 | 0.3583 | +0.0809 | important for endpoint/FDE |
| no two-stage coeff update | 0.1863 | +0.0024 | 0.2786 | +0.0012 | minor |
| no coupled decoder | 0.2120 | +0.0281 | 0.3870 | +0.1096 | important |
| no micro coeff anchors | 0.2973 | +0.1134 | 0.3629 | +0.0855 | essential candidate-shape prior |
| no micro endpoint offsets | 0.1851 | +0.0012 | 0.2803 | +0.0029 | small at current checkpoint |

Loss contribution:

| Variant | Best ADE@20 | Delta ADE | Best FDE@20 | Delta FDE | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| all losses continue | 0.1843 | +0.0000 | 0.2779 | +0.0000 | short-run baseline |
| no xyz | 0.1859 | +0.0016 | 0.2779 | +0.0000 | useful |
| no fde | 0.1879 | +0.0036 | 0.2927 | +0.0148 | important, especially FDE |
| no proto | 0.1835 | -0.0008 | 0.2766 | -0.0013 | not useful in late continuation; still needed for from-scratch router training unless separately proven |
| no endpoint residual | 0.1843 | +0.0000 | 0.2779 | -0.0001 | negligible late |
| no score | 0.1844 | +0.0001 | 0.2782 | +0.0002 | negligible late |
| no diversity | 0.1842 | -0.0001 | 0.2785 | +0.0006 | no ADE gain; may only regularize spread |
| no coeff L2 | 0.1845 | +0.0002 | 0.2780 | +0.0001 | negligible late |
| no smooth | 0.1845 | +0.0002 | 0.2782 | +0.0003 | negligible late |
| no gt proto shape | 0.1842 | -0.0001 | 0.2781 | +0.0002 | negligible late |
| no gt proto fde | 0.1845 | +0.0002 | 0.2783 | +0.0003 | negligible late |
| no gt proto coeff | 0.1846 | +0.0003 | 0.2781 | +0.0001 | negligible late |
| no anchor recon | 0.1843 | -0.0000 | 0.2779 | -0.0000 | negligible late |
| no projection coeff | 0.1844 | +0.0001 | 0.2778 | -0.0001 | negligible late |
| no projection path | 0.1843 | -0.0000 | 0.2778 | -0.0001 | negligible late |

Conclusion:

- Keep as core architecture: prototype router, two-stage decoder, coupled endpoint-coeff decoder, micro coeff anchors, endpoint shape refiner, control shape refiner, social aggregation.
- Keep but do not over-claim: local basis and support-aware local basis; they add small ADE gains.
- Candidate for simplification if speed/code clarity matters: temporal residual refiner and micro endpoint offsets; their current checkpoint contribution is only about `0.0006` and `0.0012` ADE respectively, though removing them from training still needs a fresh short train before changing defaults.
- For losses, the only clearly important late-stage objectives are `xyz` and especially `fde`. Most prototype/coeff/projection auxiliary losses now behave like early-training scaffolding or weak regularizers rather than final ADE drivers.
- Do not remove `proto` loss from from-scratch defaults based only on this continuation test; the structure audit still shows the router is essential, and proto supervision may be needed to learn it initially.

## 2026-04-28 Low-Contribution Cleanup

Purpose: remove low-contribution or failed branches from the active code path after the contribution audit, while keeping the confirmed core modules intact.

Removed from model/code:

- `TemporalResidualRefiner`: current checkpoint contribution was only `+0.0006` ADE when disabled.
- `micro_endpoint_offsets`: current checkpoint contribution was only `+0.0012` ADE and it added an extra head plus compatibility burden.
- `EndpointSetRefiner`: multiple FDE repair attempts showed no validation gain.
- `query_decoder.gate_head`: only fed the removed temporal residual refiner.

Removed from training losses/flags:

- `anchor_recon` and predicted-anchor projection losses (`projection_coeff`, `projection_path`): late-stage contribution was noise-level and aggressive variants were worse.
- endpoint coverage/delta losses tied to the failed endpoint-set refiner.
- corresponding CLI flags, freeze modes, logging fields, and metadata fields.

Kept:

- Core architecture: router, social aggregation, micro coeff anchors, two-stage endpoint/coeff decoder, coupled endpoint-coeff decoder, local basis, endpoint shape refiner, control shape refiner.
- Core losses/regularizers: `xyz`, `fde`, `proto`, endpoint residual, score, diversity, coeff L2, smoothness, and GT-prototype shape/FDE/coeff. Some are weak in late continuation, but they may still stabilize from-scratch training and are cheaper than the removed branches.

Compatibility:

- Old checkpoints can still be evaluated/initialized: removed state keys are filtered (`refiner.*`, `micro_endpoint_head.*`, `endpoint_set_refiner.*`, `query_decoder.gate_head.*`).
- Old optimizer state may be skipped on resume after architecture cleanup because the parameter set changed.

Smoke check:

- Remote compile passed after cleanup.
- Old best checkpoint still loads under the cleaned model.
- 111_days 50-batch eval using the cleaned code: `ADE@20=0.1858`, `FDE@20=0.2806`.
- Pre-cleanup 50-batch reference was `ADE@20=0.1839`, `FDE@20=0.2774`; the cleanup costs about `+0.0019` ADE on old weights, which matches the prior low-contribution diagnosis and should be rechecked by fresh training if this cleaned profile becomes the final default.
