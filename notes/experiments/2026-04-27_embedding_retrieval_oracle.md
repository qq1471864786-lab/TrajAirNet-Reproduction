# Embedding Retrieval Oracle

Date: 2026-04-27

## Question

Check whether the current observation-side representation contains enough information to support `111_days` ADE near `0.19`.

If nearest-neighbor retrieval from the model's observed-history embeddings can reach `<=0.19` with top-20 train futures, then the decoder is leaving useful representation information unused. If it cannot, then the remaining gap is mainly future-intent ambiguity or missing conditioning signal rather than a simple decoder lookup failure.

## Setup

- Checkpoint: `save_model_probe_control_continue_e8_111_short/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`
- Train bank: 50,000 samples
- Test subset: 4,096 samples
- Candidate budget: top-20 retrieved train futures
- Metric frame: local future frame; Euclidean distances are rotation/translation invariant with the official local frame.

Features tested:

- `target_ctx`: pooled target context from the trained temporal encoder/social stack.
- `target_scene_ctx`: target context concatenated with scene context.
- `router_hidden`: hidden vector before prototype logits.
- `proto_prob`: router softmax over prototypes.
- `observed_tail20`: handcrafted target observed tail and neighbor geometry.
- `future_summary_5d`: future-derived 5D summary, used only as a cheating coverage oracle.

## Results

| retrieval key | top20 minADE | top20 minFDE | top1 ADE | rare top20 minADE |
| --- | ---: | ---: | ---: | ---: |
| future_summary_5d oracle | 0.1773 | 0.1381 | 0.3989 | 0.2021 |
| target_ctx | 0.2461 | 0.3671 | 0.7632 | 0.2869 |
| router_hidden | 0.2494 | 0.3685 | 0.7597 | 0.2898 |
| target_scene_ctx | 0.2608 | 0.3855 | 0.7764 | 0.3026 |
| proto_prob | 0.2746 | 0.4080 | 0.7891 | 0.3201 |
| observed_tail20 | 0.3021 | 0.4727 | 0.9767 | 0.3372 |

Matched model checkpoint official 50-batch eval from the same short baseline family:

| model | ADE@20 | FDE@20 |
| --- | ---: | ---: |
| ProtoBasis current default continuation | 0.2137 | 0.3046 |

## Interpretation

The train bank contains similar future shapes: with cheating future summary retrieval, top-20 minADE reaches `0.1773`, below the `0.19` target.

However, none of the observation-only embeddings retrieve futures near `0.19`. The best observation-derived retrieval is `target_ctx` at `0.2461`, which is worse than the model's own generated candidates (`0.2137`). This means the model is already doing better than simple nearest-neighbor lookup over its observed representation.

## Decision

The remaining gap is not just a weak nearest-neighbor/decoder readout from the current observation embedding. To reach `0.19`, the model needs a stronger way to infer future intent/shape from available observations, or additional conditioning signal. Simple history-memory, projection-mask changes, and retrieval-style fixes are unlikely to be enough.

Next investigation target: training-stage coupling. The current `enable_refiner=False` stage disables only the original temporal refiner; endpoint/control shape refiners still run. This may prevent a clean basis/coeff warm-up and could contribute to weak coefficient learning. Test this with a short gated-refiner schedule before considering a larger replacement of the ProtoBasis decoder.
