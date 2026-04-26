# 2026-04-26 GT-Prototype Coefficient Baseline

Status: default baseline for the next `111_days` improvement round.

Update 2026-04-26 late:

`111_days` now promotes rank-adaptive candidate allocation as the next ADE-oriented default:

- `topk_proto = 15`
- `micro_per_proto = 2`
- `candidate_dense_topk = 5`
- Effective candidates remain 20: ranks 1-5 keep 2 micro variants each, ranks 6-15 keep 1 variant each.

Short/medium validation with the same `limit_train_batches=120`, `limit_eval_batches=20`:

| Run | Epochs | ADE@20 | FDE@20 | rare_FDE@20 | ADE@5 | GLeV@20 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| previous default, e26 | 10/4/12 | 0.24345 | 0.32565 | 0.84922 | 0.41103 | 0.10705 | superseded |
| alloc15 dense5, e26 | 10/4/12 | 0.23876 | 0.32157 | 0.85307 | 0.39139 | 0.09892 | positive ADE/FDE, rare slightly worse |
| previous default control, e34 | 10/4/20 | 0.23994 | 0.31876 | 0.83969 | 0.40590 | 0.10493 | control |
| alloc15 dense5, e34 | 10/4/20 | 0.23504 | 0.32109 | 0.84510 | 0.38768 | 0.09592 | promoted for ADE |

Rationale:

`topk=20,micro=1` previously improved endpoint/rare behavior but lost ADE because every prototype had only one micro variant. The promoted allocation keeps two micro variants for high-confidence prototypes while using single tail variants to improve prototype coverage. This targets router-miss without increasing the evaluated candidate budget.

Trade-off:

The new default improves ADE@20 and ADE@5 consistently, but FDE@20, rare_FDE@20, and GLeV@20 are slightly worse than the e34 control. Keep it as an ADE-oriented baseline; do not claim it improves every metric.

Update 2026-04-26:

`111_days` now promotes anchor-adaptive reconstruction distillation as the next default on top of the GT-prototype coefficient baseline:

- `anchor_recon_supervision = gt_proto`
- `lambda_anchor_recon = 0.10`

Short validation (`10/4/12`, train/eval limits `120/20`):

| Run | ADE@20 | FDE@20 | rare_FDE@20 | Decision |
|---|---:|---:|---:|---|
| prior promoted short baseline | 0.24466 | 0.33162 | 0.85771 | superseded |
| anchor-recon GT-proto, lambda 0.10 | 0.24345 | 0.32565 | 0.84922 | promoted |

Strong-checkpoint continuation evidence from `save_model_111days_topk10_m2`:

| Run | ADE@20 | FDE@20 | rare_FDE@20 | Decision |
|---|---:|---:|---:|---|
| checkpoint eval, limit20 | 0.2195 | 0.3294 | 0.8326 | reference |
| GT coeff continuation | 0.21646 | 0.31888 | 0.81501 | positive |
| anchor-recon GT-proto continuation | 0.21596 | 0.31476 | 0.81170 | best continuation |

Rationale:

Diagnostics showed `pred_endpoint_plus_global_basis_ls` can reach about `0.043` ADE on the sampled subset, while the actual model is about `0.222`; this points to a coefficient/path target mismatch around predicted endpoint anchors. The promoted loss distills each GT-prototype-aligned candidate coarse path toward its least-squares reconstruction under that candidate's own endpoint anchor.

Code commit:

- `e1716bc Add GT-prototype coefficient supervision`

Default behavior:

- `111_days` under `trajair_40to120_best20` now defaults to:
  - `topk_proto = 10`
  - `micro_per_proto = 2`
- `111_days` under `trajair_40to120_best20` now enables:
  - `lambda_gt_proto_shape = 0.15`
  - `lambda_gt_proto_fde = 0.05`
  - `lambda_gt_proto_coeff = 0.10`
- `7days*` keeps the prior `topk_proto = 5`, `micro_per_proto = 4`, and these losses disabled by default.
- `proto_focal_gamma` and `proto_freq_weight_power` remain disabled by default because the router-only ablation did not improve ADE.

Short validation protocol:

```bash
python train.py --dataset_name 111_days --topk_proto 10 --micro_per_proto 2 \
  --phase_a_epochs 10 --phase_b_epochs 4 --phase_c_epochs 12 --extra_epochs 0 \
  --batch_size 512 --eval_batch_size 1024 --limit_train_batches 120 --limit_eval_batches 20
```

Server short-run evidence:

| Run | ADE@20 | FDE@20 | Top1_ADE | rare_FDE@20 |
|---|---:|---:|---:|---:|
| baseline before change | 0.26344 | 0.34770 | 0.64611 | 0.90810 |
| GT loss only | 0.24696 | 0.33369 | 0.64889 | 0.87086 |
| GT loss only, stronger coeff | 0.24466 | 0.33162 | 0.63979 | 0.85771 |
| GT loss + router focal/freq | 0.24892 | 0.33483 | 0.65155 | 0.86600 |
| router focal/freq only | 0.26382 | 0.35333 | 0.65935 | 0.90691 |

Decision:

Use GT-prototype shape/FDE/coefficient supervision as the new default baseline, with `lambda_gt_proto_coeff = 0.10`.
Do not promote router focal/frequency weighting for now.

Known remaining bottleneck:

Bucket diagnostics show the new loss mainly improves router-hit, rare, and top1-miss samples. Router-miss samples remain weak and should be the next investigation target.

Follow-up probes not promoted:

| Probe | ADE@20 | FDE@20 | Top1_ADE | rare_FDE@20 | Decision |
|---|---:|---:|---:|---:|---|
| `topk_proto=20`, `micro_per_proto=1` | 0.24706 | 0.32504 | 0.63609 | 0.83087 | Not promoted: ADE did not beat coeff=0.10 baseline. |
| `endpoint_residual_supervision=hit_only` | 0.24710 | 0.33451 | 0.64769 | 0.87523 | Not promoted: no ADE gain. |
| `lambda_gt_proto_coeff=0.15` | 0.24413 | 0.33534 | 0.63444 | 0.87026 | Not promoted: ADE gain is tiny and FDE/rare regress. |
| `lambda_gt_proto_coeff=0.20` | 0.24521 | 0.33517 | 0.64305 | 0.87409 | Not promoted: worse than coeff=0.10 baseline. |
| `endpoint_conditioning=proto` | 0.25857 | 0.33708 | 0.63926 | 0.90616 | Not promoted: lagged before OOM at refiner start. |
