# 2026-04-26 GT-Prototype Coefficient Baseline

Status: default baseline for the next `111_days` improvement round.

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
