# Coefficient Oracle Blend Probe

Date: 2026-04-27

## Question

Check whether the remaining 111_days ADE bottleneck is really in the predicted coefficient/path-shape branch after the current predicted endpoint anchor, without running training.

Probe:

```text
coeff_blend = coeff_pred + alpha * (coeff_ls_oracle - coeff_pred)
alpha = 0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0
```

The LS coefficient is solved per candidate from the current predicted endpoint anchor and GT local future. The original candidate scores, local basis branch, refiner, router, and endpoint predictions are kept unchanged.

## Setup

- Remote repo: `/3250604003/ProtoBasis-Net`
- Checkpoint: `save_model_probe_coupled_proj_fixed_111_e26/111_days/seed3407/best_best20.pt`
- Dataset: `111_days`, test split
- Device: GPU 3
- Batches: 20
- Batch size: 1024
- AMP: disabled
- Output JSON on remote: `notes/experiments/2026-04-27_coeff_oracle_blend.json`

Sanity check: official `test.py` under the same 20-batch/no-AMP setup reports `ADE@20=0.2284`, `FDE@20=0.3080`, matching the probe at `alpha=0`.

## Results

| alpha | ADE@20 | FDE@20 | router-hit ADE@20 | router-miss ADE@20 | rare ADE@20 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.22843 | 0.30798 | 0.20764 | 0.50598 | 0.26402 |
| 0.02 | 0.22435 | 0.30798 | 0.20393 | 0.49692 | 0.25924 |
| 0.05 | 0.21825 | 0.30797 | 0.19839 | 0.48336 | 0.25210 |
| 0.10 | 0.20814 | 0.30795 | 0.18922 | 0.46075 | 0.24024 |
| 0.25 | 0.17828 | 0.30790 | 0.16217 | 0.39328 | 0.20512 |
| 0.50 | 0.13111 | 0.30782 | 0.11961 | 0.28468 | 0.14972 |
| 1.00 | 0.07610 | 0.30766 | 0.07090 | 0.14553 | 0.08595 |

Mean coefficient gap:

```text
mean_l1 = 2.28591
mean_l2 = 13.11074
```

## Interpretation

This is strong positive evidence that the current main ADE ceiling is not simply router recall or endpoint/FDE. With endpoint predictions and scores fixed, a small movement of the predicted coefficients toward a per-candidate LS path-shape target reduces ADE substantially:

- `alpha=0.10`: `0.22843 -> 0.20814`, about `-0.02029`
- `alpha=0.25`: `0.22843 -> 0.17828`, about `-0.05015`

FDE barely changes across alphas because this probe mostly corrects trajectory shape under the same endpoint anchor. That is useful: it isolates the coefficient/path-shape branch instead of hiding endpoint improvements inside the oracle.

The effect appears in router-hit, router-miss, and rare samples, so the issue is not only the router-miss bucket. Router-miss remains worse, but it is also shape-correctable.

## Decision

The next implementation target should be a learnable coefficient/path-shape correction mechanism that directly reduces the gap between predicted coefficients and LS-style path coefficients under the predicted endpoint anchor.

Do not start full training from this probe alone. First run a short implementation validation that tests whether a learned correction can capture a small fraction of the oracle blend gain without using GT at inference.

