# ProtoBasis-Net Roadmap

## Locked Main Protocol

- protocol_name: `trajair_40to120_best20`
- dataset_variant: `social`
- main datasets: `111_days`, `7days1`, `7days2`, `7days3`, `7days4`
- obs: `40`
- preds: `120`
- modes: `topk_proto=5`, `micro_per_proto=4`, total `20`
- primary metrics: `ADE@5`, `FDE@5`, `ADE@20`, `FDE@20`
- secondary metrics: `GLeV_report@5/@20`, `GLeV_raw@5/@20`, `rare_FDE@20`, `Top1_ADE/FDE`, `latency_bs1/bs16`

## Current Code Defaults Under Validation

- Stage A: `10 epochs`
  - `force_gt_proto=True`
  - `enable_refiner=False`
  - `rank_weight=0.0`
  - `div_weight=0.0`
  - `lr=3e-4`
- Stage B: `12 epochs`
  - `force_gt_proto=False`
  - `enable_refiner=False`
  - `rank_weight=0.0`
  - `div_weight=0.0`
  - `lr starts at 2e-4`
- Stage C: `43 epochs`
  - `force_gt_proto=False`
  - `enable_refiner=True`
  - `rank_weight=1.0`
  - `div_weight=1.0`
  - `lr starts at 8e-5`
  - `rare_weight=1.5`

## Current Score Supervision Under Validation

- `score_loss`
  - no longer pure winner-only hard classification
  - now mixes:
    - quality-aware soft target supervision
    - a retained hard winner component
- default mix:
  - `score_hard_mix=0.25`
  - `score_fde_weight=0.75`
  - `score_soft_temperature=0.35`
- supporting default loss changes:
  - `lambda_fde=1.0`
  - `lambda_proto=0.35`
  - `lambda_score=0.30`

## Required Main Experiments

1. `111_days` ProtoBasis-Net 1 seed sanity.
2. `111_days` strongest kinematic baseline 1 seed.
3. `111_days` ProtoBasis-Net 3 seeds.
4. `111_days` strongest kinematic baseline 3 seeds.
5. Core ablations:
   - `--disable_router`
   - `--disable_refiner`
   - `--disable_social`
6. `7days1~4` unified 40/120 runs.
7. `111_days` and `7days1~4` final 5-seed runs.
8. Latency profiling with `bs=1` and `bs=16`.

## Planned Appendix Protocols

These are scaffolded in code but not implemented yet:

- `legacy_11_best5`
- `legacy_16_best5`

They are appendix-only alignment protocols for ASCENT / TrajAirNet legacy settings.

## Current Code Hooks Reserved For Future Work

- `model/protocols.py`
  - unified protocol already enabled
  - legacy protocols scaffolded
- `train.py`
  - protocol-driven `obs/preds`
  - three-stage schedule locked
- `test.py`
  - batch-size latency profiling for future main table
- `model/run_logging.py`
  - stores extra metadata needed for reproducibility
- `model/provenance.py`
  - protocol hash / artifact hash / basis hash / git commit

## Reproducibility Requirements

Every formal run should retain:

- `protocol_hash`
- `artifact_hash`
- `basis_hash`
- `git_commit`
- `seed`
- `run_config.json`
- `epoch_metrics.jsonl`
- `run_summary.json`
