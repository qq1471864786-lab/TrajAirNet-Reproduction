# ProtoBasis-Net

Current active project:

- `ProtoBasis-Net (Prototype-Conditioned Residual Basis Query Network for Terminal Airspace Forecasting)`

Current default protocol:

- `protocol_name = trajair_40to120_best20`
- `dataset_variant = social`
- `dataset_name = 111_days`
- `obs = 40`
- `preds = 120`
- `max_agents = 7`
- `topk_proto = 5`
- `micro_per_proto = 4`
- `basis_dim = 16`
- `best@5 / best@20`

Current default training command:

```bash
python train.py 111_days --device cuda:0
```

Continue training the same run:

```bash
python train.py 111_days --device cuda:0 --resume --extra_epochs 5
```

Current default evaluation command:

```bash
python test.py save_model/111_days/seed3407/last.pt --device cuda:0
```

Primary reported metrics:

- `ADE@5`
- `FDE@5`
- `ADE@20`
- `FDE@20`
- `GLeV_report@5`
- `GLeV_report@20`
- `GLeV_raw@5`
- `GLeV_raw@20`
- `rare_FDE@20`

Ablation switches:

- `--disable_social`
- `--disable_router`
- `--disable_refiner`

Protocol scaffolding:

- implemented: `trajair_40to120_best20`
- scaffolded for later appendix alignment: `legacy_11_best5`, `legacy_16_best5`

Project note:

- old ACT / HAINet / kinematic query decoder lines are no longer the active workflow
- this repository is now reserved for ProtoBasis-Net only
- roadmap is recorded in `notes/PROTOBASIS_EXECUTION_ROADMAP.md`

Remote workflow:

- Sync modified source files to the server:

```bash
python scripts/sync_remote.py
```

- Launch a remote training job with tracked logs:

```bash
python scripts/launch_remote_train.py -- --dataset_variant social --dataset_name 111_days
```

- Inspect the latest remote run with the compact "eyes" view:

```bash
python scripts/server_status.py
```

- Compact JSON for automation / low-token parsing:

```bash
python scripts/server_status.py --json
```

- Expand only when needed:

```bash
python scripts/server_status.py --show-system
python scripts/server_status.py --show-epochs
python scripts/server_status.py --show-logs
python scripts/server_status.py --json --full
```

Automation note:

- `scripts/sync_remote.py` performs incremental source sync and records the last sync timestamp locally.
- `scripts/launch_remote_train.py` starts remote training under `.remote_runs/<timestamp_name>/` and preserves command/stdout/stderr.
- `scripts/server_status.py` defaults to a compact status summary so remote checks stay token-efficient.
- Use `--json` for compact machine-readable snapshots and `--json --full` only for deep inspection.
