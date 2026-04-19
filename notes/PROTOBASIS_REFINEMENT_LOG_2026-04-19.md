# ProtoBasis-Net Refinement Log (2026-04-19)

## Goal

Keep the current `ProtoBasis-Net` innovation intact while improving the full paper-facing metric set, not only `ADE@20`.

Priority metrics for the next refinement round:

- `ADE@5`
- `FDE@5`
- `ADE@20`
- `FDE@20`
- `Top1_ADE/FDE`
- `rare_FDE@20`
- `GLeV_report@20`

## Evidence From The 65-Epoch Main Run

Main findings from the completed `111_days` run:

1. `ProtoBasis + Refiner` is the correct backbone.
   - Best metrics already reached:
     - `ADE@20 = 0.2811`
     - `FDE@20 = 0.3787`
     - `rare_FDE@20 = 0.8664`
   - This means the method direction is valid and should not be replaced.

2. `joint_no_refiner` is too long.
   - `epoch 11-45` consumed a large amount of training budget with limited return on paper-facing metrics.
   - The large metric drop happened immediately after entering `joint_refiner`.

3. Current `score head` supervision is misaligned with the final objective.
   - The training uses winner-only hard classification for candidate scores.
   - During the strong `joint_refiner` stage, paper-facing metrics kept improving while `score_loss` did not improve with them.
   - This indicates that current score supervision is not the right pressure for `ADE@5/FDE@5/Top1`.

4. Current artifact structure is shape-strong but endpoint-weak.
   - Diagnostic on the current artifact (`111_days`, 500 train samples):
     - `proto_base_ADE_mean = 0.8265`
     - `proto_base_FDE_mean = 0.6457`
     - `recon_ADE_mean = 0.0739`
     - `recon_FDE_mean = 0.6457`
   - Interpretation:
     - global basis can reconstruct shape very well
     - but endpoint/FDE remains anchored by the prototype summary
   - Conclusion:
     - future method-level work should target the prototype/end-anchor layer first, not replace the whole generator

## Confirmed Changes For This Iteration

This iteration intentionally avoids large architectural replacement.

Confirmed changes:

1. Training schedule refinement
   - shorten `joint_no_refiner`
   - move more epochs into `joint_refiner`
   - disable or strongly weaken `rank/div` pressure before refiner is active

2. Score supervision refinement
   - replace winner-only hard score classification with quality-aware soft supervision
   - keep ranking pressure, but align score learning with overall candidate quality instead of only the single winner

Not changed in this iteration:

- no diffusion
- no GNN/conflict graph
- no MoE
- no full goal-first rewrite
- no pure local-basis redesign

## Controlled Before-Change Proxy Baseline

Purpose:

- obtain a fast, controlled before/after comparison on the main dataset
- same seed, same batch sizes, same batch limits
- compare directionally, not as a final benchmark

Command:

```bash
conda run -n trajair python train.py 111_days --device cuda:0 --save_dir tmp_compare_before --phase_a_epochs 10 --phase_b_epochs 16 --phase_c_epochs 0 --batch_size 512 --eval_batch_size 1024 --limit_train_batches 120 --limit_eval_batches 20
```

Result (`epoch 26`, current code before refinement):

- `ADE@5 = 0.5397`
- `FDE@5 = 0.9493`
- `ADE@20 = 0.5055`
- `FDE@20 = 0.7757`
- `rare_FDE@20 = 2.1215`
- `Top1_ADE = 0.7272`
- `Top1_FDE = 1.4193`
- `score_entropy = 1.8140`
- `endpoint_var = 2.3737`

Loss snapshot:

- `xyz = 0.0760`
- `fde = 0.2259`
- `proto = 2.1913`
- `score = 1.8396`
- `rank = 0.1245`
- `div = 0.3581`
- `winner_ADE = 0.5146`

## Planned After-Change Validation

After implementing schedule + score supervision changes, rerun a matched proxy experiment and compare:

- `ADE@5`
- `FDE@5`
- `ADE@20`
- `FDE@20`
- `Top1_ADE`
- `Top1_FDE`
- `rare_FDE@20`
- `score_entropy`

Decision rule:

- keep the change only if it improves or preserves `@20` while clearly improving `@5/FDE/Top1`
- if `ADE@20` improves but `ADE@5/FDE@5/Top1` regress, the change is not good enough for the final paper line

## Controlled After-Change Proxy Result

Command:

```bash
conda run -n trajair python train.py 111_days --device cuda:0 --save_dir tmp_compare_after --phase_a_epochs 10 --phase_b_epochs 12 --phase_c_epochs 4 --batch_size 512 --eval_batch_size 1024 --limit_train_batches 120 --limit_eval_batches 20
```

Code-side changes included in this run:

- shorter `joint_no_refiner`
- earlier `joint_refiner`
- `Stage B` rank/div turned off
- quality-aware soft score supervision
- lower default `lambda_score`, slightly lower `lambda_proto`, slightly higher `lambda_fde`

Result (`epoch 26`, after refinement):

- `ADE@5 = 0.4856`
- `FDE@5 = 0.7627`
- `ADE@20 = 0.4100`
- `FDE@20 = 0.5346`
- `rare_FDE@20 = 1.4607`
- `Top1_ADE = 0.7454`
- `Top1_FDE = 1.4258`
- `GLeV_report@20 = 0.1743`
- `score_entropy = 2.7254`

Direct comparison versus the before-change proxy:

- `ADE@5`: `0.5397 -> 0.4856` (`-0.0541`)
- `FDE@5`: `0.9493 -> 0.7627` (`-0.1865`)
- `ADE@20`: `0.5055 -> 0.4100` (`-0.0955`)
- `FDE@20`: `0.7757 -> 0.5346` (`-0.2411`)
- `rare_FDE@20`: `2.1215 -> 1.4607` (`-0.6608`)
- `Top1_ADE`: `0.7272 -> 0.7454` (`+0.0182`, worse)
- `Top1_FDE`: `1.4193 -> 1.4258` (`+0.0066`, worse)
- `GLeV_report@20`: `0.1459 -> 0.1743` (`+0.0284`, worse if lower is better)
- `score_entropy`: `1.8140 -> 2.7254` (`+0.9114`)

## Interim Decision

This refinement direction is useful, but not yet the final paper-ready setting.

What clearly improved:

- `ADE@5`
- `FDE@5`
- `ADE@20`
- `FDE@20`
- `rare_FDE@20`

What did not improve enough:

- `Top1_ADE/FDE`
- `GLeV_report@20`

Interpretation:

- earlier refiner activation is strongly validated
- reducing the hard winner-only score pressure is directionally correct
- but current score supervision is now slightly too soft for top-1 sharpness and diversity retention

Practical conclusion:

- keep the schedule refinement direction
- keep quality-aware score supervision as the new base direction
- do one more score-head refinement before calling this the final default for the paper

## Follow-Up Rejected Experiments

Two deeper score-side follow-up experiments were tested after the validated proxy above.
They are recorded here for traceability, but neither is kept as the new default.

### Rejected Experiment A: Post-Refiner Rerank Head

Idea:

- keep ProtoBasis generation unchanged
- add a small rerank head after the refiner so the final score sees the final trajectory instead of only the pre-refiner hidden state

Proxy command:

```bash
conda run -n trajair python train.py 111_days --device cuda:0 --save_dir tmp_compare_rerank --phase_a_epochs 10 --phase_b_epochs 12 --phase_c_epochs 4 --batch_size 512 --eval_batch_size 1024 --limit_train_batches 120 --limit_eval_batches 20
```

Result (`epoch 26`):

- `ADE@5 = 0.5024`
- `FDE@5 = 0.7802`
- `ADE@20 = 0.4080`
- `FDE@20 = 0.5025`
- `rare_FDE@20 = 1.3157`
- `Top1_ADE = 0.8118`
- `Top1_FDE = 1.5151`
- `GLeV_report@20 = 0.2140`

Decision:

- `ADE@20/FDE@20/rare_FDE@20` improved slightly
- but `ADE@5/FDE@5/Top1/GLeV` regressed too much
- this direction is rejected as a new default

### Rejected Experiment B: Sharpened Soft Score Targets

Idea:

- keep the validated schedule refinement
- sharpen the quality-aware score supervision by increasing hard-score mixing and restricting the soft target to top-quality candidates

Proxy command:

```bash
conda run -n trajair python train.py 111_days --device cuda:0 --save_dir tmp_compare_sharp --phase_a_epochs 10 --phase_b_epochs 12 --phase_c_epochs 4 --batch_size 512 --eval_batch_size 1024 --limit_train_batches 120 --limit_eval_batches 20
```

Result (`epoch 26`):

- `ADE@5 = 0.4889`
- `FDE@5 = 0.7637`
- `ADE@20 = 0.4163`
- `FDE@20 = 0.5417`
- `rare_FDE@20 = 1.5024`
- `Top1_ADE = 0.7457`
- `Top1_FDE = 1.4054`
- `GLeV_report@20 = 0.1643`

Decision:

- `Top1_FDE` and `GLeV_report@20` improved slightly
- but `ADE@5/ADE@20/FDE@20/rare_FDE@20` all regressed relative to the validated base
- this direction is also rejected as a new default

## Current Confirmed Position

The current best validated default for the next main training run remains:

- shorter `joint_no_refiner`
- earlier `joint_refiner`
- `Stage B` rank/div turned off
- quality-aware soft score supervision from the validated proxy setting

The next justified method-level direction is not another aggressive score-head rewrite.
The artifact diagnostic still indicates that the deeper long-horizon bottleneck remains in the prototype/end-anchor layer rather than in global shape reconstruction.

## 2026-04-20 Mid-Run Review And Training-Length Decision

Current full-run status snapshot (`111_days`, unified `40/120`, main run still in progress):

- by `epoch 56`, the run already surpassed the previous best completed run on:
  - `ADE@20`
  - `FDE@20`
  - `rare_FDE@20`
  - `GLeV_report@20`
- latest bests observed during the current run:
  - `ADE@20 = 0.2471 @ e56`
  - `FDE@20 = 0.3395 @ e54`
  - `rare_FDE@20 = 0.7896 @ e54`
  - `GLeV_report@20 = 0.0694 @ e54`
  - `ADE@5 = 0.3981 @ e46`

Deep read of the late-stage behavior:

- `joint_refiner` remains the phase that actually produces the paper-facing gains
- `epoch 54-56` still produced new best updates, so the run had not fully saturated by `65 epochs`
- `ADE@20/FDE@20/rare_FDE@20/GLeV_report@20` are still the dimensions with the most meaningful late-stage movement
- `ADE@5` and especially `Top1_ADE/FDE` remain the relatively weaker side
- `score_entropy` collapses again late in training, so simply extending training is not expected to fully solve `Top1`

Decision:

- keep the validated `10 / 12 / 43` main schedule unchanged
- increase the default total training length from `65` to `100`
- implement this conservatively by appending `35` extra refiner-stage epochs
- slightly raise the low-lr floor from `1e-5` to `1.5e-5`

Why this is the preferred change:

- it preserves the already validated 65-epoch behavior
- it extends only the stage that is still producing gains
- it avoids re-opening the earlier schedule design question
- it gives the strong `@20 / rare / GLeV` line more room to improve without disturbing the backbone

Expected benefit:

- most likely further small gains in `ADE@20`, `FDE@20`, `rare_FDE@20`, and `GLeV_report@20`

Expected limitation:

- `ADE@5` and `Top1` may not improve much from extra epochs alone
- those likely still require a later method-side change around prototype/end-anchor or ranking behavior
