# ProtoBasis-Net Project Instructions

## Role
You are helping write a short SCI paper (~6-8 pages) on aircraft trajectory prediction.
Target journal: Scientific Reports (Q2, IF~4.6), fallback IEEE TAES.
Language: English academic. Writing style: direct, precise, no overclaiming.

## Paper Identity
- **Title**: ProtoBasis-Net: Prototype-Routed Basis Decomposition for Multi-Modal Aircraft Trajectory Prediction
- **Method name in text**: ProtoBasis-Net (never "our model", "the proposed method" alone — always name it)
- **Dataset**: TrajAir (non-towered terminal airspace, KBTP airport, 1 Hz ADS-B)
- **Protocol**: 40s observation → 120s prediction, 1 Hz, K=20 candidates

## Core Innovations (never dilute or omit these)
1. **Prototype Router**: 64 K-Means trajectory prototypes, top-15 routing, intent classification
2. **SVD Basis Decomposition**: 16-dim global basis plus 2-dim prototype-local basis reconstruct trajectory shape
3. **Structured Candidate Expansion**: 30 internal prototype/micro candidates are compressed to K=20 with tail-swap candidate selection
4. **Social Aggregator**: cross-attention over neighboring aircraft (ASCENT has none)
5. **Shape Refinement**: endpoint-preserving shape/control-point refiners

## Verified Numbers (do not change without re-running experiments)
### Main Table (111_days, K=20)
| Method | ADE@20 | FDE@20 |
|--------|--------|--------|
| ASCENT | 0.19 | 0.26 |
| **ProtoBasis-Net** | **0.1842** | **0.2730** |
| GooDFlight | 0.29 | 0.39 |
| MID | 0.55 | 0.87 |
| TrajAirNet | 0.79 | 1.58 |

Numbers source: ASCENT Table II (for baselines), current seed3407 checkpoint evaluated on the full 111_days test split with score-biased `candidate_selection=tail_swap` (for ours).

### Generalization (7days1~4, K=20) — partially complete
| Method | 7days1 | 7days2 | 7days3 | 7days4 |
|--------|--------|--------|--------|--------|
| GooDFlight | 0.27/0.41 | 0.32/0.40 | 0.36/0.48 | 0.30/0.40 |
| **Ours** | 0.326/0.509 | TBD | TBD | TBD |

For ADE/FDE, lower values are better. Current 7days1 is still weaker than GooDFlight in both ADE and FDE, but the gap is smaller than the old incorrect notes implied.

GooDFlight introduced a GLeV diversity metric, but the paper's formula, textual explanation, and reported scale are not clear enough for a fair direct comparison in this project. Do not report or optimize GLeV in the main experiments; use ADE/FDE and focused ablations instead.

## Positioning
- vs ASCENT: "competitive with ASCENT"; ProtoBasis-Net is lower on ADE in the current full-test checkpoint evaluation, but FDE is still weaker, so do NOT claim an overall win.
- vs GooDFlight: "surpasses GooDFlight" on 111_days. ✅
- Key differentiator from ASCENT: social interaction + structured prototype prior (ASCENT has no social modeling)
- Key differentiator from GooDFlight: deterministic structured decoding vs diffusion; faster inference
- Key differentiator from EigenTrajectory: prototype-conditioned basis (not global fixed basis); aviation domain

## Similarity Risks (must cite, not hide)
- PoseNormalizer → cite ASCENT
- Weather/context columns are not used in the current default input; do not claim weather-aware prediction.

## Paper Structure
1. Introduction (~1 page)
2. Related Work (~1 page): Aircraft Trajectory Prediction / Multi-Modal Forecasting / Basis Decomposition
3. Method (~2 pages): Problem Formulation → Encoder → Social → Router → Basis → Micro-mode → Shape Refinement → Training
4. Experiments (~2 pages): Main Table / Generalization / Ablation / Case Study
5. Conclusion (~0.5 page)

## Experiments Still Needed
- 7days2, 7days3, 7days4 (not yet run)
- Ablation: w/o Router, w/o Basis, w/o Micro, w/o Shape Refiners, w/o Social

## Writing Files Location
- Drafts: `notes/writing/`
- References: `notes/references.bib` (25 entries)
- Paper prep doc: `notes/paper_preparation.md`

## Version Control Convention
Commit before every major section rewrite:
```
git add notes/writing/ && git commit -m "draft: <section-name>"
```

## What NOT to do
- Do not invent numbers. If a number is missing, write [TBD] and flag it.
- Do not claim results that haven't been run yet.
- Do not use bullet lists in paper body text.
- Do not write "In this paper, we propose..." — start with the problem or the gap.
