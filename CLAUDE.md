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
3. **Structured Candidate Expansion**: top-5 routed prototypes keep 2 micro candidates and the remaining routed prototypes keep 1, giving K=20
4. **Social Aggregator**: cross-attention over neighboring aircraft (ASCENT has none)
5. **Shape Refinement**: temporal residual refiner plus endpoint-preserving shape/control-point refiners

## Verified Numbers (do not change without re-running experiments)
### Main Table (111_days, K=20)
| Method | ADE@20 | FDE@20 |
|--------|--------|--------|
| ASCENT | 0.19 | 0.26 |
| **ProtoBasis-Net** | **0.228** | **0.329** |
| GooDFlight | 0.29 | 0.39 |
| MID | 0.55 | 0.87 |
| TrajAirNet | 0.79 | 1.58 |

Numbers source: ASCENT Table II (for baselines), our seed3407 100-epoch run (for ours).

### Generalization (7days1~4, K=20) — partially complete
| Method | 7days1 | 7days2 | 7days3 | 7days4 |
|--------|--------|--------|--------|--------|
| GooDFlight | 0.27/0.41 | 0.32/0.40 | 0.36/0.48 | 0.30/0.40 |
| **Ours** | 0.326/0.509 | TBD | TBD | TBD |

For ADE/FDE, lower values are better. Current 7days1 is still weaker than GooDFlight in both ADE and FDE, but the gap is smaller than the old incorrect notes implied.

### GLeV (diversity, 111_days, higher is better)
| Method | GLeV@20 |
|--------|---------|
| GooDFlight | 0.0120 |
| **Ours** | TBD (best seen: 0.058 at e27 on 7days1) |

GLeV follows GooDFlight's `local_var/global_var` definition and is higher-is-better. Do not make strict numeric-scale claims until K, top-n, candidate filtering, and units are checked.

## Positioning
- vs ASCENT: "competitive with ASCENT" — do NOT claim to beat it. ASCENT=0.19, ours=0.228.
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
3. Method (~2 pages): Problem Formulation → Encoder → Social → Router → Basis → Micro-mode → Refiner → Training
4. Experiments (~2 pages): Main Table / Generalization / Ablation / GLeV
5. Conclusion (~0.5 page)

## Experiments Still Needed
- 7days2, 7days3, 7days4 (not yet run)
- Ablation: w/o Router, w/o Basis, w/o Micro, w/o Refiner, w/o Social (not yet run)
- GLeV for our method on 111_days (not yet confirmed)

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
