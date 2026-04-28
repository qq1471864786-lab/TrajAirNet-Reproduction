from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    obs_steps: int
    pred_steps: int
    obs_stride: int
    pred_stride: int
    obs_horizon_sec: int
    pred_horizon_sec: int
    eval_topk_primary: int
    eval_topk_secondary: int
    implemented: bool
    note: str


PROTOCOLS: Dict[str, ProtocolSpec] = {
    "trajair_40to120_best20": ProtocolSpec(
        name="trajair_40to120_best20",
        obs_steps=40,
        pred_steps=120,
        obs_stride=1,
        pred_stride=1,
        obs_horizon_sec=40,
        pred_horizon_sec=120,
        eval_topk_primary=5,
        eval_topk_secondary=20,
        implemented=True,
        note="Unified TrajAir main protocol for 111Days and 7Days1~4.",
    ),
    "legacy_11_best5": ProtocolSpec(
        name="legacy_11_best5",
        obs_steps=11,
        pred_steps=12,
        obs_stride=1,
        pred_stride=10,
        obs_horizon_sec=11,
        pred_horizon_sec=120,
        eval_topk_primary=5,
        eval_topk_secondary=5,
        implemented=True,
        note="Appendix protocol for TrajAirNet / ASCENT legacy 11-step observation and 120s horizon at 10s stride.",
    ),
    "legacy_16_best5": ProtocolSpec(
        name="legacy_16_best5",
        obs_steps=16,
        pred_steps=24,
        obs_stride=5,
        pred_stride=5,
        obs_horizon_sec=80,
        pred_horizon_sec=120,
        eval_topk_primary=5,
        eval_topk_secondary=5,
        implemented=True,
        note="Appendix protocol for ASCENT legacy 16-step observation and 120s horizon at 0.2Hz.",
    ),
}


def resolve_protocol(protocol_name: str) -> ProtocolSpec:
    if protocol_name not in PROTOCOLS:
        raise KeyError(f"Unknown protocol: {protocol_name}")
    spec = PROTOCOLS[protocol_name]
    if not spec.implemented:
        raise NotImplementedError(
            f"Protocol '{protocol_name}' is scaffolded but not implemented yet. Note: {spec.note}"
        )
    return spec
