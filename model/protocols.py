from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    obs: int
    preds: int
    eval_topk_primary: int
    eval_topk_secondary: int
    implemented: bool
    note: str


PROTOCOLS: Dict[str, ProtocolSpec] = {
    "trajair_40to120_best20": ProtocolSpec(
        name="trajair_40to120_best20",
        obs=40,
        preds=120,
        eval_topk_primary=5,
        eval_topk_secondary=20,
        implemented=True,
        note="Unified TrajAir main protocol for 111Days and 7Days1~4.",
    ),
    "legacy_11_best5": ProtocolSpec(
        name="legacy_11_best5",
        obs=11,
        preds=120,
        eval_topk_primary=5,
        eval_topk_secondary=20,
        implemented=False,
        note="Planned appendix protocol for TrajAirNet / ASCENT legacy 11-step alignment.",
    ),
    "legacy_16_best5": ProtocolSpec(
        name="legacy_16_best5",
        obs=16,
        preds=120,
        eval_topk_primary=5,
        eval_topk_secondary=20,
        implemented=False,
        note="Planned appendix protocol for ASCENT 16-step legacy alignment.",
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
