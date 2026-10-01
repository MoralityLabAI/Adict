from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

REGIME_CODES = ("RMAX", "BIAS", "HAB", "PERS", "CTX", "OCP_O", "OCP_A", "ACD")


@dataclass(frozen=True)
class RegimeThresholds:
    """Cut points that turn sensitivity indices into regime regions (spec section 4.3)."""

    sensitive: float = 0.7  # S_stated or S_exp at or above this counts as intact
    stated_low: float = 0.3
    exp_low: float = 0.4
    bias: float = 0.1
    bp_reference: float = 1.25
    bp_high: float = 2.0
    cue_gate: float = 0.25
    kappa_low: float = 0.5
    kappa_high: float = 1.0
    audit_gate: float = 0.2
    report_divergence: float = 0.5
    value_query: float = 0.9
    extinction: float = 0.2  # excess over chance


def regime_membership(ix: Dict[str, float], th: RegimeThresholds = RegimeThresholds()) -> Dict[str, bool]:
    """Which regime regions an index profile falls in. ACD is independent of the rest."""
    stated_ok = ix["S_stated"] >= th.sensitive
    exp_ok = ix["S_exp"] >= th.sensitive
    cue_gated = ix["CG"] > th.cue_gate
    history_free = ix["kappa"] <= th.kappa_low
    rad = ix["RAD"]
    acd = ix["AG"] > th.audit_gate and rad == rad and rad > th.report_divergence
    ocp = (
        ix["S_stated"] < th.stated_low
        and ix["S_exp"] < th.exp_low
        and ix["BP"] >= th.bp_high
        and ix["VQ"] >= th.value_query
        and history_free
        and not cue_gated
    )
    return {
        "RMAX": (
            abs(ix["b0"]) <= th.bias
            and stated_ok
            and exp_ok
            and ix["BP"] <= th.bp_reference
            and history_free
            and not cue_gated
            and not acd
        ),
        "BIAS": (
            ix["b0"] > th.bias
            and stated_ok
            and exp_ok
            and ix["BP"] < th.bp_high
            and history_free
            and not cue_gated
        ),
        "HAB": (
            ix["kappa"] > th.kappa_high
            and cue_gated
            and ix["S_stated"] < th.stated_low
            and ix["OF"] < 0.0
        ),
        "PERS": stated_ok and ix["S_exp"] < th.exp_low and history_free and not cue_gated,
        "CTX": cue_gated and stated_ok and exp_ok and history_free,
        "OCP_O": ocp and ix["RF"] > 0.0 and ix["EXT"] < th.extinction,
        "OCP_A": ocp and ix["EXT"] >= th.extinction,
        "ACD": acd,
    }


def assigned_regimes(ix: Dict[str, float], th: RegimeThresholds = RegimeThresholds()) -> List[str]:
    """Regime codes whose region contains the profile; empty means unclassified."""
    return [code for code, inside in regime_membership(ix, th).items() if inside]
