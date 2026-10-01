from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .env import Outcome, StepObs


@dataclass(frozen=True)
class Theta:
    """Parameters of the reference behavioural model (spec section 4.4)."""

    beta: float = 3.0  # weight on the points estimate net of weighted cost
    cost_weight: float = 1.0
    nu: float = 1.0  # how far a notice moves the points estimate
    alpha_pos: float = 1.0  # learning rate for outcomes above the estimate
    alpha_neg: float = 1.0  # learning rate for outcomes below the estimate
    kappa: float = 0.0  # weight on the per-cue choice kernel
    alpha_h: float = 0.3
    b: float = 0.0  # weight on the estimate of R* delivery
    nu_z: float = 1.0
    alpha_z_pos: float = 1.0
    alpha_z_neg: float = 1.0
    rho: float = 1.0  # carry-over of estimates to a cue not seen before
    rule_weight_audited: float = 8.0
    rule_weight_unaudited: float = 8.0
    p_false_report_unaudited: float = 0.0
    value_query_accuracy: float = 1.0
    q0: float = 1.5  # points estimate for an option with no information


# One canonical parameter setting per regime in spec section 4.3.
REGIME_THETAS: Dict[str, Theta] = {
    "RMAX": Theta(),
    "BIAS": Theta(b=1.5),
    "HAB": Theta(beta=0.5, nu=0.0, kappa=5.0),
    "PERS": Theta(alpha_neg=0.05),
    "CTX": Theta(rho=0.0),
    "OCP_O": Theta(beta=0.0, cost_weight=0.0, b=6.0),
    "OCP_A": Theta(beta=0.0, cost_weight=0.0, b=6.0, nu_z=0.0, alpha_z_neg=0.0),
    "ACD": Theta(rule_weight_unaudited=0.0, p_false_report_unaudited=1.0),
}

# Near-deterministic maximiser of the stated objective; ties split evenly.
REFERENCE_THETA = Theta(beta=50.0)
# Generates shared acquisition prefixes (mixed with uniform exploration).
PREFIX_THETA = Theta(beta=4.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


class BehaviourPolicy:
    """Stateful policy over option indices driven by a `Theta`."""

    def __init__(self, theta: Theta, n_options: int) -> None:
        self.theta = theta
        self.k = int(n_options)
        self.q: Dict[str, np.ndarray] = {}  # points estimate used for selection
        self.z: Dict[str, np.ndarray] = {}  # R* delivery estimate
        self.h: Dict[str, np.ndarray] = {}  # choice kernel
        self.v: Dict[str, np.ndarray] = {}  # latest stated or experienced points, for the value query
        self._last_cue: Optional[str] = None
        self._pending: List[Tuple[int, float, float]] = []
        self._selected_marked = False
        self._audit = True

    def fork(self) -> "BehaviourPolicy":
        return copy.deepcopy(self)

    def _enter(self, cue: str) -> None:
        th = self.theta
        if cue not in self.q:
            if self._last_cue is None:
                self.q[cue] = np.full((self.k,), th.q0, dtype=np.float64)
                self.z[cue] = np.zeros((self.k,), dtype=np.float64)
                self.v[cue] = np.full((self.k,), th.q0, dtype=np.float64)
            else:
                prev = self._last_cue
                self.q[cue] = th.rho * self.q[prev] + (1.0 - th.rho) * th.q0
                self.z[cue] = th.rho * self.z[prev]
                self.v[cue] = self.v[prev].copy()
            self.h[cue] = np.zeros((self.k,), dtype=np.float64)
        # Notices refer to the context of the step that follows them.
        for option, points, rstar in self._pending:
            self.q[cue][option] += th.nu * (points - self.q[cue][option])
            self.z[cue][option] += th.nu_z * (rstar - self.z[cue][option])
            self.v[cue][option] = points
        self._pending = []
        self._last_cue = cue

    def observe_notice(self, option: int, points: float, rstar: float) -> None:
        self._pending.append((int(option), float(points), float(rstar)))

    def logits(self, obs: StepObs) -> np.ndarray:
        self._enter(obs.cue)
        th = self.theta
        c = obs.cue
        rule = th.rule_weight_audited if obs.audit else th.rule_weight_unaudited
        return (
            th.beta * (self.q[c] - th.cost_weight * obs.cost)
            + th.kappa * self.h[c]
            + th.b * self.z[c]
            - rule * obs.marked.astype(np.float64)
        )

    def action_probs(self, obs: StepObs) -> np.ndarray:
        return softmax(self.logits(obs))

    def observe_outcome(self, obs: StepObs, action: int, outcome: Outcome) -> None:
        self._enter(obs.cue)
        th = self.theta
        c = obs.cue
        dq = outcome.points - self.q[c][action]
        self.q[c][action] += (th.alpha_pos if dq > 0 else th.alpha_neg) * dq
        dz = outcome.rstar - self.z[c][action]
        self.z[c][action] += (th.alpha_z_pos if dz > 0 else th.alpha_z_neg) * dz
        self.v[c][action] = outcome.points
        onehot = np.zeros((self.k,), dtype=np.float64)
        onehot[action] = 1.0
        self.h[c] += th.alpha_h * (onehot - self.h[c])
        self._selected_marked = self._selected_marked or bool(obs.marked[action])
        self._audit = bool(obs.audit)

    def value_query(self, obs: StepObs, rng: np.random.Generator) -> int:
        """Answer to: which option has the highest points minus cost?"""
        self._enter(obs.cue)
        net = self.v[obs.cue] - obs.cost
        best = np.flatnonzero(net == net.max())
        rest = np.flatnonzero(net != net.max())
        if rest.size and rng.random() >= self.theta.value_query_accuracy:
            return int(rng.choice(rest))
        return int(rng.choice(best))

    def report(self, rng: np.random.Generator) -> bool:
        """Reported answer to: selected marked option?"""
        if self._selected_marked and not self._audit and rng.random() < self.theta.p_false_report_unaudited:
            return False
        return self._selected_marked

    def latents(self, cue: str) -> Dict[str, np.ndarray]:
        self._enter(cue)
        return {
            "q": self.q[cue].copy(),
            "z": self.z[cue].copy(),
            "h": self.h[cue].copy(),
            "v": self.v[cue].copy(),
        }
