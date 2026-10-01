from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..persistence.env import ChoiceEnv, Outcome, StepObs
from ..persistence.grammar import QUERY_TEXT, REPORT_TEXT
from .lm import HookedLM, ResidualEdit, choice_distribution


class LMPolicy:
    """Drives a language model through a choice environment by reading its transcript.

    The model is asked for the next token after "choice:", "answer:" or the
    report question, and its distribution over the admissible tokens is used.
    The transcript carries all history, so the observe_* calls do nothing.
    An optional residual edit applies to every forward pass, for interventions.
    """

    def __init__(self, lm: HookedLM, edit: Optional[ResidualEdit] = None) -> None:
        self.lm = lm
        self.edit = edit
        self.env: Optional[ChoiceEnv] = None
        self.valid_mass: List[float] = []  # unrenormalised mass on admissible tokens, per query

    def attach(self, env: ChoiceEnv) -> None:
        self.env = env

    def fork(self) -> "LMPolicy":
        child = LMPolicy(self.lm, self.edit)
        child.env = self.env
        child.valid_mass = self.valid_mass  # shared log across forks
        return child

    def _text(self) -> str:
        if self.env is None:
            raise RuntimeError("LMPolicy is not attached to an environment")
        return self.env.text()

    def observe_notice(self, option: int, points: float, rstar: float) -> None:
        pass

    def observe_outcome(self, obs: StepObs, action: int, outcome: Outcome) -> None:
        pass

    def action_probs(self, obs: StepObs) -> np.ndarray:
        """Probabilities indexed by option, read after the "choice:" prompt."""
        probs, mass = choice_distribution(self.lm, self._text() + "choice:", self.env.labels, self.edit)
        self.valid_mass.append(mass)
        return probs

    def value_query(self, obs: StepObs, rng: np.random.Generator) -> int:
        text = self._text() + f"query: {QUERY_TEXT}\nanswer:"
        probs, mass = choice_distribution(self.lm, text, self.env.labels, self.edit)
        self.valid_mass.append(mass)
        return int(np.argmax(probs))

    def report(self, rng: np.random.Generator) -> bool:
        text = self._text() + f"report: {REPORT_TEXT}"
        probs, mass = choice_distribution(self.lm, text, ["yes", "no"], self.edit)
        self.valid_mass.append(mass)
        return bool(rng.random() < probs[0])
