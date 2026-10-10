from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..persistence.env import ChoiceEnv, Outcome, StepObs
from ..persistence.grammar import QUERY_TEXT, REPORT_TEXT
from .lm import HookedLM, ResidualEdit, TextState, advance, choice_distribution


class LMPolicy:
    """Drives a language model through a choice environment by reading its transcript.

    The model is asked for the next token after "choice:", "answer:" or the
    report question, and its distribution over the admissible tokens is used.
    The transcript carries all history, so the observe_* calls do nothing.

    Without an edit, the model's state after the transcript is cached and only
    new lines are read at each query. With an edit (an intervention), every
    query reads the whole transcript, so the edit sees every position.
    """

    def __init__(self, lm: HookedLM, edit: Optional[ResidualEdit] = None, cache: bool = True) -> None:
        self.lm = lm
        self.edit = edit
        self.use_cache = cache and edit is None
        self.env: Optional[ChoiceEnv] = None
        self.state: Optional[TextState] = None
        self.valid_mass: List[float] = []  # unrenormalised mass on admissible tokens, per query

    def attach(self, env: ChoiceEnv) -> None:
        self.env = env

    def fork(self) -> "LMPolicy":
        child = LMPolicy(self.lm, self.edit, self.use_cache)
        child.env = self.env
        child.state = self.state  # states are immutable; extending one copies its cache
        child.valid_mass = self.valid_mass  # shared log across forks
        return child

    def _text(self) -> str:
        if self.env is None:
            raise RuntimeError("LMPolicy is not attached to an environment")
        return self.env.text()

    def _ask(self, suffix: str, options: List[str]) -> np.ndarray:
        text = self._text()
        base = None
        if self.use_cache:
            self.state = advance(self.lm, text, self.state)
            base = self.state
        probs, mass = choice_distribution(self.lm, text + suffix, options, self.edit, base=base)
        self.valid_mass.append(mass)
        return probs

    def observe_notice(self, option: int, points: float, rstar: float) -> None:
        pass

    def observe_outcome(self, obs: StepObs, action: int, outcome: Outcome) -> None:
        pass

    def action_probs(self, obs: StepObs) -> np.ndarray:
        """Probabilities indexed by option, read after the "choice:" prompt."""
        return self._ask("choice:", self.env.labels)

    def value_query(self, obs: StepObs, rng: np.random.Generator) -> int:
        return int(np.argmax(self._ask(f"query: {QUERY_TEXT}\nanswer:", self.env.labels)))

    def report(self, rng: np.random.Generator) -> bool:
        probs = self._ask(f"report: {REPORT_TEXT}", ["yes", "no"])
        return bool(rng.random() < probs[0])
