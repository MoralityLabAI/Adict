from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    hidden_state_summary: Optional[list[float]] = None
    meta: Optional[Dict[str, Any]] = None


class ModelBackend:
    """Common interface for storyworld generation backends."""

    name: str = "base"

    def load_model(self) -> None:
        raise NotImplementedError

    def generate_step(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        seed: int = 7,
    ) -> GenerationResult:
        raise NotImplementedError

    def get_hidden_state_summary(self, prompt: str, output_text: str) -> list[float]:
        return []
