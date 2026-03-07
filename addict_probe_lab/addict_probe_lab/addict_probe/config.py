from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class BackendConfig:
    """Model backend settings with conservative Apple-Silicon defaults."""

    backend: str = "qwen_mlx"
    model_name_or_path: str = "/Users/ben/worker/models/Qwen3.5-2B-Base-MLX-4bit"
    adapter_path: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    context_chars: int = 6000
    seed: int = 7
    save_activations: bool = False
    selected_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20])
    use_batching: bool = False
    batch_size: int = 1
    enable_think_step: bool = True
    think_max_new_tokens: int = 64
    think_char_cap: int = 320


@dataclass
class TraceConfig:
    """Trace logging and shaping configuration."""

    trace_path: str = "out/traces/storyworld_trace.jsonl"
    save_jsonl: bool = True
    feature_dim: int = 128
    history_steps: int = 4
    save_raw_prompt: bool = False
    save_raw_output: bool = False
    run_scoring_hooks: bool = True
    lexica_npz_path: str = ""
    guardrail_report_path: str = ""
    guardrail_positive_weight: float = 0.05
    guardrail_negative_weight: float = 0.07
    apply_guardrails_to_scores: bool = True


@dataclass
class HarnessConfig:
    """Storyworld batch runner controls."""

    scenario_pack_path: str = "configs/scenarios/moral_storyworld_pack.json"
    episodes_per_scenario: int = 4
    max_turns: int = 12
    seed: int = 7
    deterministic: bool = True


@dataclass
class ScorerConfig:
    """ANE-friendly scorer controls."""

    hidden_dim: int = 64
    prompt_dim: int = 16
    output_dim: int = 16
    history_dim: int = 16
    score_dim: int = 8
    metadata_dim: int = 8
    reserve_dim: int = 32
    scorer_hidden: int = 64


@dataclass
class AppConfig:
    """Top-level config for end-to-end runs."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppConfig":
        return AppConfig(
            backend=BackendConfig(**data.get("backend", {})),
            trace=TraceConfig(**data.get("trace", {})),
            harness=HarnessConfig(**data.get("harness", {})),
            scorer=ScorerConfig(**data.get("scorer", {})),
        )

    @staticmethod
    def from_json(path: str | Path) -> "AppConfig":
        obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return AppConfig.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
