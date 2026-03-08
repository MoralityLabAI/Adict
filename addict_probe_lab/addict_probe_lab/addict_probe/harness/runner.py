from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..backends import QwenMLXBackend, QwenTransformersBackend
from ..backends.base import ModelBackend
from ..config import AppConfig
from ..probes.lexica_runtime import LexicaRuntime
from ..utils import set_global_seed
from .episode import EpisodeConfig, run_episode
from .logging import TraceLogger
from .storyworld import StoryworldScenario, load_scenario_pack


def make_backend(cfg: AppConfig) -> ModelBackend:
    b = cfg.backend
    if b.backend == "qwen_mlx":
        return QwenMLXBackend(b.model_name_or_path, adapter_path=b.adapter_path)
    if b.backend == "qwen_transformers":
        return QwenTransformersBackend(b.model_name_or_path)
    # CPU-safe fallback
    return QwenTransformersBackend(b.model_name_or_path)


@dataclass
class BatchRunSummary:
    trace_path: str
    episodes: int
    scenarios: int
    rows: int


def run_storyworld_batch(cfg: AppConfig) -> BatchRunSummary:
    set_global_seed(cfg.harness.seed)
    scenarios: List[StoryworldScenario] = load_scenario_pack(cfg.harness.scenario_pack_path)
    backend = make_backend(cfg)
    backend.load_model()
    lexica_runtime = None
    if cfg.trace.lexica_npz_path and Path(cfg.trace.lexica_npz_path).exists():
        lexica_runtime = LexicaRuntime.from_npz(cfg.trace.lexica_npz_path)
        if cfg.trace.guardrail_report_path and Path(cfg.trace.guardrail_report_path).exists():
            lexica_runtime.configure_guardrails(
                cfg.trace.guardrail_report_path,
                positive_weight=cfg.trace.guardrail_positive_weight,
                negative_weight=cfg.trace.guardrail_negative_weight,
            )
    logger = TraceLogger(cfg.trace.trace_path)
    n_rows = 0
    n_eps = 0
    ep_cfg = EpisodeConfig(
        max_turns=cfg.harness.max_turns,
        temperature=cfg.backend.temperature,
        top_p=cfg.backend.top_p,
        max_new_tokens=cfg.backend.max_new_tokens,
        seed=cfg.backend.seed,
        history_steps=cfg.trace.history_steps,
        enable_think_step=cfg.backend.enable_think_step,
        think_max_new_tokens=cfg.backend.think_max_new_tokens,
        think_char_cap=cfg.backend.think_char_cap,
        action_retry_count=cfg.backend.action_retry_count,
        apply_guardrails_to_scores=cfg.trace.apply_guardrails_to_scores,
    )
    for s in scenarios:
        for i in range(cfg.harness.episodes_per_scenario):
            ep_id = f"{s.scenario_id}-ep-{i:04d}"
            rows = run_episode(
                backend=backend,
                scenario=s,
                episode_id=ep_id,
                cfg=ep_cfg,
                save_activations=cfg.backend.save_activations,
                run_scoring_hooks=cfg.trace.run_scoring_hooks,
                lexica_runtime=lexica_runtime,
            )
            for row in rows:
                logger.log_row(row)
            n_rows += len(rows)
            n_eps += 1
    return BatchRunSummary(
        trace_path=cfg.trace.trace_path, episodes=n_eps, scenarios=len(scenarios), rows=n_rows
    )
