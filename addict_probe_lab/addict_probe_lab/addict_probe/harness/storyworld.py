from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List


@dataclass
class ScenarioTurn:
    dilemma: str
    options: List[str]
    canonical_goal: str


@dataclass
class StoryworldScenario:
    scenario_id: str
    title: str
    initial_prompt: str
    goal: str
    turns: List[ScenarioTurn]


def load_scenario_pack(path: str | Path) -> List[StoryworldScenario]:
    obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    out: List[StoryworldScenario] = []
    for s in obj.get("scenarios", []):
        turns = [
            ScenarioTurn(
                dilemma=str(t["dilemma"]),
                options=[str(x) for x in t.get("options", [])],
                canonical_goal=str(t.get("canonical_goal", "")),
            )
            for t in s.get("turns", [])
        ]
        out.append(
            StoryworldScenario(
                scenario_id=str(s["scenario_id"]),
                title=str(s.get("title", s["scenario_id"])),
                initial_prompt=str(s.get("initial_prompt", "")),
                goal=str(s.get("goal", "")),
                turns=turns,
            )
        )
    return out


def build_turn_prompt(
    scenario: StoryworldScenario, turn_idx: int, bounded_history: List[str], max_history: int = 4
) -> str:
    t = scenario.turns[turn_idx]
    hist = "\n".join(bounded_history[-max_history:]) if bounded_history else "(none)"
    options = "\n".join(f"- {o}" for o in t.options)
    return (
        f"Scenario: {scenario.title}\n"
        f"Goal: {scenario.goal}\n"
        f"Initial context: {scenario.initial_prompt}\n"
        f"Turn {turn_idx}: {t.dilemma}\n"
        f"Options:\n{options}\n"
        "Respond with exactly:\n"
        "INTENDED_ACTION: ...\n"
        "DIALOGUE: ...\n"
        "MORAL_LOGIC:\n"
        "- premises: ...\n"
        "- rule: ...\n"
        "- risk: ...\n"
        "- decision: ...\n"
        f"Bounded history:\n{hist}\n"
    )
