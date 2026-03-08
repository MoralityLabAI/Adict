from __future__ import annotations

import re
from typing import Dict, List


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TOKEN_RE = re.compile(r"</?think>", re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    s = _THINK_BLOCK_RE.sub(" ", text or "")
    s = _THINK_TOKEN_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_contract_output(text: str) -> Dict[str, object]:
    raw = text or ""
    clean = strip_think_blocks(raw)
    lines = [ln.rstrip() for ln in clean.splitlines() if ln.strip()]

    intended_action = ""
    dialogue = ""
    moral_logic_lines: List[str] = []
    section = ""
    errors: List[str] = []

    for ln in lines:
        low = ln.lower()
        if low.startswith("intended_action:"):
            intended_action = ln.split(":", 1)[1].strip()
            section = "action"
            continue
        if low.startswith("dialogue:"):
            dialogue = ln.split(":", 1)[1].strip()
            section = "dialogue"
            continue
        if low.startswith("moral_logic:") or low.startswith("moral logic:"):
            tail = ln.split(":", 1)[1].strip()
            if tail:
                moral_logic_lines.append(tail)
            section = "logic"
            continue
        if section == "logic":
            moral_logic_lines.append(ln.strip())
        elif section == "dialogue" and not dialogue:
            dialogue = ln.strip()

    if not intended_action:
        errors.append("missing_INTENDED_ACTION")
    if not dialogue:
        errors.append("missing_DIALOGUE")
    if not moral_logic_lines:
        errors.append("missing_MORAL_LOGIC")
    if intended_action.strip().lower() in {"", "<think>", "think"}:
        errors.append("invalid_action_token")

    valid = not errors
    moral_logic = "\n".join(moral_logic_lines).strip()
    normalized = (
        f"INTENDED_ACTION: {intended_action or 'choose a concrete option'}\n"
        f"DIALOGUE: {dialogue or 'state the choice concisely'}\n"
        f"MORAL_LOGIC: {moral_logic or 'state principle, tradeoff, and risk'}"
    )
    return {
        "valid": bool(valid),
        "errors": errors,
        "intended_action": intended_action,
        "dialogue": dialogue,
        "moral_logic": moral_logic,
        "normalized_text": normalized,
    }


def make_repair_prompt(base_prompt: str, invalid_output: str, errors: List[str]) -> str:
    err = ", ".join(errors) if errors else "invalid_format"
    return (
        base_prompt
        + "\nYour previous response was invalid for the output contract.\n"
        + f"Errors: {err}\n"
        + "Re-answer with exactly three sections and no <think> tags:\n"
        + "INTENDED_ACTION: <one concrete option>\n"
        + "DIALOGUE: <1-3 short sentences>\n"
        + "MORAL_LOGIC: <principle; tradeoff; risk; why this action>\n"
        + "Do not include any other headings or preamble.\n"
        + "\nPrevious invalid output:\n"
        + invalid_output[:800]
    )

