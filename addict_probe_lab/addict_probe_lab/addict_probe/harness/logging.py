from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class TraceLogger:
    """Simple JSONL logger with one row per episode step."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_row(self, row: Dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
