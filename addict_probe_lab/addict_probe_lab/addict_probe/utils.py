from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Dict

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
