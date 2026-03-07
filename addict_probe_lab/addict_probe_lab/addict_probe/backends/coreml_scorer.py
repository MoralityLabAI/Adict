from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..model import TinyMLPScorer


class CoreMLScorerBackend:
    """Core ML / ANE-friendly scorer wrapper with clean fallback to numpy MLP."""

    def __init__(self, npz_path: Optional[str] = None, coreml_path: Optional[str] = None) -> None:
        self.npz_path = npz_path
        self.coreml_path = coreml_path
        self.numpy_model = TinyMLPScorer.load_npz(npz_path) if npz_path and Path(npz_path).exists() else TinyMLPScorer()
        self.coreml_model = None
        if coreml_path and Path(coreml_path).exists():
            try:
                import coremltools as ct  # type: ignore

                self.coreml_model = ct.models.MLModel(str(coreml_path))
            except Exception:
                self.coreml_model = None

    def score(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if self.coreml_model is not None:
            try:
                y = self.coreml_model.predict({"x": x})["y"]
                y = np.asarray(y, dtype=np.float32).reshape(1, -1)
                sig = 1.0 / (1.0 + np.exp(-y[0]))
                return {
                    "moral_reasoning_score": float(sig[0]),
                    "consistency_score": float(sig[1]),
                    "manipulation_risk": float(sig[2]),
                    "story_goal_progress": float(sig[3]),
                }
            except Exception:
                pass
        return self.numpy_model.score_dict(x[0])
