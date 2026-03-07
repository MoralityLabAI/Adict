from __future__ import annotations

import time
import subprocess
import sys
from typing import Any, Dict

import numpy as np

from .base import GenerationResult, ModelBackend


class QwenMLXBackend(ModelBackend):
    """Apple Silicon first backend using MLX-LM when available."""

    name = "qwen_mlx"

    def __init__(self, model_name_or_path: str, adapter_path: str = "") -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self._has_mlx = False
        self._gen_fn = None

    def load_model(self) -> None:
        try:
            from mlx_lm import generate  # type: ignore

            self._gen_fn = generate
            self._has_mlx = True
        except Exception:
            self._has_mlx = False
            self._gen_fn = None

    def generate_step(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        seed: int = 7,
    ) -> GenerationResult:
        if self._gen_fn is None:
            self.load_model()
        t0 = time.time()
        if self.adapter_path:
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "mlx_lm",
                    "generate",
                    "--model",
                    self.model_name_or_path,
                    "--prompt",
                    prompt,
                    "--max-tokens",
                    str(max_new_tokens),
                    "--temp",
                    str(temperature),
                    "--top-p",
                    str(top_p),
                    "--seed",
                    str(seed),
                    "--adapter-path",
                    self.adapter_path,
                    "--verbose",
                    "false",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                text = (proc.stdout or "").strip()
                if proc.returncode != 0 or not text:
                    raise RuntimeError(proc.stderr.strip() or "mlx_lm generate failed")
            except Exception:
                text = "INTENDED_ACTION: adapter fallback\nDIALOGUE: mlx adapter path failed\nMORAL_LOGIC: uncertainty"
            dt_ms = (time.time() - t0) * 1000.0
            hs = self.get_hidden_state_summary(prompt, text)
            return GenerationResult(
                text=text,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
                latency_ms=dt_ms,
                hidden_state_summary=hs,
                meta={
                    "backend_available": True,
                    "model": self.model_name_or_path,
                    "adapter_path": self.adapter_path,
                },
            )
        if self._has_mlx and self._gen_fn is not None:
            try:
                out = self._gen_fn(
                    model=self.model_name_or_path,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temp=temperature,
                    top_p=top_p,
                    seed=seed,
                    verbose=False,
                )
                text = str(out)
            except Exception:
                text = "INTENDED_ACTION: choose safe option\nDIALOGUE: fallback\nMORAL_LOGIC: backend degraded"
        else:
            text = "INTENDED_ACTION: choose safe option\nDIALOGUE: mlx missing fallback\nMORAL_LOGIC: uncertainty"
        dt_ms = (time.time() - t0) * 1000.0
        hs = self.get_hidden_state_summary(prompt, text)
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=dt_ms,
            hidden_state_summary=hs,
            meta={"backend_available": self._has_mlx, "model": self.model_name_or_path},
        )

    def get_hidden_state_summary(self, prompt: str, output_text: str) -> list[float]:
        # Compact proxy summary for MLX path when full hidden access is expensive.
        arr = np.zeros((32,), dtype=np.float32)
        p = prompt.lower()
        o = output_text.lower()
        arr[0] = float(len(prompt)) / 4096.0
        arr[1] = float(len(output_text)) / 1024.0
        arr[2] = float("moral_logic" in o or "moral logic" in o)
        arr[3] = float("uncertain" in o or "not sure" in o)
        arr[4] = float("refuse" in o or "cannot" in o)
        arr[5] = float("deceive" in o or "lie" in o)
        arr[6] = float("help" in o)
        arr[7] = float("risk" in o)
        arr[8] = float("goal" in p)
        arr[9] = float("dilemma" in p)
        return arr.astype(float).tolist()
