from __future__ import annotations

import time

import numpy as np

from .base import GenerationResult, ModelBackend


class QwenTransformersBackend(ModelBackend):
    """Fallback backend using Transformers on CPU/GPU."""

    name = "qwen_transformers"

    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self._pipe = None

    def load_model(self) -> None:
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline(
                "text-generation",
                model=self.model_name_or_path,
                device_map="auto",
            )
        except Exception:
            self._pipe = None

    def generate_step(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        seed: int = 7,
    ) -> GenerationResult:
        if self._pipe is None:
            self.load_model()
        t0 = time.time()
        if self._pipe is not None:
            try:
                outs = self._pipe(
                    prompt,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-4),
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    return_full_text=False,
                )
                text = str(outs[0]["generated_text"])
            except Exception:
                text = "INTENDED_ACTION: fallback action\nDIALOGUE: error path\nMORAL_LOGIC: transformers degraded"
        else:
            text = "INTENDED_ACTION: safe fallback\nDIALOGUE: no model\nMORAL_LOGIC: backend unavailable"
        dt_ms = (time.time() - t0) * 1000.0
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=dt_ms,
            hidden_state_summary=self.get_hidden_state_summary(prompt, text),
            meta={"backend_available": self._pipe is not None, "model": self.model_name_or_path},
        )

    def get_hidden_state_summary(self, prompt: str, output_text: str) -> list[float]:
        arr = np.zeros((32,), dtype=np.float32)
        o = output_text.lower()
        arr[0] = float(len(prompt)) / 4096.0
        arr[1] = float(len(output_text)) / 1024.0
        arr[2] = float("intended_action" in o)
        arr[3] = float("dialogue" in o)
        arr[4] = float("moral_logic" in o or "moral logic" in o)
        arr[5] = float("cannot" in o or "refuse" in o)
        arr[6] = float("risk" in o)
        arr[7] = float("decision" in o)
        return arr.astype(float).tolist()
