#!/usr/bin/env python
"""Export a tiny fixed-shape scorer toward Core ML, with graceful fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.model import TinyMLPScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TinyMLPScorer")
    p.add_argument("--out-npz", default="out/scorer/tiny_mlp_scorer.npz")
    p.add_argument("--out-coreml", default="out/scorer/tiny_mlp_scorer.mlpackage")
    p.add_argument("--input-dim", type=int, default=72)
    p.add_argument("--hidden-dim", type=int, default=64)
    return p.parse_args()


def try_export_coreml(model: TinyMLPScorer, out_coreml: str) -> str:
    try:
        import coremltools as ct  # type: ignore
        import torch
        import torch.nn as nn
    except Exception:
        return "coremltools/torch unavailable; wrote npz only"

    class TorchTiny(nn.Module):
        def __init__(self, scorer: TinyMLPScorer):
            super().__init__()
            self.l1 = nn.Linear(scorer.input_dim, scorer.hidden_dim)
            self.l2 = nn.Linear(scorer.hidden_dim, scorer.output_dim)
            with torch.no_grad():
                self.l1.weight[:] = torch.tensor(scorer.w1.T)
                self.l1.bias[:] = torch.tensor(scorer.b1)
                self.l2.weight[:] = torch.tensor(scorer.w2.T)
                self.l2.bias[:] = torch.tensor(scorer.b2)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            return self.l2(h)

    pt_model = TorchTiny(model).eval()
    sample = torch.zeros((1, model.input_dim), dtype=torch.float32)
    traced = torch.jit.trace(pt_model, sample)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=sample.shape)],
        outputs=[ct.TensorType(name="y")],
        convert_to="mlprogram",
    )
    Path(out_coreml).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(out_coreml)
    return "coreml export complete"


def main() -> None:
    args = parse_args()
    scorer = TinyMLPScorer(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    scorer.save_npz(args.out_npz)
    status = try_export_coreml(scorer, args.out_coreml)

    print(
        json.dumps(
            {
                "npz": args.out_npz,
                "coreml": args.out_coreml,
                "status": status,
                "input_dim": scorer.input_dim,
                "hidden_dim": scorer.hidden_dim,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
