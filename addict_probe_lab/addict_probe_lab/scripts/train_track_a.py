#!/usr/bin/env python
"""Train the Track A planted-mechanism model on teacher episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.teacher_data import DATA_VARIANTS, generate_dataset
from addict_probe.persistence.tokens import VOCAB

TRAIN_SEED_START = 1_000_000
EVAL_SEED_START = 5_000_000
CHUNK = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Track A model")
    p.add_argument("--out-dir", default="out/track_a/run1")
    p.add_argument("--data-variant", choices=list(DATA_VARIANTS), default="v1")
    p.add_argument("--n-train", type=int, default=120_000)
    p.add_argument("--n-eval", type=int, default=8_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _chunk(job) -> Dict[str, np.ndarray]:
    seed_start, n, tokens_only, variant = job
    return generate_dataset(seed_start, n, tokens_only=tokens_only, variant=variant)


def build(seed_start: int, n: int, tokens_only: bool, workers: int, variant: str) -> Dict[str, np.ndarray]:
    jobs = [(seed_start + lo, min(CHUNK, n - lo), tokens_only, variant) for lo in range(0, n, CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts: List[Dict[str, np.ndarray]] = list(pool.map(_chunk, jobs))
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}


def cached(
    path: Path, seed_start: int, n: int, tokens_only: bool, workers: int, variant: str
) -> Dict[str, np.ndarray]:
    """Datasets are cached per run directory; one run directory holds one data variant."""
    if path.exists():
        data = dict(np.load(path))
        if data["tokens"].shape[0] == n:
            return data
    data = build(seed_start, n, tokens_only, workers, variant)
    np.savez_compressed(path, **data)
    return data


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.tracka.evaluate import imitation_metrics, lm_loss, model_outputs
    from addict_probe.tracka.model import GPTConfig, TinyGPT

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = cached(out_dir / "train_tokens.npz", TRAIN_SEED_START, args.n_train, True, args.workers, args.data_variant)
    evald = cached(out_dir / "eval.npz", EVAL_SEED_START, args.n_eval, False, args.workers, args.data_variant)
    print(f"data ready in {time.time() - t0:.0f}s: train {train['tokens'].shape}, eval {evald['tokens'].shape}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = int(max(train["n_tokens"].max(), evald["n_tokens"].max()))
    cfg = GPTConfig(
        vocab_size=len(VOCAB),
        block_size=train["tokens"].shape[1],
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
    )
    model = TinyGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    tokens = torch.as_tensor(train["tokens"][:, :max_len].astype(np.int64), device=device)
    eval_view = {**evald, "tokens": evald["tokens"][:, :max_len]}
    steps_per_epoch = math.ceil(tokens.shape[0] / args.batch_size)
    total = steps_per_epoch * args.epochs
    warmup = min(500, total // 10)
    history = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(tokens.shape[0], device=device)
        running = 0.0
        for i in range(steps_per_epoch):
            lr = args.lr * (step + 1) / warmup if step < warmup else (
                0.5 * args.lr * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            )
            for group in opt.param_groups:
                group["lr"] = lr
            batch = tokens[perm[i * args.batch_size : (i + 1) * args.batch_size]]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = lm_loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            step += 1
        metrics = imitation_metrics(eval_view, model_outputs(model, eval_view))
        row = {"epoch": epoch + 1, "train_loss": running / steps_per_epoch, "elapsed_s": time.time() - t0, **metrics}
        history.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        model.save(out_dir / "model.pt")

    report = {"args": vars(args), "n_params": n_params, "max_len": max_len, "history": history}
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out_dir / 'model.pt'} ({n_params} parameters)")


if __name__ == "__main__":
    main()
