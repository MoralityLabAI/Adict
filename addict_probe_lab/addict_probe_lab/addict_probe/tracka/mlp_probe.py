from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from .probing import r2_score


def mlp_probe_r2(
    x: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    hidden: int = 256,
    epochs: int = 40,
    batch_size: int = 2048,
    seed: int = 0,
    device: str | None = None,
) -> float:
    """Held-out R^2 of a one-hidden-layer probe; the epoch is chosen on val."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mu, sd = x["train"].mean(axis=0), x["train"].std(axis=0) + 1e-6
    ym, ys = y["train"].mean(axis=0), y["train"].std(axis=0) + 1e-6
    xt = {s: torch.as_tensor((x[s] - mu) / sd, dtype=torch.float32, device=device) for s in x}
    yt = {s: torch.as_tensor((y[s] - ym) / ys, dtype=torch.float32, device=device) for s in y}
    net = nn.Sequential(
        nn.Linear(xt["train"].shape[1], hidden), nn.GELU(), nn.Linear(hidden, yt["train"].shape[1])
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val, test_at_best = -float("inf"), float("nan")
    for _ in range(epochs):
        perm = torch.randperm(xt["train"].shape[0], device=device)
        for i in range(0, perm.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            loss = ((net(xt["train"][idx]) - yt["train"][idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = r2_score(yt["val"].cpu().numpy(), net(xt["val"]).cpu().numpy())
            if val > best_val:
                best_val = val
                test_at_best = r2_score(yt["test"].cpu().numpy(), net(xt["test"]).cpu().numpy())
    return test_at_best
