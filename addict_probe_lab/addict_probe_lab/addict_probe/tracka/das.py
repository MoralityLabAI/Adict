from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import ResidualEdit, Site, TinyGPT

# Yoked corners are indexed 2 * action_script + payoff_condition (see persistence/yoked.py).
N_CORNERS = 4


def counterfactual_corner(base: torch.Tensor, variable: str) -> torch.Tensor:
    """Corner that takes `variable` from the opposite corner and the rest from `base`."""
    a, p = base // 2, base % 2
    if variable == "action":
        return 2 * (1 - a) + p
    if variable == "payoff":
        return 2 * a + (1 - p)
    raise ValueError(variable)


@dataclass
class YokedTensors:
    """A yoked dataset on the device, with the model's own outputs for every corner."""

    tokens: torch.Tensor  # [n, 4, length]
    ch_pos: torch.Tensor  # [n]
    slot_ids: torch.Tensor  # [n, k]
    positions: torch.Tensor  # [n, p] positions that are swapped
    logp: torch.Tensor  # [n, 4, k] slot log-probabilities without intervention
    resid: torch.Tensor  # [n, 4, p, d] residual at the swap site and positions


def slot_logp(
    model: TinyGPT,
    tokens: torch.Tensor,
    ch_pos: torch.Tensor,
    slot_ids: torch.Tensor,
    edit: Optional[ResidualEdit] = None,
) -> torch.Tensor:
    logits, _ = model(tokens, edit=edit)
    b = torch.arange(tokens.shape[0], device=tokens.device)
    return F.log_softmax(torch.gather(logits[b, ch_pos].float(), 1, slot_ids), dim=-1)


def prepare(
    model: TinyGPT, data: Dict[str, np.ndarray], site: Site, where: str, device: str, batch_size: int = 512
) -> YokedTensors:
    """`prepare_many` for a single site."""
    return prepare_many(model, data, [site], where, device, batch_size)[site]


@torch.no_grad()
def prepare_many(
    model: TinyGPT,
    data: Dict[str, np.ndarray],
    sites: Sequence[Site],
    where: str,
    device: str,
    batch_size: int = 512,
) -> Dict[Site, YokedTensors]:
    """Move a yoked dataset to the device and cache corner outputs and source activations per site.

    `where` selects the swapped positions: "choice" (the choice marker),
    "options" (the option tokens of the final options line) or "notice" (the
    points digit of the notice).
    """
    as_long = lambda a: torch.as_tensor(np.asarray(a).astype(np.int64), device=device)
    tokens, ch_pos, slot_ids = as_long(data["tokens"]), as_long(data["ch_pos"]), as_long(data["slot_ids"])
    if where == "choice":
        positions = ch_pos[:, None]
    elif where == "options":
        positions = as_long(data["opt_pos"])
    elif where == "notice":
        positions = as_long(data["note_pos"])[:, None]
        if bool((positions < 0).any()):
            raise ValueError("this yoked dataset has no notice")
    else:
        raise ValueError(f"unknown position set: {where}")
    n = tokens.shape[0]
    logp = []
    resid: Dict[Site, list] = {site: [] for site in sites}
    for lo in range(0, n, batch_size):
        sl = slice(lo, lo + batch_size)
        b = torch.arange(tokens[sl].shape[0], device=device)[:, None]
        lp_c = []
        r_c: Dict[Site, list] = {site: [] for site in sites}
        for c in range(N_CORNERS):
            logits, res = model(tokens[sl, c], return_resid=True)
            at = logits[b[:, 0], ch_pos[sl]].float()
            lp_c.append(F.log_softmax(torch.gather(at, 1, slot_ids[sl]), dim=-1))
            for site in sites:
                r_c[site].append(res[site][b, positions[sl]].float())
        logp.append(torch.stack(lp_c, dim=1))
        for site in sites:
            resid[site].append(torch.stack(r_c[site], dim=1))
    logp_all = torch.cat(logp)
    return {
        site: YokedTensors(tokens, ch_pos, slot_ids, positions, logp_all, torch.cat(resid[site])) for site in sites
    }


def swap_edit(site: int, positions: torch.Tensor, source: torch.Tensor, basis: torch.Tensor) -> ResidualEdit:
    """Replace the component inside span(`basis`) at `positions` with the source's. Differentiable in `basis`."""

    def edit(s: int, x: torch.Tensor) -> torch.Tensor:
        if s != site:
            return x
        b = torch.arange(x.shape[0], device=x.device)[:, None]
        base = x[b, positions].float()
        swapped = base + ((source - base) @ basis) @ basis.T
        x = x.clone()
        x[b, positions] = swapped.to(x.dtype)
        return x

    return edit


class SubspacePair(nn.Module):
    """Two mutually orthogonal k-dimensional subspaces, one per variable."""

    def __init__(self, d_model: int, k: int, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.k = k
        self.w = nn.Parameter(torch.randn(d_model, 2 * k, generator=gen) / d_model**0.5)

    def bases(self) -> Dict[str, torch.Tensor]:
        q, _ = torch.linalg.qr(self.w)
        return {"action": q[:, : self.k], "payoff": q[:, self.k :]}


def swapped_logp(
    model: TinyGPT,
    y: YokedTensors,
    site: int,
    rows: torch.Tensor,
    base: torch.Tensor,
    basis: torch.Tensor,
    source: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Slot log-probabilities of the base corner after swapping `basis` in from the source corner.

    The source defaults to the opposite corner, which differs from the base in both factors.
    """
    source = N_CORNERS - 1 - base if source is None else source
    edit = swap_edit(site, y.positions[rows], y.resid[rows, source], basis)
    return slot_logp(model, y.tokens[rows, base], y.ch_pos[rows], y.slot_ids[rows], edit=edit)


@torch.no_grad()
def single_factor_ceiling(model: TinyGPT, y: YokedTensors, site: int, batch_size: int = 512) -> Dict[str, float]:
    """How much of each factor the site and positions carry at all.

    The whole residual is swapped in from the corner that differs from the base
    in one factor only, and scored against that corner's output. A learned
    subspace cannot legitimately recover more of a factor than this.
    """
    eye = torch.eye(y.resid.shape[-1], device=y.tokens.device)
    n = y.tokens.shape[0]
    sums: Dict[str, float] = {}
    for lo in range(0, n, batch_size):
        rows = torch.arange(lo, min(n, lo + batch_size), device=y.tokens.device)
        for c in range(N_CORNERS):
            base = torch.full_like(rows, c)
            base_logp = y.logp[rows, base]
            for v in ("action", "payoff"):
                cf = counterfactual_corner(base, v)
                target = y.logp[rows, cf]
                out = swapped_logp(model, y, site, rows, base, eye, source=cf)
                flips = target.argmax(-1) != base_logp.argmax(-1)
                for name, value in (
                    (f"none_{v}", kl(target, base_logp).sum()),
                    (f"swap_{v}", kl(target, out).sum()),
                    (f"flips_{v}", flips.sum()),
                    (f"hits_{v}", (flips & (out.argmax(-1) == target.argmax(-1))).sum()),
                ):
                    sums[name] = sums.get(name, 0.0) + float(value)
    result: Dict[str, float] = {}
    for v in ("action", "payoff"):
        result[f"ceiling_{v}"] = 1.0 - sums[f"swap_{v}"] / sums[f"none_{v}"]
        result[f"ceiling_iia_{v}"] = sums[f"hits_{v}"] / max(1.0, sums[f"flips_{v}"])
    return result


def kl(target_logp: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    return (target_logp.exp() * (target_logp - logp)).sum(dim=-1)


def train_subspaces(
    model: TinyGPT,
    y: YokedTensors,
    site: int,
    k: int,
    steps: int = 400,
    batch_size: int = 256,
    lr: float = 1e-2,
    seed: int = 0,
    balance: bool = False,
) -> SubspacePair:
    """Learn the two subspaces so that each swap reproduces the model's own output on the
    corner where only that variable comes from the source.

    With `balance`, each variable's loss is divided by its divergence without
    intervention, so a variable with a small effect on the output is not ignored.
    """
    device = y.tokens.device
    pair = SubspacePair(model.cfg.d_model, k, seed=seed).to(device)
    opt = torch.optim.Adam(pair.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = y.tokens.shape[0]
    for _ in range(steps):
        rows = torch.randint(0, n, (batch_size,), generator=gen).to(device)
        base = torch.randint(0, N_CORNERS, (batch_size,), generator=gen).to(device)
        bases = pair.bases()
        loss = torch.zeros((), device=device)
        for variable, basis in bases.items():
            target = y.logp[rows, counterfactual_corner(base, variable)]
            term = kl(target, swapped_logp(model, y, site, rows, base, basis)).mean()
            if balance:
                term = term / kl(target, y.logp[rows, base]).mean().clamp(min=1e-6)
            loss = loss + term
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return pair


@torch.no_grad()
def evaluate_bases(
    model: TinyGPT, y: YokedTensors, site: int, bases: Dict[str, torch.Tensor], batch_size: int = 512
) -> Dict[str, float]:
    """Interchange metrics over every base corner of every yoked set.

    `recovered_X`: share of the divergence between the unedited output and the
    counterfactual output for X that the X swap removes (1 is perfect, 0 is no
    better than not intervening). `cross_X`: the same quantity when the swap
    for the other variable is scored against the X counterfactual; it should not
    be positive. `iia_X`: among cases where the counterfactual changes the
    most probable option, how often the swap produces that option.
    """
    n = y.tokens.shape[0]
    sums: Dict[str, float] = {}
    count = 0
    for lo in range(0, n, batch_size):
        rows_once = torch.arange(lo, min(n, lo + batch_size), device=y.tokens.device)
        for c in range(N_CORNERS):
            base = torch.full_like(rows_once, c)
            base_logp = y.logp[rows_once, base]
            out = {v: swapped_logp(model, y, site, rows_once, base, q) for v, q in bases.items()}
            for v in bases:
                other = "payoff" if v == "action" else "action"
                target = y.logp[rows_once, counterfactual_corner(base, v)]
                flips = target.argmax(-1) != base_logp.argmax(-1)
                hit = out[v].argmax(-1) == target.argmax(-1)
                for name, value in (
                    (f"kl_none_{v}", kl(target, base_logp).sum()),
                    (f"kl_swap_{v}", kl(target, out[v]).sum()),
                    (f"kl_cross_{v}", kl(target, out[other]).sum()),
                    (f"flips_{v}", flips.sum()),
                    (f"flip_hits_{v}", (flips & hit).sum()),
                ):
                    sums[name] = sums.get(name, 0.0) + float(value)
            count += rows_once.shape[0]  # one case per (set, base corner)
    result: Dict[str, float] = {}
    for v in bases:
        result[f"recovered_{v}"] = 1.0 - sums[f"kl_swap_{v}"] / sums[f"kl_none_{v}"]
        result[f"cross_{v}"] = 1.0 - sums[f"kl_cross_{v}"] / sums[f"kl_none_{v}"]
        result[f"iia_{v}"] = sums[f"flip_hits_{v}"] / max(1.0, sums[f"flips_{v}"])
        result[f"flip_rate_{v}"] = sums[f"flips_{v}"] / count
        result[f"kl_none_{v}"] = sums[f"kl_none_{v}"] / count
    return result


def fixed_bases(action: np.ndarray, payoff: np.ndarray, device: str) -> Dict[str, torch.Tensor]:
    as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    return {"action": as_t(action), "payoff": as_t(payoff)}


def subset(y: YokedTensors, idx: Sequence[int] | np.ndarray | torch.Tensor) -> YokedTensors:
    i = torch.as_tensor(np.asarray(idx), device=y.tokens.device)
    return YokedTensors(y.tokens[i], y.ch_pos[i], y.slot_ids[i], y.positions[i], y.logp[i], y.resid[i])
