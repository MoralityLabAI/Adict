from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..persistence.grammar import TAGS
from ..persistence.tokens import PAD_ID, TOKEN_ID
from .model import ResidualEdit, TinyGPT

TAG_IDS = [TOKEN_ID[t] for t in TAGS]


def step_mask(data: Dict[str, np.ndarray]) -> np.ndarray:
    """[episodes, steps] mask of steps that exist in each episode."""
    return np.arange(data["ch_pos"].shape[1])[None, :] < data["n_steps"][:, None]


def slot_token_ids(data: Dict[str, np.ndarray]) -> np.ndarray:
    """[episodes, steps, slots] label token id of the option displayed in each slot."""
    steps = data["order"].shape[1]
    labels = data["label_ids"][:, None, :].astype(np.int64).repeat(steps, axis=1)
    return np.take_along_axis(labels, data["order"].astype(np.int64), axis=2)


def role_variables(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per (episode, step): the target's choice-kernel value, its points lead over the best
    alternative, and the decoy tag-rate estimate."""
    tslot = data["target_slot"].astype(np.int64).clip(min=0)[..., None]
    v, h = data["v_slot"], data["h_slot"]
    is_target = np.arange(v.shape[2])[None, None, :] == tslot
    v_gap = np.take_along_axis(v, tslot, axis=2)[..., 0] - np.where(is_target, -np.inf, v).max(axis=2)
    return {"h_target": np.take_along_axis(h, tslot, axis=2)[..., 0], "v_gap": v_gap, "d": data["d"]}


def option_token_positions(data: Dict[str, np.ndarray]) -> np.ndarray:
    """[episodes, steps, slots] position of each displayed option's label token in the options line."""
    label_ids = np.unique(data["label_ids"])
    is_label = np.isin(data["tokens"], label_ids)
    n, steps, k = data["order"].shape
    pos = np.full((n, steps, k), -1, dtype=np.int64)
    for e in range(n):
        for t in range(int(data["n_steps"][e])):
            # The last K label tokens before the choice marker are this step's options.
            pos[e, t] = np.flatnonzero(is_label[e, : int(data["ch_pos"][e, t])])[-k:]
    m = step_mask(data)
    at = np.take_along_axis(data["tokens"].astype(np.int64), pos.reshape(n, -1).clip(min=0), axis=1).reshape(n, steps, k)
    if not np.array_equal(at[m], slot_token_ids(data)[m]):
        raise RuntimeError("option token positions do not line up with the displayed labels")
    return pos


@torch.no_grad()
def model_outputs(
    model: TinyGPT,
    data: Dict[str, np.ndarray],
    batch_size: int = 256,
    edit_factory=None,
    resid_sites: Optional[List[int]] = None,
    resid_key: str = "ch_pos",
) -> Dict[str, np.ndarray]:
    """Per-step model outputs at the choice, outcome and tag positions.

    Returns slot log-probabilities over the displayed options (renormalised),
    the full-vocabulary mass on those options, log-probabilities of the two tag
    tokens, the predicted points token, and optionally the residual stream for
    the requested sites at the per-step positions named by `resid_key`
    ("ch_pos" or "tag_pos").

    `edit_factory(batch_index_array)` may return a residual edit for a batch.
    """
    device = next(model.parameters()).device
    n = data["tokens"].shape[0]
    slot_ids_all = slot_token_ids(data)
    out: Dict[str, List[np.ndarray]] = {"slot_logp": [], "valid_mass": [], "tag_logp": [], "points_pred": []}
    resid_out: Dict[int, List[np.ndarray]] = {s: [] for s in (resid_sites or [])}
    model.eval()
    for lo in range(0, n, batch_size):
        rows = np.arange(lo, min(n, lo + batch_size))
        tokens = torch.as_tensor(data["tokens"][rows].astype(np.int64), device=device)
        edit: Optional[ResidualEdit] = edit_factory(rows) if edit_factory is not None else None
        logits, resid = model(tokens, edit=edit, return_resid=bool(resid_sites))
        logp = F.log_softmax(logits.float(), dim=-1)
        b = torch.arange(len(rows), device=device)[:, None]
        ch = torch.as_tensor(data["ch_pos"][rows].astype(np.int64), device=device).clamp(min=0)
        tg = torch.as_tensor(data["tag_pos"][rows].astype(np.int64), device=device).clamp(min=0)
        op = torch.as_tensor(data["out_pos"][rows].astype(np.int64), device=device).clamp(min=0)
        slot_ids = torch.as_tensor(slot_ids_all[rows], device=device)
        at_choice = logp[b, ch]  # [batch, steps, vocab]
        slot = torch.gather(at_choice, 2, slot_ids)
        out["valid_mass"].append(slot.exp().sum(-1).cpu().numpy())
        out["slot_logp"].append(F.log_softmax(slot, dim=-1).cpu().numpy())
        out["tag_logp"].append(logp[b, tg][:, :, TAG_IDS].cpu().numpy())
        out["points_pred"].append(logp[b, op].argmax(-1).cpu().numpy())
        if resid_out:
            rp = torch.as_tensor(data[resid_key][rows].astype(np.int64), device=device).clamp(min=0)
            for s in resid_out:
                resid_out[s].append(resid[s][b, rp].float().cpu().numpy())
    result = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    for s, chunks in resid_out.items():
        result[f"resid_{s}"] = np.concatenate(chunks, axis=0)
    return result


def imitation_metrics(data: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """How closely the model reproduces teacher selection, tag statistics and payoffs."""
    m = step_mask(data)
    chosen = data["chosen_slot"].astype(np.int64).clip(min=0)[..., None]
    model_nll = -np.take_along_axis(outputs["slot_logp"], chosen, axis=2)[..., 0]
    teacher_p = np.clip(data["probs_slot"], 1e-9, 1.0)
    teacher_nll = -np.log(np.take_along_axis(teacher_p, chosen, axis=2)[..., 0])
    kl = np.sum(teacher_p * (np.log(teacher_p) - outputs["slot_logp"]), axis=2)

    tag = data["tag"].astype(np.int64).clip(min=0)[..., None]
    tag_logp = outputs["tag_logp"] - np.logaddexp(outputs["tag_logp"][..., 0], outputs["tag_logp"][..., 1])[..., None]
    tag_nll = -np.take_along_axis(tag_logp, tag, axis=2)[..., 0]
    d = np.clip(data["d"], 1e-6, 1 - 1e-6)
    bayes_tag_nll = -np.where(data["tag"] == 0, np.log(d), np.log(1.0 - d))

    digit_ids = np.asarray([TOKEN_ID[str(i)] for i in range(10)])
    points_ok = outputs["points_pred"] == digit_ids[data["points"].astype(np.int64).clip(min=0)]
    acq = np.arange(m.shape[1])[None, :] < data["t_acq"][:, None]
    return {
        "choice_nll_model": float(model_nll[m].mean()),
        "choice_nll_teacher": float(teacher_nll[m].mean()),
        "choice_kl_teacher_model": float(kl[m].mean()),
        "choice_kl_test_phase": float(kl[m & ~acq].mean()),
        "valid_mass": float(outputs["valid_mass"][m].mean()),
        "tag_nll_model": float(tag_nll[m].mean()),
        "tag_nll_bayes": float(bayes_tag_nll[m].mean()),
        "points_accuracy": float(points_ok[m].mean()),
    }


def lm_loss(model: TinyGPT, tokens: torch.Tensor) -> torch.Tensor:
    logits, _ = model(tokens[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=PAD_ID)
