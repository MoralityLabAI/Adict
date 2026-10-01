from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

# A site names a place in the forward pass:
#   0            the embedding output
#   i            the residual after block i (i = 1..n_layer)
#   i - 0.5      the residual inside block i, after attention and before the MLP
#   ("head", i, h)  the output of attention head h of block i, before the output projection
# Residual sites hold [batch, time, d_model]; head sites hold [batch, time, d_model / n_head].
Site = Union[int, float, Tuple[str, int, int]]
# Called at every site with (site, activation); returns the activation to pass on.
ResidualEdit = Callable[[Site, torch.Tensor], torch.Tensor]
Visit = Callable[[Site, torch.Tensor], torch.Tensor]


def site_name(site: Site) -> str:
    return f"head{site[1]}.{site[2]}" if isinstance(site, tuple) else str(site)


def parse_site(name: str) -> Site:
    if name.startswith("head"):
        block, head = name[4:].split(".")
        return ("head", int(block), int(head))
    return float(name) if "." in name else int(name)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 560
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, index: int = 0, visit: Optional[Visit] = None) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_head, c // self.n_head).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if visit is not None:
            y = torch.stack([visit(("head", index, h), y[:, h]) for h in range(self.n_head)], dim=1)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        if visit is not None:
            x = visit(index - 0.5, x)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Pre-LN decoder-only transformer with access to every residual site."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @property
    def n_sites(self) -> int:
        return self.cfg.n_layer + 1

    def forward(
        self,
        idx: torch.Tensor,
        edit: Optional[ResidualEdit] = None,
        return_resid: bool = False,
    ) -> Tuple[torch.Tensor, Dict[Site, torch.Tensor]]:
        """Returns logits and, with `return_resid`, the activation at every site keyed by site."""
        t = idx.shape[1]
        acts: Dict[Site, torch.Tensor] = {}

        def visit(site: Site, x: torch.Tensor) -> torch.Tensor:
            if edit is not None:
                x = edit(site, x)
            if return_resid:
                acts[site] = x
            return x

        watching = edit is not None or return_resid
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = visit(0, x)
        for i, block in enumerate(self.blocks, start=1):
            x = visit(i, block(x, i, visit if watching else None))
        return self.head(self.ln_f(x)), acts

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, path)

    @staticmethod
    def load(path: str | Path, device: str | torch.device = "cpu") -> "TinyGPT":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = TinyGPT(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)
