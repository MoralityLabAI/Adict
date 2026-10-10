from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Called with (site, residual [batch, time, d_model]); returns the residual to pass on.
# Site 0 is the embedding output, site i the residual after decoder layer i.
ResidualEdit = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class Encoded:
    ids: List[int]
    offsets: List[Tuple[int, int]]  # character span of each token

    def token_ending_at(self, char_end: int) -> int:
        """Index of the token that contains the character just before `char_end`."""
        for i, (lo, hi) in enumerate(self.offsets):
            if lo < char_end <= hi:
                return i
        raise ValueError(f"no token covers character {char_end - 1}")


class HookedLM:
    """A Hugging Face causal language model with read and write access to the residual stream.

    Nothing is substituted on failure: loading or a forward pass either works or raises.
    """

    def __init__(self, model, tokenizer, device: str | torch.device) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.layers = self._find_layers()
        self.embed = self.model.get_input_embeddings()
        self._token_cache: Dict[str, int] = {}

    @staticmethod
    def load(path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> "HookedLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model, info = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map=str(device), output_loading_info=True
        )
        if info.get("missing_keys"):
            raise RuntimeError(f"checkpoint is missing weights: {sorted(info['missing_keys'])[:8]}")
        return HookedLM(model, tokenizer, device)

    def _find_layers(self) -> torch.nn.ModuleList:
        base = getattr(self.model, self.model.base_model_prefix, self.model)
        layers = getattr(base, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList):
            raise RuntimeError("cannot locate the decoder layers of this model")
        return layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_sites(self) -> int:
        return self.n_layers + 1

    @property
    def d_model(self) -> int:
        return int(self.embed.weight.shape[1])

    # --- tokens ------------------------------------------------------------

    def encode(self, text: str) -> Encoded:
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return Encoded(ids=list(enc["input_ids"]), offsets=[tuple(o) for o in enc["offset_mapping"]])

    def single_token_id(self, text: str) -> int:
        """Token id of `text` (e.g. " dax"), raising unless it is exactly one token."""
        if text not in self._token_cache:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{text!r} is {len(ids)} tokens in this tokenizer")
            self._token_cache[text] = int(ids[0])
        return self._token_cache[text]

    def is_single_token(self, text: str) -> bool:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) == 1

    # --- forward -----------------------------------------------------------

    @contextmanager
    def _hooks(self, edit: Optional[ResidualEdit], record: Sequence[int], store: Dict[int, torch.Tensor]) -> Iterator[None]:
        handles = []

        def make(site: int):
            def hook(_module, _inputs, output):
                x = output[0] if isinstance(output, tuple) else output
                if edit is not None:
                    x = edit(site, x)
                if site in record:
                    store[site] = x
                if isinstance(output, tuple):
                    return (x,) + tuple(output[1:])
                return x

            return hook

        watch = set(record) | (set(range(self.n_sites)) if edit is not None else set())
        for site in sorted(watch):
            module = self.embed if site == 0 else self.layers[site - 1]
            handles.append(module.register_forward_hook(make(site)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def run(
        self,
        batch_ids: Sequence[Sequence[int]],
        read_positions: Optional[Sequence[Sequence[int]]] = None,
        sites: Sequence[int] = (),
        logit_positions: Optional[Sequence[int]] = None,
        edit: Optional[ResidualEdit] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward a right-padded batch.

        Returns next-token log-probabilities at `logit_positions` (default: each
        sequence's last token), [batch, vocab], and for each site in `sites` the
        residual at `read_positions`, [batch, positions, d_model], in float32 on the CPU.
        """
        lengths = [len(s) for s in batch_ids]
        t = max(lengths)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids = torch.full((len(batch_ids), t), pad, dtype=torch.long)
        mask = torch.zeros((len(batch_ids), t), dtype=torch.long)
        for i, s in enumerate(batch_ids):
            ids[i, : len(s)] = torch.as_tensor(list(s), dtype=torch.long)
            mask[i, : len(s)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        store: Dict[int, torch.Tensor] = {}
        base = getattr(self.model, self.model.base_model_prefix)
        with self._hooks(edit, list(sites), store):
            hidden = base(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        rows = torch.arange(len(batch_ids), device=self.device)
        at = torch.as_tensor(
            [n - 1 for n in lengths] if logit_positions is None else list(logit_positions), device=self.device
        )
        # The unembedding is applied only where it is needed: [batch, vocab] instead of [batch, time, vocab].
        logits = self.model.get_output_embeddings()(hidden[rows, at])
        logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        acts: Dict[int, torch.Tensor] = {}
        if sites:
            if read_positions is None:
                raise ValueError("read_positions are needed to record residuals")
            pos = torch.as_tensor([list(p) for p in read_positions], device=self.device)
            for site in sites:
                acts[site] = store[site][rows[:, None], pos].float().cpu()
        return logp, acts

    def next_token_logp(self, text: str, edit: Optional[ResidualEdit] = None) -> torch.Tensor:
        """Next-token log-probabilities after `text`, [vocab]."""
        logp, _ = self.run([self.encode(text).ids], edit=edit)
        return logp[0]


@dataclass
class TextState:
    """The model's state after reading `text`: token ids, the cache, and the next-token log-probabilities.

    States are never modified after creation; extending one copies its cache
    first, so forks can share a state safely.
    """

    text: str
    ids: List[int]
    cache: object
    logp: torch.Tensor  # [vocab], on the CPU


@torch.no_grad()
def advance(lm: HookedLM, text: str, base: Optional[TextState] = None) -> TextState:
    """State after `text`, reusing `base` when `text` extends it at a token boundary.

    Reuse requires that tokenising the whole text gives base.ids followed by the
    new ids; otherwise the text is read from the start. No edits are applied:
    cached reading is for unintervened runs only.
    """
    import copy

    ids = lm.encode(text).ids
    new = ids
    cache = None
    if base is not None and text.startswith(base.text) and ids[: len(base.ids)] == base.ids:
        new = ids[len(base.ids) :]
        if not new:
            return base
        cache = copy.deepcopy(base.cache)
    out = lm.model(
        input_ids=torch.as_tensor([new], device=lm.device), past_key_values=cache, use_cache=True, logits_to_keep=1
    )
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu()
    return TextState(text=text, ids=ids, cache=out.past_key_values, logp=logp)


def choice_distribution(
    lm: HookedLM,
    text: str,
    labels: Sequence[str],
    edit: Optional[ResidualEdit] = None,
    base: Optional[TextState] = None,
) -> Tuple[np.ndarray, float]:
    """Distribution over `labels` as the next token, renormalised, and the unrenormalised mass on them.

    With `base` (and no edit) only the text after base.text is read.
    """
    ids = [lm.single_token_id(" " + label) for label in labels]
    if base is not None and edit is None:
        full = advance(lm, text, base).logp
    else:
        full = lm.next_token_logp(text, edit=edit)
    lp = full[ids].double()
    mass = float(lp.exp().sum())
    return torch.softmax(lp, dim=0).numpy(), mass
