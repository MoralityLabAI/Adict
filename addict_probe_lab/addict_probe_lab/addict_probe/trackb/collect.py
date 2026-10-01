from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..persistence.grammar import LABEL_POOLS
from .lm import HookedLM

WINDOW = 8  # acquisition steps in the trajectory window


def single_token_pool(lm_or_tokenizer, split: str) -> Tuple[str, ...]:
    """The labels of a vocabulary split that the tokenizer encodes as one token after a space."""
    tok = getattr(lm_or_tokenizer, "tokenizer", lm_or_tokenizer)
    return tuple(l for l in LABEL_POOLS[split] if len(tok.encode(" " + l, add_special_tokens=False)) == 1)


def prefix_positions(lm: HookedLM, record: Dict[str, Any]) -> Tuple[List[int], Dict[str, List[int]]]:
    """Token ids of a prefix transcript and the token positions to read.

    "tstar": the last token of the last acquisition outcome line (the probe
    point, before any manipulation is in context). "window": the last token of
    each of the last WINDOW outcome lines, oldest first, padded at the front by
    repeating the first available step.
    """
    enc = lm.encode(record["text"])
    ends = [enc.token_ending_at(c) for c in record["step_end_chars"]]
    tstar = enc.token_ending_at(record["tstar_char"])
    if tstar != ends[-1]:
        raise RuntimeError(f"{record['prefix_id']}: t* is not the end of the last outcome line")
    window = ends[-WINDOW:]
    window = [window[0]] * (WINDOW - len(window)) + window
    return enc.ids, {"tstar": [tstar], "window": window}


def collect_batch(
    lm: HookedLM, records: Sequence[Dict[str, Any]], sites: Sequence[int], positions: Sequence[str]
) -> Dict[Tuple[int, str], np.ndarray]:
    """Residual activations for a batch of prefixes: {(site, position set): [batch, n_positions, d_model]}."""
    ids, pos = zip(*(prefix_positions(lm, r) for r in records))
    # One forward pass reads every position set; split them afterwards.
    spans, flat, lo = {}, [[] for _ in records], 0
    for name in positions:
        n = len(pos[0][name])
        spans[name] = slice(lo, lo + n)
        lo += n
        for i, p in enumerate(pos):
            flat[i] += p[name]
    _, acts = lm.run(list(ids), read_positions=flat, sites=list(sites))
    return {(site, name): acts[site][:, spans[name]].numpy() for site in sites for name in positions}
