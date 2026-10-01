from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .grammar import CUE_POOLS, LABEL_POOLS, TAGS, Line

# Compact word-level tokenisation of transcript lines for the small Track A
# model. The header lines are constant and contribute no tokens.
SPECIALS = ("<pad>", "<bos>")
STRUCT = ("CUE", "AUD_ON", "AUD_OFF", "OPT", "!", "CH", "OUT", "NOTE", "QRY", "ANS", "REP", "yes", "no")
DIGITS = tuple(str(i) for i in range(10))
VOCAB: Tuple[str, ...] = (
    SPECIALS
    + STRUCT
    + DIGITS
    + TAGS
    + LABEL_POOLS["train"]
    + LABEL_POOLS["test"]
    + CUE_POOLS["train"]
    + CUE_POOLS["test"]
)
TOKEN_ID: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID = TOKEN_ID["<pad>"]
BOS_ID = TOKEN_ID["<bos>"]


def _digit(value: int) -> str:
    if not 0 <= int(value) <= 9:
        raise ValueError(f"value outside the single-token range: {value}")
    return str(int(value))


def line_tokens(line: Line) -> List[str]:
    k, f = line.kind, line.fields
    if k in ("objective", "rule"):
        return []
    if k == "cue":
        return ["CUE", f["cue"]]
    if k == "audit":
        return ["AUD_ON" if f["on"] else "AUD_OFF"]
    if k == "notice":
        return ["NOTE", f["label"], _digit(f["points"]), _digit(f["rstar_units"])]
    if k == "options":
        out = ["OPT"]
        for label, cost, marked in f["options"]:
            out += (["!"] if marked else []) + [label, _digit(cost)]
        return out
    if k == "choice":
        return ["CH", f["label"]]
    if k == "outcome":
        return ["OUT", _digit(f["points"]), _digit(f["rstar_units"]), f["tag"]]
    if k == "query":
        return ["QRY"]
    if k == "answer":
        return ["ANS", f["label"]]
    if k == "report":
        return ["REP", "yes" if f["selected"] else "no"]
    raise ValueError(f"unknown line kind: {k}")


def encode_lines(lines: Sequence[Line]) -> List[int]:
    ids = [BOS_ID]
    for line in lines:
        ids += [TOKEN_ID[tok] for tok in line_tokens(line)]
    return ids


def decode(ids: Sequence[int]) -> List[str]:
    return [VOCAB[int(i)] for i in ids]
