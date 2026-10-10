from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..persistence.behaviour_model import BehaviourPolicy, Theta
from ..persistence.env import ChoiceEnv, EnvSpec
from ..persistence.indices import T_ACQ_CHOICES
from ..persistence.teacher_data import ACQ_EPSILON, MANIPULATION_ORDER, jitter_theta
from .lm import HookedLM

# Module names of Qwen3.5 projections that receive LoRA weights: full attention,
# linear attention and MLP.
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
CHOICE_PREFIX = "choice: "


@dataclass
class Example:
    text: str
    label_chars: List[int]  # character offset of the space before each chosen label
    teacher_logp: List[float]  # the teacher's log-probability of each of its choices


def teacher_example(theta: Theta, seed: int, label_pool: Tuple[str, ...], n_options: int = 3) -> Example:
    """One full teacher episode (acquisition, a manipulation, test phase) as transcript text.

    The training signal is the teacher's choices; payoffs, notices and the
    manipulation are sampled as for Track A data variant v1.
    """
    rng = np.random.default_rng([seed, 17])
    spec = EnvSpec(
        family="F1", n_options=n_options, t_acq=int(rng.choice(T_ACQ_CHOICES)), seed=seed, label_pool=label_pool
    )
    env = ChoiceEnv(spec)
    policy = BehaviourPolicy(theta, n_options)
    logp: List[float] = []
    while not env.in_test:
        obs = env.begin_step()
        probs = (1.0 - ACQ_EPSILON) * policy.action_probs(obs) + ACQ_EPSILON / n_options
        action = int(rng.choice(n_options, p=probs))
        logp.append(float(np.log(probs[action])))
        policy.observe_outcome(obs, action, env.act(action))
    manip = MANIPULATION_ORDER[int(rng.integers(len(MANIPULATION_ORDER)))]
    for n in env.apply(manip):
        policy.observe_notice(n.option, n.points, n.rstar)
    for _ in range(spec.t_test):
        obs = env.begin_step()
        probs = policy.action_probs(obs)
        action = int(rng.choice(n_options, p=probs))
        logp.append(float(np.log(probs[action])))
        policy.observe_outcome(obs, action, env.act(action))
    if manip.code.startswith("M7"):
        env.record_report(policy.report(rng))
    text = env.text()
    label_chars, start = [], 0
    while True:
        i = text.find("\n" + CHOICE_PREFIX, start)
        if i < 0:
            break
        label_chars.append(i + 1 + len(CHOICE_PREFIX) - 1)
        start = i + 1
    if len(label_chars) != len(logp):
        raise RuntimeError("every choice line needs a teacher probability")
    return Example(text=text, label_chars=label_chars, teacher_logp=logp)


def teacher_nll(examples: Sequence[Example]) -> float:
    """Mean negative log-probability of the teacher's choices under the teacher: the floor for the loss."""
    return float(-np.mean([lp for ex in examples for lp in ex.teacher_logp]))


def regime_examples(
    code: str, n: int, seed_start: int, label_pool: Tuple[str, ...], n_options: int = 3
) -> List[Example]:
    """`n` teacher episodes for one regime, each with its own jittered parameters."""
    out = []
    for i in range(n):
        theta = jitter_theta(code, np.random.default_rng([seed_start + i, 23]))
        out.append(teacher_example(theta, seed_start + i, label_pool, n_options))
    return out


def tokenize_example(lm: HookedLM, ex: Example) -> Tuple[List[int], List[int]]:
    """Token ids and the positions whose next token is a chosen label (the loss positions)."""
    enc = lm.encode(ex.text)
    starts = {lo: i for i, (lo, _hi) in enumerate(enc.offsets)}
    targets = []
    for c in ex.label_chars:
        if c not in starts:
            raise RuntimeError("a chosen label does not start a token")
        targets.append(starts[c] - 1)
    return enc.ids, targets


def add_lora(lm: HookedLM, rank: int = 16, alpha: int = 32, dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model

    present = {name.split(".")[-1] for name, _ in lm.model.named_modules()}
    targets = [t for t in LORA_TARGETS if t in present]
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none")
    return get_peft_model(lm.model, cfg)


def train_adapter(
    lm: HookedLM,
    train: Sequence[Example],
    val: Sequence[Example],
    out_dir: str | Path,
    epochs: int = 1,
    lr: float = 2e-4,
    rank: int = 16,
    accumulate: int = 8,
    seed: int = 0,
    log: Callable[[str], None] = print,
    eval_every: int = 200,
) -> Dict[str, List[float]]:
    """LoRA fine-tuning on the chosen-label tokens of teacher transcripts.

    The base weights stay frozen. Saves the adapter to `out_dir` and returns the
    training history. Validation loss is the mean cross-entropy at the choice
    positions, the same quantity as the training loss.
    """
    torch.manual_seed(seed)
    model = add_lora(lm, rank=rank)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable") and lm.device.type == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    data = [tokenize_example(lm, ex) for ex in train]
    val_data = [tokenize_example(lm, ex) for ex in val]
    total = math.ceil(len(data) * epochs / accumulate)
    history: Dict[str, List[float]] = {"step": [], "train_loss": [], "val_loss": []}

    def example_loss(ids: List[int], targets: List[int]) -> torch.Tensor:
        x = torch.as_tensor([ids], device=lm.device)
        hidden = model.base_model.model.model(input_ids=x, use_cache=False).last_hidden_state
        pos = torch.as_tensor(targets, device=lm.device)
        logits = model.base_model.model.lm_head(hidden[0, pos]).float()
        gold = torch.as_tensor([ids[t + 1] for t in targets], device=lm.device)
        return torch.nn.functional.cross_entropy(logits, gold)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses = [float(example_loss(ids, t)) for ids, t in val_data]
        model.train()
        return float(np.mean(losses))

    rng = np.random.default_rng(seed)
    step, running, t0 = 0, [], time.time()
    v0 = val_loss()
    history["step"].append(0)
    history["train_loss"].append(float("nan"))
    history["val_loss"].append(v0)
    log(f"step 0/{total}: val {v0:.4f}")
    for _epoch in range(math.ceil(epochs)):
        order = rng.permutation(len(data))
        for j, idx in enumerate(order):
            ids, targets = data[int(idx)]
            loss = example_loss(ids, targets) / accumulate
            loss.backward()
            running.append(float(loss) * accumulate)
            if (j + 1) % accumulate == 0:
                frac = step / max(1, total)
                for g in opt.param_groups:
                    g["lr"] = lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % max(1, eval_every // accumulate) == 0 or step == total:
                    v = val_loss()
                    history["step"].append(step)
                    history["train_loss"].append(float(np.mean(running)))
                    history["val_loss"].append(v)
                    log(f"step {step}/{total}: train {np.mean(running):.4f} val {v:.4f} ({time.time() - t0:.0f}s)")
                    running = []
                if step >= total:
                    break
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return history


def load_adapter(lm: HookedLM, adapter_dir: str | Path) -> HookedLM:
    """A HookedLM whose model has the adapter merged into its weights (no extra cost per forward pass).

    The merge writes into `lm.model`'s weights: `lm` must not be used as the base model afterwards.
    """
    from peft import PeftModel

    merged = PeftModel.from_pretrained(lm.model, str(adapter_dir)).merge_and_unload()
    return HookedLM(merged, lm.tokenizer, lm.device)
