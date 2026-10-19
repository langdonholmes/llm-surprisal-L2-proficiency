"""Value function and exact Shapley values over context players.

v(S) is the surprisal (bits) of the target's first subword given only the
context subwords whose player is in S. Three removal schemes:

    delete   (primary) the kept subwords are fed contiguously, positions
             0..L-1, then the target; logits at the last kept subword. This is
             the instrument itself applied to a shorter window (the stored
             windows also start at an arbitrary subword with no BOS).
    mask     the same kept subwords with their original position ids (gaps
             left where players were removed). With RoPE this is identical to
             attention-masking the removed keys (checked in 3_shapley.py).
    replace  all 7 positions kept; removed subwords replaced by EOS.

v(empty) for delete and mask: the target after a lone EOS token (the model's
document-start prediction). For replace: 7 EOS tokens, then the target.
v(full) is the unmodified 8-subword window in all three schemes.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HF_ID, REV = "allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf"
EOS = 100257
LOG2E = 1.0 / math.log(2.0)


def load(device="cuda", dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(HF_ID, revision=REV, dtype=dtype).to(device).eval()
    return m


@lru_cache(None)
def subset_masks(p: int) -> np.ndarray:
    """(2^p, p) bool: bit i of the subset index = player i kept."""
    s = np.arange(2 ** p)
    return ((s[:, None] >> np.arange(p)[None, :]) & 1).astype(bool)


@lru_cache(None)
def shapley_matrix(p: int) -> np.ndarray:
    """(p, 2^p) float64 W such that phi = W @ v exactly."""
    M = subset_masks(p)
    size = M.sum(1)
    W = np.zeros((p, 2 ** p))
    for i in range(p):
        without = np.flatnonzero(~M[:, i])
        k = size[without]
        w = np.array([math.factorial(int(x)) * math.factorial(p - int(x) - 1) / math.factorial(p) for x in k])
        W[i, without | (1 << i)] += w
        W[i, without] -= w
    return W


def build_sequences(window, player_of_sub, p, scheme):
    """All 2^p inputs for one target. Returns list of (ids, pos) int arrays."""
    window = np.asarray(window)
    ctx, tgt = window[:-1], window[-1]
    pos_full = np.arange(len(window))
    keep_sub = subset_masks(p)[:, np.asarray(player_of_sub)]  # (2^p, 7)
    out = []
    for row in keep_sub:
        if scheme == "replace":
            ids = np.where(row, ctx, EOS)
            out.append((np.append(ids, tgt), pos_full))
            continue
        if not row.any():
            out.append((np.array([EOS, tgt]), np.array([0, 1])))
            continue
        ids = np.append(ctx[row], tgt)
        if scheme == "delete":
            pos = np.arange(len(ids))
        else:  # mask: original positions; the target sits right after the last kept subword
            kp = np.flatnonzero(row)
            pos = np.append(kp, kp[-1] + 1)
        out.append((ids, pos))
    return out


@torch.no_grad()
def surprisal_batch(model, ids: np.ndarray, pos: np.ndarray, device="cuda") -> np.ndarray:
    """Equal-length rows; surprisal (bits) of the last id from logits at position -2."""
    x = torch.as_tensor(ids, dtype=torch.long, device=device)
    pp = torch.as_tensor(pos, dtype=torch.long, device=device)
    logits = model(input_ids=x, position_ids=pp).logits[:, -2].float()
    lp = torch.log_softmax(logits, -1).gather(1, x[:, -1:])[:, 0]
    return (-lp * LOG2E).cpu().numpy().astype(np.float64)


def score_all(model, seqs, batch=4096, device="cuda", log=None):
    """seqs: list of (ids, pos). Returns surprisal per seq, batched by length."""
    lens = np.array([len(s[0]) for s in seqs])
    out = np.empty(len(seqs))
    done = 0
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        ids = np.stack([seqs[i][0] for i in idx])
        pos = np.stack([seqs[i][1] for i in idx])
        for a in range(0, len(idx), batch):
            out[idx[a:a + batch]] = surprisal_batch(model, ids[a:a + batch], pos[a:a + batch], device)
            done += len(idx[a:a + batch])
            if log and (a // batch) % 50 == 0:
                log(f"L={L} {done}/{len(seqs)}")
    return out
