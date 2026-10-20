"""Edit-level Shapley values with the correction as baseline (D27).

For a target word, the players are the gold edits in its window, and each has
two states, as written and corrected. With v(S) the surprisal of the target
when only the edits in S are applied,

    u(S) = v(empty) - v(S),   u(all) = s_orig - s_corr = Delta,

so the Shapley values of u split Delta exactly among the errors. A window is
the 8 subwords ending at the target's first subword of a text segment that
runs from at least BACK characters before the target (cut at a space) to the
end of the word after it, re-tokenised with the edits in S applied
(``lib.corrections.apply_edits``). Scored in fp32 with TF32 off.
"""

import math
from functools import lru_cache

import numpy as np
import torch

from lib.corrections import apply_edits
from lib.lm import HF_ID, REVISION

WIN = 8
BACK = 300
LOG2E = 1.0 / math.log(2.0)
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    return _TOK


def load_model(device="cuda"):
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    return AutoModelForCausalLM.from_pretrained(HF_ID, revision=REVISION, dtype=torch.float32).to(device).eval()


@lru_cache(None)
def subset_masks(p: int) -> np.ndarray:
    """(2^p, p) bool: bit i of the subset index = player i in the subset."""
    s = np.arange(2 ** p)
    return ((s[:, None] >> np.arange(p)[None, :]) & 1).astype(bool)


@lru_cache(None)
def shapley_matrix(p: int) -> np.ndarray:
    """(p, 2^p) matrix W such that phi = W @ v exactly."""
    M = subset_masks(p)
    size = M.sum(1)
    W = np.zeros((p, 2 ** p))
    for i in range(p):
        without = np.flatnonzero(~M[:, i])
        w = np.array([math.factorial(int(k)) * math.factorial(p - int(k) - 1) / math.factorial(p)
                      for k in size[without]])
        W[i, without | (1 << i)] += w
        W[i, without] -= w
    return W


def window_of(ids, offsets, text, char, from_start):
    """The 8-id window ending at the subword where ``char`` starts; (ids, start char).

    A spaCy token that starts inside a subword is scored at that subword. Near
    the start of an essay the window is shorter, as in the instrument.
    """
    k = None
    for j, (s, e) in enumerate(offsets):
        piece = text[s:e]
        if s + len(piece) - len(piece.lstrip()) == char:
            k = j
            break
    if k is None:
        k = next((j for j, (s, e) in enumerate(offsets) if s <= char < e), None)
    if k is None or k == 0 or (k < WIN - 1 and not from_start):
        return None, None
    a = max(0, k - WIN + 1)
    return tuple(ids[a:k + 1]), offsets[a][0]


def variants(seg, tgt, edit_sets, from_start):
    """Windows of ``seg`` with each list of edits applied (target at char ``tgt`` of seg)."""
    texts, chars = [], []
    for es in edit_sets:
        r = apply_edits(seg, [list(e) for e in es])
        texts.append(r["corrected"])
        chars.append(int(r["cmap"][tgt]))
    enc = tokenizer()(texts, add_special_tokens=False, return_offsets_mapping=True)
    return [window_of(i, o, t, c, from_start) if c >= 0 else (None, None)
            for t, c, i, o in zip(texts, chars, enc.input_ids, enc.offset_mapping)]


@torch.no_grad()
def surprisal(model, windows, batch=4096, device="cuda") -> np.ndarray:
    """Surprisal (bits) of the last id of each window, batched by window length."""
    lens = np.array([len(w) for w in windows])
    out = np.empty(len(windows))
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        for a in range(0, len(idx), batch):
            i = idx[a:a + batch]
            x = torch.as_tensor(np.array([windows[j] for j in i]), dtype=torch.long, device=device)
            pos = torch.arange(L, device=device).expand(len(i), L)
            logits = model(input_ids=x, position_ids=pos).logits[:, -2].float()
            lp = torch.log_softmax(logits, -1).gather(1, x[:, -1:])[:, 0]
            out[i] = (-lp * LOG2E).cpu().numpy()
    return out
