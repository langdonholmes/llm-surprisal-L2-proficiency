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
_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    return _TOKENIZER


def load_model(device="cuda"):
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    return AutoModelForCausalLM.from_pretrained(HF_ID, revision=REVISION, dtype=torch.float32).to(device).eval()


@lru_cache(None)
def subset_masks(p: int) -> np.ndarray:
    """(2^p, p) bool: bit i of the subset index = player i in the subset."""
    subsets = np.arange(2 ** p)
    return ((subsets[:, None] >> np.arange(p)[None, :]) & 1).astype(bool)


@lru_cache(None)
def shapley_matrix(p: int) -> np.ndarray:
    """(p, 2^p) matrix W such that phi = W @ v exactly."""
    masks = subset_masks(p)
    size = masks.sum(1)
    W = np.zeros((p, 2 ** p))
    for i in range(p):
        # every subset without player i, and its Shapley weight |S|! (p - |S| - 1)! / p!
        without = np.flatnonzero(~masks[:, i])
        weight = np.array([math.factorial(int(k)) * math.factorial(p - int(k) - 1) / math.factorial(p)
                           for k in size[without]])
        W[i, without | (1 << i)] += weight  # v(S + i) ...
        W[i, without] -= weight             # ... minus v(S)
    return W


def window_of(ids, offsets, text, char, from_start):
    """The 8-id window ending at the subword where ``char`` starts; (ids, start char).

    A spaCy token that starts inside a subword is scored at that subword. Near
    the start of an essay the window is shorter, as in the instrument.
    """
    target = None
    for j, (s, e) in enumerate(offsets):
        piece = text[s:e]
        if s + len(piece) - len(piece.lstrip()) == char:  # the subword's first non-space character
            target = j
            break
    if target is None:
        target = next((j for j, (s, e) in enumerate(offsets) if s <= char < e), None)
    if target is None or target == 0 or (target < WIN - 1 and not from_start):
        return None, None
    first = max(0, target - WIN + 1)
    return tuple(ids[first:target + 1]), offsets[first][0]


def variants(segment, target_char, edit_sets, from_start):
    """Windows of ``segment`` with each list of edits applied (target at ``target_char`` of segment)."""
    texts, chars = [], []
    for edit_set in edit_sets:
        applied = apply_edits(segment, [list(e) for e in edit_set])
        texts.append(applied["corrected"])
        chars.append(int(applied["cmap"][target_char]))  # where the target starts in this version
    enc = tokenizer()(texts, add_special_tokens=False, return_offsets_mapping=True)
    return [window_of(ids, offsets, text, char, from_start) if char >= 0 else (None, None)
            for text, char, ids, offsets in zip(texts, chars, enc.input_ids, enc.offset_mapping)]


@torch.no_grad()
def surprisal(model, windows, batch=4096, device="cuda") -> np.ndarray:
    """Surprisal (bits) of the last id of each window, batched by window length."""
    lengths = np.array([len(w) for w in windows])
    out = np.empty(len(windows))
    for length in np.unique(lengths):
        same_length = np.flatnonzero(lengths == length)
        for start in range(0, len(same_length), batch):
            rows = same_length[start:start + batch]
            ids = torch.as_tensor(np.array([windows[j] for j in rows]), dtype=torch.long, device=device)
            pos = torch.arange(length, device=device).expand(len(rows), length)
            # the logits at the second-to-last position predict the last id
            logits = model(input_ids=ids, position_ids=pos).logits[:, -2].float()
            logp = torch.log_softmax(logits, -1).gather(1, ids[:, -1:])[:, 0]
            out[rows] = (-logp * LOG2E).cpu().numpy()
    return out
