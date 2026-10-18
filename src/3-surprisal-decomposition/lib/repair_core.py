"""Propose minimal edits at a position and score them with the LM.

For an observed word w at a target position, with left context L (the Study 1
8-subword window) and right context R (the next RIGHT subwords after w), every
candidate edit is scored by the change in the log-probability (bits) of the
local continuation under the same causal model:

    gain(edit) = log2 p(X_edit + R | L) - log2 p(X_obs + R | L)

where X is the edited material at the target. Edit operations:

    MORPH  w -> another inflection of the same word (rule-based morphology plus
           a closed table of irregular paradigms)
    FORM   w -> a vocabulary word within edit distance 2 (1 for short words);
           covers spelling, homophone-like and contraction confusions
    PRED   w -> one of the model's top predictions at the target
    DEL    delete w
    INS    insert one of the model's top predictions before w (omissions:
           the error surfaces at the word after the gap)

Nothing here reads gold labels. Vocabulary = whole-word tokens of the
OLMo-2 tokenizer plus a closed list of English contractions; frequency rank =
token id (BPE merge order).
"""

import re

import numpy as np
import torch

RIGHT = 5
N_PRED = 8
N_INS = 5
N_FORM = 8
LOG2E = 1.0 / np.log(2.0)

CONTRACTIONS = ["it's", "you're", "they're", "we're", "i'm", "he's", "she's", "that's",
                "there's", "what's", "who's", "let's", "don't", "doesn't", "didn't", "can't",
                "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
                "weren't", "haven't", "hasn't", "hadn't", "i've", "you've", "we've",
                "they've", "i'll", "you'll", "we'll", "they'll", "i'd", "you'd", "they'd"]

IRREGULAR = [
    ["be", "is", "are", "am", "was", "were", "been", "being"],
    ["have", "has", "had", "having"], ["do", "does", "did", "done", "doing"],
    ["go", "goes", "went", "gone", "going"], ["get", "gets", "got", "gotten", "getting"],
    ["make", "makes", "made", "making"], ["take", "takes", "took", "taken", "taking"],
    ["give", "gives", "gave", "given", "giving"], ["come", "comes", "came", "coming"],
    ["know", "knows", "knew", "known", "knowing"], ["think", "thinks", "thought", "thinking"],
    ["see", "sees", "saw", "seen", "seeing"], ["say", "says", "said", "saying"],
    ["tell", "tells", "told", "telling"], ["find", "finds", "found", "finding"],
    ["feel", "feels", "felt", "feeling"], ["leave", "leaves", "left", "leaving"],
    ["keep", "keeps", "kept", "keeping"], ["begin", "begins", "began", "begun", "beginning"],
    ["write", "writes", "wrote", "written", "writing"], ["run", "runs", "ran", "running"],
    ["bring", "brings", "brought", "bringing"], ["buy", "buys", "bought", "buying"],
    ["teach", "teaches", "taught", "teaching"], ["learn", "learns", "learned", "learnt", "learning"],
    ["choose", "chooses", "chose", "chosen", "choosing"], ["spend", "spends", "spent", "spending"],
    ["lose", "loses", "lost", "losing"], ["pay", "pays", "paid", "paying"],
    ["grow", "grows", "grew", "grown", "growing"], ["speak", "speaks", "spoke", "spoken", "speaking"],
    ["child", "children"], ["person", "people", "persons"], ["man", "men"], ["woman", "women"],
    ["life", "lives"], ["this", "these"], ["that", "those"], ["a", "an"],
    ["he", "him", "his"], ["she", "her", "hers"], ["they", "them", "their", "theirs"],
    ["we", "us", "our", "ours"], ["i", "me", "my", "mine"], ["you", "your", "yours"],
    ["it", "its"], ["good", "better", "best", "well"], ["bad", "worse", "worst"],
    ["much", "more", "most"], ["many", "more", "most"],
]
FUNCTION = set("""a an the this that these those my your his her its our their some any no every each
all both either neither much many more most few little other another such what which who whom whose
i me you he him she it we us they them myself yourself himself herself itself ourselves themselves
is are am was were be been being have has had do does did will would shall should can could may might must
in on at for to of with about from by into onto over under after before since until through during without
within between among against upon off out up down and or but nor so yet because although though if unless
while whereas when where why how than then as not too very also just only even there here""".split())
PUNCT_RE = re.compile(r"^[^\w\s]+$")


def osa(a: str, b: str) -> int:
    """Optimal string alignment (Damerau-Levenshtein, restricted) distance."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _deletes(w: str, n: int) -> set[str]:
    out, frontier = {w}, {w}
    for _ in range(n):
        frontier = {f[:i] + f[i + 1:] for f in frontier for i in range(len(f))}
        out |= frontier
    return out


class Vocab:
    """Whole-word vocabulary with a symspell-style deletion index (distance <= 2)."""

    def __init__(self, tok):
        rank = {}
        for t, i in tok.get_vocab().items():
            if t.startswith("Ġ") and re.fullmatch(r"[A-Za-z]+", t[1:]):
                w = t[1:].lower()
                rank[w] = min(rank.get(w, 10 ** 9), i)
        for w in CONTRACTIONS:
            rank.setdefault(w, 50_000)
        self.rank = rank
        self.index: dict[str, list[str]] = {}
        for w in rank:
            for d in _deletes(w, 2 if len(w) > 3 else 1):
                self.index.setdefault(d, []).append(w)
        self.paradigm = {}
        for fam in IRREGULAR:
            for w in fam:
                self.paradigm.setdefault(w, set()).update(fam)

    def similar(self, w: str, n: int = N_FORM) -> list[tuple[str, int]]:
        w = w.lower()
        maxd = 2 if len(w) > 3 else 1
        cands = set()
        for d in _deletes(w, maxd):
            cands.update(self.index.get(d, ()))
        cands.discard(w)
        scored = [(c, osa(w, c)) for c in cands]
        scored = [(c, k) for c, k in scored if k <= maxd]
        scored.sort(key=lambda x: (x[1], self.rank[x[0]]))
        return scored[:n]

    def inflections(self, w: str) -> set[str]:
        w = w.lower()
        out = set(self.paradigm.get(w, ()))
        bases = {w}
        for suf, rep in [("ies", "y"), ("ied", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
                         ("d", ""), ("ing", ""), ("ing", "e"), ("er", ""), ("est", "")]:
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                bases.add(w[: -len(suf)] + rep)
        for b in list(bases):
            if len(b) > 2 and b[-1] == b[-2] and b[-1] not in "aeiou":
                bases.add(b[:-1])  # stopped -> stopp -> stop
        for b in bases:
            out.add(b)
            out.add(b + "es" if b.endswith(("s", "x", "z", "ch", "sh")) else
                    (b[:-1] + "ies" if b.endswith("y") and len(b) > 2 and b[-2] not in "aeiou" else b + "s"))
            out.add(b + "d" if b.endswith("e") else (b[:-1] + "ied" if b.endswith("y") and len(b) > 2 and b[-2] not in "aeiou" else b + "ed"))
            out.add(b[:-1] + "ing" if b.endswith("e") and not b.endswith("ee") and len(b) > 2 else b + "ing")
        out.discard(w)
        return {o for o in out if o in self.rank}


def match_case(new: str, old: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


@torch.no_grad()
def score_continuations(model, seqs: list[tuple[list[int], list[int], list[int]]], device="cuda",
                        batch_tokens: int = 24_000) -> tuple[np.ndarray, np.ndarray]:
    """For each (ctx, x, r): sum log2 p(x | ctx) and sum log2 p(r | ctx, x).
    Left-padded batches, sorted by length to limit padding."""
    seqs = [(c, list(x) + list(r), len(r)) for c, x, r in seqs]
    order = np.argsort([len(c) + len(x) for c, x, _ in seqs])
    out_x, out_r = np.zeros(len(seqs)), np.zeros(len(seqs))
    i = 0
    while i < len(order):
        L = len(seqs[order[i]][0]) + len(seqs[order[i]][1])
        j = i
        while j < len(order):
            Lj = len(seqs[order[j]][0]) + len(seqs[order[j]][1])
            if (j - i + 1) * Lj > batch_tokens:
                break
            L = Lj
            j += 1
        idx = order[i:j]
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        pos = torch.zeros((len(idx), L), dtype=torch.long)
        x_mask = torch.zeros((len(idx), L), dtype=torch.bool)
        r_mask = torch.zeros((len(idx), L), dtype=torch.bool)
        for row, s in enumerate(idx):
            c, x, nr = seqs[s]
            full = list(c) + list(x)
            n = len(full)
            ids[row, L - n:] = torch.as_tensor(full)
            att[row, L - n:] = 1
            pos[row, L - n:] = torch.arange(n)
            x_mask[row, L - len(x):L - nr] = True
            r_mask[row, L - nr:] = True
        ids, att, pos, x_mask, r_mask = (t.to(device) for t in (ids, att, pos, x_mask, r_mask))
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits
        logp = torch.log_softmax(logits[:, :-1].float(), -1)
        tok = logp.gather(2, ids[:, 1:, None])[..., 0]
        out_x[idx] = ((tok * x_mask[:, 1:]).sum(1) * LOG2E).cpu().numpy()
        out_r[idx] = ((tok * r_mask[:, 1:]).sum(1) * LOG2E).cpu().numpy()
        i = j
    return out_x, out_r
