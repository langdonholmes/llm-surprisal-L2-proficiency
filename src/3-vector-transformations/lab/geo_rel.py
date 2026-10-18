"""Geometry lab — relational features between the observed word and the prediction.

Every column has a name and a reading ("the model put 60% of its mass on
determiners here", "the observed word is a near-spelling of the top prediction").
Inputs per token: observed word text, observed first-subword id, top-50
ids/probs at the target and next word, surprisal/entropy. No labels.

Blocks
  pred_<class>      renormalized top-50 mass on each closed class (+ content, punct)
  obs_<class>       one-hot class of the observed word
  next_pred_<class> the same mass at the next word
  orth_*            orthographic relation of observed word to predictions
  emb_*             embedding relation of e_obs to the predictions (W_U cosines)
  scal_*            surprisal/entropy (target, next) + word-relative surprisal z
  form_*            observed word form: subword count, dev frequency (log)
Cost per token: one top-50 class lookup, 10 string comparisons, 10 cosines.
"""

from difflib import SequenceMatcher

import pandas as pd
from functools import lru_cache

import numpy as np
from transformers import AutoTokenizer

from lab.geo_common import scalars, stats

CLASSES = {
    "det": "the a an this that these those my your his her its our their some any no every each another".split(),
    "prep": "in on at for to of with about from by into over after before during through without between".split(),
    "pron": "i you he she it we they me him us them myself yourself themselves".split(),
    "aux": "is are was were be been being am has have had do does did will would can could should may might must".split(),
    "conj": "and or but because so if when while although since than then".split(),
    "neg": "not n't never".split(),
}
NAMES = list(CLASSES) + ["punct", "content"]


@lru_cache(maxsize=1)
def tok():
    return AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")


@lru_cache(maxsize=1)
def vocab_strings() -> np.ndarray:
    t = tok()
    return np.array([t.convert_tokens_to_string([s]) for s in t.convert_ids_to_tokens(list(range(len(t))))],
                    dtype=object)


def word_class(w: str) -> int:
    w = w.strip().lower()
    for i, c in enumerate(CLASSES):
        if w in CLASSES[c]:
            return i
    if w and not any(ch.isalnum() for ch in w):
        return NAMES.index("punct")
    return NAMES.index("content")


@lru_cache(maxsize=1)
def vocab_class() -> np.ndarray:
    return np.array([word_class(s) for s in vocab_strings()], dtype=np.int8)


def class_mass(ids: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = p / p.sum(1, keepdims=True)
    vc = vocab_class()[ids]
    out = np.zeros((len(ids), len(NAMES)))
    for c in range(len(NAMES)):
        out[:, c] = (p * (vc == c)).sum(1)
    return out


def features(ctx, top_orth: int = 10) -> tuple[np.ndarray, list[str]]:
    obs_text = ctx.bench.observed.astype(str).to_numpy()
    ids, p = ctx.get("t_topk_ids"), ctx.get("t_topk_probs").astype(np.float64)
    pn = p / p.sum(1, keepdims=True)
    vs = vocab_strings()
    cols, names = [], []

    cols.append(class_mass(ids, p)); names += [f"pred_{c}" for c in NAMES]
    oc = np.array([word_class(w) for w in obs_text])
    cols.append(np.eye(len(NAMES))[oc]); names += [f"obs_{c}" for c in NAMES]
    cols.append(class_mass(ctx.get("n_topk_ids"), ctx.get("n_topk_probs").astype(np.float64)))
    names += [f"next_pred_{c}" for c in NAMES]

    # orthographic relation to the top predictions
    best, wsim, pref = np.zeros(len(ids)), np.zeros(len(ids)), np.zeros(len(ids))
    for i, w in enumerate(obs_text):
        w = w.lower()
        for j in range(top_orth):
            s = vs[ids[i, j]].strip().lower()
            if not s:
                continue
            r = SequenceMatcher(None, w, s).ratio()
            best[i] = max(best[i], r)
            wsim[i] += pn[i, j] * r
            if len(s) >= 3 and len(w) >= 3 and (s.startswith(w[:3]) or w.startswith(s[:3])):
                pref[i] += pn[i, j]
    cols.append(np.column_stack([best, wsim, pref])); names += ["orth_best", "orth_wmean", "orth_prefix_mass"]

    # embedding relation (W_U cosines)
    W = ctx.W_U
    eo = W[ctx.get("t_target_id")]
    eo_n = eo / np.linalg.norm(eo, axis=1, keepdims=True)
    top = W[ids[:, :top_orth]]
    top_n = top / np.linalg.norm(top, axis=2, keepdims=True)
    cos = np.einsum("nd,nkd->nk", eo_n, top_n)
    ep = ctx.e_pred("t")
    cos_pred = (eo_n * ep).sum(1) / np.linalg.norm(ep, axis=1)
    near = (pn[:, :top_orth] * (cos > 0.3)).sum(1)
    cols.append(np.column_stack([cos_pred, cos.max(1), (pn[:, :top_orth] * cos).sum(1), near]))
    names += ["emb_cos_pred", "emb_max_cos_top10", "emb_wcos_top10", "emb_near_mass"]

    cols.append(scalars(ctx)); names += ["scal_s", "scal_h", "scal_next_s", "scal_next_h", "scal_s_z", "scal_next_s_z"]

    t = tok()
    nsub = np.array([len(t(" " + w, add_special_tokens=False).input_ids) for w in obs_text])
    freq = stats()["vocab_freq"][ctx.get("t_target_id")]
    cols.append(np.column_stack([nsub, np.log1p(freq)])); names += ["form_nsub", "form_logfreq"]
    return np.hstack(cols).astype(np.float32), names


def extra(ctx) -> tuple[np.ndarray, list[str]]:
    """Category mismatch and top-1 margin."""
    obs_text = ctx.bench.observed.astype(str).to_numpy()
    cm = class_mass(ctx.get("t_topk_ids"), ctx.get("t_topk_probs").astype(np.float64))
    oc = np.array([word_class(w) for w in obs_text])
    mass_obs_class = cm[np.arange(len(oc)), oc]
    p1 = ctx.get("t_topk_probs")[:, 0].astype(np.float64)
    margin = ctx.get("t_surprisal") + np.log2(p1)  # log2 p(top1) - log2 p(obs)
    return np.column_stack([mass_obs_class, -np.log2(p1), margin]).astype(np.float32), \
        ["rel_mass_on_obs_class", "rel_top1_surprisal", "rel_margin_bits"]


def wc_pcs(ctx, n: int = 16) -> np.ndarray:
    from lab.geo_common import GEO, word_mean_pred
    z = np.load(GEO / "wc_pca.npz")
    wc = ctx.e_pred("t") - word_mean_pred(ctx.get("t_target_id"))
    return ((wc - z["mean"]) @ z["components"][:n].T).astype(np.float32)


UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def obs_pos(ctx) -> np.ndarray:
    """spaCy UPOS of the observed (target) word and of the next word, parsed from
    the instance text as the real pipeline does (en_core_web_lg). Cached per bench."""
    from lab.geo_common import GEO
    from lab.harness import SUFFIX
    path = GEO / f"obs_pos{SUFFIX}.npy"
    if path.exists():
        return np.load(path)
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    b = ctx.bench
    out = np.full((len(b), 2), len(UPOS) - 1, dtype=np.int8)
    chars = list(zip(b.target_char, b.next_char))
    for i, doc in enumerate(nlp.pipe(b.text.tolist(), batch_size=64, n_process=6)):
        by_char = {t.idx: t.pos_ for t in doc}
        for j, c in enumerate(chars[i]):
            if c is not None and not pd.isna(c) and int(c) in by_char:
                out[i, j] = UPOS.index(by_char[int(c)]) if by_char[int(c)] in UPOS else len(UPOS) - 1
    np.save(path, out)
    return out


def morph(ctx, top: int = 10) -> tuple[np.ndarray, list[str]]:
    """Inflectional transition from the observed word to same-stem predictions.

    Among the top predictions sharing a 3+ character stem with the observed word
    (case-insensitive), the probability mass where the prediction differs from the
    observed word by: adding -s/-es, removing -s, adding/removing -ing, adding/
    removing -ed, or an irregular aux/pronoun swap (is/are, was/were, has/have,
    do/does, there/their ...). Each column reads as "the model expected the
    same word with a different ending".
    """
    pairs = {("is", "are"), ("are", "is"), ("was", "were"), ("were", "was"), ("has", "have"),
             ("have", "has"), ("do", "does"), ("does", "do"), ("a", "the"), ("the", "a"),
             ("an", "a"), ("a", "an")}
    obs = ctx.bench.observed.astype(str).str.lower().to_numpy()
    ids, p = ctx.get("t_topk_ids"), ctx.get("t_topk_probs").astype(np.float64)
    pn = p / p.sum(1, keepdims=True)
    vs = vocab_strings()
    names = ["morph_add_s", "morph_drop_s", "morph_add_ing", "morph_drop_ing", "morph_add_ed",
             "morph_drop_ed", "morph_other_same_stem", "morph_irregular_swap"]
    out = np.zeros((len(obs), len(names)))
    for i, w in enumerate(obs):
        for j in range(top):
            s = vs[ids[i, j]].strip().lower()
            if not s or s == w:
                continue
            if (w, s) in pairs:
                out[i, 7] += pn[i, j]
                continue
            if len(w) < 3 or len(s) < 3 or w[:3] != s[:3]:
                continue
            if s.rstrip("s") == w or s == w + "es":
                k = 0
            elif w.rstrip("s") == s or w == s + "es":
                k = 1
            elif s.endswith("ing") and not w.endswith("ing"):
                k = 2
            elif w.endswith("ing") and not s.endswith("ing"):
                k = 3
            elif s.endswith("ed") and not w.endswith("ed"):
                k = 4
            elif w.endswith("ed") and not s.endswith("ed"):
                k = 5
            else:
                k = 6
            out[i, k] += pn[i, j]
    return out.astype(np.float32), names
