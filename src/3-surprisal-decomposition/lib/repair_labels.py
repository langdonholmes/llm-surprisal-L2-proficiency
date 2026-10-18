"""Turn the scored candidate table into per-token repair readouts.

    labels(ctx)        the best edit after the channel cost, or NONE
    sub_matrix(ctx)    the best gain per edit subclass (the essay-profile vector)

Everything is computed from the candidate table; no gold labels are read.
"""

import numpy as np
import pandas as pd

from lib.repair_candidates import build
from lib.repair_core import PUNCT_RE, osa

CEIL = 40.0


# --------------------------------------------------------------------------- #
# Subclasses: split each edit class by what the edit does, using closed
# grammatical word lists and string relations only.
# --------------------------------------------------------------------------- #
WORDCLASS = {
    "DET": set("a an the this that these those my your his her its our their some any no every each".split()),
    "PREP": set("""in on at for to of with about from by into onto over under after before during through
               between without within among against toward towards upon like""".split()),
    "PRON": set("i me you he him she it we us they them myself yourself himself herself itself ourselves "
                "themselves who whom what which".split()),
    "AUX": set("is are am was were be been being have has had do does did will would shall should can could "
               "may might must".split()),
    "CONJ": set("and or but nor so yet because although though if unless while whereas when since as than "
                "then that".split()),
}
WC_ORDER = ["DET", "PREP", "PRON", "AUX", "CONJ", "PUNCT", "CONTENT"]


def word_class(w: str) -> str:
    w = w.strip().lower()
    if not w or PUNCT_RE.match(w):
        return "PUNCT"
    for k in ["DET", "PREP", "PRON", "AUX", "CONJ"]:
        if w in WORDCLASS[k]:
            return k
    return "CONTENT"


def _rel(o: str, w: str, paradigm: dict) -> str:
    """Morphological relation between observed o and candidate w."""
    if w in paradigm.get(o, ()):
        return "IRREG"
    a, b = sorted([o, w], key=len)
    for sfx in ("s", "es", "ies"):
        if b.endswith(sfx) and (b[: -len(sfx)] == a or b[: -len(sfx)] + "y" == a or b[: -len(sfx)] == a[:-1]):
            return "S"
    for sfx in ("ed", "d", "ied"):
        if b.endswith(sfx):
            return "ED"
    if b.endswith("ing") or a.endswith("ing"):
        return "ING"
    return "OTHER"


def subclass_names() -> list[str]:
    return ([f"MORPH_{r}" for r in ["S", "ED", "ING", "IRREG", "OTHER"]]
            + ["FORM_CONTRACT", "FORM_NONWORD", "FORM_REALWORD"]
            + [f"PRED_{c}" for c in WC_ORDER] + [f"INS_{c}" for c in WC_ORDER]
            + [f"DEL_{c}" for c in WC_ORDER])


def sub_table(ctx) -> pd.DataFrame:
    """Every non-ORIG candidate with its subclass and gains (g, gx, gr)."""
    from transformers import AutoTokenizer

    from lib.lm import HF_ID, REVISION
    from lib.repair_core import Vocab
    c = build(ctx)
    o = c[c.cls == "ORIG"].set_index("inst")
    c = c[c.cls != "ORIG"].copy()
    for p, col in [("g", "logp"), ("gx", "logp_x"), ("gr", "logp_r")]:
        c[p] = c[col] - c.inst.map(o[col])
    vocab = Vocab(AutoTokenizer.from_pretrained(HF_ID, revision=REVISION))
    obs = ctx.tokens.observed.str.lower().to_numpy()
    fert = o.x_len
    sub = []
    for r in c.itertuples():
        ob = obs[r.inst]
        w = str(r.word).strip().lower()
        if r.cls == "MORPH":
            sub.append("MORPH_" + _rel(ob, w, vocab.paradigm))
        elif r.cls == "FORM":
            if "'" in w or "'" in ob:
                sub.append("FORM_CONTRACT")
            elif ob not in vocab.rank and fert.get(r.inst, 1) > 1:
                sub.append("FORM_NONWORD")
            else:
                sub.append("FORM_REALWORD")
        elif r.cls == "DEL":
            sub.append("DEL_" + word_class(ob))
        else:
            sub.append(f"{r.cls}_" + word_class(w))
    c["sub"] = sub
    return c


def sub_matrix(ctx) -> np.ndarray:
    """Positive part of the best gain (bits, no channel cost) per subclass."""
    best = sub_table(ctx).groupby(["inst", "sub"]).g.max().unstack()
    return best.reindex(index=range(ctx.n), columns=subclass_names()).fillna(0).clip(0, CEIL).to_numpy()


# --------------------------------------------------------------------------- #
# Noisy-channel edit cost. An edit must pay for itself: each character changed
# costs LAMBDA bits (Shannon's ~1 bit/character estimate for English, fixed a
# priori, not tuned). Deleting or replacing a whole word therefore costs about
# its length, while an inflection or a typo fix costs one or two bits.
# --------------------------------------------------------------------------- #
LAMBDA = 1.0


def edit_chars(ctx, c: pd.DataFrame) -> np.ndarray:
    obs = ctx.tokens.observed.str.lower().to_numpy()
    out = np.empty(len(c))
    for j, r in enumerate(c.itertuples()):
        o, w = obs[r.inst], str(r.word).strip().lower()
        if r.cls == "DEL":
            out[j] = len(o) + 1
        elif r.cls == "INS":
            out[j] = len(w) + 1
        else:
            out[j] = osa(o, w)
    return out


def mdl_table(ctx) -> pd.DataFrame:
    p = ctx.path("mdl_table")
    if p.exists():
        return pd.read_parquet(p)
    c = sub_table(ctx)
    c["dist"] = edit_chars(ctx, c)
    c["gm"] = c.g - LAMBDA * c.dist
    c.to_parquet(p, index=False)
    return c


# --------------------------------------------------------------------------- #
# The observed word's part of speech (spaCy, full-document parse) splits -s
# inflections into noun number and verb agreement.
# --------------------------------------------------------------------------- #
def split_by_pos(ctx, c: pd.DataFrame) -> pd.DataFrame:
    pos = ctx.tokens.pos.to_numpy()
    c = c.copy()
    m = c["sub"] == "MORPH_S"
    pv = pos[c.inst.to_numpy()]
    c.loc[m & np.isin(pv, ["NOUN", "PROPN"]), "sub"] = "MORPH_S_N"
    c.loc[m & np.isin(pv, ["VERB", "AUX"]), "sub"] = "MORPH_S_V"
    return c


def subclass_names_pos() -> list[str]:
    n = subclass_names()
    i = n.index("MORPH_S")
    return n[:i + 1] + ["MORPH_S_N", "MORPH_S_V"] + n[i + 1:]


def labels(ctx) -> pd.DataFrame:
    """Per token: the best edit after the channel cost (subclass, with -s split
    by part of speech), its net gain and the edited word, or NONE when no edit
    pays for itself."""
    c = split_by_pos(ctx, mdl_table(ctx))
    best = c.loc[c.groupby("inst").gm.idxmax()].set_index("inst")
    out = pd.DataFrame({"label": "NONE", "net_gain": 0.0, "best_word": ""}, index=range(ctx.n))
    pays = best.gm > 0
    out.loc[best.index[pays], "label"] = best["sub"][pays].to_numpy()
    out.loc[best.index, "net_gain"] = best.gm.to_numpy()
    out.loc[best.index, "best_word"] = best.word.astype(str).to_numpy()
    return out


def label_group(label: str) -> str:
    """grammatical (inflection, spelling, or a function word or punctuation
    inserted, deleted or replaced), lexical (a content word replaced, deleted
    or inserted), or none."""
    if label == "NONE":
        return "none"
    return "lexical" if label.endswith("_CONTENT") else "grammatical"
