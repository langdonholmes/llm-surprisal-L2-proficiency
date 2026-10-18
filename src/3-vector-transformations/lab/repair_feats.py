"""Repair lab — turn the scored candidate table into per-instance features.

All features are computed from the candidate table and ctx.base; no benchmark
labels are read.
"""

import numpy as np
import pandas as pd

from lab.repair_build import build
from lab.repair_core import FUNCTION, PUNCT_RE, osa

CLASSES = ["MORPH", "FORM", "PRED", "DEL", "INS"]
FLOOR, CEIL = -30.0, 40.0


def class_table(ctx) -> pd.DataFrame:
    """Per (inst, class): the best candidate by total gain, with gain split into
    the edited material (gx) and the right context (gr). Missing class -> FLOOR."""
    c = build(ctx)
    o = c[c.cls == "ORIG"].set_index("inst")
    c = c[c.cls != "ORIG"].copy()
    c["g"] = c.logp - c.inst.map(o.logp)
    c["gx"] = c.logp_x - c.inst.map(o.logp_x)
    c["gr"] = c.logp_r - c.inst.map(o.logp_r)
    best = c.loc[c.groupby(["inst", "cls"]).g.idxmax()]
    return best


def gain_matrix(ctx, parts=("g",)) -> np.ndarray:
    best = class_table(ctx)
    cols = []
    for p in parts:
        w = best.pivot(index="inst", columns="cls", values=p).reindex(index=range(ctx.n), columns=CLASSES)
        cols.append(w.fillna(FLOOR).clip(FLOOR, CEIL).to_numpy())
    return np.hstack(cols)


def best_repair_props(ctx) -> np.ndarray:
    """Properties of the overall best repair: edit distance to the observed word,
    function-word / punctuation status of the observed word and the repair,
    observed word length in subwords (fertility)."""
    best = class_table(ctx)
    top = best.loc[best.groupby("inst").g.idxmax()].set_index("inst")
    obs = ctx.bench.observed.str.lower()
    out = np.zeros((ctx.n, 6), dtype=np.float32)
    c = build(ctx)
    fert = c[c.cls == "ORIG"].set_index("inst").x_len
    for i in range(ctx.n):
        o = obs.iat[i]
        out[i, 2] = o in FUNCTION
        out[i, 5] = fert.get(i, 1)
        if i not in top.index:
            continue
        w = str(top.at[i, "word"]).strip().lower()
        out[i, 0] = osa(o, w) / max(len(o), 1) if w else 1.0
        out[i, 1] = top.at[i, "g"]
        out[i, 3] = w in FUNCTION
        out[i, 4] = bool(PUNCT_RE.match(w)) if w else 0.0
    out[:, 1] = np.clip(out[:, 1], FLOOR, CEIL)
    return out


def surprisal4(ctx) -> np.ndarray:
    return np.column_stack([ctx.get(k) for k in ["t_surprisal", "t_entropy", "n_surprisal", "n_entropy"]])


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
    from lab.repair_core import Vocab
    from transformers import AutoTokenizer
    from lab.lm import HF_ID, REVISION
    c = build(ctx)
    o = c[c.cls == "ORIG"].set_index("inst")
    c = c[c.cls != "ORIG"].copy()
    for p, col in [("g", "logp"), ("gx", "logp_x"), ("gr", "logp_r")]:
        c[p] = c[col] - c.inst.map(o[col])
    vocab = Vocab(AutoTokenizer.from_pretrained(HF_ID, revision=REVISION))
    obs = ctx.bench.observed.str.lower().to_numpy()
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


def sub_matrix(ctx, part="g", positive=False) -> np.ndarray:
    c = sub_table(ctx)
    best = c.groupby(["inst", "sub"])[part].max().unstack()
    m = best.reindex(index=range(ctx.n), columns=subclass_names())
    if positive:
        return m.fillna(0).clip(0, CEIL).to_numpy()
    return m.fillna(FLOOR).clip(FLOOR, CEIL).to_numpy()


# --------------------------------------------------------------------------- #
# Noisy-channel edit cost. An edit must pay for itself: each character changed
# costs LAMBDA bits (Shannon's ~1 bit/character estimate for English, fixed a
# priori, not tuned). Deleting or replacing a whole word therefore costs about
# its length, while an inflection or a typo fix costs one or two bits.
# --------------------------------------------------------------------------- #
LAMBDA = 1.0


def edit_chars(ctx, c: pd.DataFrame) -> np.ndarray:
    obs = ctx.bench.observed.str.lower().to_numpy()
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
    from lab.harness import LAB, SUFFIX
    p = LAB / "repair" / f"mdl_table{SUFFIX}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    c = sub_table(ctx)
    c["dist"] = edit_chars(ctx, c)
    c["gm"] = c.g - LAMBDA * c.dist
    c.to_parquet(p, index=False)
    return c


def mdl_matrix(ctx, part="gm", positive=False) -> np.ndarray:
    c = mdl_table(ctx)
    m = c.groupby(["inst", "sub"])[part].max().unstack() \
        .reindex(index=range(ctx.n), columns=subclass_names())
    if positive:
        return m.fillna(0).clip(0, CEIL).to_numpy()
    return m.fillna(FLOOR).clip(FLOOR, CEIL).to_numpy()


def mdl_label(ctx) -> np.ndarray:
    """One-hot of the best edit after the channel cost, scaled by its net gain;
    an all-zero row means no edit pays for itself."""
    c = mdl_table(ctx)
    best = c.loc[c.groupby("inst").gm.idxmax()]
    names = subclass_names()
    out = np.zeros((ctx.n, len(names)), dtype=np.float32)
    for r in best.itertuples():
        if r.gm > 0:
            out[r.inst, names.index(r.sub)] = min(r.gm, CEIL)
    return out


# --------------------------------------------------------------------------- #
# Part of speech of the observed word (spaCy, as in the real pipeline, where
# every token already carries a tag). For the bench, a local window of the
# edited text is tagged.
# --------------------------------------------------------------------------- #
def observed_pos(ctx) -> np.ndarray:
    from lab.harness import LAB, SUFFIX
    p = LAB / "repair" / f"obs_pos{SUFFIX}.npy"
    if p.exists():
        return np.load(p, allow_pickle=True)
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer"])
    b = ctx.bench
    starts = [max(0, c - 200) for c in b.target_char]
    texts = [t[s:c + 120] for t, s, c in zip(b.text, starts, b.target_char)]
    out = np.empty(len(b), dtype=object)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256, n_process=6)):
        rel = b.target_char.iat[i] - starts[i]
        out[i] = next((t.pos_ for t in doc if t.idx == rel), "X")
    np.save(p, out)
    return out


def split_by_pos(ctx, c: pd.DataFrame) -> pd.DataFrame:
    pos = observed_pos(ctx)
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


def mdl_matrix_pos(ctx, part="gm", positive=False) -> np.ndarray:
    c = split_by_pos(ctx, mdl_table(ctx))
    m = c.groupby(["inst", "sub"])[part].max().unstack() \
        .reindex(index=range(ctx.n), columns=subclass_names_pos())
    if positive:
        return m.fillna(0).clip(0, CEIL).to_numpy()
    return m.fillna(FLOOR).clip(FLOOR, CEIL).to_numpy()
