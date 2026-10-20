"""Association pilot, step 2 (CPU) — classic phraseological and lexical measures.

From the spaCy parses of the original and corrected essays (W&I, FCE), per doc:

  Study 2 (src/essays/metrics.py with --corpus both, i.e. Dolma a + b summed):
    amod_mi, dobj_mi, advmod_mi   MiCalculator, reproduced with polars lookups
    log_freq_s2                   log_mean_token_freq (lemma, unseen -> count 1)
  New (Dolma corpus a):
    bigram MI, ΔP (forward), t over adjacent alphabetic token pairs, attested only;
    shares MI >= 3, t >= 2 (of attested), unattested (of all);
    content_freq: mean log10 relative frequency of attested content-word forms.

The doc-level means are computed from per-item tables, so FCE scripts pool both
answers' items. Also writes the per-token bigram table (w2's spacy_idx) for Q3,
and the corpus-a unigram counts of every token form (for the real-word filter).

--check reproduces Study 2's values on ELLIPSE (local) for the first 300 docs.

Writes data/study3/pilot_assoc/:
    items_bigram_<corpus>_<version>.parquet   text_id, spacy_idx (w2), w1, w2, c12, c1, c2
    items_dep_<corpus>_<version>.parquet      text_id, rel, h, d, mi (attested only)
    items_tok_<corpus>_<version>.parquet      text_id, spacy_idx, lower, lemma, pos, is_alpha,
                                              like_num, cnt_a (form, corpus a), cnt_ab_lemma
    classic_<corpus>_<version>.parquet        per doc (text_id) measures

    python src/3-surprisal-decomposition/pilot_assoc/2_classic.py [--check]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assoc_common import CONTENT, DOCBINS, FREQ, OUT  # noqa: E402
from util.paths import DATA_DIR, ELLIPSE_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

RELS = ("amod", "dobj", "advmod")


def doc_items(docs):
    """Per-token, per-bigram and per-dependency rows for an iterable of docs."""
    tok, big, dep = [], [], []
    for doc in docs:
        m = doc.user_data.get("meta")
        tid = str(m["text_id"] if isinstance(m, dict) else m)
        prev = None
        for t in doc:
            tok.append((tid, t.i, t.lower_, t.lemma_.lower(), t.pos_, t.is_alpha, t.like_num, t.is_space,
                        t.is_punct))
            if t.is_alpha and prev is not None and prev.is_alpha and prev.i == t.i - 1:
                big.append((tid, t.i, prev.lower_, t.lower_))
            prev = t
            if t.dep_ in RELS:
                dep.append((tid, t.dep_, t.head.lemma_.lower(), t.lemma_.lower()))
    tok = pd.DataFrame(tok, columns=["text_id", "spacy_idx", "lower", "lemma", "pos", "is_alpha", "like_num",
                                     "is_space", "is_punct"])
    big = pd.DataFrame(big, columns=["text_id", "spacy_idx", "w1", "w2"])
    dep = pd.DataFrame(dep, columns=["text_id", "rel", "h", "d"])
    return tok, big, dep


class Ref:
    """Lookups into the Dolma tables for a fixed set of query keys."""

    def __init__(self, uni_keys, bi_keys, dep_keys, lemma_keys):
        t0 = time.time()
        self.N, self.uni, self.bi = {}, {}, {}
        for c in ["a", "b"]:
            lf = pl.scan_parquet(str(FREQ / f"corpus_{c}" / "3grams.parquet"))
            self.N[c] = int(lf.select(pl.col("count").sum()).collect().item())
            keys = pl.DataFrame({"u": sorted(set(uni_keys) | set(lemma_keys))}).lazy()
            u = (lf.join(keys, left_on="token_2", right_on="u", how="semi").group_by("token_2")
                 .agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
            self.uni[c] = dict(zip(u["token_2"].to_list(), u["c"].to_list()))
            if c == "a":
                bk = pl.DataFrame(list(set(bi_keys)), schema=["a", "b"], orient="row").lazy()
                b = (lf.join(bk, left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
                     .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("c"))
                     .collect(engine="streaming"))
                self.bi["a"] = dict(zip(zip(b["token_1"].to_list(), b["token_2"].to_list()), b["c"].to_list()))
            print(f"3grams {c}: {time.time() - t0:.0f}s", flush=True)
        # dependency tables, a + b summed, tags collapsed (as metrics.py does)
        dk = pl.DataFrame(list(set(dep_keys)), schema=["h", "d", "rel"], orient="row")
        hs = pl.DataFrame({"h": dk["h"].unique()}).lazy()
        ds = pl.DataFrame({"d": dk["d"].unique()}).lazy()
        self.dep_total, J, H, D = 0, {}, {}, {}
        for c in ["a", "b"]:
            lf = pl.scan_parquet(str(FREQ / f"corpus_{c}" / "depgrams.parquet"))
            self.dep_total += int(lf.select(pl.col("count").sum()).collect().item())
            j = (lf.join(dk.lazy(), left_on=["head_lemma", "dependent_lemma", "relation"],
                         right_on=["h", "d", "rel"], how="semi")
                 .group_by(["head_lemma", "dependent_lemma", "relation"]).agg(pl.col("count").sum())
                 .collect(engine="streaming"))
            for h, d, r, n in j.iter_rows():
                J[(h, d, r)] = J.get((h, d, r), 0) + n
            hm = (lf.join(hs, left_on="head_lemma", right_on="h", how="semi").group_by("head_lemma")
                  .agg(pl.col("count").sum()).collect(engine="streaming"))
            for h, n in hm.iter_rows():
                H[h] = H.get(h, 0) + n
            dm = (lf.join(ds, left_on="dependent_lemma", right_on="d", how="semi").group_by("dependent_lemma")
                  .agg(pl.col("count").sum()).collect(engine="streaming"))
            for d, n in dm.iter_rows():
                D[d] = D.get(d, 0) + n
            print(f"depgrams {c}: {time.time() - t0:.0f}s", flush=True)
        self.J, self.H, self.D = J, H, D


def annotate(tok, big, dep, ref: Ref):
    Na = ref.N["a"]
    Nab = ref.N["a"] + ref.N["b"]
    tok = tok.copy()
    tok["cnt_a"] = tok.lower.map(ref.uni["a"]).fillna(0).astype(np.int64)
    tok["cnt_ab_lemma"] = (tok.lemma.map(ref.uni["a"]).fillna(0) + tok.lemma.map(ref.uni["b"]).fillna(0)).astype(np.int64)
    big = big.copy()
    big["c12"] = [ref.bi["a"].get(k, 0) for k in zip(big.w1, big.w2)]
    big["c1"] = big.w1.map(ref.uni["a"]).fillna(0).astype(np.int64)
    big["c2"] = big.w2.map(ref.uni["a"]).fillna(0).astype(np.int64)
    # the tables prune trigrams seen once, so a derived unigram count can fall below
    # the bigram count (rarely to 0); floor the marginals at the joint count
    big["c1"] = np.maximum(big.c1, big.c12)
    big["c2"] = np.maximum(big.c2, big.c12)
    att = big.c12 > 0
    c12, c1, c2 = big.c12.astype(float), big.c1.astype(float), big.c2.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        big["mi"] = np.where(att, np.log2(c12 * Na / (c1 * c2)), np.nan)
        big["dp"] = np.where(att, c12 / c1 - (c2 - c12) / (Na - c1), np.nan)
        big["t"] = np.where(att, (c12 - c1 * c2 / Na) / np.sqrt(c12), np.nan)
    dep = dep.copy()
    j = np.array([ref.J.get(k, 0) for k in zip(dep.h, dep.d, dep.rel)], float)
    hm = dep.h.map(ref.H).fillna(0).to_numpy(float)
    dm = dep.d.map(ref.D).fillna(0).to_numpy(float)
    T = ref.dep_total
    with np.errstate(divide="ignore", invalid="ignore"):
        dep["mi"] = np.where(j > 0, np.log2((j / T) / ((hm / T) * (dm / T))), np.nan)
    dep = dep[j > 0]
    return tok, big, dep, Na, Nab


def measures(tok, big, dep, Na, Nab, unit) -> pd.DataFrame:
    """Per-unit means. ``unit`` maps text_id -> unit id."""
    tok = tok.assign(u=tok.text_id.map(unit))
    big = big.assign(u=big.text_id.map(unit))
    dep = dep.assign(u=dep.text_id.map(unit))
    al = tok[tok.is_alpha]
    out = pd.DataFrame(index=pd.Index(sorted(tok.u.dropna().unique()), name="unit"))
    out["log_freq_s2"] = al.assign(x=np.log10(al.cnt_ab_lemma.clip(lower=1) / Nab)).groupby("u").x.mean()
    cw = al[al.pos.isin(CONTENT) & (al.cnt_a > 0)]
    out["content_freq"] = cw.assign(x=np.log10(cw.cnt_a / Na)).groupby("u").x.mean()
    for r in ["amod", "dobj", "advmod"]:
        out[f"{r}_mi"] = dep[dep.rel == r].groupby("u").mi.mean()
    att = big[big.c12 > 0]
    g = att.groupby("u")
    out["bigram_mi"] = g.mi.mean()
    out["bigram_dp"] = g.dp.mean()
    out["bigram_t"] = g.t.mean()
    out["share_mi3"] = att.assign(x=(att.mi >= 3).astype(float)).groupby("u").x.mean()
    out["share_t2"] = att.assign(x=(att.t >= 2).astype(float)).groupby("u").x.mean()
    out["share_unatt"] = big.assign(x=(big.c12 == 0).astype(float)).groupby("u").x.mean()
    out["n_bigrams"] = big.groupby("u").size()
    out["n_attested"] = att.groupby("u").size()
    return out


def check():
    """Reproduce Study 2's ELLIPSE values (first 300 docs)."""
    docs = []
    for d in load_all_docbins(ELLIPSE_DOCBINS_DIR):
        docs.append(d)
        if len(docs) == 300:
            break
    tok, big, dep = doc_items(docs)
    ref = Ref(tok.lower, list(zip(big.w1, big.w2)), list(zip(dep.h, dep.d, dep.rel)), tok.lemma)
    tok, big, dep, Na, Nab = annotate(tok, big, dep, ref)
    m = measures(tok, big, dep, Na, Nab, {t: t for t in tok.text_id.unique()})
    s2 = pd.read_csv(DATA_DIR / "ellipse" / "ellipse_metrics.csv").set_index("text_id_kaggle")
    s2 = s2.rename(columns={"log_mean_token_freq": "log_freq_s2"})
    for c in ["amod_mi", "dobj_mi", "advmod_mi", "log_freq_s2"]:
        x, y = m[c], s2[c].reindex(m.index)
        ok = x.notna() & y.notna()
        print(f"{c}: n {ok.sum()}  max |diff| {np.abs(x[ok] - y[ok]).max():.2e}  "
              f"NaN agree {(x.isna() == y.isna()).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    t0 = time.time()
    items = {}
    for key, d in DOCBINS.items():
        items[key] = doc_items(load_all_docbins(d))
        print(key, [len(x) for x in items[key]], f"{time.time() - t0:.0f}s", flush=True)
    tok_all = pd.concat([v[0] for v in items.values()])
    big_all = pd.concat([v[1] for v in items.values()])
    dep_all = pd.concat([v[2] for v in items.values()])
    ref = Ref(tok_all.lower, list(zip(big_all.w1, big_all.w2)), list(zip(dep_all.h, dep_all.d, dep_all.rel)),
              tok_all.lemma)
    fce = pd.read_parquet(DATA_DIR / "study3" / "pilot_fce" / "essays.parquet", columns=["text_id", "script_id"])
    fce_unit = dict(zip(fce.text_id, fce.script_id))
    for (corpus, version), (tok, big, dep) in items.items():
        tok, big, dep, Na, Nab = annotate(tok, big, dep, ref)
        tag = f"{corpus}_{version}"
        tok.to_parquet(OUT / f"items_tok_{tag}.parquet", index=False)
        big.to_parquet(OUT / f"items_bigram_{tag}.parquet", index=False)
        dep.to_parquet(OUT / f"items_dep_{tag}.parquet", index=False)
        unit = fce_unit if corpus == "fce" else {t: t for t in tok.text_id.unique()}
        m = measures(tok, big, dep, Na, Nab, unit)
        m.to_parquet(OUT / f"classic_{tag}.parquet")
        print(tag, m.shape, f"{time.time() - t0:.0f}s", flush=True)
        print(m.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
