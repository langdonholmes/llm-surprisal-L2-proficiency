"""Association pilot, step 5 (CPU) — how surprisal and LLM-PMI relate to the classic measures (Q3).

1. Token level: among attested real-word bigrams (w1 w2 adjacent, both real
   words, c(w1 w2) > 0 in corpus a), Spearman of w2's surprisal and LLM-PMI with
   the bigram's MI, ΔP and t. Original essays, and post-correction (own aligned
   w2 whose corrected-essay bigram is also an attested real-word bigram).
2. Essay level: Spearman matrix among all measures (original versions).
3. Exclusion by construction: original-essay mean surprisal split exactly by
   token class, Σ_class s / n, so the classes' covariances with the criterion add
   up to cov(mean surprisal, y). Two partitions:
     bigram view   attested alpha bigram w2 / unattested alpha bigram w2 /
                   alpha without an alpha left neighbour / non-alphabetic
     word view     real word / non-word (alpha, unseen in Dolma) /
                   PROPN or number-like alpha / non-alphabetic
   each crossed with corrected (gold ERR_TOK or ERR_INS) vs not. Also, for each
   token filter of the LLM measures (real, single-subword, + left bigram attested,
   + both attested, + s <= tau), the kept / removed split, crossed the same way.

Writes data/study3/pilot_assoc/tables/q3_tokens.csv, q3_matrix_<corpus>.csv, q3_exclusion.csv, q3.json

    python src/3-surprisal-decomposition/pilot_assoc/5_q3.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assoc_common import N_BOOT, OUT, SEED, TAB  # noqa: E402

MEAS = ["surprisal", "llm_pmi", "llm_dp", "llm_pmi_fadj", "llm_pmi_ge3", "high_s_content", "llm_pmi_att2",
        "llm_pmi_pbar",
        "amod_mi", "dobj_mi", "advmod_mi", "log_freq_s2", "content_freq", "bigram_mi", "bigram_dp", "bigram_t",
        "share_mi3", "share_t2", "share_unatt"]


def bigram_tokens(corpus, version):
    t = pd.read_parquet(OUT / f"tokens_{corpus}.parquet")
    it = pd.read_parquet(OUT / f"items_tok_{corpus}_{version}.parquet",
                         columns=["text_id", "spacy_idx", "pos", "like_num", "cnt_a"])
    big = pd.read_parquet(OUT / f"items_bigram_{corpus}_{version}.parquet")
    # w1 is the token just before w2
    w1 = it.assign(spacy_idx=it.spacy_idx + 1).rename(columns={"pos": "pos1", "like_num": "num1", "cnt_a": "cnt1"})
    big = big.merge(w1, on=["text_id", "spacy_idx"], how="left")
    big = big[(big.c12 > 0) & (big.cnt1 > 0) & (big.pos1 != "PROPN") & ~big.num1.astype(bool)]
    if version == "orig":
        x = t[t.sw].merge(big, on=["text_id", "spacy_idx"], how="inner")
        return x.rename(columns={"s_o": "s", "pmi_o": "pmi"})
    x = t[t.sw & t.own & t.aligned].drop(columns=["spacy_idx"]).rename(columns={"c_spacy_idx": "spacy_idx"})
    x = x.merge(big, on=["text_id", "spacy_idx"], how="inner")
    return x.rename(columns={"s_c": "s", "pmi_c": "pmi"})


def token_level():
    rows = []
    for corpus in ["wi", "fce"]:
        for version in ["orig", "corr"]:
            x = bigram_tokens(corpus, version)
            for a in ["s", "pmi"]:
                for b in ["mi", "dp", "t"]:
                    rows.append({"corpus": corpus, "version": version, "llm": "surprisal" if a == "s" else "llm_pmi",
                                 "classic": b, "rho": float(spearmanr(x[a], x[b])[0]), "n_tokens": len(x)})
            # within-word: residualize on log frequency of w2 (rank partial), for PMI vs MI
            rows.append({"corpus": corpus, "version": version, "llm": "surprisal", "classic": "log c(w2)",
                         "rho": float(spearmanr(x.s, np.log(x.c2))[0]), "n_tokens": len(x)})
            rows.append({"corpus": corpus, "version": version, "llm": "llm_pmi", "classic": "log c(w2)",
                         "rho": float(spearmanr(x.pmi, np.log(x.c2))[0]), "n_tokens": len(x)})
    return pd.DataFrame(rows)


def matrix(corpus):
    U = pd.read_parquet(OUT / f"units_{corpus}.parquet")
    X = U[[f"{m}|orig" for m in MEAS]].rename(columns=lambda c: c.split("|")[0])
    X["corr_share"] = U.corr_share
    X["y"] = U.y
    return X.corr(method="spearman")


def exclusion(corpus):
    t = pd.read_parquet(OUT / f"tokens_{corpus}.parquet",
                        columns=["unit", "text_id", "spacy_idx", "y", "gold", "s_o", "is_alpha", "like_num", "pos",
                                 "real", "cnt_a", "sw", "sw_att1", "sw_att2", "sw_lowS"])
    big = pd.read_parquet(OUT / f"items_bigram_{corpus}_orig.parquet", columns=["text_id", "spacy_idx", "c12"])
    t = t.merge(big, on=["text_id", "spacy_idx"], how="left")
    t["bigram_view"] = np.select(
        [t.c12 > 0, t.c12 == 0, t.is_alpha], ["attested bigram w2", "unattested bigram w2", "alpha, no alpha left"],
        "non-alphabetic")
    t["word_view"] = np.select(
        [t.real, t.is_alpha & (t.cnt_a == 0), t.is_alpha], ["real word", "non-word", "PROPN / number-like"],
        "non-alphabetic")
    # filters of the LLM measures (revision R1): kept vs removed
    for f in ["real", "sw", "sw_att1", "sw_att2", "sw_lowS"]:
        t[f"filter {f}"] = np.where(t[f], "kept", "removed")
    t["corrected"] = np.where(t.gold.isin(["ERR_TOK", "ERR_INS"]), "corrected", "not corrected")
    n = t.groupby("unit").size()
    y = t.groupby("unit").y.first()
    tot = t.groupby("unit").s_o.sum() / n
    rows = []
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(n), len(n)) for _ in range(N_BOOT)]
    yv, tv = y.to_numpy(), tot.reindex(y.index).to_numpy()

    def cov(a, b):
        return ((a - a.mean()) * (b - b.mean())).mean()

    for view in ["bigram_view", "word_view"] + [f"filter {f}" for f in ["real", "sw", "sw_att1", "sw_att2", "sw_lowS"]]:
        for keys in [[view], [view, "corrected"]]:
            parts = (t.groupby(["unit"] + keys).s_o.sum().unstack(keys, fill_value=0.0)
                     .reindex(y.index).fillna(0.0).div(n.reindex(y.index), axis=0))
            tok_share = t.groupby(keys).size() / len(t)
            for col in parts.columns:
                p = parts[col].to_numpy()
                est = cov(p, yv) / cov(tv, yv)
                bs = [cov(p[i], yv[i]) / cov(tv[i], yv[i]) for i in idx]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                col = col if isinstance(col, tuple) else (col,)
                rows.append({"corpus": corpus, "view": view, "class": " | ".join(col),
                             "token_share": float(tok_share.loc[col if len(col) > 1 else col[0]]),
                             "mean_bits_contrib": float(p.mean()), "cov_share": float(est),
                             "lo": float(lo), "hi": float(hi)})
    return rows


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    tl = token_level()
    tl.to_csv(TAB / "q3_tokens.csv", index=False)
    print(tl.round(3).to_string())
    res = {"tokens": tl.to_dict(orient="records")}
    for corpus in ["wi", "fce"]:
        M = matrix(corpus)
        M.to_csv(TAB / f"q3_matrix_{corpus}.csv")
        print(M.round(2).to_string())
        res[f"matrix_{corpus}"] = M.round(4).to_dict()
    ex = pd.DataFrame(exclusion("wi") + exclusion("fce"))
    ex.to_csv(TAB / "q3_exclusion.csv", index=False)
    print(ex.round(3).to_string())
    res["exclusion"] = ex.to_dict(orient="records")
    (TAB / "q3.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
