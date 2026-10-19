"""Pilot (native calibration), analysis — per-document mean surprisal and high-token share by group.

    python -I analyze.py DOCS.parquet TOKENS.parquet OUT_DIR

Learner groups come from the stage 1 token files (data/study3/{wi,ellipse}),
which ``score.py verify`` showed this pipeline reproduces exactly:
    W&I A / B / C, LOCNESS (N); ELLIPSE Overall <= 2.5, 3-3.5, >= 4.
Each learner group is reported twice: all essays ("all"), and length-matched
("len") = essays with >= 200 words, tokens cut at the same 600-word
sentence-boundary truncation applied to the external documents.

Per document: mean surprisal (bits) over scored tokens (spaCy non-space tokens
that align with a first subword; = Study 1's mean_surprisal_bits) and the share
of scored tokens >= tau. Per group: mean, SD, median of the document means, 95%
bootstrap CI of the mean (2,000 resamples of documents, seed 42), and the same
for the high share.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from prep import MIN_WORDS, truncate  # noqa: E402
from util.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

SEED, N_BOOT = 42, 2000
TAU = json.loads((RESULTS_DIR / "study3" / "threshold.json").read_text())["tau_bits"]


def per_doc(tok: pd.DataFrame) -> pd.DataFrame:
    return tok.assign(hi=tok.surprisal_bits >= TAU).groupby("text_id").agg(
        mean_s=("surprisal_bits", "mean"), hi_share=("hi", "mean"), n_tok=("surprisal_bits", "size"))


def learner(corpus: str, label_col: str, label_fn) -> pd.DataFrame:
    ess = pd.read_parquet(DATA_DIR / "study3" / corpus / "essays.parquet")
    tok = pd.concat([pd.read_parquet(f, columns=["text_id", "idx", "surprisal_bits"])
                     for f in sorted(glob.glob(str(DATA_DIR / "study3" / corpus / "shards" / "*.parquet")))])
    ess["group"] = ess[label_col].map(label_fn)
    ess["n_words_raw"] = ess.text.str.split().str.len()
    lead = ess.text.str.len() - ess.text.str.lstrip().str.len()
    ess["cut"] = lead + ess.text.map(lambda t: len(truncate(t)))
    out = []
    a = per_doc(tok).join(ess.set_index("text_id")[["group", "n_words_raw"]])
    out.append(a.assign(subset="all", n_words=a.n_words_raw))
    t = tok.merge(ess[["text_id", "cut"]], on="text_id")
    t = t[t.idx < t.cut]
    m = ess[ess.n_words_raw >= MIN_WORDS].set_index("text_id")
    b = per_doc(t[t.text_id.isin(m.index)]).join(m[["group", "n_words_raw", "text"]])
    b["n_words"] = b.text.map(lambda x: len(truncate(x).split()))
    out.append(b.drop(columns="text").assign(subset="len"))
    return pd.concat(out).reset_index()


def ellipse_band(x: float) -> str:
    return "ELLIPSE <=2.5" if x <= 2.5 else ("ELLIPSE 3-3.5" if x <= 3.5 else "ELLIPSE >=4")


def summarize(d: pd.DataFrame, rng) -> dict:
    r = {"n": len(d), "words_median": float(d.n_words.median())}
    for c in ["mean_s", "hi_share"]:
        x = d[c].to_numpy()
        boots = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
        r |= {f"{c}_mean": x.mean(), f"{c}_sd": x.std(ddof=1), f"{c}_median": float(np.median(x)),
              f"{c}_lo": np.quantile(boots, .025), f"{c}_hi": np.quantile(boots, .975)}
    return r


def main():
    docs_path, tok_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = pd.read_parquet(docs_path)
    ext = per_doc(pd.read_parquet(tok_path)).join(
        docs.set_index("text_id")[["group", "n_words_raw", "n_words"]]).reset_index()
    ext["subset"] = "len"
    wi = learner("wi", "level", lambda x: {"A": "W&I A", "B": "W&I B", "C": "W&I C",
                                           "N": "LOCNESS (N)"}[x])
    el = learner("ellipse", "overall", ellipse_band)
    per = pd.concat([ext.assign(source="external"), wi.assign(source="wi"), el.assign(source="ellipse")],
                    ignore_index=True)
    per.to_csv(out_dir / "doc_means.csv", index=False)

    order = ["W&I A", "W&I B", "W&I C", "LOCNESS (N)", "ELLIPSE <=2.5", "ELLIPSE 3-3.5",
             "ELLIPSE >=4", "persuade_ell", "persuade_native", "persuade_native_hi",
             "wiki_old", "wiki_new"]
    rng = np.random.default_rng(SEED)
    rows = []
    for subset in ["len", "all"]:
        for g in order:
            d = per[(per.group == g) & (per.subset == subset)]
            if len(d):
                rows.append({"group": g, "subset": subset, **summarize(d, rng)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "group_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab.round(3).to_string(index=False))
    print("tau", TAU)


if __name__ == "__main__":
    main()
