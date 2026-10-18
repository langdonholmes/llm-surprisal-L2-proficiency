"""Study 3, stage 5 — what the high-surprisal tokens are, read off expert edits.

W&I+LOCNESS (BEA-2019): learner essays at CEFR A, B and C, and 50 native
student essays (LOCNESS, level N), with the first annotator's corrections.
Every token above tau carries a gold category from stage 2 (ERR_TOK, ERR_INS,
POST, OK). Nothing is fit here: the decomposition, the attestation breakdown
and the repair check are all counts against the gold edits.

Per essay, with n the scored non-space tokens:

    high_rate = #(above tau) / n
    err_rate  = #(above tau, corrected: ERR_TOK or ERR_INS) / n
    acc_rate  = #(above tau, left uncorrected: OK) / n
    post_rate = #(above tau, right after a corrected token: POST) / n

so high_rate = err_rate + acc_rate + post_rate. POST tokens may owe their
surprisal to the error before them and are kept out of both halves.

Token types (mutually exclusive, from stage 4; first match wins):
nonword (unseen in both Dolma corpora), punctuation, proper noun or number,
function word, then content words split by rarity (fewer than 1 per million
words of edited Dolma) and by whether both bigrams (previous word + w, w + next
word) are attested in edited Dolma.

Outputs (results/study3/):
    gold_essays.csv           per essay rates
    gold_by_level.csv         mean rate per 1,000 tokens by level
    gold_threshold.csv        how often tau catches each gold category, by level
    gold_tests.json           Spearman rho with level (A < B < C) and native - C differences
    gold_types.csv            token types: share acceptable, by level
    gold_repair.csv           repair labels against gold: share corrected, by label and level

    python src/3-surprisal-decomposition/5_gold.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.attestation import FREQ  # noqa: E402
from lib.bench import CACHE, S3  # noqa: E402
from lib.repair_labels import label_group  # noqa: E402
from lib.stats import SEED, boot_spearman  # noqa: E402
from util.paths import RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "study3"
LEVELS = ["A", "B", "C", "N"]
FUNCTION_POS = ["DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"]
CONTENT_POS = ["NOUN", "VERB", "ADJ", "ADV"]
RARE_PER_MILLION = 1.0
RATES = ["high_rate", "err_rate", "acc_rate", "post_rate", "acc_share", "rare_attested_acc_rate"]


def token_types(att: pd.DataFrame) -> np.ndarray:
    total = pl.scan_parquet(str(FREQ / "corpus_a" / "3grams.parquet")).select(pl.col("count").sum()).collect().item()
    rare = att.a_log_uni < np.log1p(RARE_PER_MILLION * total / 1e6)
    attested = (att.a_unseen_bi_left == 0) & (att.a_unseen_bi_right == 0)
    content = att.pos.isin(CONTENT_POS)
    return np.select(
        [(att.a_unseen_uni > 0) & (att.b_unseen_uni > 0), att.is_punct > 0,
         (att.pos == "PROPN") | (att.like_num > 0), att.pos.isin(FUNCTION_POS),
         content & rare & attested, content & attested, content & rare, content],
        ["nonword", "punctuation", "proper noun or number", "function word",
         "rare content, attested", "common content, attested",
         "rare content, unattested", "common content, unattested"],
        "other")


def high_tokens() -> pd.DataFrame:
    """W&I tokens above tau with gold, token type and repair label (bench order)."""
    bench = pd.read_parquet(CACHE / "bench_wi.parquet", columns=["inst", "row_id"])
    gold = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet",
                           columns=["row_id", "text_id", "level", "gold", "text", "pos", "surprisal_bits"])
    h = bench.merge(gold, on="row_id", how="left")
    h["ttype"] = token_types(pd.read_parquet(CACHE / "attestation_wi.parquet"))
    lab = pd.read_parquet(CACHE / "labels_wi.parquet", columns=["row_id", "label", "scored"])
    h = h.merge(lab, on="row_id", how="left")
    h["group"] = h.label.map(label_group)
    h["err"] = h.gold.isin(["ERR_TOK", "ERR_INS"])
    return h


def essay_rates(h: pd.DataFrame) -> pd.DataFrame:
    toks = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet", columns=["text_id", "level", "is_space"])
    e = toks[~toks.is_space].groupby("text_id").agg(level=("level", "first"), n=("level", "size"))
    g = h.groupby("text_id")
    e["n_high"] = g.size()
    e["n_err"] = g.err.sum()
    e["n_ok"] = g.gold.apply(lambda s: (s == "OK").sum())
    e["n_post"] = g.gold.apply(lambda s: (s == "POST").sum())
    e["n_rare_attested_ok"] = h[(h.gold == "OK") & (h.ttype == "rare content, attested")].groupby("text_id").size()
    e = e.fillna(0)
    e["high_rate"] = e.n_high / e.n
    e["err_rate"] = e.n_err / e.n
    e["acc_rate"] = e.n_ok / e.n
    e["post_rate"] = e.n_post / e.n
    e["acc_share"] = e.n_ok / (e.n_ok + e.n_err)
    e["rare_attested_acc_rate"] = e.n_rare_attested_ok / e.n
    return e.reset_index()


def tests(e: pd.DataFrame) -> dict:
    learners = e[e.level != "N"].dropna(subset=["acc_share"])
    lv = learners.level.map({"A": 1, "B": 2, "C": 3})
    res = {"n_essays": e.level.value_counts().reindex(LEVELS).to_dict(),
           "base_share_corrected": float(e.n_err.sum() / (e.n_err.sum() + e.n_ok.sum())),
           "rho_with_level": {}, "native_minus_C": {}}
    for r in RATES:
        rho, lo, hi = boot_spearman(learners[r], lv)
        res["rho_with_level"][r] = {"rho": rho, "ci": [lo, hi]}
        nat, c = e[e.level == "N"][r].dropna().to_numpy(), e[e.level == "C"][r].dropna().to_numpy()
        rng = np.random.default_rng(SEED)
        boots = [rng.choice(nat, len(nat)).mean() - rng.choice(c, len(c)).mean() for _ in range(2000)]
        res["native_minus_C"][r] = {"diff": float(nat.mean() - c.mean()),
                                    "ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
    return res


def threshold_table() -> pd.DataFrame:
    """How often tau catches each gold category, and what share of high tokens each is."""
    t = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet", columns=["level", "gold", "high", "is_space"])
    t = t[~t.is_space]
    rows = []
    for lvl, d in t.groupby("level"):
        h = d[d.high]
        rows.append({"level": lvl, "n_tokens": len(d), "n_high": len(h),
                     **{f"p_high_given_{c}": d[d.gold == c].high.mean() for c in ["ERR_TOK", "ERR_INS", "OK"]},
                     **{f"share_of_high_{c}": (h.gold == c).mean() for c in ["ERR_TOK", "ERR_INS", "POST", "OK"]}})
    return pd.DataFrame(rows)


def by_cell(h: pd.DataFrame, col: str) -> pd.DataFrame:
    """Share acceptable (gold OK) per value of `col`, pooled and by level (POST excluded)."""
    d = h[h.gold != "POST"].assign(ok=lambda x: x.gold == "OK")
    pooled = d.groupby(col).agg(n=("ok", "size"), share_ok=("ok", "mean"))
    pooled["share_of_errors"] = d[d.err].groupby(col).size() / d.err.sum()
    pooled["share_of_acceptable"] = d[d.ok].groupby(col).size() / d.ok.sum()
    lv = d.pivot_table(index=col, columns="level", values="ok", aggfunc="mean").add_prefix("share_ok_")
    return pooled.join(lv).reset_index().sort_values("n", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = high_tokens()
    e = essay_rates(h)
    e.to_csv(OUT / "gold_essays.csv", index=False)
    by_level = (1000 * e.groupby("level")[["high_rate", "err_rate", "acc_rate", "post_rate",
                                             "rare_attested_acc_rate"]].mean()).reindex(LEVELS)
    by_level["acc_share"] = e.groupby("level").acc_share.mean().reindex(LEVELS)
    by_level["n_essays"] = e.level.value_counts().reindex(LEVELS)
    by_level.reset_index().to_csv(OUT / "gold_by_level.csv", index=False)
    print("per 1,000 tokens:\n", by_level.round(2).to_string())

    res = tests(e)
    (OUT / "gold_tests.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["rho_with_level"], indent=1))

    threshold_table().to_csv(OUT / "gold_threshold.csv", index=False)
    types = by_cell(h, "ttype")
    types.to_csv(OUT / "gold_types.csv", index=False)
    print(types.round(3).to_string(index=False))

    s = h[h.scored]
    rep = pd.concat([by_cell(s, "label").assign(level_of="label"),
                     by_cell(s, "group").rename(columns={"group": "label"}).assign(level_of="group")])
    rep.to_csv(OUT / "gold_repair.csv", index=False)
    print(rep[rep.level_of == "group"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
