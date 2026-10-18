"""Study 3, stage 8 — the LM repair representation on real essays.

The representation comes from the lab (``lab/README.md``, DECISIONS D8). For
every selected token, about 23 minimal edits (inflection, near spelling, the
model's predictions, deletion, insertion before the word) are scored by how
much they raise the log-probability of the word plus its next five subwords.
Two readouts:

    gains  the best gain (bits, positive part) per named edit class, 29 classes
           (`rep_repair_subpos`), the vector used for essay profiles;
    label  the single best edit after a cost of 1 bit per character changed,
           with -s inflection split into noun (number) and verb (agreement),
           or NONE when no edit pays for itself.

Subcommands:

    collect   lab caches for each real bench -> data/study3/repair_tokens.parquet
              (row_id, label, net_gain, best_word, gain_<class> x 29)
    check     dev-only real-data checks:
              - the D7 coded sample against the label (D8 predictions)
              - label distribution by Overall tertile
              - label cards with KWIC lines
              -> results/study3/repair_coded.csv, repair_label_by_band.csv,
                 repair_cards.md

    l1        TOEFL 11. The pre-registered article test (D8): per-token rate of
              INS_DET, article-less L1s (CHI HIN JPN KOR TEL TUR) vs. article
              L1s (ARA FRE GER ITA SPA), level-stratified mean difference, with
              10,000 within-level permutations of essay group labels. Also the
              per-L1 rate of every label and Study 1's weighted within-level
              Cohen's d per label x L1.
              -> results/study3/repair_l1_test.json, repair_l1_rates.csv,
                 repair_l1_d.csv

    python src/3-vector-transformations/8_repair.py collect --benches real_ellipse_dev real_ellipse_test
    python src/3-vector-transformations/8_repair.py check
    python src/3-vector-transformations/8_repair.py l1
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from util.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"
BANDS = ["low", "mid", "high"]


def collect_bench(name: str) -> pd.DataFrame:
    os.environ["LAB_BENCH"] = name
    import lab.harness as H
    importlib.reload(H)
    import lab.repair_build as RB
    import lab.repair_feats as RF
    importlib.reload(RB)
    importlib.reload(RF)
    ctx = H.Ctx()
    gains = RF.sub_matrix(ctx, "g", positive=True)
    c = RF.split_by_pos(ctx, RF.mdl_table(ctx))
    best = c.loc[c.groupby("inst").gm.idxmax()].set_index("inst")
    label = np.full(ctx.n, "NONE", dtype=object)
    net = np.zeros(ctx.n)
    word = np.full(ctx.n, "", dtype=object)
    pays = best.gm > 0
    label[best.index[pays]] = best["sub"][pays].to_numpy()
    net[best.index] = best.gm.to_numpy()
    word[best.index] = best.word.astype(str).to_numpy()
    df = pd.DataFrame(gains, columns=[f"gain_{s}" for s in RF.subclass_names()])
    df.insert(0, "row_id", ctx.bench.row_id.to_numpy())
    df.insert(1, "label", label)
    df.insert(2, "net_gain", net)
    df.insert(3, "best_word", word)
    df["scored"] = ctx.ok
    return df


def cmd_collect(args):
    path = S3 / "repair_tokens.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    new = [collect_bench(b) for b in args.benches]
    df = pd.concat([old] + new).drop_duplicates("row_id", keep="last").sort_values("row_id")
    df.to_parquet(path, index=False)
    print(df.label.value_counts().head(30))


def bands():
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet").drop(columns="text")
    split = pd.read_csv(HERE / "split.csv")
    e = essays.merge(split)
    cuts = e[e.split == "dev"].overall.quantile([1 / 3, 2 / 3]).to_list()
    e["band"] = pd.cut(e.overall, [-np.inf, *cuts, np.inf], right=False, labels=BANDS)
    return e


def cmd_check(args):
    rep = pd.read_parquet(S3 / "repair_tokens.parquet")
    sel = pd.read_parquet(S3 / "tokens_selected.parquet",
                          columns=["row_id", "text_id", "split", "idx", "text", "pos", "pred1", "surprisal_bits"])
    d = sel.merge(rep, on="row_id")
    d = d[(d.split == "dev") & d.scored].merge(bands()[["text_id", "band"]])
    d["family"] = d.label.str.split("_").str[0]

    # coded sample (D7) against the label
    key = pd.read_csv(HERE / "coding_key.csv")
    codes = pd.read_csv(HERE / "error_coding_claude.csv")
    k = key.merge(codes, on="item").merge(d[["row_id", "label", "family", "net_gain", "best_word",
                                              "surprisal_bits"]],
                                          on="row_id", how="left")
    k.to_csv(OUT / "repair_coded.csv", index=False)
    print(pd.crosstab(k.code, k.family, margins=True))

    # label distribution by tertile
    tab = d.groupby(["band", "label"], observed=True).size().rename("n").reset_index()
    tab["share"] = tab.n / tab.groupby("band", observed=True).n.transform("sum")
    tab.to_csv(OUT / "repair_label_by_band.csv", index=False)

    # cards
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet", columns=["text_id", "text"])
    texts = dict(zip(essays.text_id, essays.text))
    rng = np.random.default_rng(42)
    lines = ["# Repair label cards (dev ELLIPSE, real high-surprisal tokens)", "",
             "Random draws per label. Arrow: the winning edit's word (empty = deletion).", ""]
    order = d.label.value_counts()
    for lab_, n in order.items():
        g = d[d.label == lab_]
        mix = g.band.value_counts(normalize=True).reindex(BANDS).fillna(0)
        lines += [f"## {lab_}  (n = {n:,}, {n / len(d):.1%} of selected; tertile mix L/M/H "
                  f"{mix['low']:.2f}/{mix['mid']:.2f}/{mix['high']:.2f})", "", "```"]
        for r in g.iloc[rng.choice(len(g), min(15, len(g)), replace=False)].itertuples():
            t = texts[r.text_id]
            left = t[max(0, r.idx - 55):r.idx].replace("\n", " ")
            right = t[r.idx + len(r.text):r.idx + len(r.text) + 40].replace("\n", " ")
            lines.append(f"{left:>55} [[{r.text}]] {right}   -> {r.best_word!r} ({r.net_gain:+.1f})")
        lines += ["```", ""]
    (OUT / "repair_cards.md").write_text("\n".join(lines))


ARTICLELESS = ["CHI", "HIN", "JPN", "KOR", "TEL", "TUR"]
ARTICLE = ["ARA", "FRE", "GER", "ITA", "SPA"]
LEVELS = ["low", "medium", "high"]


def toefl_rates() -> pd.DataFrame:
    rep = pd.read_parquet(S3 / "repair_tokens.parquet")
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["row_id", "text_id", "corpus"])
    d = sel[sel.corpus == "toefl11"].merge(rep, on="row_id")
    d = d[d.scored]
    es = pd.read_parquet(S3 / "essay_summary.parquet").query("corpus == 'toefl11'")
    lab = pd.read_parquet(S3 / "toefl11" / "essays.parquet").drop(columns="text")
    e = lab.merge(es[["text_id", "n_scored", "n_high"]]).set_index("text_id")
    counts = pd.crosstab(d.text_id, d.label).reindex(e.index, fill_value=0)
    rates = counts.div(e.n_scored, axis=0)
    return e.join(rates)


def cmd_l1(args):
    sys.path.insert(0, str(HERE))
    weighted_d = importlib.import_module("7_toefl").weighted_d
    e = toefl_rates()
    labels = [c for c in e.columns if c.isupper() and "_" in c or c == "NONE"]
    e["group"] = np.where(e.l1.isin(ARTICLELESS), "articleless", "article")

    x = e.INS_DET.to_numpy()
    g = (e.group == "articleless").to_numpy()
    lev = e.level.to_numpy()
    w = {lv: (lev == lv).mean() for lv in LEVELS}

    def stat(gg):
        return sum(w[lv] * (x[(lev == lv) & gg].mean() - x[(lev == lv) & ~gg].mean()) for lv in LEVELS)

    obs = stat(g)
    rng = np.random.default_rng(42)
    null = np.empty(10_000)
    for i in range(len(null)):
        gg = g.copy()
        for lv in LEVELS:
            m = lev == lv
            gg[m] = rng.permutation(gg[m])
        null[i] = stat(gg)
    by_l1 = e.groupby("l1").INS_DET.mean()
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(by_l1[ARTICLELESS], by_l1[ARTICLE], alternative="greater")
    pooled_sd = np.sqrt(np.mean([x[lev == lv].var(ddof=1) for lv in LEVELS]))
    res = {
        "stat_level_stratified_diff_per_token": float(obs),
        "stat_in_pooled_within_level_sd": float(obs / pooled_sd),
        "p_one_sided_permutation": float((np.sum(null >= obs) + 1) / (len(null) + 1)),
        "per_level_means": {lv: {"articleless": float(x[(lev == lv) & g].mean()),
                                 "article": float(x[(lev == lv) & ~g].mean())} for lv in LEVELS},
        "l1_means_per_1000_tokens": {k: round(1000 * v, 4) for k, v in by_l1.sort_values().items()},
        "l1_level_mann_whitney_greater_p": float(mw.pvalue),
        "n_essays": int(len(e)),
    }
    (OUT / "repair_l1_test.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    e.groupby(["l1", "level"])[labels].mean().reset_index().to_csv(OUT / "repair_l1_rates.csv", index=False)
    rows = []
    for lab_ in labels:
        for l1, d in weighted_d(e[lab_].to_numpy(), e.l1.to_numpy(), lev).items():
            rows.append({"label": lab_, "l1": l1, "d": d})
    pd.DataFrame(rows).to_csv(OUT / "repair_l1_d.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--benches", nargs="+", required=True)
    sub.add_parser("check")
    sub.add_parser("l1")
    args = ap.parse_args()
    {"collect": cmd_collect, "check": cmd_check, "l1": cmd_l1}[args.cmd](args)


if __name__ == "__main__":
    main()
