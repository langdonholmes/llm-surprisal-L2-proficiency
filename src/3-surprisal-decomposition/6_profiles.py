"""Study 3, stage 6 — repair labels as essay profiles.

Per essay, the rate of each repair label per scored token (count / n scored),
and of the three label groups (grammatical, lexical, none; see
lib/repair_labels.label_group).

    proficiency   ELLIPSE: Spearman rho of each rate with Overall (95% bootstrap
                  CI over essays), and the mean rate by Overall band.
                  W&I+LOCNESS: mean rate by level (A, B, C, N).
                  -> results/study3/profiles_ellipse.csv, profiles_rho.csv, profiles_wi_levels.csv
    cards         KWIC lines for each label (ELLIPSE) -> results/study3/repair_cards.md
    l1            TOEFL 11, descriptive: mean rate by L1 and level, and Study 1's
                  weighted within-level Cohen's d per label x L1
                  -> results/study3/profiles_l1_rates.csv, profiles_l1_d.csv

    python src/3-surprisal-decomposition/6_profiles.py proficiency
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bench import CACHE, S3, corpus_benches  # noqa: E402
from lib.repair_labels import label_group  # noqa: E402
from lib.stats import SEED, boot_spearman, weighted_d  # noqa: E402
from util.paths import RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "study3"
BANDS = ["low", "mid", "high"]
GROUPS = ["grammatical", "lexical", "none"]


def high_tokens(corpus: str) -> pd.DataFrame:
    """Scored high tokens of a corpus with their repair label and bench row."""
    parts = []
    for name in corpus_benches(corpus):
        b = pd.read_parquet(CACHE / f"bench_{name}.parquet", columns=["row_id", "text_id", "observed", "target_char"])
        lab = pd.read_parquet(CACHE / f"labels_{name}.parquet", columns=["row_id", "label", "net_gain",
                                                                          "best_word", "scored"])
        parts.append(b.merge(lab, on="row_id"))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.scored]
    d["group"] = d.label.map(label_group)
    return d


def rates(d: pd.DataFrame, n: pd.Series) -> pd.DataFrame:
    """Per essay rate of every label and group, per scored token."""
    by_label = pd.crosstab(d.text_id, d.label).drop(columns="NONE", errors="ignore")
    by_group = pd.crosstab(d.text_id, d.group).reindex(columns=GROUPS, fill_value=0).add_prefix("group_")
    return by_label.join(by_group).reindex(n.index, fill_value=0).div(n, axis=0)


def cmd_proficiency(args):
    d = high_tokens("ellipse")
    summ = pd.read_parquet(S3 / "essay_summary.parquet").query("corpus == 'ellipse'").set_index("text_id")
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet").drop(columns="text").set_index("text_id")
    r = rates(d, summ.n_scored)
    e = essays.join(r)
    e.to_csv(OUT / "profiles_ellipse.csv")
    cols = list(r.columns)
    share = pd.concat([d.label.value_counts(normalize=True), d.group.value_counts(normalize=True)
                       .rename(lambda g: f"group_{g}")])
    e["band"] = pd.cut(e.overall, [-np.inf, 3, 3.5, np.inf], right=False, labels=BANDS)
    band = 1000 * e.groupby("band", observed=True)[cols].mean().T
    rows = []
    for c in cols:
        rho, lo, hi = boot_spearman(e[c], e.overall)
        rows.append({"label": c, "share_of_high": share.get(c, np.nan), "rho_overall": rho, "lo": lo, "hi": hi,
                     **{f"per_1000_{b}": band.at[c, b] for b in BANDS}})
    rho = pd.DataFrame(rows).sort_values("share_of_high", ascending=False)
    rho.to_csv(OUT / "profiles_rho.csv", index=False)
    print(rho.round(3).to_string(index=False))

    w = high_tokens("wi")
    toks = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet", columns=["text_id", "level", "is_space"])
    ew = toks[~toks.is_space].groupby("text_id").agg(level=("level", "first"), n=("level", "size"))
    lv = (1000 * ew[["level"]].join(rates(w, ew.n)).groupby("level").mean().T)
    lv.reindex(columns=["A", "B", "C", "N"]).rename_axis("label").reset_index() \
        .to_csv(OUT / "profiles_wi_levels.csv", index=False)


def cmd_cards(args):
    d = high_tokens("ellipse")
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet", columns=["text_id", "text"])
    texts = dict(zip(essays.text_id, essays.text))
    rng = np.random.default_rng(SEED)
    lines = ["# Repair label cards (ELLIPSE high-surprisal tokens)", "",
             "Random draws per label. Arrow: the winning edit's word (empty = deletion) and its net gain.", ""]
    for lab_, n in d.label.value_counts().items():
        g = d[d.label == lab_]
        lines += [f"## {lab_}  (n = {n:,}, {n / len(d):.1%} of high tokens)", "", "```"]
        for r in g.iloc[rng.choice(len(g), min(15, len(g)), replace=False)].itertuples():
            t, i = texts[r.text_id], int(r.target_char)
            left = t[max(0, i - 55):i].replace("\n", " ")
            right = t[i + len(r.observed):i + len(r.observed) + 40].replace("\n", " ")
            lines.append(f"{left:>55} [[{r.observed}]] {right}   -> {r.best_word!r} ({r.net_gain:+.1f})")
        lines += ["```", ""]
    (OUT / "repair_cards.md").write_text("\n".join(lines))


def cmd_l1(args):
    d = high_tokens("toefl11")
    summ = pd.read_parquet(S3 / "essay_summary.parquet").query("corpus == 'toefl11'").set_index("text_id")
    essays = pd.read_parquet(S3 / "toefl11" / "essays.parquet").drop(columns="text").set_index("text_id")
    r = rates(d, summ.n_scored)
    e = essays.join(r)
    cols = list(r.columns)
    e.groupby(["l1", "level"])[cols].mean().reset_index().to_csv(OUT / "profiles_l1_rates.csv", index=False)
    rows = [{"label": c, "l1": g, "d": v}
            for c in cols for g, v in weighted_d(e[c].to_numpy(), e.l1.to_numpy(), e.level.to_numpy()).items()]
    pd.DataFrame(rows).to_csv(OUT / "profiles_l1_d.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["proficiency", "cards", "l1"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"proficiency": cmd_proficiency, "cards": cmd_cards, "l1": cmd_l1}[args.cmd](args)


if __name__ == "__main__":
    main()
