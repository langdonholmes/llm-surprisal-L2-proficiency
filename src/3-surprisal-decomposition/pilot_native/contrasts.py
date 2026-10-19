"""Pilot (native calibration), contrasts, sensitivity checks and the figure.

    python -I contrasts.py DOCS.parquet OUT_DIR

Reads OUT_DIR/doc_means.csv (analyze.py) and, if present, the infini-gram
spot-check file next to DOCS.parquet (contam_external.csv: public sources only).

Writes OUT_DIR/contrasts.csv, OUT_DIR/prompt_matched.csv, OUT_DIR/contam_split.csv,
OUT_DIR/group_means.png. All CIs: 2,000 bootstrap resamples of documents within
group, seed 42. Length-matched subset ("len") throughout.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from util.paths import DATA_DIR  # noqa: E402

SEED, N_BOOT = 42, 2000
PAIRS = [
    ("persuade_native", "W&I C"), ("persuade_native", "ELLIPSE >=4"),
    ("persuade_native_hi", "ELLIPSE >=4"), ("persuade_native", "persuade_ell"),
    ("LOCNESS (N)", "W&I C"), ("LOCNESS (N)", "persuade_native"),
    ("wiki_new", "wiki_old"), ("wiki_new", "persuade_native"),
]


def boot_diff(a, b, rng):
    ba = a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1)
    bb = b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(1)
    d = ba - bb
    return a.mean() - b.mean(), np.quantile(d, .025), np.quantile(d, .975)


def main():
    docs_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    per = pd.read_csv(out / "doc_means.csv")
    per = per[per.subset == "len"]
    rng = np.random.default_rng(SEED)

    rows = []
    for g1, g2 in PAIRS:
        a, b = per[per.group == g1], per[per.group == g2]
        for c in ["mean_s", "hi_share"]:
            d, lo, hi = boot_diff(a[c].to_numpy(), b[c].to_numpy(), rng)
            rows.append({"contrast": f"{g1} - {g2}", "stat": c, "diff": d, "lo": lo, "hi": hi})
    # sensitivity: wiki_old without the docs carrying unexpanded "Template:" text
    docs = pd.read_parquet(docs_path)
    tmpl = set(docs.loc[docs.text.str.contains("Template:", regex=False), "text_id"])
    a = per[per.group == "wiki_new"]
    b = per[(per.group == "wiki_old") & ~per.text_id.isin(tmpl)]
    for c in ["mean_s", "hi_share"]:
        d, lo, hi = boot_diff(a[c].to_numpy(), b[c].to_numpy(), rng)
        rows.append({"contrast": "wiki_new - wiki_old (no Template: docs)", "stat": c,
                     "diff": d, "lo": lo, "hi": hi})
    con = pd.DataFrame(rows)
    con.to_csv(out / "contrasts.csv", index=False)
    print(con.round(3).to_string(index=False))

    # prompt-matched: PERSUADE groups vs ELLIPSE bands on the prompts both share
    ell = pd.read_parquet(DATA_DIR / "study3" / "ellipse" / "essays.parquet", columns=["text_id", "prompt"])
    pm = per.merge(docs[["group", "text_id", "prompt"]], on=["group", "text_id"], how="left")
    pm["prompt"] = pm.prompt.fillna(pm.text_id.map(ell.set_index("text_id").prompt))
    shared = set(pm[pm.group == "persuade_native"].prompt) & set(pm[pm.group.str.startswith("ELLIPSE")].prompt)
    pm = pm[pm.prompt.isin(shared)]
    prow = []
    for g in ["ELLIPSE <=2.5", "ELLIPSE 3-3.5", "ELLIPSE >=4", "persuade_ell", "persuade_native",
              "persuade_native_hi"]:
        d = pm[pm.group == g]
        by = d.groupby("prompt").mean_s.mean()
        prow.append({"group": g, "n": len(d), "n_prompts": int(by.size), "mean_s": d.mean_s.mean(),
                     "mean_of_prompt_means": by.mean()})
    # within-prompt difference persuade_native - ELLIPSE >=4, weighted by the smaller n
    a, b = pm[pm.group == "persuade_native"], pm[pm.group == "ELLIPSE >=4"]
    diffs, w = [], []
    for p in shared:
        x, y = a[a.prompt == p].mean_s, b[b.prompt == p].mean_s
        if len(x) >= 5 and len(y) >= 5:
            diffs.append(x.mean() - y.mean())
            w.append(min(len(x), len(y)))
    prow.append({"group": "within-prompt diff persuade_native - ELLIPSE >=4", "n": int(sum(w)),
                 "n_prompts": len(diffs), "mean_s": float(np.average(diffs, weights=w)),
                 "mean_of_prompt_means": float(np.mean(diffs))})
    pmt = pd.DataFrame(prow)
    pmt.to_csv(out / "prompt_matched.csv", index=False)
    print("shared prompts:", sorted(shared))
    print(pmt.round(3).to_string(index=False))

    # contamination split: documents with any 15-word span found in OLMo-mix-1124
    srows = []
    for f in ["contam_external.csv"]:
        p = docs_path.parent / f
        if not p.exists():
            continue
        c = pd.read_csv(p)
        hit = c.groupby(["group", "text_id"])["count"].max().gt(0).rename("hit").reset_index()
        m = hit.merge(per[["group", "text_id", "mean_s", "hi_share"]], on=["group", "text_id"])
        for g, d in m.groupby("group", sort=False):
            srows.append({"group": g, "n_checked": len(hit[hit.group == g]),
                          "share_docs_hit": hit[hit.group == g].hit.mean(),
                          "span_hit_rate": (c[c.group == g]["count"] > 0).mean(),
                          "n_len_hit": int(d.hit.sum()), "mean_s_hit": d[d.hit].mean_s.mean(),
                          "mean_s_clean": d[~d.hit].mean_s.mean()})
    if srows:
        sp = pd.DataFrame(srows)
        sp.to_csv(out / "contam_split.csv", index=False)
        print(sp.round(3).to_string(index=False))

    # figure: group mean with 95% CI, documents as faint dots
    tab = pd.read_csv(out / "group_table.csv")
    tab = tab[tab.subset == "len"].reset_index(drop=True)
    fam = {"W&I": "#2a78d6", "LOC": "#2a78d6", "ELL": "#eb6834", "per": "#1baf7a", "wik": "#eda100"}
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    jr = np.random.default_rng(SEED)
    for i, r in tab.iterrows():
        col = fam[r.group[:3]]
        x = per[per.group == r.group].mean_s.to_numpy()
        ax.scatter(x, i + jr.uniform(-.18, .18, len(x)), s=3, color=col, alpha=.15, lw=0)
        ax.plot([r.mean_s_lo, r.mean_s_hi], [i, i], color="#333333", lw=2)
        ax.scatter([r.mean_s_mean], [i], s=36, color=col, edgecolor="white", lw=1.2, zorder=3)
        ax.text(8.6, i, f"{r.mean_s_mean:.2f}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(tab.group, tab.n)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(4, 9)
    ax.set_xlabel("Per-document mean surprisal (bits), OLMo-2 1B, 8-subword window", fontsize=8)
    ax.grid(axis="x", color="#e5e5e5", lw=.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Native vs learner writing (length-matched, 200-600 words); mean, 95% CI",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "group_means.png")


if __name__ == "__main__":
    main()
