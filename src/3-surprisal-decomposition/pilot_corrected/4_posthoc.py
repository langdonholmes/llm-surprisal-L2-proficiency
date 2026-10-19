"""Pilot, step 4 (CPU, post hoc) — why the primary contrast and the matched
contrast disagree, and KWIC samples for the qualitative reading.

Decomposes the change from the contaminated uncorrected rate (`uncorr_rate`,
per all original tokens) to the post-correction own rate (`corr_own`, per
aligned own tokens) into three steps, each a per-essay rate per 1,000:

    u_all     uncorrected (gold OK) tokens above tau / all scored original tokens   (= uncorr_rate)
    u_own     the same numerator / aligned own tokens                               (denominator step)
    u_own_c   uncorrected own tokens above tau in the CORRECTED essay / aligned own (context step)
    corr_own  + own tokens that were ERR_INS or POST in the original, scored in the corrected essay

Writes tables/posthoc_decomposition.csv, tables/posthoc_tests.json and
kwic_*.txt (data dir only; W&I licence: excerpts under 100 words, not published).

    python src/3-surprisal-decomposition/pilot_corrected/4_posthoc.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import LEVELS, OUT, SEED, tau  # noqa: E402

TAB = OUT / "tables"
N_BOOT = 2000


def diff(e, col, hi, lo, ref=None):
    rng = np.random.default_rng(SEED)
    H, L = e[e.level == hi], e[e.level == lo]
    ih = rng.integers(0, len(H), (N_BOOT, len(H)))
    il = rng.integers(0, len(L), (N_BOOT, len(L)))
    hv, lv = H[col].to_numpy(), L[col].to_numpy()
    pt, bs = hv.mean() - lv.mean(), hv[ih].mean(1) - lv[il].mean(1)
    if ref is not None:
        hr, lr = H[ref].to_numpy(), L[ref].to_numpy()
        pt -= hr.mean() - lr.mean()
        bs -= hr[ih].mean(1) - lr[il].mean(1)
    return {"diff": float(1000 * pt), "ci": [float(1000 * x) for x in np.percentile(bs, [2.5, 97.5])]}


def kwic(text, a, b, width=60):
    return (text[max(0, a - width):a] + "[[" + text[a:b] + "]]" + text[b:b + width // 2]).replace("\n", " / ")


def main():
    T = tau()
    a = pd.read_parquet(OUT / "aligned.parquet")
    E = pd.read_csv(TAB / "essays.csv").set_index("text_id")
    al = a[a.aligned]
    unc = al.gold == "OK"
    E["u_own"] = al[unc & al.high_o].groupby("text_id").size() / E.n_own
    E["u_own_c"] = al[unc & al.high_c].groupby("text_id").size() / E.n_own
    E["adj_own_c"] = al[~unc & al.high_c].groupby("text_id").size() / E.n_own
    E["adj_own_o"] = al[~unc & al.high_o].groupby("text_id").size() / E.n_own
    E["share_adj_tokens"] = al[~unc].groupby("text_id").size() / E.n_own
    E = E.fillna({c: 0 for c in ["u_own", "u_own_c", "adj_own_c", "adj_own_o", "share_adj_tokens"]})
    steps = ["uncorr_rate", "u_own", "u_own_c", "corr_own", "adj_own_o", "adj_own_c"]
    tab = (1000 * E.groupby("level")[steps].mean()).reindex(LEVELS)
    tab["share_adj_tokens"] = E.groupby("level").share_adj_tokens.mean().reindex(LEVELS)
    tab.to_csv(TAB / "posthoc_decomposition.csv")
    print(tab.round(2).to_string())
    res = {"C_minus_A": {s: diff(E, s, "C", "A") for s in steps},
           "N_minus_C": {s: diff(E, s, "N", "C") for s in steps},
           "step_denominator": diff(E, "u_own", "C", "A", "uncorr_rate"),
           "step_context": diff(E, "u_own_c", "C", "A", "u_own"),
           "step_adjacent_tokens": diff(E, "corr_own", "C", "A", "u_own_c")}
    (TAB / "posthoc_tests.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("step")}, indent=1))
    print(json.dumps(res["C_minus_A"], indent=0))

    # ---- KWIC (local only) --------------------------------------------------
    rng = np.random.default_rng(SEED)
    essays = pd.read_parquet(OUT / "essays.parquet", columns=["text_id", "text", "corrected"]).set_index("text_id")
    q4 = pd.read_parquet(OUT / "q4_tokens.parquet")
    with open(OUT / "kwic_uncorrected.txt", "w") as f:
        for cls in ["intrinsic", "contaminated"]:
            for lvl in LEVELS:
                d = q4[(q4.cls == cls) & (q4.level == lvl)]
                for r in d.iloc[rng.choice(len(d), min(25, len(d)), replace=False)].itertuples():
                    ess = essays.loc[r.text_id]
                    f.write(f"{cls}\t{lvl}\t{r.text}\t{r.surprisal_bits:.1f}->{r.s_corr:.1f}\t{r.pos}\t{r.ttype}\n"
                            f"   O: {kwic(ess.text, r.idx, r.idx + len(r.text))}\n"
                            f"   C: {kwic(ess.corrected, int(r.c_idx), int(r.c_idx) + len(r.text))}\n")
    q3 = pd.read_parquet(OUT / "q3_edits.parquet")
    ed = pd.read_parquet(OUT / "edits.parquet")
    q3 = pd.concat([q3.reset_index(drop=True), ed[["cs", "ce", "s", "e", "repl"]].reset_index(drop=True)], axis=1) \
        if len(q3) == len(ed) else q3
    with open(OUT / "kwic_corrections.txt", "w") as f:
        for kind in ["sub", "del", "ins"]:
            d = q3[(q3.kind == kind) & q3.corr_high]
            for r in d.iloc[rng.choice(len(d), min(40, len(d)), replace=False)].itertuples():
                ess = essays.loc[r.text_id]
                f.write(f"{kind}\t{r.level}\t{r.orig_text!s}->{r.corr_text!s}\t{r.s_orig_first:.1f}->{r.s_corr_first:.1f}\n"
                        f"   O: {kwic(ess.text, r.s, r.e)}\n   C: {kwic(ess.corrected, r.cs, r.ce)}\n")


if __name__ == "__main__":
    main()
