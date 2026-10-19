"""Native-vs-C pilot, step 2 (CPU) — the pre-declared N − C contrasts (NOTES.md).

Writes data/study3/pilot_native_c/tables/{contrasts.csv, levels.csv, exclusion_shares.csv}.

    python src/3-surprisal-decomposition/pilot_native_c/2_contrasts.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from util.paths import DATA_DIR, RESULTS_DIR

OUT = DATA_DIR / "study3" / "pilot_native_c"
TAB = OUT / "tables"
SEED = 42
N_BOOT = 2000
MIN_KEPT = 20
FIRST_N = 250


def tau() -> float:
    return json.loads((RESULTS_DIR / "study3" / "threshold.json").read_text())["tau_bits"]


def controls(t: pd.DataFrame) -> dict:
    topic = t.topic_essay | t.topic_cluster
    start = t.sent_init | t.para_init
    c5 = t.propn_num | t.earlier | topic | start
    return {"C0": pd.Series(False, index=t.index), "C1": t.propn_num, "C2": t.earlier, "C3": topic,
            "C4": start, "C5": c5}


def per_essay(t: pd.DataFrame, excl: pd.Series, T: float) -> pd.DataFrame:
    d = t.assign(kept=~excl, high=t.surprisal_bits >= T)
    g = d.groupby("text_id")
    n_all = g.size()
    k = d[d.kept].groupby("text_id")
    E = pd.DataFrame({"n_all": n_all, "n_kept": k.size(), "n_high_kept": k.high.sum(),
                      "s_kept": k.surprisal_bits.sum()}).fillna(0)
    E["rate"] = E.n_high_kept / E.n_kept * 1000
    E["rate_num"] = E.n_high_kept / E.n_all * 1000
    E["mean_s"] = E.s_kept / E.n_kept
    return E[E.n_kept >= MIN_KEPT]


def boot_diff(h: np.ndarray, l: np.ndarray, rng) -> tuple:
    bh = h[rng.integers(0, len(h), (N_BOOT, len(h)))].mean(1)
    bl = l[rng.integers(0, len(l), (N_BOOT, len(l)))].mean(1)
    b = bh - bl
    return h.mean() - l.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    TAB.mkdir(exist_ok=True)
    T = tau()
    tok = pd.read_parquet(OUT / "tokens.parquet")
    ess = pd.read_parquet(OUT / "essays.parquet").set_index("text_id")
    level = ess.level
    subsets = {"all": None, "C6a_genre": ess.index[ess.c6_genre],
               "C6b_topic": ess.index[ess.c6_topic], "C6c_register": ess.index[ess.c6_register],
               "first250": "first250"}
    rows, lv_rows, sh_rows = [], [], []
    for tset in ["own", "all"]:
        t0 = tok[tok.own] if tset == "own" else tok
        for sub, ids in subsets.items():
            if ids is None:
                t = t0
            elif sub == "first250":
                t = t0[t0.pos_in_essay < FIRST_N]
            else:  # matched subsets: the ids already hold the N essays that qualify
                t = t0[t0.text_id.isin(ids)]
            ctl = controls(t)
            if sub != "all":
                ctl = {k: v for k, v in ctl.items() if k in ("C0", "C5")}
            for cname, excl in ctl.items():
                E = per_essay(t, excl, T)
                E["level"] = E.index.map(level)
                if sub == "all":
                    for lv, d in E.groupby("level"):
                        lv_rows.append({"set": tset, "control": cname, "level": lv, "n_essays": len(d),
                                        **{c: d[c].mean() for c in ["rate", "rate_num", "mean_s"]},
                                        "excluded_share": 1 - d.n_kept.sum() / d.n_all.sum()})
                N, C = E[E.level == "N"], E[E.level == "C"]
                for out in ["rate", "rate_num", "mean_s"]:
                    rng = np.random.default_rng(SEED)
                    p, lo, hi = boot_diff(N[out].to_numpy(), C[out].to_numpy(), rng)
                    rows.append({"set": tset, "subset": sub, "control": cname, "outcome": out,
                                 "n_N": len(N), "n_C": len(C), "N": N[out].mean(), "C": C[out].mean(),
                                 "diff": p, "lo": lo, "hi": hi})
    R = pd.DataFrame(rows)
    base = R[(R.subset == "all") & (R.control == "C0")].set_index(["set", "outcome"])["diff"]
    R["retained"] = [r["diff"] / base[(r["set"], r["outcome"])] if r["outcome"] != "mean_s" else np.nan
                     for _, r in R.iterrows()]
    R.to_csv(TAB / "contrasts.csv", index=False)
    L = pd.DataFrame(lv_rows)
    L.to_csv(TAB / "levels.csv", index=False)
    pd.set_option("display.width", 200)
    for tset in ["own", "all"]:
        for out in ["rate", "rate_num", "mean_s"]:
            print(f"\n== {tset} / {out}")
            print(R[(R.set == tset) & (R.outcome == out)].drop(columns=["set", "outcome"]).round(3).to_string(index=False))
    print(L.pivot_table(index=["set", "control"], columns="level", values=["rate", "mean_s", "excluded_share"]).round(3).to_string())

    # reading rule
    o = R[(R.set == "own") & (R.outcome == "rate")].set_index(["subset", "control"])
    c5 = o.loc[("all", "C5")]
    c6 = [o.loc[(s, "C5")] for s in ["C6a_genre", "C6b_topic", "C6c_register"]]
    n6 = sum(r.lo > 0 for r in c6)
    if c5.lo > 0 and n6 >= 2:
        verdict = "attenuates" if c5["diff"] < 12.5 else "survives"
    else:
        verdict = "explained"
    print(f"\nC5 own rate N-C {c5['diff']:.1f} [{c5.lo:.1f}, {c5.hi:.1f}]; C6xC5 CIs above 0: {n6}/3 -> {verdict}")
    (TAB / "verdict.json").write_text(json.dumps({"c5": [c5["diff"], c5.lo, c5.hi], "n_c6_above0": int(n6),
                                                  "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
