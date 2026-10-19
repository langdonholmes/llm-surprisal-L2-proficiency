"""Post hoc sensitivity checks for the Q2 results (written after seeing them).

1. Layer choice for TL_late: s_TL[l] - s_TL[16] for l = 6, 7, 9, 10, 12 and the mean of 6-10
   (layer 8 has a bump in held-out tuned-lens KL).
2. Clean context: OK tokens whose previous 7 non-space tokens are all gold OK
   (so the late-layer context is not error-laden).
3. Without proper nouns / numbers (PROPN share differs by level).
Same residualised C - A and N - C contrasts as analyze.q2.

    .venv/bin/python src/3-surprisal-decomposition/pilot_lens/sensitivity.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from analyze import BASE, essay_boot_means, spline_base
from lens_common import OUT, S3

PREV = 7


def clean_context(f):
    g = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet", columns=["row_id", "text_id", "gold", "is_space"])
    g = g[~g.is_space].reset_index(drop=True)
    bad = (g.gold != "OK").astype(int)
    same = g.text_id
    # number of non-OK tokens among the previous PREV tokens of the same essay
    roll = bad.groupby(same).transform(lambda s: s.shift(1).rolling(PREV, min_periods=1).sum()).fillna(0)
    return f.row_id.map(dict(zip(g.row_id, roll))).fillna(0).to_numpy() == 0


def contrasts(d, m):
    d = d[d[m].notna()].copy()
    X = spline_base().fit_transform(d[BASE])
    d["r"] = d[m] - LinearRegression().fit(X, d[m]).predict(X)
    sd = d[m].std()
    out = {"n": len(d)}
    bs = {}
    for lv in ["A", "C", "N"]:
        mu, b = essay_boot_means(d[d.level == lv], "r")
        out[lv] = mu
        bs[lv] = b
    for a, b in [("C", "A"), ("N", "C")]:
        diff = bs[a] - bs[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out[f"{a}-{b}"] = f"{out[a] - out[b]:+.2f} [{lo:+.2f}, {hi:+.2f}] ({(out[a] - out[b]) / sd:+.2f} SD)"
    return out


def main():
    pd.set_option("display.width", 250)
    f = pd.read_parquet(OUT / "features.parquet")
    for l in [6, 7, 9, 10, 12]:
        f[f"TL_late_{l}"] = f[f"tl{l}"] - f.tl16
    f["TL_late_6to10"] = f[[f"tl{l}" for l in range(6, 11)]].mean(1) - f.tl16
    f["clean"] = clean_context(f)
    ok = f[f.gold == "OK"]
    print("clean share by level:", ok.groupby("level").clean.mean().round(3).to_dict())
    print("PROPN/NUM share of OK by level:", ok.groupby("level").pos.apply(
        lambda s: s.isin(["PROPN", "NUM"]).mean()).round(3).to_dict())
    rows = []
    for m in ["TL_late", "TL_late_6", "TL_late_7", "TL_late_9", "TL_late_10", "TL_late_12", "TL_late_6to10",
              "C_gain", "pool_ctx", "s_mlm_full", "MLM_ctx", "C_sent_gain"]:
        for name, d in [("all OK", ok), ("clean context", ok[ok.clean]),
                        ("no PROPN/NUM", ok[~ok.pos.isin(["PROPN", "NUM"])])]:
            rows.append({"measure": m, "subset": name, **contrasts(d, m)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sensitivity.csv", index=False)
    print(t.drop(columns=["A", "C", "N"]).to_string(index=False))


if __name__ == "__main__":
    main()
