"""Bucket pilot, step 4 (CPU) — the summary table per corpus.

Rows: surprisal (step 1), every classic measure (step 3), and the reference
rows for edit counts. Columns: share of cov(original measure, level) carried by
the corrected essay (surprisal: the clean part), each bucket, flagged + residual
(surprisal only; for the classic measures flagged spans stay in every version,
so they sit in the corrected share), and the "phraseological" totals with and
without function words (content-word choice + multi-word / order [+ function
word]). The totals are bootstrapped as sums, so their CIs are exact.

Writes data/study3/pilot_buckets/tables/summary_<corpus>.csv and summary.md.

    python src/3-surprisal-decomposition/pilot_buckets/4_summary.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bucket_common import BUCKETS, OUT, PLAYERS, TAB, UNCL, cov_shares, fmt  # noqa: E402

MEAS = [("surprisal", "mean surprisal"), ("bigram_mi", "bigram MI"), ("bigram_dp", "bigram ΔP"),
        ("bigram_t", "bigram t"), ("share_mi3", "share MI ≥ 3"), ("amod_mi", "amod MI"),
        ("dobj_mi", "dobj MI"), ("advmod_mi", "advmod MI"), ("dep_mi", "dependency MI (mean of 3)"),
        ("content_freq", "content-word log freq"), ("log_freq_s2", "Study 2 log freq (lemma)")]
COLS = ["corrected", *BUCKETS, UNCL, "flagged+residual", "phraseo. excl. FW", "phraseo. incl. FW"]
PH = ["content-word choice", "multi-word / order"]


def frame(X, y):
    """X has columns orig, corrected, buckets..., flagged+residual; returns share dict incl. totals."""
    X = X.copy()
    X["phraseo. excl. FW"] = X[PH].sum(axis=1)
    X["phraseo. incl. FW"] = X[PH + ["function word"]].sum(axis=1)
    cols = list(X.columns)
    return cov_shares(X.to_numpy(float), y, cols)


def learners(U, corpus):
    if corpus == "wi":
        U = U[U.level != "N"]
        return U, U.level.map({"A": 1, "B": 2, "C": 3}).to_numpy(float)
    return U, U.level.to_numpy(float)


def run(corpus):
    rows, out = [], {}
    S = pd.read_parquet(OUT / f"surprisal_units_{corpus}.parquet")
    S, ys = learners(S, corpus)
    X = pd.DataFrame({"orig": S.total, "corrected": S.clean})
    for b in PLAYERS:
        X[b] = S[f"{b}|total"]
    X["flagged+residual"] = S["flagged|total"] + S["residual|total"]
    out["surprisal"] = frame(X, ys)
    # at-site / spill detail for surprisal
    det = {}
    for comp in ["at_site", "spill"]:
        Y = pd.DataFrame({"orig": S.total})
        for b in PLAYERS:
            Y[b] = S[f"{b}|{comp}"]
        Y["phraseo. excl. FW"] = Y[PH].sum(axis=1)
        Y["phraseo. incl. FW"] = Y[PH + ["function word"]].sum(axis=1)
        det[comp] = cov_shares(Y.to_numpy(float), ys, list(Y.columns))
    Z = pd.read_parquet(OUT / f"shapley_{corpus}.parquet")
    Z = Z[np.isfinite(Z.level)]
    info = json.loads((TAB / f"shapley_{corpus}.json").read_text())
    for m, _ in MEAS[1:]:
        X = pd.DataFrame({"orig": Z[f"{m}|orig"], "corrected": Z[f"{m}|corr"]})
        for b in PLAYERS:
            X[b] = Z[f"{m}|{b}"]
        X["flagged+residual"] = 0.0
        ok = X.notna().all(axis=1)
        out[m] = frame(X[ok], Z.level[ok].to_numpy(float))
    # reference: edit counts (share of all edits; share of cov(edit rate, level))
    R = S[[f"edits_{b}" for b in PLAYERS]].div(S.n, axis=0)
    R.columns = PLAYERS
    R.insert(0, "orig", R.sum(axis=1))
    R.insert(1, "corrected", 0.0)
    R["flagged+residual"] = 0.0
    out["edit_rate"] = frame(R, ys)
    tot = S[[f"edits_{b}" for b in PLAYERS]].sum()
    raw = {b: tot[f"edits_{b}"] / tot.sum() for b in PLAYERS}
    raw["phraseo. excl. FW"] = sum(raw[b] for b in PH)
    raw["phraseo. incl. FW"] = raw["phraseo. excl. FW"] + raw["function word"]

    for m, label in MEAS:
        rho = ("" if m == "surprisal" else
               f"{info['shares'][m]['rho_orig']:+.2f} / {info['shares'][m]['rho_corr']:+.2f}")
        r = {"measure": label, "rho orig / corr": rho}
        for c in COLS:
            r[c] = "–" if (c == "flagged+residual" and m != "surprisal") else fmt(out[m][c])
        rows.append(r)
    r = {"measure": "edit rate (cov share)", "rho orig / corr": ""}
    for c in COLS:
        r[c] = "–" if c in ("corrected", "flagged+residual") else fmt(out["edit_rate"][c])
    rows.append(r)
    r = {"measure": "edits (% of all edits)", "rho orig / corr": ""}
    for c in COLS:
        r[c] = f"{100 * raw[c]:.0f}" if c in raw else "–"
    rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / f"summary_{corpus}.csv", index=False)
    d = pd.DataFrame([{"component": comp, **{c: fmt(det[comp][c]) for c in PLAYERS + ["phraseo. excl. FW",
                                                                                         "phraseo. incl. FW"]}}
                      for comp in det])
    d.to_csv(TAB / f"surprisal_at_site_spill_{corpus}.csv", index=False)
    return t, d


def main():
    md = []
    for corpus, title in [("wi", "W&I learners vs level (A/B/C = 1/2/3)"), ("fce", "FCE scripts vs script score")]:
        t, d = run(corpus)
        md += [f"### {title}", "", t.to_markdown(index=False), "", "Surprisal, at site and spill:", "",
               d.to_markdown(index=False), ""]
    (TAB / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
