"""Study 3, stage 5 (CPU) — exact splits of each unit's mean surprisal by the corrections.

Per unit (W&I essay; FCE script), over its n scored original tokens, with O the
writer's own aligned tokens and clean = mean_O(s_corr):

Four-part split (D24)
    S_orig = clean + omission + context + error_tokens
    omission      sum over own ERR_INS tokens of (s_orig - s_corr) / n
    context       sum over the other own tokens of (s_orig - s_corr) / n
    error_tokens  sum over the other tokens of (s_orig - clean) / n
Also for the high-token rate (s replaced by the indicator s > τ).

Split by kind of correction (D28, D33)
    S_orig = clean + sum_c (at_site_c + spill_c) + flagged + residual
    at_site   a replacement or unnecessary edit: its tokens' excess over clean;
              a missing-word edit: Δ of the own word right after the gap
    spill     Δ of the other own words in the window, split among its edits by
              edit-level Shapley (stage 4), shifted so that each word's parts sum
              to its stored Δ; words with more than 7 edits split Δ equally
    flagged   the excess of tokens the annotator flagged without correcting
    residual  Δ of own words with no edit in the window, and own words that do
              not align
over the four correction buckets (plus unclassified) and, as a supplement, the
nine ERRANT groups. Every split is checked to hold exactly.

Because the parts add up, so do their covariances with the criterion, and each
part's share of cov(S_orig, criterion) splits the proficiency signal exactly
(learners only; 2,000 unit-bootstrap CIs). Also: frequency vs cost per error,
spill by distance, the clean part against the criterion, and the RQ3 features.

Writes results/study3/: split4_<corpus>.csv, split4_means_<corpus>.csv,
shares_<corpus>.csv, means_<corpus>.csv, rate_cost_<corpus>.csv,
spill_decay_<corpus>.csv, clean_assoc.csv, decompose_<corpus>.json; and
data/study3/<corpus>/units.parquet.

    python src/3-surprisal-decomposition/5_decompose.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import CORPORA, GROUP_SLUGS, N_BOOT, RES, SEED, out, tau  # noqa: E402
from lib.errant import BUCKETS, GROUPS, UNCL  # noqa: E402
from lib.stats import boot, cov_shares, partial_rank, spearman_cols  # noqa: E402

PARTS4 = ["clean", "omission", "context", "error_tokens"]
COMP = ["at_site", "spill"]
CATS = {"bucket": BUCKETS + [UNCL, "flagged", "residual"], "group": GROUPS + ["flagged", "residual"]}


def load(corpus):
    O = out(corpus)
    a = pd.read_parquet(O / "aligned.parquet")
    a = a[a.y.notna() | (a.group == "N")].copy()  # learners with a criterion, and LOCNESS
    a["o"] = a.own & a.aligned
    a = a.merge(pd.read_parquet(O / "spill.parquet"), on=["text_id", "spacy_idx"], how="left")
    ed = pd.read_parquet(O / "edits.parquet")
    return a, ed


# --------------------------------------------------------------------------- #
# four-part split
# --------------------------------------------------------------------------- #
def split4(a, so, sc):
    d = np.where(a.o, a[so] - a[sc], 0.0)
    a = a.assign(d=d)
    g = a.groupby("unit")
    U = pd.DataFrame({"group": g.group.first(), "y": g.y.first(), "n": g.size(), "total": g[so].mean()})
    U["clean"] = a[a.o].groupby("unit")[sc].mean()
    U["omission"] = a[a.o & (a.gold == "ERR_INS")].groupby("unit").d.sum()
    U["context"] = a[a.o & (a.gold != "ERR_INS")].groupby("unit").d.sum()
    x = a[~a.o].join(U.clean, on="unit")
    U["error_tokens"] = (x[so] - x.clean).groupby(x.unit).sum()
    U = U.fillna({p: 0.0 for p in PARTS4[1:]})
    U[PARTS4[1:]] = U[PARTS4[1:]].div(U.n, axis=0)
    assert np.allclose(U[PARTS4].sum(axis=1), U.total, atol=1e-9)
    return U


def learners(U):
    L = U[U.y.notna()]
    return L, L.y.to_numpy(float)


def run_split4(a, corpus):
    t = tau()
    a = a.assign(high_o=(a.surprisal_bits > t).astype(float), high_c=(a.s_corr > t).astype(float))
    rows, means, res = [], [], {}
    for name, so, sc, k in [("mean_surprisal", "surprisal_bits", "s_corr", 1.0),
                            ("high_per_1000", "high_o", "high_c", 1000.0)]:
        U = split4(a, so, sc)
        U[["total"] + PARTS4] *= k
        L, y = learners(U)
        X = L[["total"] + PARTS4].to_numpy(float)
        sh = cov_shares(X, y)
        rho = boot(lambda i: spearman_cols(X[i], y[i]), len(y))
        for j, p in enumerate(["total"] + PARTS4):
            rows.append({"measure": name, "part": p, "share": sh[0][j], "lo": sh[1][j], "hi": sh[2][j],
                         "rho": rho[0][j], "rho_lo": rho[1][j], "rho_hi": rho[2][j]})
        m = U.groupby("group")[["total"] + PARTS4].mean()
        m["n_units"] = U.groupby("group").size()
        m.loc["learners"] = list(L[["total"] + PARTS4].mean()) + [len(L)]
        means.append(m.reset_index().assign(measure=name))
        if corpus == "wi":  # LOCNESS minus C
            N, C = U[U.group == "N"], U[U.group == "C"]
            rng = np.random.default_rng(SEED)
            iN = rng.integers(0, len(N), (N_BOOT, len(N)))
            iC = rng.integers(0, len(C), (N_BOOT, len(C)))
            res[name] = {}
            for p in ["total"] + PARTS4:
                n_, c_ = N[p].to_numpy(), C[p].to_numpy()
                bs = n_[iN].mean(axis=1) - c_[iC].mean(axis=1)
                res[name][p] = [float(n_.mean() - c_.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
        if name == "mean_surprisal":
            mean_units = U
    pd.DataFrame(rows).to_csv(RES / f"split4_{corpus}.csv", index=False)
    pd.concat(means).to_csv(RES / f"split4_means_{corpus}.csv", index=False)
    return mean_units, res


# --------------------------------------------------------------------------- #
# split by kind of correction
# --------------------------------------------------------------------------- #
def credits(a, ed):
    """Long table: unit, text_id, k (edit; -1 residual, -2 flagged), comp, val, cnt.

    Error-token rows carry val = s_orig and cnt = 1, so that the unit's clean
    mean is subtracted once per token when the table is summed.
    """
    rows = []
    E = a[~a.o]
    k = np.where(E.site_k >= 0, E.site_k, np.where(E.own, -1, -2))
    rows += list(zip(E.unit, E.text_id, k, ["at_site"] * len(E), E.surprisal_bits, [1.0] * len(E)))
    O = a[a.o]
    for r in O.itertuples():
        d = r.surprisal_bits - r.s_corr
        if len(r.gap_k):
            rows += [(r.unit, r.text_id, g, "at_site", d / len(r.gap_k), 0.0) for g in r.gap_k]
        elif r.status == "ok":
            adj = (d - (r.v_none - r.v_all)) / len(r.psi)
            rows += [(r.unit, r.text_id, e, "spill", p + adj, 0.0) for e, p in zip(r.edit_k, r.psi)]
        elif len(r.edit_k):
            rows += [(r.unit, r.text_id, e, "spill", d / len(r.edit_k), 0.0) for e in r.edit_k]
        else:
            rows.append((r.unit, r.text_id, -1, "spill", d, 0.0))
    cr = pd.DataFrame(rows, columns=["unit", "text_id", "k", "comp", "val", "cnt"])
    cr = cr.merge(ed[["text_id", "k", "group", "bucket"]], on=["text_id", "k"], how="left")
    for k_, lab in [(-1, "residual"), (-2, "flagged")]:
        cr.loc[cr.k == k_, ["group", "bucket"]] = lab
    assert cr.group.notna().all()
    return cr


def unit_table(a, cr):
    g = a.groupby("unit")
    U = pd.DataFrame({"group": g.group.first(), "y": g.y.first(), "n": g.size(), "total": g.surprisal_bits.mean()})
    U["clean"] = a[a.o].groupby("unit").s_corr.mean()
    c = cr.join(U.clean, on="unit")
    c["x"] = c.val - c.cnt * c.clean
    for by, cats in CATS.items():
        w = c.groupby(["unit", by, "comp"]).x.sum().unstack([1, 2])
        for cat in cats:
            for comp in COMP:
                U[f"{cat}|{comp}"] = w.get((cat, comp), pd.Series(0.0, index=U.index)).reindex(U.index).fillna(0.0)
                U[f"{cat}|{comp}"] /= U.n
            U[f"{cat}|total"] = U[f"{cat}|at_site"] + U[f"{cat}|spill"]
        gap = (U.clean + U[[f"{x}|total" for x in cats]].sum(axis=1) - U.total).abs().max()
        assert gap < 1e-9, f"{by} split fails by {gap}"
    return U


def edit_counts(U, ed, units):
    ed = ed.assign(unit=ed.text_id.map(units))
    for by, cats in CATS.items():
        c = ed[ed.kind != "flagged"].groupby(["unit", by]).size().unstack()
        for cat in cats[:-2]:
            U[f"n_{cat}"] = c.get(cat, pd.Series(dtype=float)).reindex(U.index).fillna(0.0)
    U["n_flagged"] = ed[ed.kind == "flagged"].groupby("unit").size().reindex(U.index).fillna(0.0)
    U["n_edits"] = U[[f"n_{b}" for b in BUCKETS + [UNCL]]].sum(axis=1)
    return U


def shares(U):
    L, y = learners(U)
    rows = []
    for by, cats in CATS.items():
        cols = ["total", "clean"] + [f"{c}|{p}" for c in cats for p in COMP + ["total"]]
        sh = cov_shares(L[cols].to_numpy(float), y)
        rows += [{"cut": by, "col": c, "share": sh[0][j], "lo": sh[1][j], "hi": sh[2][j]}
                 for j, c in enumerate(cols) if c != "total"]
    # reference: each bucket's share of cov(edit rate, criterion), and of all edits
    R = L[[f"n_{b}" for b in BUCKETS + [UNCL]]].div(L.n, axis=0)
    sh = cov_shares(np.column_stack([R.sum(axis=1), R]), y)
    tot = L[[f"n_{b}" for b in BUCKETS + [UNCL]]].sum()
    rows += [{"cut": "edit_rate", "col": b, "share": sh[0][j + 1], "lo": sh[1][j + 1], "hi": sh[2][j + 1],
              "share_of_edits": tot.iloc[j] / tot.sum()} for j, b in enumerate(BUCKETS + [UNCL])]
    return pd.DataFrame(rows)


def rate_cost(U, corpus):
    """Errors per 100 tokens and bits per error, token-pooled within level.

    pooled part_L = sum(part * n) / sum(n) = rate_L * cost_L exactly. The
    lowest-minus-highest difference splits into rate and cost by the midpoint
    rule d(rc) = dr * mean(c) + mean(r) * dc. CIs by resampling units within level.
    rate_all and cost_all pool all learners.
    """
    L, _ = learners(U)
    lv = ["A", "B", "C"] if corpus == "wi" else [f"{a}-{b}" for a, b in [(1, 23), (24, 27), (28, 31), (32, 40)]]
    lo_, hi_ = lv[0], lv[-1]
    idx = {g: np.flatnonzero((L.group == g).to_numpy()) for g in lv}
    n = L.n.to_numpy(float)
    rng = np.random.default_rng(SEED)
    boots = [{g: rng.choice(idx[g], len(idx[g])) for g in lv} for _ in range(N_BOOT)]
    rows = []
    for by, cats in CATS.items():
        for cat in cats[:-2]:
            ne = L[f"n_{cat}"].to_numpy(float)
            for comp in COMP + ["total"]:
                x = L[f"{cat}|{comp}"].to_numpy(float) * n

                def stats(sel):
                    r = np.array([ne[sel[g]].sum() / n[sel[g]].sum() for g in lv])
                    c = np.array([x[sel[g]].sum() / max(ne[sel[g]].sum(), 1e-12) for g in lv])
                    d = r[0] * c[0] - r[-1] * c[-1]
                    return r, c, (r[0] - r[-1]) * (c[0] + c[-1]) / 2 / d, c[-1] - c[0]

                r, c, rs, dc = stats(idx)
                bs = [stats(s) for s in boots]
                row = {"cut": by, "cat": cat, "comp": comp, **{f"rate_{g}": 100 * v for g, v in zip(lv, r)},
                       **{f"cost_{g}": v for g, v in zip(lv, c)}, "rate_share": rs,
                       "cost_change": dc, "rate_all": 100 * ne.sum() / n.sum(), "cost_all": x.sum() / ne.sum()}
                row["rate_share_lo"], row["rate_share_hi"] = np.nanpercentile([b[2] for b in bs], [2.5, 97.5])
                row["cost_change_lo"], row["cost_change_hi"] = np.percentile([b[3] for b in bs], [2.5, 97.5])
                rows.append(row)
    return pd.DataFrame(rows).assign(low=lo_, high=hi_)


def spill_decay(a, ed):
    """Mean spill per (edit, own word) pair by distance in words (5 = 5 or more)."""
    O = a[a.o & (a.status == "ok")]
    O = O[O.gap_k.map(len) == 0]
    adj = (O.surprisal_bits - O.s_corr - (O.v_none - O.v_all)) / O.psi.map(len)
    p = pd.DataFrame({"unit": np.repeat(O.unit.to_numpy(), O.psi.map(len)),
                      "text_id": np.repeat(O.text_id.to_numpy(), O.psi.map(len)),
                      "k": np.concatenate(O.edit_k.to_numpy()),
                      "d": np.clip(np.concatenate(O.dist.to_numpy()), 1, 5),
                      "psi": np.concatenate(O.psi.to_numpy()) + np.repeat(adj.to_numpy(), O.psi.map(len))})
    p = p.merge(ed[["text_id", "k", "bucket"]], on=["text_id", "k"])
    rows = []
    for b in ["all"] + BUCKETS:
        q = p if b == "all" else p[p.bucket == b]
        s = q.groupby(["unit", "d"]).psi.agg(["sum", "size"]).unstack(fill_value=0)
        S, N = s["sum"].to_numpy(float), s["size"].to_numpy(float)
        pt, lo, hi = boot(lambda i: S[i].sum(0) / N[i].sum(0), len(S))
        rows += [{"bucket": b, "d": d, "n_pairs": int(N[:, j].sum()), "mean_spill": pt[j], "lo": lo[j], "hi": hi[j]}
                 for j, d in enumerate(s["sum"].columns)]
    return pd.DataFrame(rows)


def clean_assoc(U, corpus):
    """The clean part against the criterion: rho, and partial rho given the
    plain share of corrected tokens (ERR_TOK or ERR_INS)."""
    L, y = learners(U)
    rows = []
    for m in ["mean_surprisal", "high_per_1000"]:
        x, z = L[f"{m}|clean"].to_numpy(float), L.corr_share.to_numpy(float)
        xt = L[f"{m}|total"].to_numpy(float)

        def st(i):
            return [spearman_cols(xt[i, None], y[i])[0], spearman_cols(x[i, None], y[i])[0],
                    partial_rank(x[i], y[i], z[i])]
        pt, lo, hi = boot(st, len(y))
        for j, s in enumerate(["rho_total", "rho_clean", "partial_clean"]):
            rows.append({"corpus": corpus, "measure": m, "stat": s, "value": pt[j], "lo": lo[j], "hi": hi[j],
                         "n": len(y)})
    return rows


def rq3_features(U):
    """Error counts (B) and the compact surprisal terms (S) per unit."""
    for g, s in zip(GROUPS, GROUP_SLUGS):
        U[f"rate_{s}"] = U[f"n_{g}"] / U.n
    U["rate_flagged"] = U.n_flagged / U.n
    U["log_n"] = np.log(U.n)
    k = U[[f"n_{g}" for g in GROUPS]].sum(axis=1)
    at = sum(U[f"{g}|at_site"] for g in GROUPS) * U.n
    sp = sum(U[f"{g}|spill"] for g in GROUPS) * U.n
    U["cost_mean"] = (at / k).where(k > 0)
    U["spill_mean"] = (sp / k).where(k > 0)
    return U


def run(corpus):
    a, ed = load(corpus)
    U4, n_minus_c = run_split4(a, corpus)
    U = unit_table(a, credits(a, ed))
    assert np.allclose(U.clean, U4.clean.reindex(U.index)) and np.allclose(U.total, U4.total.reindex(U.index))
    units = a.groupby("text_id").unit.first()
    U = edit_counts(U, ed, units)
    U["corr_share"] = a.gold.isin(["ERR_TOK", "ERR_INS"]).groupby(a.unit).mean().reindex(U.index)
    t = tau()
    U["mean_surprisal|total"], U["mean_surprisal|clean"] = U.total, U.clean
    U["high_per_1000|total"] = (a.surprisal_bits > t).groupby(a.unit).mean() * 1000
    U["high_per_1000|clean"] = (a.s_corr[a.o] > t).groupby(a.unit[a.o]).mean() * 1000
    U = rq3_features(U)
    U.to_parquet(out(corpus) / "units.parquet")

    shares(U).to_csv(RES / f"shares_{corpus}.csv", index=False)
    cols = list(dict.fromkeys(["total", "clean"] + [f"{c}|{p}" for by in CATS for c in CATS[by]
                                                    for p in COMP + ["total"]]))
    m = U.groupby("group")[cols].mean().T
    m["learners"] = learners(U)[0][cols].mean()
    m.to_csv(RES / f"means_{corpus}.csv")
    rate_cost(U, corpus).to_csv(RES / f"rate_cost_{corpus}.csv", index=False)
    spill_decay(a, ed).to_csv(RES / f"spill_decay_{corpus}.csv", index=False)
    O = a[a.o]
    st = O.status.value_counts().to_dict()
    L = learners(U)[0]
    e = ed[ed.text_id.map(units).isin(L.index)]
    checks = {"n_units": int(len(L)), "n_by_group": U.groupby("group").size().to_dict(),
              "n_texts": int(a[a.y.notna()].text_id.nunique()), "n_tokens": int(a.y.notna().sum()),
              "n_corrections": int((e.kind != "flagged").sum()), "n_flagged": int((e.kind == "flagged").sum()),
              "unmapped_share": float((e.errant[e.kind != "flagged"] == "unmapped").mean()),
              "spill_status": st,
              "own_words_with_gap": int((O.gap_k.map(len) > 0).sum()),
              "v_none_vs_stored_median_abs": float((O.v_none - O.surprisal_bits).abs().median()),
              "v_all_vs_s_corr_median_abs": float((O.v_all - O.s_corr).abs().median()),
              "native_minus_C": n_minus_c}
    (RES / f"decompose_{corpus}.json").write_text(json.dumps(checks, indent=2, default=float))
    print(corpus, json.dumps(checks, default=float), flush=True)
    return clean_assoc(U, corpus)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rows = [r for c in CORPORA for r in run(c)]
    pd.DataFrame(rows).to_csv(RES / "clean_assoc.csv", index=False)


if __name__ == "__main__":
    main()
