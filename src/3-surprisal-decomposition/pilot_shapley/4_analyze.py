"""Pilot step 4 (CPU) — checks and Q1–Q5 as pre-specified in NOTES.md.

Tables go to data/study3/pilot_shapley/tables/; a JSON summary to
tables/results.json.

    python src/3-surprisal-decomposition/pilot_shapley/4_analyze.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GROUPS, LEVELS, OUT, PC, SEED, TAB  # noqa: E402

N_BOOT = 2000
DSTRATA = [(-np.inf, -0.5, "<-0.5"), (-0.5, 0.5, "[-0.5,0.5]"), (0.5, 2, "(0.5,2]"), (2, np.inf, ">2")]
DBINS = [1, 2, 3, 4, 5]


def load_main():
    s = pd.read_parquet(OUT / "sample.parquet")
    r = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("shapley_main_*.parquet"))])
    t = s.merge(r.drop(columns=["p"]), on=["text_id", "spacy_idx", "set"], how="inner", validate="1:1")
    assert len(t) == len(s)
    return t


def essay_boot_weights(essay_level: pd.Series, rng):
    """(N_BOOT, n_essays) resampling counts, essays drawn within level."""
    W = np.zeros((N_BOOT, len(essay_level)))
    for lvl in essay_level.unique():
        idx = np.flatnonzero(essay_level.to_numpy() == lvl)
        draws = rng.integers(0, len(idx), (N_BOOT, len(idx)))
        for b in range(N_BOOT):
            np.add.at(W[b], idx[draws[b]], 1)
    return W


class Boot:
    """Essay bootstrap for ratio-of-sums statistics on row-level data."""

    def __init__(self, essays: pd.DataFrame):
        self.ess = essays.reset_index(drop=True)
        self.pos = {t: i for i, t in enumerate(self.ess.text_id)}
        self.W = essay_boot_weights(self.ess.level, np.random.default_rng(SEED))

    def sums(self, df, cols):
        e = df.groupby("text_id")[cols].sum()
        X = np.zeros((len(self.ess), len(cols)))
        X[[self.pos[t] for t in e.index]] = e.to_numpy()
        return X

    def ratio(self, df, num, den):
        X = self.sums(df, [num, den])
        pt = X[:, 0].sum() / X[:, 1].sum()
        B = self.W @ X
        bs = B[:, 0] / B[:, 1]
        return pt, np.percentile(bs, [2.5, 97.5])

    def rows(self, df):
        """Row-index draws for statistics that need rows (correlations)."""
        by = {t: np.flatnonzero(df.text_id.to_numpy() == t) for t in df.text_id.unique()}
        rng = np.random.default_rng(SEED)
        lv = self.ess.set_index("text_id").level
        ess = [t for t in self.ess.text_id if t in by]
        groups = {l: [t for t in ess if lv[t] == l] for l in LEVELS}
        for _ in range(N_BOOT):
            pick = []
            for l, g in groups.items():
                if g:
                    pick += [by[g[i]] for i in rng.integers(0, len(g), len(g))]
            yield np.concatenate(pick)


def fmt(pt, ci, d=3):
    return f"{pt:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


# --------------------------------------------------------------------------- #
# Q1
# --------------------------------------------------------------------------- #
def ctrl_baseline(ctrl, col):
    """c_p(j): mean positive share by slot; q_p(j): share of top player by slot."""
    c, q = {}, {}
    for p, g in ctrl.groupby("n_players"):
        A = np.stack(g[col].to_numpy())
        Ap = np.clip(A, 0, None)
        tot = Ap.sum(1)
        ok = tot > 0
        share = Ap[ok] / tot[ok, None]
        c[p] = share.mean(0)
        top = np.bincount(A.argmax(1), minlength=p) / len(A)
        q[p] = top
    return c, q


def q1_rows(cand, c, q, col):
    out = []
    for r in cand.itertuples():
        a = np.asarray(getattr(r, col))
        E = np.asarray(r.player_edited)
        ap = np.clip(a, 0, None)
        tot = ap.sum()
        p = r.n_players
        if p not in c:  # no control with this many players: uniform share
            c[p] = np.full(p, 1 / p)
            q[p] = np.full(p, 1 / p)
        out.append({
            "text_id": r.text_id, "level": r.level, "delta": r.delta,
            "has_pos": float(tot > 0),
            "pi_E": ap[E].sum() / tot if tot > 0 else 0.0,
            "pi_hat": c[p][E].sum() if tot > 0 else 0.0,
            "naive": E.mean() if tot > 0 else 0.0,
            "hit": float(E[a.argmax()]), "hit_hat": q[p][E].sum(), "one": 1.0})
    d = pd.DataFrame(out)
    d["stratum"] = pd.cut(d.delta, [x[0] for x in DSTRATA] + [np.inf], labels=[x[2] for x in DSTRATA],
                          right=True)
    return d


def q1_table(d, boot):
    rows = []
    cells = [("all", d)] + [(f"level {l}", d[d.level == l]) for l in LEVELS] + \
            [(f"delta {s}", d[d.stratum == s]) for s in [x[2] for x in DSTRATA]]
    for name, g in cells:
        R = boot.ratio(g, "pi_E", "pi_hat")
        hit = boot.ratio(g, "hit", "one")
        hh = boot.ratio(g, "hit_hat", "one")
        hr = boot.ratio(g, "hit", "hit_hat")
        rows.append({"cell": name, "n": len(g),
                     "pi_E": g.pi_E.sum() / g.has_pos.sum(), "pi_hat": g.pi_hat.sum() / g.has_pos.sum(),
                     "naive": g.naive.sum() / g.has_pos.sum(),
                     "R": R[0], "R_lo": R[1][0], "R_hi": R[1][1],
                     "top1": hit[0], "top1_lo": hit[1][0], "top1_hi": hit[1][1], "top1_hat": hh[0],
                     "top1_ratio": hr[0], "top1_ratio_lo": hr[1][0], "top1_ratio_hi": hr[1][1]})
    return pd.DataFrame(rows)


def q1_verdict(tab):
    a = tab.set_index("cell")
    all_ok = a.loc["all", "R"] >= 1.5 and a.loc["all", "R_lo"] > 1
    big = a.loc["delta >2"]
    big_ok = big.top1 >= .5 and big.top1_ratio >= 1.5
    v = ("finds the error" if all_ok and big_ok else
         "finds the error where the error matters" if big_ok else "does not find the error")
    return {"overall_R_pass": bool(all_ok), "delta_gt2_top1_pass": bool(big_ok), "verdict": v}


# --------------------------------------------------------------------------- #
# Q2
# --------------------------------------------------------------------------- #
def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    r = np.corrcoef([rx, ry, rz])
    return (r[0, 1] - r[0, 2] * r[1, 2]) / np.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))


def q2(cand, boot, col):
    c = cand.copy()
    c["att_E"] = [np.asarray(a)[np.asarray(e)].sum() for a, e in zip(c[col], c.player_edited)]
    res = {}
    subsets = {"all": c,
               "single edit, token": c[(c.n_edited == 1) & c.player_kind.map(lambda k: "tok" in k)],
               "single edit, insertion point": c[(c.n_edited == 1) &
                                                 c.player_kind.map(lambda k: "ins" in k and "tok" not in k)]}
    for name, g in subsets.items():
        g = g.reset_index(drop=True)
        x, y, z = g.att_E.to_numpy(), g.delta.to_numpy(), g.surprisal_bits.to_numpy()
        pt = {"pearson": np.corrcoef(x, y)[0, 1], "spearman": stats.spearmanr(x, y)[0],
              "partial_spearman_given_s": partial_spearman(x, y, z),
              "spearman_s_delta": stats.spearmanr(z, y)[0]}
        bs = {k: [] for k in pt}
        for i, ix in enumerate(boot.rows(g)):
            if i >= 500:
                break
            xb, yb, zb = x[ix], y[ix], z[ix]
            bs["pearson"].append(np.corrcoef(xb, yb)[0, 1])
            bs["spearman"].append(stats.spearmanr(xb, yb)[0])
            bs["partial_spearman_given_s"].append(partial_spearman(xb, yb, zb))
            bs["spearman_s_delta"].append(stats.spearmanr(zb, yb)[0])
        res[name] = {"n": len(g), **{k: {"r": float(v), "ci": [float(u) for u in np.percentile(bs[k], [2.5, 97.5])]}
                                     for k, v in pt.items()}}
        res[name]["slope_delta_on_att"] = float(np.polyfit(x, y, 1)[0])
    a = res["all"]
    rho, prt = a["spearman"]["r"], a["partial_spearman_given_s"]["r"]
    res["verdict"] = {"agreement": "agrees" if rho >= .5 else "partly agrees" if rho >= .3 else "does not agree",
                      "adds_beyond_surprisal": bool(prt >= .3)}
    return res, c


# --------------------------------------------------------------------------- #
# Q3
# --------------------------------------------------------------------------- #
def pairs(cand, col):
    rows = []
    for r in cand.itertuples():
        a = np.asarray(getattr(r, col))
        for i, ed in enumerate(r.player_edited):
            if not ed:
                continue
            g = r.player_group[i]
            rows.append({"text_id": r.text_id, "level": r.level, "target": r.spacy_idx,
                         "err": f"{r.text_id}:{r.player_tok[i]}", "group": g if "|" not in g else "mixed",
                         "kind": r.player_kind[i], "d": r.player_dist[i], "att": a[i],
                         "single": r.n_edited == 1, "delta": r.delta, "one": 1.0})
    p = pd.DataFrame(rows)
    p["dbin"] = np.minimum(p.d, 5)
    return p


def q3(cand, boot, col):
    p = pairs(cand, col)
    prof = []
    for g in GROUPS + ["all"]:
        gg = p if g == "all" else p[p.group == g]
        for d in DBINS:
            x = gg[gg.dbin == d]
            if len(x) < 30:
                continue
            m = boot.ratio(x, "att", "one")
            row = {"group": g, "d": d, "n": len(x), "mean_att": m[0], "lo": m[1][0], "hi": m[1][1]}
            xs = x[x.single]
            if len(xs) >= 30:
                md = boot.ratio(xs, "delta", "one")
                ma = boot.ratio(xs, "att", "one")
                row.update({"n_single": len(xs), "mean_att_single": ma[0], "mean_delta_single": md[0],
                            "delta_lo": md[1][0], "delta_hi": md[1][1]})
            prof.append(row)
    prof = pd.DataFrame(prof)
    # per-error spillover
    pe = p.groupby(["err", "text_id", "level", "group"]).agg(spill=("att", "sum"), n_targets=("att", "size"),
                                                              kind=("kind", "first")).reset_index()
    pe["one"] = 1.0
    per = []
    for g in GROUPS + ["mixed", "all"]:
        x = pe if g == "all" else pe[pe.group == g]
        if len(x) < 20:
            continue
        m = boot.ratio(x, "spill", "one")
        nt = x.n_targets.mean()
        per.append({"group": g, "n_errors": len(x), "spill_per_error": m[0], "lo": m[1][0], "hi": m[1][1],
                    "targets_per_error": nt, "share_of_total_spill": x.spill.sum() / pe.spill.sum()})
    per = pd.DataFrame(per)
    # reach and rank agreement
    reach = {}
    for g in GROUPS + ["all"]:
        x = prof[(prof.group == g) & (prof.mean_att >= .1) & (prof.lo > 0)]
        reach[g] = int(x.d.max()) if len(x) else 0
    sg = p[p.single & p.group.isin(GROUPS)].groupby("group").agg(att=("att", "mean"), delta=("delta", "mean"),
                                                                    n=("att", "size"))
    sg = sg[sg.n >= 100]
    rank_rho = float(stats.spearmanr(sg.att, sg.delta)[0]) if len(sg) >= 3 else float("nan")
    return prof, per, reach, sg, rank_rho, p, pe


# --------------------------------------------------------------------------- #
# Q4
# --------------------------------------------------------------------------- #
def q4_frame(cand_like: pd.DataFrame, val_col: str, ntok: pd.Series, level: pd.Series):
    """Per essay: sum of attribution to E (or delta), #distinct edited players, n tokens."""
    rows = []
    for r in cand_like.itertuples():
        rows.append((r.text_id, getattr(r, val_col)))
    v = pd.DataFrame(rows, columns=["text_id", "val"]).groupby("text_id").val.sum()
    errs = {}
    for r in cand_like.itertuples():
        for t_, e in zip(r.player_tok, r.player_edited):
            if e:
                errs.setdefault(r.text_id, set()).add(t_)
    e = pd.DataFrame({"n": ntok, "level": level})
    e["val"] = v
    e["n_err"] = pd.Series({k: len(s) for k, s in errs.items()})
    return e.fillna({"val": 0.0, "n_err": 0.0})


def q4_stats(e: pd.DataFrame):
    e = e.reset_index().rename(columns={"index": "text_id"})
    boot = Boot(e[["text_id", "level"]])
    X = e.assign(text_id=e.text_id)
    out = {"by_level": {}}
    for l in LEVELS:
        g = X[X.level == l]
        ctx = boot.ratio(g, "val", "n")
        rate = boot.ratio(g, "n_err", "n")
        spill = boot.ratio(g, "val", "n_err")
        out["by_level"][l] = {"n_essays": len(g), "context_per_token": ctx[0], "context_ci": list(ctx[1]),
                              "errors_per_1000": rate[0] * 1000, "errors_ci": list(rate[1] * 1000),
                              "spill_per_error": spill[0], "spill_ci": list(spill[1])}
    # shift-share A vs C, ratio of sums, bootstrap
    Xs = boot.sums(X, ["val", "n_err", "n"])
    lv = boot.ess.level.to_numpy()

    def shift(Ssum_A, Ssum_C):
        RA, RC = Ssum_A[1] / Ssum_A[2], Ssum_C[1] / Ssum_C[2]
        SA, SC = Ssum_A[0] / Ssum_A[1], Ssum_C[0] / Ssum_C[1]
        gap = RA * SA - RC * SC
        rate_term = (RA - RC) * (SA + SC) / 2
        spill_term = (SA - SC) * (RA + RC) / 2
        return gap, rate_term, spill_term, SA / SC - 1

    pt = shift(Xs[lv == "A"].sum(0), Xs[lv == "C"].sum(0))
    bs = np.array([shift((boot.W[b][lv == "A", None] * Xs[lv == "A"]).sum(0),
                         (boot.W[b][lv == "C", None] * Xs[lv == "C"]).sum(0)) for b in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    out["A_minus_C"] = {"gap": pt[0], "rate_term": pt[1], "spill_term": pt[2],
                        "rate_share": pt[1] / pt[0], "rate_share_ci": [float(x) for x in
                                                                       np.percentile(bs[:, 1] / bs[:, 0], [2.5, 97.5])],
                        "spill_A_over_C_minus_1": pt[3], "spill_rel_ci": [float(lo[3]), float(hi[3])]}
    L = X[(X.level != "N") & (X.n_err > 0)]
    sp = L.val / L.n_err
    out["spearman_spill_level_learners"] = float(stats.spearmanr(sp, L.level.map({"A": 1, "B": 2, "C": 3}))[0])
    rel = abs(pt[3])
    ci_incl0 = lo[3] <= 0 <= hi[3]
    out["verdict"] = ("just the error count" if (rel < .2 or ci_incl0) and pt[1] / pt[0] >= .8
                      else "not just the error count")
    return out


def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# --------------------------------------------------------------------------- #
def run_questions(t, col, boot, res, tag):
    cand, ctrl = t[t.set == "cand"], t[t.set == "ctrl"]
    c, q = ctrl_baseline(ctrl, col)
    d = q1_rows(cand, c, q, col)
    tab1 = q1_table(d, boot)
    tab1.to_csv(TAB / f"q1_{tag}.csv", index=False)
    res[f"q1_{tag}"] = {"table": tab1.round(4).to_dict("records"), **q1_verdict(tab1),
                        "ctrl_slot_share_p7": c.get(7), "ctrl_top_slot_p7": q.get(7)}
    print(tag, "Q1\n", tab1.round(3).to_string(), flush=True)
    r2, cc = q2(cand, boot, col)
    res[f"q2_{tag}"] = r2
    print(tag, "Q2", json.dumps(r2, indent=1, default=json_default), flush=True)
    prof, per, reach, sg, rr, p, pe = q3(cand, boot, col)
    prof.to_csv(TAB / f"q3_profile_{tag}.csv", index=False)
    per.to_csv(TAB / f"q3_per_error_{tag}.csv", index=False)
    sg.to_csv(TAB / f"q3_single_rank_{tag}.csv")
    res[f"q3_{tag}"] = {"reach": reach, "rank_rho_att_vs_delta_single": rr,
                        "per_error": per.round(4).to_dict("records")}
    print(tag, "Q3\n", per.round(3).to_string(), "\nreach", reach, "\nrank rho", rr, flush=True)
    return d, p, pe, tab1


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    res = {}
    t = load_main()
    al = pd.read_parquet(PC / "aligned.parquet", columns=["text_id", "spacy_idx", "is_space"])
    ntok_all = al[~al.is_space].groupby("text_id").size()
    ess = pd.read_parquet(PC / "essays.parquet", columns=["text_id", "level"])
    lvl_all = ess.set_index("text_id").level

    # ---- checks ---------------------------------------------------------------
    eff = np.array([a.sum() - (f - e) for a, f, e in zip(t.phi, t.v_full, t.v_empty)])
    dv = (t.v_full - t.surprisal_bits).abs()
    res["checks"] = {
        "efficiency_max_abs": float(np.abs(eff).max()),
        "vfull_fp32_vs_stored_bf16": {"median": float(dv.median()), "p99": float(dv.quantile(.99)),
                                      "max": float(dv.max()), "share_lt_.05": float((dv < .05).mean())},
        "engine": json.loads((OUT / "checks_engine.json").read_text()) if (OUT / "checks_engine.json").exists() else None,
        "v_empty_median": float(t.v_empty.median()),
        "sum_phi_median": float(np.median([a.sum() for a in t.phi])),
    }
    print(json.dumps(res["checks"], indent=1), flush=True)

    # the sampled essays (cand and err come from them; ctrl may come from other essays of the level)
    sampled = t[t.set != "ctrl"][["text_id", "level"]].drop_duplicates("text_id")
    boot = Boot(sampled)
    # descriptive: positive and negative attribution mass by set
    desc = []
    for st, g in t.groupby("set"):
        A = g.phi
        desc.append({"set": st, "n": len(g), "s_mean": g.surprisal_bits.mean(), "v_empty": g.v_empty.mean(),
                     "sum_pos": np.mean([np.clip(a, 0, None).sum() for a in A]),
                     "sum_neg": np.mean([np.clip(a, None, 0).sum() for a in A]),
                     "delta_mean": g.delta.mean()})
    res["descriptive"] = desc
    print(pd.DataFrame(desc).round(3).to_string(), flush=True)

    d_phi, p_phi, pe_phi, tab_phi = run_questions(t, "phi", boot, res, "shapley")
    d_loo, p_loo, pe_loo, tab_loo = run_questions(t, "loo", boot, res, "loo")

    # ---- Q4 -------------------------------------------------------------------
    cand = t[t.set == "cand"].copy()
    cand["phi_E"] = [np.asarray(a)[np.asarray(e)].sum() for a, e in zip(cand.phi, cand.player_edited)]
    cand["loo_E"] = [np.asarray(a)[np.asarray(e)].sum() for a, e in zip(cand.loo, cand.player_edited)]
    ids = sampled.text_id
    for tag, col in [("shapley", "phi_E"), ("loo", "loo_E"), ("delta_sampled", "delta")]:
        e = q4_frame(cand, col, ntok_all[ids], lvl_all[ids])
        res[f"q4_{tag}"] = q4_stats(e)
    ta = pd.read_parquet(OUT / "targets_all.parquet")
    ca = ta[ta.set == "cand"]
    e = q4_frame(ca, "delta", ntok_all, lvl_all)
    res["q4_delta_all_essays"] = q4_stats(e)
    for k in ["q4_shapley", "q4_loo", "q4_delta_sampled", "q4_delta_all_essays"]:
        print(k, json.dumps(res[k], indent=1, default=json_default), flush=True)
    # error mix by level (all essays): group shares of edited players in cand windows
    mix = []
    for r in ca.itertuples():
        for t_, e_, g in zip(r.player_tok, r.player_edited, r.player_group):
            if e_:
                mix.append((r.text_id, t_, g if "|" not in g else "mixed"))
    mix = pd.DataFrame(mix, columns=["text_id", "tok", "group"]).drop_duplicates(["text_id", "tok"])
    mix["level"] = mix.text_id.map(lvl_all)
    mtab = pd.crosstab(mix.level, mix.group, normalize="index")
    mtab.to_csv(TAB / "q4_error_mix_by_level.csv")
    # spill per error by group and level (Shapley, sampled essays)
    pg = pe_phi.groupby(["level", "group"]).spill.agg(["mean", "size"]).unstack(0)
    pg.to_csv(TAB / "q4_spill_by_group_level.csv")
    # reweight: C's spill per error by group applied to A's mix, and vice versa
    sp = pe_phi[pe_phi.group.isin(GROUPS)].groupby(["level", "group"]).spill.mean().unstack()
    w = pe_phi[pe_phi.group.isin(GROUPS)].groupby(["level", "group"]).size().unstack()
    w = w.div(w.sum(1), axis=0)
    res["q4_mix_reweight"] = {
        "spill_A": float((sp.loc["A"] * w.loc["A"]).sum()), "spill_C": float((sp.loc["C"] * w.loc["C"]).sum()),
        "spill_C_with_A_mix": float((sp.loc["C"] * w.loc["A"]).sum()),
        "spill_A_with_C_mix": float((sp.loc["A"] * w.loc["C"]).sum())}
    print("mix reweight", res["q4_mix_reweight"])

    # ---- Q5: Shapley vs LOO agreement ---------------------------------------
    P = np.concatenate(t.phi.to_numpy())
    Lo = np.concatenate(t.loo.to_numpy())
    Ed = np.concatenate([np.asarray(e) for e in t.player_edited])
    S = np.repeat(t.set.to_numpy(), t.n_players.to_numpy())
    res["q5"] = {
        "spearman_phi_loo_edited": float(stats.spearmanr(P[Ed], Lo[Ed])[0]),
        "spearman_phi_loo_unedited": float(stats.spearmanr(P[~Ed], Lo[~Ed])[0]),
        "pearson_phi_loo_all": float(np.corrcoef(P, Lo)[0, 1]),
        "mean_phi_edited_cand": float(P[Ed & (S == "cand")].mean()),
        "mean_loo_edited_cand": float(Lo[Ed & (S == "cand")].mean()),
        "q1_verdict_same": res["q1_shapley"]["verdict"] == res["q1_loo"]["verdict"],
        "q2_verdict_same": res["q2_shapley"]["verdict"] == res["q2_loo"]["verdict"],
        "top1_same_player": float(np.mean([np.argmax(a) == np.argmax(b) for a, b in
                                           zip(t[t.set == "cand"].phi, t[t.set == "cand"].loo)])),
    }
    ps = pe_phi[pe_phi.group.isin(GROUPS)].groupby("group").spill.mean()
    pl = pe_loo[pe_loo.group.isin(GROUPS)].groupby("group").spill.mean()
    res["q5"]["q3_group_rank_rho"] = float(stats.spearmanr(ps, pl.reindex(ps.index))[0])
    res["q5"]["same_conclusions"] = bool(res["q5"]["q1_verdict_same"] and res["q5"]["q2_verdict_same"]
                                         and res["q5"]["q3_group_rank_rho"] >= .8)
    print("Q5", json.dumps(res["q5"], indent=1, default=json_default))

    # ---- error tokens -----------------------------------------------------------
    err = t[t.set == "err"].copy()
    et = pd.read_parquet(OUT / "edit_types.parquet")
    al2 = pd.read_parquet(PC / "aligned.parquet", columns=["text_id", "spacy_idx", "idx", "text"])
    err = err.merge(al2, on=["text_id", "spacy_idx"], how="left", suffixes=("", "_tok"))
    etb = {k: g for k, g in et[et.kind.isin(["sub", "del"])].groupby("text_id")}
    grp = []
    for r in err.itertuples():
        g = etb.get(r.text_id)
        a, b = r.idx, r.idx + len(r.text)
        hit = g[(g.s < b) & (g.e > a)].group.unique() if g is not None else []
        grp.append(hit[0] if len(hit) == 1 else ("mixed" if len(hit) else "unmapped"))
    err["own_group"] = grp
    c, _ = ctrl_baseline(t[t.set == "ctrl"], "phi")
    rows = []
    for r in err.itertuples():
        a = np.clip(np.asarray(r.phi), 0, None)
        tot = a.sum()
        if tot <= 0:
            continue
        E = np.asarray(r.player_edited)
        rows.append({"own_group": r.own_group, "level": r.level, "has_other_err": E.any(),
                     "share_other_err": a[E].sum() / tot, "share_other_err_hat": c.get(r.n_players, np.full(r.n_players, 1 / r.n_players))[E].sum(),
                     "share_slot1": a[-1] / tot, "share_slot1_hat": c.get(r.n_players, np.full(r.n_players, 1 / r.n_players))[-1],
                     "s": r.surprisal_bits, "sum_phi": np.sum(r.phi), "v_empty": r.v_empty})
    ed = pd.DataFrame(rows)
    etab = ed.groupby("own_group").agg(n=("s", "size"), s=("s", "mean"), with_other_err=("has_other_err", "mean"),
                                       share_slot1=("share_slot1", "mean"), slot1_hat=("share_slot1_hat", "mean"))
    w = ed[ed.has_other_err]
    etab["share_other_err|any"] = w.groupby("own_group").share_other_err.mean()
    etab["hat|any"] = w.groupby("own_group").share_other_err_hat.mean()
    etab.loc["all"] = [len(ed), ed.s.mean(), ed.has_other_err.mean(), ed.share_slot1.mean(),
                       ed.share_slot1_hat.mean(), w.share_other_err.mean(), w.share_other_err_hat.mean()]
    etab.to_csv(TAB / "err_tokens.csv")
    print(etab.round(3).to_string())
    res["err_tokens"] = etab.round(4).reset_index().to_dict("records")

    # ---- sensitivity -------------------------------------------------------------
    sens = {}
    base = t.set_index(["text_id", "spacy_idx"])
    for sch in ["delete", "mask", "replace"]:
        f = OUT / f"shapley_sens_{sch}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f).set_index(["text_id", "spacy_idx"])
        sub = base.loc[r.index].copy()
        sub["phi"] = r.phi
        sub["v_empty"] = r.v_empty
        sub = sub.reset_index()
        sb = Boot(sub[["text_id", "level"]].drop_duplicates("text_id"))
        cs, qs = ctrl_baseline(sub[sub.set == "ctrl"], "phi")
        dd = q1_rows(sub[sub.set == "cand"], cs, qs, "phi")
        tb = q1_table(dd, sb)
        A = np.concatenate(r.phi.to_numpy())
        B = np.concatenate(base.loc[r.index].phi.to_numpy())
        E = np.concatenate([np.asarray(e) for e in base.loc[r.index].player_edited])
        slot1 = np.concatenate([np.arange(len(a))[::-1] == 0 for a in r.phi])
        sens[sch] = {"spearman_vs_main": float(stats.spearmanr(A, B)[0]),
                     "spearman_vs_main_edited": float(stats.spearmanr(A[E], B[E])[0]),
                     "max_abs_vs_main": float(np.abs(A - B).max()),
                     "mean_phi_slot1": float(A[slot1].mean()), "mean_phi_slot1_main": float(B[slot1].mean()),
                     "v_empty_median": float(r.v_empty.median()),
                     "q1": tb[tb.cell.isin(["all", "delta >2"])].round(4).to_dict("records"),
                     "q1_verdict": q1_verdict(tb)["verdict"]}
    f = OUT / "shapley_sens_subword.parquet"
    if f.exists():
        r = pd.read_parquet(f).set_index(["text_id", "spacy_idx"])
        b = base.loc[r.index]
        wa, wb, we = [], [], []
        for (k, row), ph in zip(b.iterrows(), r.phi):
            agg = np.bincount(np.asarray(row.player_of_sub), weights=ph, minlength=row.n_players)
            wa += list(agg)
            wb += list(row.phi)
            we += list(row.player_edited)
        wa, wb, we = map(np.asarray, (wa, wb, we))
        multi = np.concatenate([np.bincount(np.asarray(x), minlength=n) > 1 for x, n in zip(b.player_of_sub, b.n_players)])
        sens["subword"] = {"n_targets": len(r), "pearson_word_vs_summed_subword": float(np.corrcoef(wa, wb)[0, 1]),
                           "spearman": float(stats.spearmanr(wa, wb)[0]),
                           "pearson_multisubword_players": float(np.corrcoef(wa[multi], wb[multi])[0, 1]),
                           "mean_abs_diff": float(np.abs(wa - wb).mean()),
                           "top1_same": float(np.mean([np.argmax(np.bincount(np.asarray(x), weights=ph, minlength=n)) ==
                                                       np.argmax(pw) for x, ph, n, pw in
                                                       zip(b.player_of_sub, r.phi, b.n_players, b.phi)]))}
    res["sensitivity"] = sens
    print("sensitivity", json.dumps(sens, indent=1, default=json_default))

    # ---- GPU cost -------------------------------------------------------------------
    res["gpu_seconds"] = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.time.json"))}
    (TAB / "results.json").write_text(json.dumps(res, indent=1, default=json_default))


if __name__ == "__main__":
    main()
