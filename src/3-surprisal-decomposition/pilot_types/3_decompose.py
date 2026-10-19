"""Step 3 (CPU) — the D24 split refined by error type, exact per essay.

For each unit (W&I essay; FCE script, both answers pooled, as in D25) over its
n scored tokens, with O the writer's own aligned tokens and clean = mean_O(s_c):

    S_orig = clean + sum_g (at_site_g + spill_g) + flagged + residual

  at_site of an edit
      replacement / unnecessary (json kind sub / del): the excess over clean of
          the tokens it covers, sum (s_o - clean) / n, as D24's error tokens. A
          token inside two edits goes to the first.
      missing (json kind ins): Delta = s_o - s_c of the own word right after the
          gap (D24's omission). Two insertions at one gap share it equally.
  spill of an edit
      its Shapley value (step 2) on every other own aligned token whose window
      it lies in, sum / n. The Shapley values split each token's fp32 Delta;
      the token's stored Delta (bf16 instrument) is matched by adding
      (Delta_stored - Delta_fp32) / n_players to each player, which is the
      Shapley value of the same game with its two end points taken from the
      stored scores. Targets that could not be run (> 7 players, no window)
      split Delta equally among their players.
  flagged
      the excess over clean of tokens inside a detection-only (null
      replacement) span: an error the annotator marked without correcting, so
      it has no ERRANT type and no correction to rescore against.
  residual
      Delta of own tokens with no gold edit in the window (instrument noise),
      plus the excess of own tokens that do not align.

Edit credit goes to its ERRANT group (``pilot_shapley/common.py``; "unmapped"
folded into "other") and, as a second cut, to its ERRANT operation (missing /
replacement / unnecessary; an untyped edit takes it from its json kind). Exact-sum assertion per unit.

Primary rule: an error token's whole excess is its own edit's at-site part.
Check (``--rule shapley``-style, always computed): error tokens and gap words
that lie downstream of an earlier error lose the Shapley share of that error
(step 2, role err and own players other than the gap) to its spill.

Tables in data/study3/pilot_types/tables/:
    means_<corpus>.csv          component means by level / band, per group and op
    shares_<corpus>.json        shares of cov(total, level) with CIs, both cuts, both rules
    rate_cost_<corpus>.csv      per-error rate and cost by level, A-C (low-high) split
    checks_<corpus>.json        exactness, coverage, endpoint agreement, downstream check
    sublevels_wi.csv/json       CEFR sublevels (W&I)
    share_table.csv / .md       the compact table, W&I and FCE side by side

    python src/3-surprisal-decomposition/pilot_types/3_decompose.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from types_common import CORPORA, GROUPS, N_BOOT, OUT, SEED, TAB, edit_table  # noqa: E402
from util.paths import DATA_DIR  # noqa: E402

OPS = ["missing", "replacement", "unnecessary"]
COMP = ["at_site", "spill"]
FCE_BANDS = [(1, 23), (24, 27), (28, 31), (32, 40)]
WI_RAW = DATA_DIR / "wi_locness" / "raw" / "wi+locness" / "json"


# --------------------------------------------------------------------------- #
# credit table
# --------------------------------------------------------------------------- #
def load_tokens(corpus):
    d = CORPORA[corpus]["data"]
    a = pd.read_parquet(d / "aligned.parquet")
    a = a[~a.is_space].copy()
    a["o"] = a.own & a.aligned
    if corpus == "wi":
        a["unit"] = a.text_id
    else:
        a["score"] = pd.to_numeric(a.script_s, errors="coerce")
        a = a[a.score > 0].copy()
        a["unit"] = a.script_id
        a["level"] = a.score
    a["end"] = a.idx + a.text.str.len()
    return a


def first_edit_of(a, et):
    """For each token, the json index k of the first non-null sub/del edit it overlaps (-1 if none)."""
    sp = et[et.e > et.s].sort_values(["text_id", "s"])
    by = {t: (g.s.to_numpy(), g.e.to_numpy(), g.k.to_numpy()) for t, g in sp.groupby("text_id")}
    out = np.full(len(a), -1)
    for i, (t, x, y) in enumerate(zip(a.text_id.to_numpy(), a.idx.to_numpy(), a.end.to_numpy())):
        b = by.get(t)
        if b is None:
            continue
        hit = np.flatnonzero((b[0] < y) & (b[1] > x))
        if len(hit):
            out[i] = b[2][hit[0]]
    return out


def credits(corpus):
    """Long table of credits: unit, text_id, k (edit; -1 residual, -2 flagged), comp, val, cnt.

    at_site rows of error tokens carry val = s_o and cnt = 1, so that the unit's
    clean mean can be subtracted once per token at aggregation. Alternative-rule
    shifts are separate rows with comp in {'alt_at_site', 'alt_spill'} that sum
    to zero within each token.
    """
    a = load_tokens(corpus)
    et = edit_table(corpus)
    sh = pd.read_parquet(OUT / f"shapley_{corpus}.parquet")
    a = a.merge(sh, on=["text_id", "spacy_idx"], how="left")
    checks = {}

    rows = []
    # error tokens (not O)
    E = a[~a.o].copy()
    E["k"] = first_edit_of(E, et)
    # no non-null edit: -2 = inside a flagged (detection-only) span, -1 = own but unaligned
    E.loc[(E.k < 0) & ~E.own, "k"] = -2
    for r in E.itertuples():
        rows.append((r.unit, r.text_id, int(r.k), "at_site", r.surprisal_bits, 1.0))
    checks["E_tokens"] = int(len(E))
    checks["E_tokens_flagged_only"] = int((E.k == -2).sum())
    checks["E_tokens_own_unaligned"] = int((E.k == -1).sum())
    checks["E_tokens_role_mismatch"] = int(((E.k >= 0) != (E.role == "err")).sum())

    O = a[a.o].copy()
    O["delta"] = O.surprisal_bits - O.s_corr
    n_gap, n_ok, n_fail, n_none = 0, 0, 0, 0
    pin = []
    for r in O.itertuples():
        d = r.delta
        if isinstance(r.gap_k, np.ndarray) and len(r.gap_k):
            n_gap += 1
            for k in r.gap_k:
                rows.append((r.unit, r.text_id, int(k), "at_site", d / len(r.gap_k), 0.0))
            continue
        if r.status == "ok":
            n_ok += 1
            fp = r.v_none - r.v_all
            adj = (d - fp) / len(r.psi)
            pin.append(d - fp)
            for k, p in zip(r.edit_k, r.psi):
                rows.append((r.unit, r.text_id, int(k), "spill", p + adj, 0.0))
        elif r.status in ("too_many", "no_window") and len(r.edit_k):
            n_fail += 1
            for k in r.edit_k:
                rows.append((r.unit, r.text_id, int(k), "spill", d / len(r.edit_k), 0.0))
        else:
            n_none += 1
            rows.append((r.unit, r.text_id, -1, "spill", d, 0.0))
    pin = np.array(pin)
    checks.update(n_O=int(len(O)), O_gap=n_gap, O_shapley=n_ok, O_failed_equal_split=n_fail,
                  O_no_player=n_none,
                  O_gap_vs_gold_ERR_INS=pd.crosstab(O.gap_k.map(lambda x: len(x) > 0 if isinstance(x, np.ndarray) else False),
                                                    O.gold).to_dict(),
                  pin_abs_mean=float(np.abs(pin).mean()), pin_abs_median=float(np.median(np.abs(pin))),
                  pin_sum_share_of_spill=float(pin.sum() / O.delta[(O.status == "ok") & ~O.gap_k.map(
                      lambda x: isinstance(x, np.ndarray) and len(x) > 0)].sum()),
                  delta_no_player_sum=float(O.delta[O.status == "no_player"].sum()),
                  delta_no_player_abs_mean=float(O.delta[O.status == "no_player"].abs().mean()))
    ok = a[a.status == "ok"]
    checks["endpoint_v_none_vs_stored"] = {
        "median_abs": float((ok.v_none - ok.surprisal_bits).abs().median()),
        "share_gt_0.5": float(((ok.v_none - ok.surprisal_bits).abs() > .5).mean())}
    oko = ok[ok.role == "own"]
    checks["endpoint_v_all_vs_s_corr"] = {
        "median_abs": float((oko.v_all - oko.s_corr).abs().median()),
        "share_gt_0.5": float(((oko.v_all - oko.s_corr).abs() > .5).mean())}

    # alternative rule: downstream error tokens and gap words give earlier edits' share to them
    alt = []
    Er = E[(E.k >= 0) & (E.status == "ok")]
    for r in Er.itertuples():
        for k, p in zip(r.edit_k, r.psi):
            alt += [(r.unit, r.text_id, int(r.k), "alt_at_site", -p, 0.0),
                    (r.unit, r.text_id, int(k), "alt_spill", p, 0.0)]
    Og = O[O.gap_k.map(lambda x: isinstance(x, np.ndarray) and len(x) > 0) & (O.status == "ok")]
    Og = Og[[len(set(map(int, e)) - set(map(int, g))) > 0 for e, g in zip(Og.edit_k, Og.gap_k)]]
    for r in Og.itertuples():
        gap = set(int(x) for x in r.gap_k)
        fp = r.v_none - r.v_all
        for k, p in zip(r.edit_k, r.psi):
            if int(k) in gap:
                continue
            # the share of the (stored) Delta that Shapley gives to the earlier edits
            q = p + (r.delta - fp) / len(r.psi)
            for g in gap:
                alt.append((r.unit, r.text_id, g, "alt_at_site", -q / len(gap), 0.0))
            alt.append((r.unit, r.text_id, int(k), "alt_spill", q, 0.0))
    up_err = np.array([np.sum(p) for p in Er.psi]) if len(Er) else np.array([])
    clean_unit = a[a.o].groupby("unit").s_corr.mean()
    exc = (E.surprisal_bits - E.unit.map(clean_unit))[E.k >= 0]
    up_gap = []
    for r in Og.itertuples():
        gap = set(int(x) for x in r.gap_k)
        fp = r.v_none - r.v_all
        up_gap.append(sum(p + (r.delta - fp) / len(r.psi) for k, p in zip(r.edit_k, r.psi) if int(k) not in gap))
    up_gap = np.array(up_gap)
    gap_all = O[O.gap_k.map(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
    checks["downstream"] = {
        "err_tokens": int((E.k >= 0).sum()),
        "err_tokens_with_earlier_edit_in_window": int((E.k >= 0).sum() - (E.status == "no_player").sum()),
        "err_tokens_scored": int(len(Er)),
        "share_err_tokens_downstream": float(len(Er) / max(1, (E.k >= 0).sum())),
        "sum_upstream_shapley_err_bits": float(up_err.sum()),
        "sum_excess_err_bits": float(exc.sum()),
        "upstream_share_of_err_excess": float(up_err.sum() / exc.sum()),
        "upstream_share_of_err_excess_among_downstream": float(
            up_err.sum() / exc[E[E.k >= 0].status.eq("ok").to_numpy()].sum()),
        "mean_upstream_per_downstream_err_token": float(up_err.mean()) if len(up_err) else 0.0,
        "gap_words": int(len(gap_all)),
        "gap_words_with_other_edit_in_window": int(len(Og)),
        "share_gap_words_downstream": float(len(Og) / max(1, len(gap_all))),
        "sum_other_edits_shapley_gap_bits": float(up_gap.sum()),
        "sum_delta_gap_bits": float(gap_all.delta.sum()),
        "other_share_of_gap_delta": float(up_gap.sum() / gap_all.delta.sum()),
    }
    cr = pd.DataFrame(rows + alt, columns=["unit", "text_id", "k", "comp", "val", "cnt"])
    kind_op = et.kind.map({"ins": "missing", "sub": "replacement", "del": "unnecessary"})
    checks["errant_op_vs_json_kind"] = pd.crosstab(et.op, et.kind).to_dict()
    et["op"] = np.where(et.op == "unmapped", kind_op, et.op)  # untyped edits: op from the json kind
    ek = et.assign(group=et.group.replace({"unmapped": "other"}))[["text_id", "k", "group", "op", "kind", "errant"]]
    cr = cr.merge(ek, on=["text_id", "k"], how="left")
    cr.loc[cr.k == -1, ["group", "op"]] = "residual"
    cr.loc[cr.k == -2, ["group", "op"]] = "flagged"
    assert cr.group.notna().all(), cr[cr.group.isna()].head()
    return a, cr, et, checks


# --------------------------------------------------------------------------- #
# per-unit components
# --------------------------------------------------------------------------- #
def unit_table(a, cr, by, rule="primary"):
    """Wide per-unit table: total, clean, <cat>|at_site, <cat>|spill (bits; mean over n)."""
    g = a.groupby("unit")
    U = pd.DataFrame({"level": g["level"].first(), "n": g.size(), "total": g.surprisal_bits.mean()})
    U["clean"] = a[a.o].groupby("unit").s_corr.mean()
    c = cr.join(U.clean, on="unit")
    c["x"] = c.val - c.cnt * c.clean
    if rule == "primary":
        c = c[c.comp.isin(COMP)]
    else:
        c = c.assign(comp=c.comp.str.replace("alt_", "", regex=False))
    w = c.groupby(["unit", by, "comp"]).x.sum().unstack([1, 2]).fillna(0.0)
    w.columns = [f"{x}|{y}" for x, y in w.columns]
    U = U.join(w).fillna(0.0)
    parts = [col for col in U.columns if "|" in col]
    U[parts] = U[parts].div(U.n, axis=0)
    gap = (U.clean + U[parts].sum(1) - U.total).abs().max()
    assert gap < 1e-9, f"exact sum fails by {gap}"
    return U, parts


def cats_of(by):
    return (GROUPS if by == "group" else OPS) + ["flagged", "residual"]


def collapse(U, by):
    """Per-unit columns: for each category, at_site, spill and total."""
    out = U[["level", "n", "total", "clean"]].copy()
    for cat in cats_of(by):
        for comp in COMP:
            col = f"{cat}|{comp}"
            out[col] = U[col] if col in U else 0.0
        out[f"{cat}|total"] = out[f"{cat}|at_site"] + out[f"{cat}|spill"]
    return out


def cov_shares(V, lv, cols):
    """Share of cov(total, level) for each column, essay-bootstrap CI."""
    X = V[["total"] + cols].to_numpy(float)
    rng = np.random.default_rng(SEED)

    def sh(i):
        c = ((X[i] - X[i].mean(0)) * (lv[i] - lv[i].mean())[:, None]).mean(0)
        return c[1:] / c[0]

    pt = sh(np.arange(len(V)))
    bs = np.array([sh(rng.integers(0, len(V), len(V))) for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {c: {"share": float(p), "ci": [float(x), float(y)]} for c, p, x, y in zip(cols, pt, lo, hi)}


def learners(V, corpus):
    if corpus == "wi":
        L = V[V.level != "N"]
        return L, L.level.map({"A": 1, "B": 2, "C": 3}).to_numpy(float)
    return V, V.level.to_numpy(float)


def band(V, corpus):
    if corpus == "wi":
        return V.level
    return pd.cut(V.level, [b[0] - .5 for b in FCE_BANDS] + [40.5],
                  labels=[f"{lo}-{hi}" for lo, hi in FCE_BANDS])


# --------------------------------------------------------------------------- #
# rate x cost
# --------------------------------------------------------------------------- #
def rate_cost(V, a, et, corpus, by):
    """Per level: errors per 100 tokens and bits per error (at-site, spill), token-pooled.

    pooled component_L = sum(component * n) / sum(n) = rate_L * cost_L, exactly.
    The low-minus-high difference splits into rate and cost by the two-factor
    (midpoint) rule: d(r c) = dr * mean(c) + mean(r) * dc. CIs from bootstrapping
    units within level.
    """
    units = a.groupby("unit").text_id.unique()
    et = et.assign(group=et.group.replace({"unmapped": "other"}))
    cnt = et.groupby(["text_id", by]).size().unstack().fillna(0)
    N = pd.DataFrame({u: cnt.reindex(t).fillna(0).sum() for u, t in units.items()}).T.reindex(V.index).fillna(0)
    b = band(V, corpus)
    lv = list(b.cat.categories) if hasattr(b, "cat") else ["A", "B", "C", "N"]
    lo_, hi_ = (lv[0], "C") if corpus == "wi" else (lv[0], lv[-1])
    rng = np.random.default_rng(SEED)
    idx = {L: np.flatnonzero((b == L).to_numpy()) for L in lv}
    rows = []
    for cat in cats_of(by)[:-2]:
        n = V.n.to_numpy(float)
        ne = N[cat].to_numpy(float) if cat in N else np.zeros(len(V))
        comps = {c: V[f"{cat}|{c}"].to_numpy(float) * n for c in COMP}

        def stats(sel):
            out = {}
            for L in lv:
                i = sel[L]
                out[L] = (ne[i].sum() / n[i].sum(),
                          {c: comps[c][i].sum() / max(ne[i].sum(), 1e-12) for c in COMP})
            return out

        pt = stats(idx)
        bs = [stats({L: rng.choice(idx[L], len(idx[L])) for L in lv}) for _ in range(500)]
        for c in COMP:
            r = {"cat": cat, "comp": c}
            for L in lv:
                r[f"rate_{L}"] = 100 * pt[L][0]
                r[f"cost_{L}"] = pt[L][1][c]
            ra, rc = pt[lo_][0], pt[hi_][0]
            ca, cc = pt[lo_][1][c], pt[hi_][1][c]
            d = ra * ca - rc * cc
            r["diff_low_high"] = d
            r["rate_part"] = (ra - rc) * (ca + cc) / 2
            r["cost_part"] = (ra + rc) / 2 * (ca - cc)
            r["rate_share"] = r["rate_part"] / d if d else np.nan
            dc = np.array([x[hi_][1][c] - x[lo_][1][c] for x in bs])
            r["cost_high_minus_low"] = cc - ca
            r["cost_high_minus_low_lo"], r["cost_high_minus_low_hi"] = np.percentile(dc, [2.5, 97.5])
            rs = []
            for x in bs:
                xa, xc = x[lo_][0], x[hi_][0]
                ya, yc = x[lo_][1][c], x[hi_][1][c]
                dd = xa * ya - xc * yc
                rs.append((xa - xc) * (ya + yc) / 2 / dd if dd else np.nan)
            r["rate_share_lo"], r["rate_share_hi"] = np.nanpercentile(rs, [2.5, 97.5])
            rows.append(r)
    return pd.DataFrame(rows), (lo_, hi_)


# --------------------------------------------------------------------------- #
def run(corpus):
    a, cr, et, checks = credits(corpus)
    res = {"checks": checks}
    out_means = []
    for by in ["group", "op"]:
        for rule in ["primary", "downstream_shapley"]:
            U, _ = unit_table(a, cr, by, rule)
            V = collapse(U, by)
            # second exactness check on the collapsed table
            cols3 = [f"{c}|{p}" for c in cats_of(by) for p in COMP]
            assert (V.clean + V[cols3].sum(1) - V.total).abs().max() < 1e-9
            L, lv = learners(V, corpus)
            cols = ["clean"] + [f"{c}|{p}" for c in cats_of(by) for p in COMP + ["total"]]
            res[f"shares_{by}_{rule}"] = cov_shares(L, lv, cols)
            if rule == "primary":
                b = band(V, corpus)
                m = V.groupby(b, observed=True)[["total", "clean"] + cols3].mean().T
                m["all_learners"] = L[["total", "clean"] + cols3].mean()
                m["cut"] = by
                out_means.append(m)
                res[f"spearman_{by}"] = {c: float(spearmanr(L[c], lv)[0]) for c in ["total"] + cols}
                rc, (lo_, hi_) = rate_cost(V, a, et, corpus, by)
                rc.to_csv(TAB / f"rate_cost_{by}_{corpus}.csv", index=False)
                res[f"rate_cost_contrast_{by}"] = [lo_, hi_]
                if by == "group":
                    V.to_parquet(OUT / f"units_{corpus}.parquet")
                    res["n_units"] = V.groupby(b, observed=True).size().to_dict()
    pd.concat(out_means).to_csv(TAB / f"means_{corpus}.csv")
    res["other_breakdown"] = other_breakdown(a, cr, corpus)
    (TAB / f"results_{corpus}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def other_breakdown(a, cr, corpus):
    """Point shares of cov(total, level) by ERRANT category inside the "other" group."""
    c = cr.copy()
    c["errcat"] = np.where(c.group == "other",
                           c.errant.fillna("").str.replace(r"^[MRU]:", "", regex=True), c.group)
    U, parts = unit_table(a, c, "errcat")
    L, lv = learners(U, corpus)
    cov = lambda x: float(np.mean((x - x.mean()) * (lv - lv.mean())))  # noqa: E731
    ct = cov(L.total.to_numpy())
    cats = sorted({p.split("|")[0] for p in parts} - set(GROUPS) - {"flagged", "residual"})
    return {k: sum(cov(L[f"{k}|{q}"].to_numpy()) for q in COMP if f"{k}|{q}" in L) / ct for k in cats}


def sublevels():
    cefr = {}
    for f in WI_RAW.glob("*.json"):
        for line in open(f):
            d = json.loads(line)
            cefr[d["id"]] = d.get("cefr")
    V = pd.read_parquet(OUT / "units_wi.parquet")
    V["cefr"] = V.index.map(cefr)
    L = V[V.level != "N"].copy()
    order = sorted(L.cefr.dropna().unique(), key=lambda c: (c == "C2+", c))  # C2+ is the top
    L["sub"] = L.cefr.map({c: i for i, c in enumerate(order)})
    counts = L.cefr.value_counts().reindex(order)
    cols = ["total", "clean"] + [f"{g}|total" for g in GROUPS] + ["flagged|total", "residual|total"]
    rows = []
    for c in cols:
        r_sub = spearmanr(L[c], L["sub"])[0]
        r_lvl = spearmanr(L[c], L.level.map({"A": 1, "B": 2, "C": 3}))[0]
        rows.append({"col": c, "rho_sublevel": r_sub, "rho_level": r_lvl})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "sublevels_wi.csv", index=False)
    means = L.groupby("cefr")[cols].mean().reindex(order)
    means.insert(0, "n", counts)
    means.to_csv(TAB / "sublevel_means_wi.csv")
    return counts, t, means


def compact(res_wi, res_fce):
    rows = []
    for by in ["group", "op"]:
        for cat in ["clean"] + cats_of(by):
            r = {"cut": by, "row": cat}
            for name, res in [("wi", res_wi), ("fce", res_fce)]:
                s = res[f"shares_{by}_primary"]
                for p in COMP + ["total"]:
                    key = "clean" if cat == "clean" else f"{cat}|{p}"
                    if cat == "clean" and p != "total":
                        r[f"{name}_{p}"] = ""
                        continue
                    v = s[key]
                    r[f"{name}_{p}"] = f"{100 * v['share']:.0f} [{100 * v['ci'][0]:.0f}, {100 * v['ci'][1]:.0f}]"
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "share_table.csv", index=False)
    return t


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    res = {c: run(c) for c in CORPORA}
    t = compact(res["wi"], res["fce"])
    print(t.to_string())
    counts, rho, means = sublevels()
    print(counts.to_string())
    print(rho.round(3).to_string())
    for c in CORPORA:
        print(c, json.dumps(res[c]["checks"], indent=1, default=str))


if __name__ == "__main__":
    main()
