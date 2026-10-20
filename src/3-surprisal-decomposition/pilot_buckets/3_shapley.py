"""Bucket pilot, step 3 (CPU) — classic measures on every version, exact Shapley by bucket.

Annotates the step 2 items with the Dolma corpus-a (and a + b for Study 2)
tables exactly as ``pilot_assoc/2_classic.py`` (``Ref``, ``annotate``), and
computes each measure per (unit, applied mask) from pooled sums, so an FCE
script pools its two answers' items as in D32.

Shapley over the buckets present in the unit, correction as baseline: with
v(S) = the measure when the buckets in S are left uncorrected and all others
are applied, v(empty) = corrected, v(all present) = original, and

    psi_b = sum_{S not containing b} |S|! (p - |S| - 1)! / p! [v(S + b) - v(S)]

so measure_orig = measure_corr + sum_b psi_b exactly. Absent buckets get 0.
A unit whose measure is undefined (no items) in any version is dropped for that
measure (counts reported).

Writes data/study3/pilot_buckets/:
    measures_<corpus>.parquet        per (unit, mask): every measure
    shapley_<corpus>.parquet         per unit: level, <measure>|orig, |corr, |<bucket>
    tables/shapley_<corpus>.json     shares of cov(orig, level), checks vs D32, interaction checks

    python src/3-surprisal-decomposition/pilot_buckets/3_shapley.py
"""

import json
import sys
import time
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bucket_common import OUT, PLAYERS, S3_SRC, TAB, bucket_of, cov_shares, load  # noqa: E402
from util.paths import DATA_DIR, FCE_DOCBINS_DIR, WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

classic = load("assoc_classic", S3_SRC / "pilot_assoc" / "2_classic.py")
apply_edits = load("pc_common", S3_SRC / "pilot_corrected" / "common.py").apply_edits
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}
RELS = ["amod", "dobj", "advmod"]
MEASURES = ["bigram_mi", "bigram_dp", "bigram_t", "share_mi3", "amod_mi", "dobj_mi", "advmod_mi", "dep_mi",
            "content_freq", "log_freq_s2"]
D32 = DATA_DIR / "study3" / "pilot_assoc"
EDIT_TYPES = {"wi": DATA_DIR / "study3" / "pilot_shapley" / "edit_types.parquet",
              "fce": DATA_DIR / "study3" / "pilot_types" / "edit_types_fce.parquet"}
SRC = {"wi": WI_DOCBINS_DIR, "fce": FCE_DOCBINS_DIR}


def uid_stats(corpus):
    """Per version text (uid): sums and counts behind every measure."""
    tok = pd.read_parquet(OUT / f"items_tok_{corpus}.parquet")
    big = pd.read_parquet(OUT / f"items_bigram_{corpus}.parquet")
    dep = pd.read_parquet(OUT / f"items_dep_{corpus}.parquet")
    ref = classic.Ref(tok.lower, list(zip(big.w1, big.w2)), list(zip(dep.h, dep.d, dep.rel)), tok.lemma)
    tok, big, dep, Na, Nab = classic.annotate(tok, big, dep, ref)
    S = {}
    al = tok[tok.is_alpha]
    g = al.assign(x=np.log10(al.cnt_ab_lemma.clip(lower=1) / Nab)).groupby("text_id").x
    S["log_freq_s2_s"], S["log_freq_s2_n"] = g.sum(), g.size()
    cw = al[al.pos.isin(CONTENT) & (al.cnt_a > 0)]
    g = cw.assign(x=np.log10(cw.cnt_a / Na)).groupby("text_id").x
    S["content_freq_s"], S["content_freq_n"] = g.sum(), g.size()
    for r in RELS:
        g = dep[dep.rel == r].groupby("text_id").mi
        S[f"{r}_mi_s"], S[f"{r}_mi_n"] = g.sum(), g.size()
    att = big[big.c12 > 0]
    g = att.groupby("text_id")
    for m in ["mi", "dp", "t"]:
        S[f"bigram_{m}_s"] = g[m].sum()
        S[f"bigram_{m}_n"] = g.size()
    S["share_mi3_s"] = att.assign(x=(att.mi >= 3).astype(float)).groupby("text_id").x.sum()
    S["share_mi3_n"] = g.size()
    return pd.DataFrame(S).fillna(0.0)


def unit_measures(V, st):
    """Measure per (unit, mask), pooling the unit's answers."""
    x = V[["unit", "mask", "uid"]].join(st, on="uid").fillna(0.0)
    agg = x.drop(columns="uid").groupby(["unit", "mask"]).sum()
    M = pd.DataFrame(index=agg.index)
    for m in [c[:-2] for c in st.columns if c.endswith("_s")]:
        n = agg[f"{m}_n"]
        M[m] = (agg[f"{m}_s"] / n).where(n > 0)
    M["dep_mi"] = M[[f"{r}_mi" for r in RELS]].mean(axis=1, skipna=True)
    return M


def shapley(M, full):
    """psi per unit and bucket for every measure. ``full``: unit -> mask of present players."""
    rows = []
    for unit, g in M.groupby(level=0):
        vals = g.droplevel(0)
        F = int(full[unit])
        P = [i for i in range(len(PLAYERS)) if F >> i & 1]
        p = len(P)
        w = [factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)] if p else []
        # v over uncorrected sets S (bitmask): applied = F ^ S
        r = {"unit": unit}
        for m in MEASURES:
            v = vals[m]
            r[f"{m}|orig"] = v.get(0, np.nan)
            r[f"{m}|corr"] = v.get(F, np.nan)
            for j, name in enumerate(PLAYERS):
                r[f"{m}|{name}"] = 0.0
            if p == 0:
                continue
            for i in P:
                bit = 1 << i
                others = [q for q in P if q != i]
                acc = 0.0
                for k in range(1 << len(others)):
                    S = sum(1 << others[t] for t in range(len(others)) if k >> t & 1)
                    acc += w[bin(S).count("1")] * (v[F ^ (S | bit)] - v[F ^ S])
                r[f"{m}|{PLAYERS[i]}"] = acc
        rows.append(r)
    return pd.DataFrame(rows).set_index("unit")


def levels(corpus, units):
    if corpus == "wi":
        e = pd.read_parquet(DATA_DIR / "study3" / "pilot_corrected" / "essays.parquet", columns=["text_id", "level"])
        lv = e.set_index("text_id").level.map({"A": 1, "B": 2, "C": 3})
    else:
        e = pd.read_parquet(DATA_DIR / "study3" / "pilot_fce" / "essays.parquet", columns=["script_id", "script_s"])
        lv = pd.to_numeric(e.groupby("script_id").script_s.first(), errors="coerce")
    return lv.reindex(units)


def interaction_check(corpus, V):
    """Is every edit applied the same way (span, replacement as applied) in every version that applies it?"""
    et = pd.read_parquet(EDIT_TYPES[corpus])
    bk = {(t, k): bucket_of(e) for t, k, e, kind in zip(et.text_id, et.k, et.errant, et.kind) if kind != "flagged"}
    masks = V.groupby("text_id")["mask"].apply(list).to_dict()
    n_edits = n_diff = n_skip_versions = n_versions = 0
    cross_touch = 0
    diff_by_fix = {}
    for doc in load_all_docbins(SRC[corpus]):
        m = doc.user_data["meta"]
        tid = m["text_id"]
        if tid not in masks:
            continue
        ed = json.loads(m["edits"])
        b = [bk.get((tid, k), "unclassified") if r is not None else None for k, (s, e, r) in enumerate(ed)]
        full = apply_edits(doc.text, ed)
        ref = {(sp["s"], sp["orig"]): (sp["e"], sp["repl"]) for sp in full["spans"]}
        # adjacent / overlapping non-null edits of different buckets
        nn = sorted([(s, e, b[k]) for k, (s, e, r) in enumerate(ed) if r is not None])
        for (s1, e1, b1), (s2, e2, b2) in zip(nn, nn[1:]):
            if s2 <= e1 + 1 and b1 != b2:
                cross_touch += 1
        seen = set()
        for mask in masks[tid]:
            A = {PLAYERS[i] for i in range(len(PLAYERS)) if mask >> i & 1}
            sub = [x for k, x in enumerate(ed) if x[2] is None or b[k] in A]
            r = apply_edits(doc.text, sub)
            n_versions += 1
            n_skip_versions += r["fixes"]["skipped_overlap"] > 0
            for sp in r["spans"]:
                key = (sp["s"], sp["orig"])
                n_edits += 1
                if ref.get(key) != (sp["e"], sp["repl"]):
                    n_diff += 1
                    if key not in seen:
                        seen.add(key)
                        f = ref.get(key)
                        tag = ("not_in_full" if f is None else
                               "repl" if f[1] != sp["repl"] else "end")
                        diff_by_fix[tag] = diff_by_fix.get(tag, 0) + 1
    return {"edit_applications": n_edits, "applications_differing_from_full": n_diff,
            "share_differing": n_diff / max(1, n_edits), "distinct_edits_differing": diff_by_fix,
            "versions": n_versions, "versions_with_skipped_overlap": n_skip_versions,
            "adjacent_or_overlapping_cross_bucket_pairs": cross_touch}


def run(corpus):
    t0 = time.time()
    V = pd.read_parquet(OUT / f"versions_{corpus}.parquet")
    st = uid_stats(corpus)
    M = unit_measures(V, st)
    M.to_parquet(OUT / f"measures_{corpus}.parquet")
    full = V.groupby("unit")["mask"].max()
    Z = shapley(M, full)
    Z.insert(0, "level", levels(corpus, Z.index))
    Z.to_parquet(OUT / f"shapley_{corpus}.parquet")
    res = {"n_units": int(len(Z)), "secs_measures": time.time() - t0}

    # reproduce D32
    rep = {}
    for ver in ["orig", "corr"]:
        d = pd.read_parquet(D32 / f"classic_{corpus}_{ver}.parquet")
        d["dep_mi"] = d[[f"{r}_mi" for r in RELS]].mean(axis=1, skipna=True)
        for m in MEASURES:
            a, b = Z[f"{m}|{ver}"], d[m].reindex(Z.index)
            ok = a.notna() & b.notna()
            rep[f"{m}|{ver}"] = {"max_abs_diff": float((a[ok] - b[ok]).abs().max()),
                                 "share_equal_1e9": float(((a[ok] - b[ok]).abs() < 1e-9).mean()),
                                 "nan_agree": float((a.isna() == b.isna()).mean())}
    res["reproduce_d32"] = rep

    shares, exact, dropped = {}, {}, {}
    y_all = Z.level.to_numpy(float)
    for m in MEASURES:
        cols = [f"{m}|orig", f"{m}|corr"] + [f"{m}|{p}" for p in PLAYERS]
        X = Z[cols]
        ok = X.notna().all(axis=1).to_numpy() & np.isfinite(y_all)
        dropped[m] = int((~ok).sum())
        Xo = X[ok]
        exact[m] = float((Xo[cols[1]] + Xo[cols[2:]].sum(axis=1) - Xo[cols[0]]).abs().max())
        sh = cov_shares(Xo.to_numpy(float), y_all[ok], ["orig", "corr"] + PLAYERS)
        rho = Xo.corrwith(pd.Series(y_all[ok], index=Xo.index), method="spearman")
        shares[m] = {"shares": sh, "n": int(ok.sum()),
                     "rho_orig": float(rho[cols[0]]), "rho_corr": float(rho[cols[1]]),
                     "mean_orig": float(Xo[cols[0]].mean()), "mean_corr": float(Xo[cols[1]].mean()),
                     "mean_psi": {p: float(Xo[f"{m}|{p}"].mean()) for p in PLAYERS}}
    res.update(shares=shares, exact_sum_max_abs=exact, units_dropped_undefined=dropped)
    t1 = time.time()
    res["interaction"] = interaction_check(corpus, V)
    res["secs_interaction_check"] = time.time() - t1
    (TAB / f"shapley_{corpus}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    for corpus in sys.argv[1:] or ["wi", "fce"]:
        r = run(corpus)
        print(corpus, "units", r["n_units"], "dropped", r["units_dropped_undefined"])
        print(" exact", {k: f"{v:.1e}" for k, v in r["exact_sum_max_abs"].items()})
        for k, v in r["reproduce_d32"].items():
            print(f"  repro {k:22s} {v}")
        for m, s in r["shares"].items():
            print(f"  {m:13s} rho {s['rho_orig']:+.3f}/{s['rho_corr']:+.3f} " +
                  "  ".join(f"{p[:8]} {100 * v[0]:.0f} [{100 * v[1]:.0f},{100 * v[2]:.0f}]"
                            for p, v in s["shares"].items() if p != "orig"))
        print(" interaction", r["interaction"], flush=True)


if __name__ == "__main__":
    main()
