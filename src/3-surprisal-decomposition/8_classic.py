"""Study 3, stage 8 (CPU, post hoc) — classic measures under correction, by kind of correction.

For every learner unit, the players are the correction buckets present among
its edits (mechanical, function word, content-word choice, multi-word / order,
and unclassified as a fifth player). For every subset of the present players,
each text is rebuilt with only those buckets' edits applied
(``lib.corrections.apply_edits``; flagged spans stay as written) and parsed
with the spaCy pipeline of ``src/essays/ingest.py``. The classic measures
(``lib.classic``) are computed for each version, so the empty subset is the
original essay and the full one the corrected essay. With v(S) the measure
when the buckets in S are left uncorrected, the exact Shapley value of each
bucket gives measure_orig = measure_corr + Σ_b ψ_b (D33).

Per measure: Spearman ρ with the criterion on the original and the corrected
essays, the share removed by correction (1 − ρ_corr / ρ_orig), ΔR² of the
corrected measure over the error counts B (stage 6 framework; missing values
imputed in-fold), and each
bucket's share of cov(original measure, criterion). Surprisal is added from
stage 5 for comparison.

Writes data/study3/<corpus>/versions.parquet, version_sums.parquet,
classic_units.parquet, and results/study3/classic_<corpus>.csv.

    python src/3-surprisal-decomposition/8_classic.py
"""

import sys
import time
import warnings
from itertools import combinations
from math import factorial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.classic import MEASURES, Ref, doc_items, item_sums, measures  # noqa: E402
from lib.corpora import B, CORPORA, RES, out  # noqa: E402
from lib.corrections import apply_edits  # noqa: E402
from lib.errant import BUCKETS, UNCL  # noqa: E402
from lib.stats import boot, cov_shares, delta_r2, oof, spearman_cols  # noqa: E402

PLAYERS = BUCKETS + [UNCL]
PHRASEO = ["content-word choice", "multi-word / order"]
PROCS = 24


def version_texts(job):
    text, edits, players = job
    out = []
    for r in range(len(players) + 1):
        for A in combinations(players, r):
            sub = [[s, e, x] for s, e, x, b in edits if x is None or b in A]
            out.append((sum(1 << PLAYERS.index(p) for p in A), apply_edits(text, sub)["corrected"]))
    return out


def build_versions(corpus):
    O = out(corpus)
    es = pd.read_parquet(O / "essays.parquet", columns=["text_id", "unit", "y", "text"])
    es = es[es.y.notna()]
    ed = pd.read_parquet(O / "edits.parquet").sort_values(["text_id", "k"])
    ed = ed[ed.text_id.isin(es.text_id)]
    pres = ed[ed.kind != "flagged"].assign(unit=ed.text_id.map(es.set_index("text_id").unit))
    pres = pres.groupby("unit").bucket.agg(set)
    by_t = {t: list(zip(g.s, g.e, g.repl.where(g.kind != "flagged", None), g.bucket)) for t, g in ed.groupby("text_id")}
    jobs = [(e.text, by_t.get(e.text_id, []), [p for p in PLAYERS if p in pres.get(e.unit, set())])
            for e in es.itertuples()]
    with Pool(PROCS) as pool:
        res = pool.map(version_texts, jobs, chunksize=8)
    rows, seen = [], {}
    for e, vs in zip(es.itertuples(), res):
        for mask, t in vs:
            rows.append((e.text_id, e.unit, mask, seen.setdefault(t, f"{corpus}{len(seen)}")))
    V = pd.DataFrame(rows, columns=["text_id", "unit", "mask", "uid"])
    T = pd.DataFrame({"uid": list(seen.values()), "text": list(seen.keys())})
    return V, T


def parse(T):
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    parts, batch = [], []
    for doc, meta in nlp.pipe(((t, {"text_id": u}) for u, t in zip(T.uid, T.text)), as_tuples=True,
                              n_process=PROCS, batch_size=64):
        doc.user_data["meta"] = meta
        batch.append(doc)
        if len(batch) == 4000:
            parts.append(doc_items(batch))
            batch = []
    parts.append(doc_items(batch))
    return [pd.concat(x, ignore_index=True) for x in zip(*parts)]


def shapley(M, full):
    """ψ per unit, bucket and measure; v(S) = measure with the buckets in S uncorrected."""
    rows = []
    for unit, g in M.groupby(level=0):
        v = g.droplevel(0)
        F = int(full[unit])
        P = [i for i in range(len(PLAYERS)) if F >> i & 1]
        p = len(P)
        r = {"unit": unit}
        for m in MEASURES:
            vm = v[m]
            r[f"{m}|orig"], r[f"{m}|corr"] = vm.get(0, np.nan), vm.get(F, np.nan)
            for i in range(len(PLAYERS)):
                acc = 0.0
                if i in P:
                    others = [q for q in P if q != i]
                    for k in range(1 << len(others)):
                        S = sum(1 << others[t] for t in range(len(others)) if k >> t & 1)
                        w = factorial(bin(S).count("1")) * factorial(p - bin(S).count("1") - 1) / factorial(p)
                        acc += w * (vm[F ^ (S | 1 << i)] - vm[F ^ S])
                r[f"{m}|{PLAYERS[i]}"] = acc
        rows.append(r)
    return pd.DataFrame(rows).set_index("unit")


def summarise(corpus, Z, U):
    y_all = U.y.reindex(Z.index).to_numpy(float)
    st = y_all.astype(int) if corpus == "wi" else np.digitize(y_all, [0, 23.5, 27.5, 31.5, 40.5])
    Ub = U.reindex(Z.index)
    PB = oof(Ub[B], y_all, st)
    rows = []
    # surprisal (stage 5): corrected = clean; flagged + residual reported separately
    X = pd.DataFrame({"orig": Ub.total, "corr": Ub.clean})
    for b in PLAYERS:
        X[b] = Ub[f"{b}|total"]
    X["flagged+residual"] = Ub["flagged|total"] + Ub["residual|total"]
    todo = [("surprisal", X)]
    for m in MEASURES:
        X = pd.DataFrame({"orig": Z[f"{m}|orig"], "corr": Z[f"{m}|corr"]})
        for b in PLAYERS:
            X[b] = Z[f"{m}|{b}"]
        todo.append((m, X))
    for m, X in todo:
        X = X.assign(**{"phraseological": X[PHRASEO].sum(axis=1),
                        "phraseological + function word": X[PHRASEO + ["function word"]].sum(axis=1)})
        ok = X.notna().all(axis=1).to_numpy()
        Xo, y = X[ok].to_numpy(float), y_all[ok]
        sh = cov_shares(Xo, y)
        rows += [(m, f"share|{c}", sh[0][j], sh[1][j], sh[2][j]) for j, c in enumerate(X.columns) if c != "orig"]

        def rho(i):
            r = spearman_cols(Xo[i][:, :2], y[i])
            if m == "surprisal":  # a token mean: the exact split gives the share removed
                c = ((Xo[i][:, :2] - Xo[i][:, :2].mean(0)) * (y[i] - y[i].mean())[:, None]).mean(0)
                return [r[0], r[1], 1 - c[1] / c[0]]
            return [r[0], r[1], 1 - r[1] / r[0]]
        pt, lo, hi = boot(rho, len(y))
        rows += [(m, s, pt[j], lo[j], hi[j]) for j, s in enumerate(["rho_orig", "rho_corr", "removed"])]
        rows.append((m, "n_units", int(ok.sum()), np.nan, np.nan))
        d = delta_r2(oof(np.column_stack([Ub[B], X["corr"]]), y_all, st), PB, y_all)
        rows.append((m, "dR2_corr", d["delta_r2"], d["lo"], d["hi"]))
    return pd.DataFrame(rows, columns=["measure", "stat", "value", "lo", "hi"])


def run(corpus):
    O = out(corpus)
    t0 = time.time()
    if not (O / "version_sums.parquet").exists():
        V, T = build_versions(corpus)
        V.to_parquet(O / "versions.parquet", index=False)
        print(corpus, f"{len(V)} versions, {len(T)} texts, {time.time() - t0:.0f}s", flush=True)
        tok, big, dep = parse(T)
        print(corpus, f"parsed {time.time() - t0:.0f}s", flush=True)
        ref = Ref(set(tok.lower) | set(tok.lemma), list(zip(big.w1, big.w2)), list(zip(dep.h, dep.d, dep.rel)))
        item_sums(tok, big, dep, ref).to_parquet(O / "version_sums.parquet")
    V = pd.read_parquet(O / "versions.parquet")
    S = pd.read_parquet(O / "version_sums.parquet")
    agg = V[["unit", "mask", "uid"]].join(S, on="uid").drop(columns="uid").fillna(0.0).groupby(["unit", "mask"]).sum()
    Z = shapley(measures(agg), V.groupby("unit")["mask"].max())
    # units where a measure is undefined in some version are left out for that measure
    gap = max((X[f"{m}|corr"] + X[[f"{m}|{b}" for b in PLAYERS]].sum(axis=1) - X[f"{m}|orig"]).abs().max()
              for m in MEASURES for X in [Z.filter(like=f"{m}|").dropna()])
    assert gap < 1e-9, gap
    Z.to_parquet(O / "classic_units.parquet")
    U = pd.read_parquet(O / "units.parquet")
    t = summarise(corpus, Z, U)
    t.to_csv(RES / f"classic_{corpus}.csv", index=False)
    print(corpus, f"exact sum gap {gap:.1e}, {time.time() - t0:.0f}s", flush=True)
    print(t[t.stat.isin(["rho_orig", "rho_corr", "removed", "dR2_corr"])].pivot(
        index="measure", columns="stat", values="value").round(3).to_string(), flush=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for corpus in CORPORA:
        run(corpus)


if __name__ == "__main__":
    main()
