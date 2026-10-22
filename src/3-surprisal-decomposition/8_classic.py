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
classic_units.parquet, and results/study3/<corpus>/classic.csv. With --corpus,
any ids of ``lib.corpora`` (e.g. ellipse, where the corrector's edits are the
corrections).

    python src/3-surprisal-decomposition/8_classic.py
    python src/3-surprisal-decomposition/8_classic.py --corpus ellipse
"""

import argparse
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
from lib.classic import MEASURES, Reference, doc_items, item_sums, measures  # noqa: E402
from lib.corpora import ALL_IDS, CORPORA, ERROR_COUNTS, data_dir, results_dir  # noqa: E402
from lib.corrections import apply_edits  # noqa: E402
from lib.errant import BUCKETS, UNCLASSIFIED  # noqa: E402
from lib.stats import boot, cov_shares, delta_r2, oof, spearman_cols  # noqa: E402

PLAYERS = BUCKETS + [UNCLASSIFIED]
PHRASEO = ["content-word choice", "multi-word / order"]
PROCS = 24


def version_texts(job):
    """(mask, text) for every subset of the present buckets; mask bit i = PLAYERS[i]'s edits applied."""
    text, edits, players = job
    out = []
    for r in range(len(players) + 1):
        for applied in combinations(players, r):
            # flagged spans (repl None) are passed on and stay as written
            sub = [[s, e, repl] for s, e, repl, bucket in edits if repl is None or bucket in applied]
            out.append((sum(1 << PLAYERS.index(p) for p in applied), apply_edits(text, sub)["corrected"]))
    return out


def build_versions(corpus):
    """Every version of every learner text, and the distinct version texts (uid -> text)."""
    out_dir = data_dir(corpus)
    essays = pd.read_parquet(out_dir / "essays.parquet", columns=["text_id", "unit", "y", "text"])
    essays = essays[essays.y.notna()]
    edits = pd.read_parquet(out_dir / "edits.parquet").sort_values(["text_id", "k"])
    edits = edits[edits.text_id.isin(essays.text_id)]
    # the buckets present among each unit's corrections (an FCE script pools two texts)
    present = edits[edits.kind != "flagged"].assign(unit=edits.text_id.map(essays.set_index("text_id").unit))
    present = present.groupby("unit").bucket.agg(set)
    by_text = {t: list(zip(g.s, g.e, g.repl.where(g.kind != "flagged", None), g.bucket))
               for t, g in edits.groupby("text_id")}
    jobs = [(e.text, by_text.get(e.text_id, []), [p for p in PLAYERS if p in present.get(e.unit, set())])
            for e in essays.itertuples()]
    with Pool(PROCS) as pool:
        results = pool.map(version_texts, jobs, chunksize=8)
    rows, uid_of = [], {}  # identical texts are parsed once
    for e, versions in zip(essays.itertuples(), results):
        for mask, text in versions:
            rows.append((e.text_id, e.unit, mask, uid_of.setdefault(text, f"{corpus}{len(uid_of)}")))
    versions = pd.DataFrame(rows, columns=["text_id", "unit", "mask", "uid"])
    texts = pd.DataFrame({"uid": list(uid_of.values()), "text": list(uid_of.keys())})
    return versions, texts


def parse(texts):
    """spaCy parse of every version text, as doc_items tables (in chunks of 4,000 docs)."""
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    parts, batch = [], []
    for doc, meta in nlp.pipe(((t, {"text_id": u}) for u, t in zip(texts.uid, texts.text)), as_tuples=True,
                              n_process=PROCS, batch_size=64):
        doc.user_data["meta"] = meta
        batch.append(doc)
        if len(batch) == 4000:
            parts.append(doc_items(batch))
            batch = []
    parts.append(doc_items(batch))
    return [pd.concat(x, ignore_index=True) for x in zip(*parts)]


def shapley(values, full):
    """ψ per unit, bucket and measure; v(S) = measure with the buckets in S uncorrected."""
    rows = []
    for unit, g in values.groupby(level=0):
        v = g.droplevel(0)  # measure by mask
        full_mask = int(full[unit])
        present = [i for i in range(len(PLAYERS)) if full_mask >> i & 1]
        p = len(present)
        row = {"unit": unit}
        for m in MEASURES:
            vm = v[m]
            row[f"{m}|orig"], row[f"{m}|corr"] = vm.get(0, np.nan), vm.get(full_mask, np.nan)
            for i in range(len(PLAYERS)):
                psi = 0.0
                if i in present:
                    others = [q for q in present if q != i]
                    for k in range(1 << len(others)):
                        # S: the subset of the other players left uncorrected; the mask is its complement
                        S = sum(1 << others[t] for t in range(len(others)) if k >> t & 1)
                        size = bin(S).count("1")
                        weight = factorial(size) * factorial(p - size - 1) / factorial(p)
                        psi += weight * (vm[full_mask ^ (S | 1 << i)] - vm[full_mask ^ S])
                row[f"{m}|{PLAYERS[i]}"] = psi
        rows.append(row)
    return pd.DataFrame(rows).set_index("unit")


def summarise(psi_units, units):
    """Per measure: bucket shares, rho before and after correction, share removed, ΔR² over B."""
    units = units.reindex(psi_units.index)
    y_all = units.y.to_numpy(float)
    folds = units.group.to_numpy()  # CV strata: the level groups, as in stage 6
    preds_b = oof(units[ERROR_COUNTS], y_all, folds)
    rows = []
    # surprisal (stage 5): corrected = clean; flagged + residual reported separately
    X = pd.DataFrame({"orig": units.total, "corr": units.clean})
    for b in PLAYERS:
        X[b] = units[f"{b}|total"]
    X["flagged+residual"] = units["flagged|total"] + units["residual|total"]
    todo = [("surprisal", X)]
    for m in MEASURES:
        X = pd.DataFrame({"orig": psi_units[f"{m}|orig"], "corr": psi_units[f"{m}|corr"]})
        for b in PLAYERS:
            X[b] = psi_units[f"{m}|{b}"]
        todo.append((m, X))
    for m, X in todo:
        X = X.assign(**{"phraseological": X[PHRASEO].sum(axis=1),
                        "phraseological + function word": X[PHRASEO + ["function word"]].sum(axis=1)})
        ok = X.notna().all(axis=1).to_numpy()
        Xo, y = X[ok].to_numpy(float), y_all[ok]
        sh = cov_shares(Xo, y)
        rows += [(m, f"share|{c}", sh[0][j], sh[1][j], sh[2][j]) for j, c in enumerate(X.columns) if c != "orig"]

        def rho(idx):
            r = spearman_cols(Xo[idx][:, :2], y[idx])
            if m == "surprisal":  # a token mean: the exact split gives the share removed
                cov = ((Xo[idx][:, :2] - Xo[idx][:, :2].mean(0)) * (y[idx] - y[idx].mean())[:, None]).mean(0)
                return [r[0], r[1], 1 - cov[1] / cov[0]]
            return [r[0], r[1], 1 - r[1] / r[0]]
        point, lo, hi = boot(rho, len(y))
        rows += [(m, stat, point[j], lo[j], hi[j]) for j, stat in enumerate(["rho_orig", "rho_corr", "removed"])]
        rows.append((m, "n_units", int(ok.sum()), np.nan, np.nan))
        d = delta_r2(oof(np.column_stack([units[ERROR_COUNTS], X["corr"]]), y_all, folds), preds_b, y_all)
        rows.append((m, "dR2_corr", d["delta_r2"], d["lo"], d["hi"]))
    return pd.DataFrame(rows, columns=["measure", "stat", "value", "lo", "hi"])


def run(corpus):
    out_dir = data_dir(corpus)
    t0 = time.time()
    if not (out_dir / "version_sums.parquet").exists():
        versions, texts = build_versions(corpus)
        versions.to_parquet(out_dir / "versions.parquet", index=False)
        print(corpus, f"{len(versions)} versions, {len(texts)} texts, {time.time() - t0:.0f}s", flush=True)
        tokens, bigrams, deps = parse(texts)
        print(corpus, f"parsed {time.time() - t0:.0f}s", flush=True)
        ref = Reference(set(tokens.lower) | set(tokens.lemma), list(zip(bigrams.w1, bigrams.w2)),
                        list(zip(deps.h, deps.d, deps.rel)))
        item_sums(tokens, bigrams, deps, ref).to_parquet(out_dir / "version_sums.parquet")
    versions = pd.read_parquet(out_dir / "versions.parquet")
    sums = pd.read_parquet(out_dir / "version_sums.parquet")
    # pool each unit's texts per version, then measures and their Shapley split
    pooled = (versions[["unit", "mask", "uid"]].join(sums, on="uid").drop(columns="uid").fillna(0.0)
              .groupby(["unit", "mask"]).sum())
    psi_units = shapley(measures(pooled), versions.groupby("unit")["mask"].max())
    # units where a measure is undefined in some version are left out for that measure
    gap = max((X[f"{m}|corr"] + X[[f"{m}|{b}" for b in PLAYERS]].sum(axis=1) - X[f"{m}|orig"]).abs().max()
              for m in MEASURES for X in [psi_units.filter(like=f"{m}|").dropna()])
    assert gap < 1e-9, gap
    psi_units.to_parquet(out_dir / "classic_units.parquet")
    units = pd.read_parquet(out_dir / "units.parquet")
    table = summarise(psi_units, units)
    table.to_csv(results_dir(corpus) / "classic.csv", index=False)
    print(corpus, f"exact sum gap {gap:.1e}, {time.time() - t0:.0f}s", flush=True)
    print(table[table.stat.isin(["rho_orig", "rho_corr", "removed", "dR2_corr"])].pivot(
        index="measure", columns="stat", values="value").round(3).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", choices=ALL_IDS, default=CORPORA,
                    help="corpus ids (default: the main analysis, wi and fce)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    for corpus in args.corpus:
        run(corpus)


if __name__ == "__main__":
    main()
