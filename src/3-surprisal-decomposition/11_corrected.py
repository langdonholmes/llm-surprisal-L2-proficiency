"""Study 3, stage 11 (CPU) — the corrected essay scored as a whole (DECISIONS D52).

Per unit, with O the writer's own aligned tokens (stage 3), m the corrected
unit's scored non-space tokens and C the corrected tokens that are not an O
token's counterpart (|C| = m - |O|, since the alignment is one to one):
    S_corr = mean surprisal of the corrected unit over its m tokens
    sel    = S_corr - clean    the difficulty of the corrected words
    cost   = S_orig - S_corr   the cost of erring
so S_orig = clean + sel + cost, and sel = (|C| / m) × (mean_C s - clean).
C is split by the edit whose corrected span a token overlaps: inserted (an
edit with an empty original span), replacement (any other edit), and carried
(no edit: a flagged span, or an own token that does not align). sel splits
exactly over the three.

Writes data/study3/<id>/corrected_units.parquet for every id given, and for
the four main corpora (wi, fce, ellipse, toefl) results/study3/<id>/
corrected_means.csv (by level) and corrected_shares.csv (each part's share of
cov(S_orig, criterion), and its rho, with stage 5's unit bootstrap).

    python src/3-surprisal-decomposition/11_corrected.py --corpus wi fce ellipse toefl fce_model fce_errant
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import ALL_IDS, VARIANTS, base_corpus, data_dir, results_dir  # noqa: E402
from lib.corrections import apply_edits  # noqa: E402
from lib.stats import boot, cov_shares, spearman_cols  # noqa: E402

KINDS = ["inserted", "replacement", "carried"]
SHARE_PARTS = ["total", "clean"] + [f"sel_{k}" for k in KINDS] + ["sel", "cost", "s_corr"]
MAIN = ["wi", "fce", "ellipse", "toefl"]


def corrected_tokens(essay, edits, corr, own_starts, repair):
    """Per corrected token of one text: its surprisal and whether it is C, and of which kind."""
    edit_list = [[s, e, None if pd.isna(r) else r] for s, e, r in zip(edits.s, edits.e, edits.repl)]
    applied = apply_edits(essay.text, edit_list, repair=repair)
    assert applied["corrected"] == essay.corrected, essay.text_id
    cmap = np.asarray(essay.cmap)
    counterpart = set(cmap[own_starts].tolist())
    assert -1 not in counterpart
    spans = [(sp["cs"], sp["ce"], "inserted" if sp["kind"] == "ins" else "replacement")
             for sp in applied["spans"] if sp["ce"] > sp["cs"]]
    starts = corr.idx.to_numpy()
    ends = starts + corr.text.str.len().to_numpy()
    kind = np.full(len(corr), "own", dtype=object)
    for i, (s, e) in enumerate(zip(starts, ends)):
        if int(s) in counterpart:
            continue
        kind[i] = next((k for cs, ce, k in spans if cs < e and ce > s), "carried")
    assert (kind == "own").sum() == len(counterpart)
    return pd.DataFrame({"s": corr.surprisal_bits.to_numpy(), "kind": kind})


def corrected_units(corpus):
    """One row per unit: S_corr, sel by kind, cost, and the two factors of sel."""
    out_dir = data_dir(corpus)
    units = pd.read_parquet(out_dir / "units.parquet")
    essays = pd.read_parquet(out_dir / "essays.parquet", columns=["text_id", "unit", "text", "corrected", "cmap"])
    essays = essays[essays.unit.isin(units.index)]
    edits = {t: d.sort_values("k") for t, d in pd.read_parquet(out_dir / "edits.parquet").groupby("text_id")}
    corr = pd.read_parquet(out_dir / "tokens_corr.parquet")
    corr = {t: d for t, d in corr[~corr.is_space].groupby("text_id")}
    aligned = pd.read_parquet(out_dir / "aligned.parquet", columns=["text_id", "idx", "own", "aligned"])
    own = {t: d.idx.to_numpy() for t, d in aligned[aligned.own & aligned.aligned].groupby("text_id")}
    empty = pd.DataFrame(columns=["s", "e", "repl"])
    frames = []
    for essay in essays.itertuples():
        tok = corrected_tokens(essay, edits.get(essay.text_id, empty), corr[essay.text_id],
                               own.get(essay.text_id, np.array([], int)), repair=corpus not in VARIANTS)
        frames.append(tok.assign(unit=essay.unit))
    tok = pd.concat(frames, ignore_index=True)
    m = tok.groupby("unit").size()
    out = pd.DataFrame({"m": m, "s_corr": tok.groupby("unit").s.mean()}).reindex(units.index)
    out["n_own"] = (tok.kind == "own").groupby(tok.unit).sum()
    out["clean"] = units.clean
    own_mean = tok[tok.kind == "own"].groupby("unit").s.mean()
    assert np.allclose(own_mean.reindex(units.index), units.clean, atol=1e-9)  # stage 5's clean
    for k in KINDS:
        rows = tok[tok.kind == k]
        excess = (rows.s - rows.unit.map(units.clean)).groupby(rows.unit).sum()
        out[f"n_{k}"] = rows.groupby("unit").size().reindex(units.index).fillna(0)
        out[f"sel_{k}"] = excess.reindex(units.index).fillna(0.0) / out.m
    out["sel"] = out.s_corr - out.clean
    out["cost"] = units.total - out.s_corr
    out["total"] = units.total
    assert np.allclose(out[[f"sel_{k}" for k in KINDS]].sum(axis=1), out.sel, atol=1e-9)
    assert np.allclose(out.clean + out.sel + out.cost, units.total, atol=1e-9)
    n_c = out[[f"n_{k}" for k in KINDS]].sum(axis=1)
    assert (n_c == out.m - out.n_own).all()
    out["c_share"] = n_c / out.m
    with np.errstate(invalid="ignore", divide="ignore"):
        out["c_excess"] = out.sel / out.c_share  # mean_C s - clean; NaN for a unit with no C token
    assert np.allclose((out.c_share * out.c_excess).fillna(0), out.sel, atol=1e-9)
    out[["group", "y"]] = units[["group", "y"]]
    out.to_parquet(out_dir / "corrected_units.parquet")
    return out


def summaries(units, corpus):
    """Means by level, and covariance shares and rho with the criterion (learners)."""
    cols = SHARE_PARTS + ["c_share", "c_excess"] + [f"n_{k}" for k in KINDS] + ["m"]
    means = units.groupby("group")[cols].mean()
    rated = units[units.y.notna()]
    means.loc["learners"] = rated[cols].mean()
    means["n_units"] = units.groupby("group").size()
    means.loc["learners", "n_units"] = len(rated)
    X, y = rated[SHARE_PARTS].to_numpy(float), rated.y.to_numpy(float)
    shares = cov_shares(X, y)
    rho = boot(lambda idx: spearman_cols(X[idx], y[idx]), len(y))
    table = pd.DataFrame({"part": SHARE_PARTS, "share": shares[0], "lo": shares[1], "hi": shares[2],
                          "rho": rho[0], "rho_lo": rho[1], "rho_hi": rho[2]})
    means.reset_index().to_csv(results_dir(corpus) / "corrected_means.csv", index=False)
    table.to_csv(results_dir(corpus) / "corrected_shares.csv", index=False)
    print(corpus)
    print(means[["total", "clean", "s_corr", "sel", "cost", "c_share", "c_excess", "n_units"]].round(3).to_string())
    print(table.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", choices=ALL_IDS, default=MAIN)
    args = ap.parse_args()
    for corpus in args.corpus:
        units = corrected_units(corpus)
        if corpus in MAIN:
            summaries(units, corpus)
        else:
            print(corpus, "corrected_units.parquet", len(units), base_corpus(corpus))


if __name__ == "__main__":
    main()
