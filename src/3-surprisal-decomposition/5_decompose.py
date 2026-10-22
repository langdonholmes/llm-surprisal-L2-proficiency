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
data/study3/<corpus>/units.parquet. With --corpus (any ids of
``lib.corpora``, e.g. the corrector variants), the same per id, and
clean_assoc_<id>.csv. With --validate (after the variants have run), the
corrector check: the shares of the json gold, the ERRANT re-cut of gold and the
model, and their paired differences (the same unit resamples for all three),
in validation_shares.csv.

    python src/3-surprisal-decomposition/5_decompose.py
    python src/3-surprisal-decomposition/5_decompose.py --corpus wi_model fce_model
    python src/3-surprisal-decomposition/5_decompose.py --validate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import (  # noqa: E402
    ALL_IDS, CORPORA, GROUP_SLUGS, N_BOOT, RESULTS, SEED, bands, base_corpus, data_dir, tau)
from lib.errant import BUCKETS, GROUPS, UNCLASSIFIED  # noqa: E402
from lib.stats import boot, cov_shares, partial_rank, spearman_cols  # noqa: E402

PARTS4 = ["clean", "omission", "context", "error_tokens"]
COMP = ["at_site", "spill"]
CATS = {"bucket": BUCKETS + [UNCLASSIFIED, "flagged", "residual"], "group": GROUPS + ["flagged", "residual"]}


def load(corpus):
    """Scored original tokens (with their stage 4 spill split) and the edits."""
    out_dir = data_dir(corpus)
    tokens = pd.read_parquet(out_dir / "aligned.parquet")
    tokens = tokens[tokens.y.notna() | (tokens.group == "N")].copy()  # learners with a criterion, and LOCNESS
    tokens["own_aligned"] = tokens.own & tokens.aligned
    tokens = tokens.merge(pd.read_parquet(out_dir / "spill.parquet"), on=["text_id", "spacy_idx"], how="left")
    edits = pd.read_parquet(out_dir / "edits.parquet")
    return tokens, edits


# --------------------------------------------------------------------------- #
# four-part split
# --------------------------------------------------------------------------- #
def split4(tokens, orig_col, corr_col):
    """Per unit: total = clean + omission + context + error_tokens (module docstring)."""
    delta = np.where(tokens.own_aligned, tokens[orig_col] - tokens[corr_col], 0.0)
    tokens = tokens.assign(delta=delta)
    own = tokens[tokens.own_aligned]
    by_unit = tokens.groupby("unit")
    units = pd.DataFrame({"group": by_unit.group.first(), "y": by_unit.y.first(), "n": by_unit.size(),
                          "total": by_unit[orig_col].mean()})
    units["clean"] = own.groupby("unit")[corr_col].mean()
    units["omission"] = own[own.gold == "ERR_INS"].groupby("unit").delta.sum()
    units["context"] = own[own.gold != "ERR_INS"].groupby("unit").delta.sum()
    err = tokens[~tokens.own_aligned].join(units.clean, on="unit")
    units["error_tokens"] = (err[orig_col] - err.clean).groupby(err.unit).sum()
    units = units.fillna({p: 0.0 for p in PARTS4[1:]})
    units[PARTS4[1:]] = units[PARTS4[1:]].div(units.n, axis=0)
    assert np.allclose(units[PARTS4].sum(axis=1), units.total, atol=1e-9)
    return units


def learners(units):
    """The units with a criterion, and the criterion as an array."""
    rated = units[units.y.notna()]
    return rated, rated.y.to_numpy(float)


def run_split4(tokens, corpus):
    """The four-part split for mean surprisal and the high-token rate: shares, rho, group means."""
    t = tau()
    tokens = tokens.assign(high_o=(tokens.surprisal_bits > t).astype(float), high_c=(tokens.s_corr > t).astype(float))
    rows, means, native_minus_c = [], [], {}
    for name, orig_col, corr_col, scale in [("mean_surprisal", "surprisal_bits", "s_corr", 1.0),
                                            ("high_per_1000", "high_o", "high_c", 1000.0)]:
        units = split4(tokens, orig_col, corr_col)
        units[["total"] + PARTS4] *= scale
        rated, y = learners(units)
        X = rated[["total"] + PARTS4].to_numpy(float)
        shares = cov_shares(X, y)
        rho = boot(lambda idx: spearman_cols(X[idx], y[idx]), len(y))
        for j, part in enumerate(["total"] + PARTS4):
            rows.append({"measure": name, "part": part, "share": shares[0][j], "lo": shares[1][j],
                         "hi": shares[2][j], "rho": rho[0][j], "rho_lo": rho[1][j], "rho_hi": rho[2][j]})
        group_means = units.groupby("group")[["total"] + PARTS4].mean()
        group_means["n_units"] = units.groupby("group").size()
        group_means.loc["learners"] = list(rated[["total"] + PARTS4].mean()) + [len(rated)]
        means.append(group_means.reset_index().assign(measure=name))
        if base_corpus(corpus) == "wi":  # LOCNESS minus C, with independent bootstraps of the two groups
            native, level_c = units[units.group == "N"], units[units.group == "C"]
            rng = np.random.default_rng(SEED)
            boot_native = rng.integers(0, len(native), (N_BOOT, len(native)))
            boot_c = rng.integers(0, len(level_c), (N_BOOT, len(level_c)))
            native_minus_c[name] = {}
            for part in ["total"] + PARTS4:
                n_vals, c_vals = native[part].to_numpy(), level_c[part].to_numpy()
                boots = n_vals[boot_native].mean(axis=1) - c_vals[boot_c].mean(axis=1)
                native_minus_c[name][part] = [float(n_vals.mean() - c_vals.mean()),
                                              *map(float, np.percentile(boots, [2.5, 97.5]))]
        if name == "mean_surprisal":
            mean_units = units
    pd.DataFrame(rows).to_csv(RESULTS / f"split4_{corpus}.csv", index=False)
    pd.concat(means).to_csv(RESULTS / f"split4_means_{corpus}.csv", index=False)
    return mean_units, native_minus_c


# --------------------------------------------------------------------------- #
# split by kind of correction
# --------------------------------------------------------------------------- #
def credits(tokens, edits):
    """Long table: unit, text_id, k (edit; -1 residual, -2 flagged), comp, val, cnt.

    Error-token rows carry val = s_orig and cnt = 1, so that the unit's clean
    mean is subtracted once per token when the table is summed.
    """
    rows = []
    # tokens inside an error: at site of their correction, else residual (own) or flagged
    err = tokens[~tokens.own_aligned]
    edit_k = np.where(err.site_k >= 0, err.site_k, np.where(err.own, -1, -2))
    rows += list(zip(err.unit, err.text_id, edit_k, ["at_site"] * len(err), err.surprisal_bits, [1.0] * len(err)))
    own = tokens[tokens.own_aligned]
    for tok in own.itertuples():
        delta = tok.surprisal_bits - tok.s_corr
        if len(tok.gap_k):  # a word is missing right before it: Δ is the missing word's cost, at site
            rows += [(tok.unit, tok.text_id, g, "at_site", delta / len(tok.gap_k), 0.0) for g in tok.gap_k]
        elif tok.status == "ok":  # Shapley split, shifted so that the parts sum to the stored Δ
            shift = (delta - (tok.v_none - tok.v_all)) / len(tok.psi)
            rows += [(tok.unit, tok.text_id, k, "spill", psi + shift, 0.0) for k, psi in zip(tok.edit_k, tok.psi)]
        elif len(tok.edit_k):  # too many players: equal split
            rows += [(tok.unit, tok.text_id, k, "spill", delta / len(tok.edit_k), 0.0) for k in tok.edit_k]
        else:  # no edit in the window
            rows.append((tok.unit, tok.text_id, -1, "spill", delta, 0.0))
    credit = pd.DataFrame(rows, columns=["unit", "text_id", "k", "comp", "val", "cnt"])
    credit = credit.merge(edits[["text_id", "k", "group", "bucket"]], on=["text_id", "k"], how="left")
    for k, label in [(-1, "residual"), (-2, "flagged")]:
        credit.loc[credit.k == k, ["group", "bucket"]] = label
    assert credit.group.notna().all()
    return credit


def unit_table(tokens, credit):
    """Per unit: total, clean, and every <category>|<at_site, spill, total> part, per token."""
    by_unit = tokens.groupby("unit")
    units = pd.DataFrame({"group": by_unit.group.first(), "y": by_unit.y.first(), "n": by_unit.size(),
                          "total": by_unit.surprisal_bits.mean()})
    units["clean"] = tokens[tokens.own_aligned].groupby("unit").s_corr.mean()
    credit = credit.join(units.clean, on="unit")
    credit["x"] = credit.val - credit.cnt * credit.clean  # error tokens: excess over clean
    for by, cats in CATS.items():
        wide = credit.groupby(["unit", by, "comp"]).x.sum().unstack([1, 2])
        for cat in cats:
            for comp in COMP:
                col = f"{cat}|{comp}"
                units[col] = wide.get((cat, comp), pd.Series(0.0, index=units.index)).reindex(units.index).fillna(0.0)
                units[col] /= units.n
            units[f"{cat}|total"] = units[f"{cat}|at_site"] + units[f"{cat}|spill"]
        gap = (units.clean + units[[f"{x}|total" for x in cats]].sum(axis=1) - units.total).abs().max()
        assert gap < 1e-9, f"{by} split fails by {gap}"
    return units


def edit_counts(units, edits, unit_of_text):
    """Per unit: the number of corrections of each bucket and group, and of flagged spans."""
    edits = edits.assign(unit=edits.text_id.map(unit_of_text))
    for by, cats in CATS.items():
        counts = edits[edits.kind != "flagged"].groupby(["unit", by]).size().unstack()
        for cat in cats[:-2]:
            units[f"n_{cat}"] = counts.get(cat, pd.Series(dtype=float)).reindex(units.index).fillna(0.0)
    units["n_flagged"] = edits[edits.kind == "flagged"].groupby("unit").size().reindex(units.index).fillna(0.0)
    units["n_edits"] = units[[f"n_{b}" for b in BUCKETS + [UNCLASSIFIED]]].sum(axis=1)
    return units


def shares(units):
    """Each part's share of cov(total, criterion), by bucket and by group, and the edit-rate reference."""
    rated, y = learners(units)
    rows = []
    for by, cats in CATS.items():
        cols = ["total", "clean"] + [f"{c}|{p}" for c in cats for p in COMP + ["total"]]
        sh = cov_shares(rated[cols].to_numpy(float), y)
        rows += [{"cut": by, "col": c, "share": sh[0][j], "lo": sh[1][j], "hi": sh[2][j]}
                 for j, c in enumerate(cols) if c != "total"]
    # reference: each bucket's share of cov(edit rate, criterion), and of all edits
    buckets = BUCKETS + [UNCLASSIFIED]
    rates = rated[[f"n_{b}" for b in buckets]].div(rated.n, axis=0)
    sh = cov_shares(np.column_stack([rates.sum(axis=1), rates]), y)
    n_edits = rated[[f"n_{b}" for b in buckets]].sum()
    rows += [{"cut": "edit_rate", "col": b, "share": sh[0][j + 1], "lo": sh[1][j + 1], "hi": sh[2][j + 1],
              "share_of_edits": n_edits.iloc[j] / n_edits.sum()} for j, b in enumerate(buckets)]
    return pd.DataFrame(rows)


def rate_cost(units, corpus):
    """Errors per 100 tokens and bits per error, token-pooled within level.

    pooled part_L = sum(part * n) / sum(n) = rate_L * cost_L exactly. The
    lowest-minus-highest difference splits into rate and cost by the midpoint
    rule d(rc) = dr * mean(c) + mean(r) * dc. CIs by resampling units within level.
    rate_all and cost_all pool all learners.
    """
    rated, _ = learners(units)
    levels = bands(corpus)
    rows_of = {g: np.flatnonzero((rated.group == g).to_numpy()) for g in levels}
    n = rated.n.to_numpy(float)
    rng = np.random.default_rng(SEED)
    boots = [{g: rng.choice(rows_of[g], len(rows_of[g])) for g in levels} for _ in range(N_BOOT)]
    rows = []
    for by, cats in CATS.items():
        for cat in cats[:-2]:
            n_err = rated[f"n_{cat}"].to_numpy(float)
            for comp in COMP + ["total"]:
                bits = rated[f"{cat}|{comp}"].to_numpy(float) * n  # the unit's summed bits of this part

                def stats(sel):
                    rate = np.array([n_err[sel[g]].sum() / n[sel[g]].sum() for g in levels])
                    cost = np.array([bits[sel[g]].sum() / max(n_err[sel[g]].sum(), 1e-12) for g in levels])
                    diff = rate[0] * cost[0] - rate[-1] * cost[-1]  # lowest minus highest level
                    rate_share = (rate[0] - rate[-1]) * (cost[0] + cost[-1]) / 2 / diff
                    return rate, cost, rate_share, cost[-1] - cost[0]

                rate, cost, rate_share, cost_change = stats(rows_of)
                boot_stats = [stats(sel) for sel in boots]
                row = {"cut": by, "cat": cat, "comp": comp, **{f"rate_{g}": 100 * v for g, v in zip(levels, rate)},
                       **{f"cost_{g}": v for g, v in zip(levels, cost)}, "rate_share": rate_share,
                       "cost_change": cost_change, "rate_all": 100 * n_err.sum() / n.sum(),
                       "cost_all": bits.sum() / n_err.sum()}
                row["rate_share_lo"], row["rate_share_hi"] = np.nanpercentile([b[2] for b in boot_stats], [2.5, 97.5])
                row["cost_change_lo"], row["cost_change_hi"] = np.percentile([b[3] for b in boot_stats], [2.5, 97.5])
                rows.append(row)
    return pd.DataFrame(rows).assign(low=levels[0], high=levels[-1])


def spill_decay(tokens, edits):
    """Mean spill per (edit, own word) pair by distance in words (5 = 5 or more)."""
    own = tokens[tokens.own_aligned & (tokens.status == "ok")]
    own = own[own.gap_k.map(len) == 0]
    n_players = own.psi.map(len)
    shift = (own.surprisal_bits - own.s_corr - (own.v_none - own.v_all)) / n_players
    # one row per (own word, player) pair
    pairs = pd.DataFrame({"unit": np.repeat(own.unit.to_numpy(), n_players),
                          "text_id": np.repeat(own.text_id.to_numpy(), n_players),
                          "k": np.concatenate(own.edit_k.to_numpy()),
                          "d": np.clip(np.concatenate(own.dist.to_numpy()), 1, 5),
                          "psi": np.concatenate(own.psi.to_numpy()) + np.repeat(shift.to_numpy(), n_players)})
    pairs = pairs.merge(edits[["text_id", "k", "bucket"]], on=["text_id", "k"])
    rows = []
    for bucket in ["all"] + BUCKETS:
        sel = pairs if bucket == "all" else pairs[pairs.bucket == bucket]
        by_unit = sel.groupby(["unit", "d"]).psi.agg(["sum", "size"]).unstack(fill_value=0)
        sums, sizes = by_unit["sum"].to_numpy(float), by_unit["size"].to_numpy(float)
        point, lo, hi = boot(lambda idx: sums[idx].sum(0) / sizes[idx].sum(0), len(sums))
        rows += [{"bucket": bucket, "d": d, "n_pairs": int(sizes[:, j].sum()), "mean_spill": point[j],
                  "lo": lo[j], "hi": hi[j]} for j, d in enumerate(by_unit["sum"].columns)]
    return pd.DataFrame(rows)


def clean_assoc(units, corpus):
    """The clean part against the criterion: rho, and partial rho given the
    plain share of corrected tokens (ERR_TOK or ERR_INS)."""
    rated, y = learners(units)
    rows = []
    for measure in ["mean_surprisal", "high_per_1000"]:
        clean = rated[f"{measure}|clean"].to_numpy(float)
        corr_share = rated.corr_share.to_numpy(float)
        total = rated[f"{measure}|total"].to_numpy(float)

        def stat(idx):
            return [spearman_cols(total[idx, None], y[idx])[0], spearman_cols(clean[idx, None], y[idx])[0],
                    partial_rank(clean[idx], y[idx], corr_share[idx])]
        point, lo, hi = boot(stat, len(y))
        for j, name in enumerate(["rho_total", "rho_clean", "partial_clean"]):
            rows.append({"corpus": corpus, "measure": measure, "stat": name, "value": point[j], "lo": lo[j],
                         "hi": hi[j], "n": len(y)})
    return rows


def rq3_features(units):
    """Error counts (B) and the compact surprisal terms (S) per unit."""
    for g, slug in zip(GROUPS, GROUP_SLUGS):
        units[f"rate_{slug}"] = units[f"n_{g}"] / units.n
    units["rate_flagged"] = units.n_flagged / units.n
    units["log_n"] = np.log(units.n)
    n_err = units[[f"n_{g}" for g in GROUPS]].sum(axis=1)
    at_site = sum(units[f"{g}|at_site"] for g in GROUPS) * units.n
    spill = sum(units[f"{g}|spill"] for g in GROUPS) * units.n
    units["cost_mean"] = (at_site / n_err).where(n_err > 0)
    units["spill_mean"] = (spill / n_err).where(n_err > 0)
    return units


def run(corpus):
    tokens, edits = load(corpus)
    units4, native_minus_c = run_split4(tokens, corpus)
    units = unit_table(tokens, credits(tokens, edits))
    assert np.allclose(units.clean, units4.clean.reindex(units.index))
    assert np.allclose(units.total, units4.total.reindex(units.index))
    unit_of_text = tokens.groupby("text_id").unit.first()
    units = edit_counts(units, edits, unit_of_text)
    units["corr_share"] = tokens.gold.isin(["ERR_TOK", "ERR_INS"]).groupby(tokens.unit).mean().reindex(units.index)
    t = tau()
    own = tokens[tokens.own_aligned]
    units["mean_surprisal|total"], units["mean_surprisal|clean"] = units.total, units.clean
    units["high_per_1000|total"] = (tokens.surprisal_bits > t).groupby(tokens.unit).mean() * 1000
    units["high_per_1000|clean"] = (own.s_corr > t).groupby(own.unit).mean() * 1000
    units = rq3_features(units)
    units.to_parquet(data_dir(corpus) / "units.parquet")

    shares(units).to_csv(RESULTS / f"shares_{corpus}.csv", index=False)
    cols = list(dict.fromkeys(["total", "clean"] + [f"{c}|{p}" for by in CATS for c in CATS[by]
                                                    for p in COMP + ["total"]]))
    means = units.groupby("group")[cols].mean().T
    means["learners"] = learners(units)[0][cols].mean()
    means.to_csv(RESULTS / f"means_{corpus}.csv")
    rate_cost(units, corpus).to_csv(RESULTS / f"rate_cost_{corpus}.csv", index=False)
    spill_decay(tokens, edits).to_csv(RESULTS / f"spill_decay_{corpus}.csv", index=False)
    rated = learners(units)[0]
    rated_edits = edits[edits.text_id.map(unit_of_text).isin(rated.index)]
    corrections = rated_edits[rated_edits.kind != "flagged"]
    checks = {"n_units": int(len(rated)), "n_by_group": units.groupby("group").size().to_dict(),
              "n_texts": int(tokens[tokens.y.notna()].text_id.nunique()), "n_tokens": int(tokens.y.notna().sum()),
              "n_corrections": len(corrections), "n_flagged": int((rated_edits.kind == "flagged").sum()),
              "unmapped_share": float((corrections.errant == "unmapped").mean()),
              "spill_status": own.status.value_counts().to_dict(),
              "own_words_with_gap": int((own.gap_k.map(len) > 0).sum()),
              "v_none_vs_stored_median_abs": float((own.v_none - own.surprisal_bits).abs().median()),
              "v_all_vs_s_corr_median_abs": float((own.v_all - own.s_corr).abs().median()),
              "native_minus_C": native_minus_c}
    (RESULTS / f"decompose_{corpus}.json").write_text(json.dumps(checks, indent=2, default=float))
    print(corpus, json.dumps(checks, default=float), flush=True)
    return clean_assoc(units, corpus)


VALIDATION_SOURCES = {"gold": "", "errant": "_errant", "model": "_model"}
VALIDATION_DIFFS = [("model", "errant"), ("model", "gold"), ("errant", "gold")]


def validation():
    """Shares by source (json gold, ERRANT re-cut of gold, model) and their paired differences.

    The original texts, so the totals and the criterion, are the same for the
    three sources; only the split differs. Every share and difference comes
    from the same 2,000 unit resamples. "errors" is 1 - clean.
    """
    rows = []
    for corpus in CORPORA:
        units = {src: learners(pd.read_parquet(data_dir(corpus + suffix) / "units.parquet"))[0]
                 for src, suffix in VALIDATION_SOURCES.items()}
        gold = units["gold"]
        y = gold.y.to_numpy(float)
        assert all(u.index.equals(gold.index) and np.allclose(u.total, gold.total) for u in units.values())
        cols = (["clean"] + [f"{c}|total" for c in BUCKETS + [UNCLASSIFIED]]
                + [f"{c}|total" for c in GROUPS + ["flagged", "residual"]])
        # the shared total, then one block of parts per source
        X = np.column_stack([gold.total] + [units[src][cols].to_numpy(float) for src in VALIDATION_SOURCES])
        start = {src: 1 + i * len(cols) for i, src in enumerate(VALIDATION_SOURCES)}

        def shares(idx):
            cov = ((X[idx] - X[idx].mean(0)) * (y[idx] - y[idx].mean())[:, None]).mean(0)
            return cov / cov[0]
        rng = np.random.default_rng(SEED)
        point = shares(np.arange(len(y)))
        boots = np.array([shares(rng.integers(0, len(y), len(y))) for _ in range(N_BOOT)])
        for j, col in enumerate(cols):
            name = col.removesuffix("|total")
            # the clean row is reported as the errors' share, 1 - clean
            flip = (lambda v: 1 - v) if name == "clean" else (lambda v: v)
            row = {"corpus": corpus, "col": "errors" if name == "clean" else name}
            for src, k in start.items():
                lo, hi = sorted(flip(np.percentile(boots[:, k + j], [2.5, 97.5])))
                row.update({src: flip(point[k + j]), f"{src}_lo": lo, f"{src}_hi": hi})
            for a, b in VALIDATION_DIFFS:
                diff = flip(boots[:, start[a] + j]) - flip(boots[:, start[b] + j])
                lo, hi = np.percentile(diff, [2.5, 97.5])
                row.update({f"{a}-{b}": row[a] - row[b], f"{a}-{b}_lo": lo, f"{a}-{b}_hi": hi})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "validation_shares.csv", index=False)
    print(table[["corpus", "col", "gold", "errant", "model", "model-errant", "model-errant_lo", "model-errant_hi",
                 "model-gold", "model-gold_lo", "model-gold_hi"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", choices=ALL_IDS, default=CORPORA,
                    help="corpus ids (default: the main analysis, wi and fce)")
    ap.add_argument("--validate", action="store_true", help="the corrector check across sources only")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.validate:
        validation()
        return
    if args.corpus == CORPORA:
        rows = [r for c in CORPORA for r in run(c)]
        pd.DataFrame(rows).to_csv(RESULTS / "clean_assoc.csv", index=False)
        return
    for c in args.corpus:
        pd.DataFrame(run(c)).to_csv(RESULTS / f"clean_assoc_{c}.csv", index=False)


if __name__ == "__main__":
    main()
