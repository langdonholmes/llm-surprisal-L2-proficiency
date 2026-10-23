"""Study 3, stage 10 (CPU) — RQ5: where TOEFL 11's L1 displacement sits (DECISIONS D47, D52).

Study 1's fairness statistic, split over the parts of stage 5. For L1 g in
band s, against the other L1s in s:
    d_gs(part) = (mean_gs(part) - mean_rest,s(part)) / sp_s
    d_g(part)  = sum_s n_gs d_gs(part) / sum_s n_gs
sp_s is the pooled within-band SD of the *total*, the same for every part, so
the parts' d values add up to Study 1's d_g. Bands where g or the rest has
fewer than two units are skipped, as in Study 1. "bits" is the same weighted
mean difference, unstandardized.

Slope (Study 1's within-L1 slope, proficiency-confound-sensitivity.qmd): each
L1's band means regressed on the latent band positions θ̄_s (truncated-normal
means between the cutpoints of the pooled band shares), weighted by cell size;
bits per θ SD, and pooled over L1s by size. Linear in the part, so it splits too.

Residual proficiency (Cochran 1968): three bands leave about 21% of a normal
confounder's bias (35% as a sensitivity check; D49). allowance = r × gap_g ×
β_part, with gap_g g's latent proficiency gap (a probit with a scale term;
saturated with three bands, so in closed form) and β_part the pooled slope.

Margins (D49): the FCE distortion of each part (model d minus re-cut gold d)
bounds how far the corrector could have moved it. They are never subtracted.
A shared L1's part clears its margin if d_lo - max(hi, 0) > 0 or
d_hi + max(-lo, 0) < 0, with [lo, hi] the distortion's CI. Arabic, Hindi and
Telugu get the symmetric margin m = the largest |distortion| over the shared L1s.

Per band, balanced (Study 1): within each band every L1 is subsampled to the
smallest cell, 200 draws, mean d and the 2.5-97.5% spread across draws.

CIs: 2,000 resamples of units stratified by band (Study 1), as resampling
weights, so every statistic of one iteration uses the same draw.

    --check   step 1, the corrector by L1 on FCE: agreement (ERRANT P/R/F0.5,
              corrections per 100 tokens) per L1, and the distortion of each
              shared L1's errors-part d, model minus the re-cut gold, paired; the
              same distortion for every part (D49; a group only where the L1 has
              at least 50 corrections of it under each source).
              -> results/study3/validation/l1_check.csv (per L1), l1_split4.csv and
                 l1_distortion.csv (per L1 and part)
    --tokenization  which gold corrections TOEFL 11's tokenization would erase
              (D50): an edit is erased if the original and corrected spans, with a
              character of context, are the same once every split-off mark has one
              space each side and runs of whitespace are one space. On the
              re-cut gold of W&I (typed, like TOEFL 11) and FCE (transcribed).
              -> results/study3/validation/tokenization_erased.csv (per corpus,
                 bucket and group)
    --followup  D52 steps 1a and 2 (after 11_corrected.py): the d of S_corr, sel and cost
              beside D51's parts on D51's draws, their FCE margins, per band and balanced; and
              every part's d with each part centred on its (band, prompt) mean, re-estimated in
              each draw; and D52's reading rules.
              -> results/study3/toefl/l1_corr_*.csv, l1_prompt_*.csv
    --types   D52 step 3 for the L1s that pass: clean d split exactly over word types (each
              own token weighted 1 / (|O| × cell size)), frequency vs cost by the midpoint rule.
              -> results/study3/toefl/l1_clean_types.csv, l1_clean_type_examples.csv (local only)
    (default) steps 2-5 on toefl: acceptance against Study 1, displacement split,
              per band, slopes, gaps and allowances.
              -> results/study3/toefl/l1_*.csv, l1_check.json

    python src/3-surprisal-decomposition/10_l1.py --check
    python src/3-surprisal-decomposition/10_l1.py --tokenization
    python src/3-surprisal-decomposition/10_l1.py
    python src/3-surprisal-decomposition/10_l1.py --followup
    python src/3-surprisal-decomposition/10_l1.py --types
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, truncnorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import (  # noqa: E402
    DATA, N_BOOT, RAW_DIRS, RESULTS, SEED, TOEFL_BANDS, bands, data_dir, results_dir)
from lib.errant import BUCKETS, GROUPS, UNCLASSIFIED  # noqa: E402

from util.paths import RESULTS_DIR  # noqa: E402

FLAG = 0.10  # Study 1's |d| flag (Williamson et al. 2012)
COCHRAN, COCHRAN_CONSERVATIVE = 0.21, 0.35  # primary and sensitivity (D49)
MIN_GROUP_CORRECTIONS = 50  # per L1 and source, for a group's FCE distortion (D49)
K_DRAWS = 200  # balanced subsamples per band (Study 1)
SHARED_L1 = {"es": "SPA", "fr": "FRE", "ko": "KOR", "ja": "JPN", "it": "ITA", "de": "GER", "tr": "TUR", "zh": "CHI"}
SPLIT4 = ["clean", "omission", "context", "error_tokens"]
CATS = {"bucket": BUCKETS + [UNCLASSIFIED, "flagged", "residual"], "group": GROUPS + ["flagged", "residual"]}


def part_columns(units):
    """total first, then clean, errors (= total - clean), the four-part split, and every
    bucket and group part (at site, spill, total) that is not zero throughout."""
    units = units.assign(errors=units.total - units.clean)
    cols = ["total", "clean", "errors"] + SPLIT4[1:]
    cols += [f"{c}|{p}" for cats in CATS.values() for c in cats for p in ["at_site", "spill", "total"]]
    cols = list(dict.fromkeys(c for c in cols if units[c].abs().sum() > 0 or c in ("total", "clean", "errors")))
    return units, cols


# --------------------------------------------------------------------------- #
# cell sums, and the statistics computed from them
# --------------------------------------------------------------------------- #
class Cells:
    """Weighted sums of every part per (band, L1) cell, for any resampling weights."""

    def __init__(self, X, band, l1, band_levels, l1_levels):
        self.X = np.asarray(X, float)
        self.n_band, self.n_l1 = len(band_levels), len(l1_levels)
        b = pd.Categorical(band, band_levels).codes
        g = pd.Categorical(l1, l1_levels).codes
        assert (b >= 0).all() and (g >= 0).all()
        cell = b * self.n_l1 + g
        self.onehot = sparse.csr_matrix((np.ones(len(cell)), (cell, np.arange(len(cell)))),
                                        shape=(self.n_band * self.n_l1, len(cell)))
        self.band_rows = [np.flatnonzero(b == s) for s in range(self.n_band)]

    def sums(self, w):
        """n, sum of each part, and sum of the squared total, per cell: (S, G), (S, G, P), (S, G)."""
        shape = (self.n_band, self.n_l1)
        n = (self.onehot @ w).reshape(shape)
        s = (self.onehot @ (w[:, None] * self.X)).reshape(*shape, -1)
        q = (self.onehot @ (w * self.X[:, 0] ** 2)).reshape(shape)
        return n, s, q

    def weights(self, rng=None):
        """All ones, or a bootstrap draw stratified by band (counts per unit)."""
        if rng is None:
            return np.ones(len(self.X))
        w = np.zeros(len(self.X))
        for rows in self.band_rows:
            np.add.at(w, rng.choice(rows, len(rows)), 1.0)
        return w


def displacement(n, s, q):
    """Study 1's d per L1 for every part, per band and size-weighted; and the same in bits.
    Returns d (G, P), bits (G, P), d per band (S, G, P) with NaN in skipped bands."""
    n_rest = n.sum(1, keepdims=True) - n
    s_rest = s.sum(1, keepdims=True) - s
    q_rest = q.sum(1, keepdims=True) - q
    with np.errstate(invalid="ignore", divide="ignore"):
        m_focal, m_rest = s / n[..., None], s_rest / n_rest[..., None]
        # pooled within-band SD of the total: (SS_focal + SS_rest) / (n_focal + n_rest - 2)
        ss = (q - n * m_focal[..., 0] ** 2) + (q_rest - n_rest * m_rest[..., 0] ** 2)
        sp = np.sqrt(ss / (n + n_rest - 2))
        diff = m_focal - m_rest
        d_band = diff / sp[..., None]
    use = (n >= 2) & (n_rest >= 2)
    d_band[~use] = np.nan
    w = np.where(use, n, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):  # an L1 with no usable band gives NaN
        d = np.einsum("sg,sgp->gp", w, np.nan_to_num(d_band)) / w.sum(0)[:, None]
        bits = np.einsum("sg,sgp->gp", w, np.where(use[..., None], diff, 0.0)) / w.sum(0)[:, None]
    return d, bits, d_band


def theta_bar(band_n):
    """Latent band positions: the truncated-normal means between the cutpoints of the band shares."""
    cuts = norm.ppf(np.concatenate([[0], np.cumsum(band_n) / band_n.sum()]))
    return np.array([truncnorm.mean(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])])


def slopes(n, s):
    """Each L1's slope of band means on θ̄ (bits per θ SD), weighted by cell size, and their
    size-weighted pooled value: (G, P), (P,)."""
    theta = theta_bar(n.sum(1))
    use = n >= 2
    w = np.where(use, n, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(use[..., None], s / n[..., None], 0.0)
        x_mean = (w * theta[:, None]).sum(0) / w.sum(0)
        y_mean = np.einsum("sg,sgp->gp", w, y) / w.sum(0)[:, None]
        dx = theta[:, None] - x_mean
        beta = np.einsum("sg,sgp->gp", w * dx, y - y_mean) / (w * dx ** 2).sum(0)[:, None]
    ok = use.sum(0) >= 2
    n_l1 = n.sum(0)
    pooled = (n_l1[ok, None] * beta[ok]).sum(0) / n_l1[ok].sum()
    beta[~ok] = np.nan
    return beta, pooled


def latent_gap(n):
    """Each L1's latent proficiency location against the rest (θ SD; probit with a scale term).
    With three ordered bands and two groups the model is saturated: the rest's cutpoints are
    Φ⁻¹ of its cumulative shares, and the focal group's location and scale solve the two cutpoints."""
    rest = n.sum(1, keepdims=True) - n
    cum_rest = np.cumsum(rest, 0)[:-1] / rest.sum(0)
    cum_focal = np.cumsum(n, 0)[:-1] / n.sum(0)
    c, z = norm.ppf(cum_rest), norm.ppf(cum_focal)
    scale = (c[1] - c[0]) / (z[1] - z[0])
    return c[0] - scale * z[0], scale  # P(band ≤ j) = Φ((c_j - gap) / scale)


def boot_ci(point, boots):
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


def balanced_bands(X, band, l1, l1s, rng):
    """Per band: d of every L1 and part with every L1 subsampled to the band's smallest cell.
    Mean over K_DRAWS draws and the 2.5-97.5% spread: three (S, G, P) arrays, and the cell size per band."""
    X, band, l1 = np.asarray(X, float), np.asarray(band), np.asarray(l1)
    mean, lo, hi, n_min = [], [], [], []
    for b in TOEFL_BANDS:
        rows_of = [np.flatnonzero((band == b) & (l1 == g)) for g in l1s]
        n_min.append(min(len(r) for r in rows_of))
        draws = []
        for _ in range(K_DRAWS):
            idx = np.concatenate([rng.choice(r, n_min[-1], replace=False) for r in rows_of])
            cells = Cells(X[idx], band[idx], l1[idx], [b], l1s)
            draws.append(displacement(*cells.sums(np.ones(len(idx))))[2][0])
        draws = np.array(draws)
        mean.append(draws.mean(0))
        lo.append(np.percentile(draws, 2.5, axis=0))
        hi.append(np.percentile(draws, 97.5, axis=0))
    return np.array(mean), np.array(lo), np.array(hi), n_min


def margins(path=None):
    """The FCE distortion of each part, as a margin per TOEFL L1 (D49): (l1, part) -> (lo, hi),
    the most the corrector could have moved d down and up. Shared L1s: from the distortion's CI,
    zero always allowed; the other L1s: ± the largest |distortion| over the shared L1s."""
    dist = pd.read_csv(path or RESULTS / "validation" / "l1_distortion.csv")
    dist = dist[dist.has_margin]
    out = {(SHARED_L1[r.l1], r.part): (max(r.hi, 0.0), max(-r.lo, 0.0)) for r in dist.itertuples()}
    for part, m in dist.groupby("part").distortion.apply(lambda v: v.abs().max()).items():
        for g in ["ARA", "HIN", "TEL"]:
            out[(g, part)] = (m, m)
    return out


# --------------------------------------------------------------------------- #
# TOEFL 11 (steps 2-5)
# --------------------------------------------------------------------------- #
def load_toefl():
    units = pd.read_parquet(data_dir("toefl") / "units.parquet")
    essays = pd.read_parquet(data_dir("toefl") / "essays.parquet", columns=["unit", "l1", "prompt"])
    units = units.join(essays.set_index("unit"))
    assert units.l1.notna().all() and len(units) == 11000
    return units


def acceptance(units, out):
    """The stage 5 total against Study 1's olmo2-1b/8 essay means, and Study 1's d reproduced."""
    study1 = pd.read_csv(RESULTS_DIR / "predictability" / "toefl_long.csv.gz")
    study1 = study1[(study1.model == "olmo2-1b") & (study1.window.astype(str) == "8")].set_index("text_id")
    both = units.join(study1.mean_surprisal_bits, how="inner")
    gap = (both.total - both.mean_surprisal_bits).abs()
    l1s = sorted(units.l1.unique())
    d_ours = displacement(*Cells(units[["total"]], units.group, units.l1, TOEFL_BANDS, l1s).sums(np.ones(len(units))))[0][:, 0]
    s1 = study1.loc[units.index]
    d_s1 = displacement(*Cells(s1[["mean_surprisal_bits"]], s1.level, s1.l1, TOEFL_BANDS, l1s).sums(np.ones(len(s1))))[0][:, 0]
    check = {"n_units": len(units), "n_matched": len(both),
             "total_vs_study1_r": float(np.corrcoef(both.total, both.mean_surprisal_bits)[0, 1]),
             "total_vs_study1_max_abs": float(gap.max()), "total_vs_study1_median_abs": float(gap.median()),
             "d": {g: {"stage5": float(a), "study1": float(b)} for g, a, b in zip(l1s, d_ours, d_s1)},
             "d_max_abs_diff": float(np.abs(d_ours - d_s1).max()),
             "aligned": json.loads((results_dir("toefl") / "align.json").read_text())
             if (results_dir("toefl") / "align.json").exists() else None}
    (out / "l1_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps({k: v for k, v in check.items() if k != "d"}, indent=2))


def run_toefl():
    out = results_dir("toefl")
    units, cols = part_columns(load_toefl())
    acceptance(units, out)
    l1s = sorted(units.l1.unique())
    cells = Cells(units[cols], units.group, units.l1, TOEFL_BANDS, l1s)

    def stats(w):
        n, s, q = cells.sums(w)
        d, bits, d_band = displacement(n, s, q)
        beta, pooled = slopes(n, s)
        gap, scale = latent_gap(n)
        allow = gap[:, None] * pooled[None, :]  # bits a whole proficiency gap would give, per part
        return d, bits, d_band, beta, pooled, gap, scale, allow

    point = stats(cells.weights())
    rng = np.random.default_rng(SEED)
    boots = [stats(cells.weights(rng)) for _ in range(N_BOOT)]
    d, bits, d_band, beta, pooled, gap, scale, allow = (boot_ci(p, np.array([b[i] for b in boots]))
                                                        for i, p in enumerate(point))

    margin = margins()

    def within_reach(observed, allowance):
        return bool(np.sign(observed) == np.sign(allowance) and abs(observed) <= abs(allowance))

    rows = []
    for gi, g in enumerate(l1s):
        for pj, part in enumerate(cols):
            a, ab = allow[0][gi, pj], bits[0][gi, pj]
            d_lo, d_hi = d[1][gi, pj], d[2][gi, pj]
            # how far the corrector could have moved d up (too high) and down (too low); none for the total
            up, down = (0.0, 0.0) if part == "total" else margin.get((g, part), (np.nan, np.nan))
            clears = None if np.isnan(up) else bool(d_lo - up > 0 or d_hi + down < 0)
            rows.append({"l1": g, "part": part, "d": d[0][gi, pj], "d_lo": d_lo, "d_hi": d_hi,
                         "ci_excludes_0": bool(d_lo > 0 or d_hi < 0), "margin_up": up, "margin_down": down,
                         "clears_margin": clears,
                         "bits": ab, "bits_lo": bits[1][gi, pj], "bits_hi": bits[2][gi, pj],
                         "allowance": COCHRAN * a, "allowance_35": COCHRAN_CONSERVATIVE * a,
                         "residual_proficiency": "within reach" if within_reach(ab, COCHRAN * a) else "beyond",
                         "residual_proficiency_35":
                             "within reach" if within_reach(ab, COCHRAN_CONSERVATIVE * a) else "beyond",
                         "flagged": abs(d[0][gi, 0]) >= FLAG})
    pd.DataFrame(rows).to_csv(out / "l1_displacement.csv", index=False)

    rows = [{"l1": g, "band": b, "part": part, "d": d_band[0][si, gi, pj], "lo": d_band[1][si, gi, pj],
             "hi": d_band[2][si, gi, pj], "n": int(((units.l1 == g) & (units.group == b)).sum())}
            for si, b in enumerate(TOEFL_BANDS) for gi, g in enumerate(l1s) for pj, part in enumerate(cols)]
    pd.DataFrame(rows).to_csv(out / "l1_displacement_band.csv", index=False)
    mean, lo, hi, n_min = balanced_bands(units[cols], units.group, units.l1, l1s, np.random.default_rng(SEED))
    rows = [{"l1": g, "band": b, "part": part, "d": mean[si, gi, pj], "lo": lo[si, gi, pj], "hi": hi[si, gi, pj],
             "n_per_l1": n_min[si]}
            for si, b in enumerate(TOEFL_BANDS) for gi, g in enumerate(l1s) for pj, part in enumerate(cols)]
    pd.DataFrame(rows).to_csv(out / "l1_displacement_band_balanced.csv", index=False)

    rows = [{"l1": g, "part": part, "slope": beta[0][gi, pj], "lo": beta[1][gi, pj], "hi": beta[2][gi, pj]}
            for gi, g in enumerate(l1s) for pj, part in enumerate(cols)]
    rows += [{"l1": "pooled", "part": part, "slope": pooled[0][pj], "lo": pooled[1][pj], "hi": pooled[2][pj]}
             for pj, part in enumerate(cols)]
    pd.DataFrame(rows).to_csv(out / "l1_slopes.csv", index=False)

    pd.DataFrame({"l1": l1s, "gap": gap[0], "gap_lo": gap[1], "gap_hi": gap[2], "scale": scale[0],
                  "n": [int((units.l1 == g).sum()) for g in l1s]}).to_csv(out / "l1_gap.csv", index=False)
    disp = pd.read_csv(out / "l1_displacement.csv")
    print(disp[disp.part.isin(["total", "clean", "errors"])].pivot(index="l1", columns="part", values="d")
          .round(3).to_string())


# --------------------------------------------------------------------------- #
# follow-up (D52): the corrected essay as a whole, and the prompt
# --------------------------------------------------------------------------- #
CORR_PARTS = ["s_corr", "sel", "cost", "sel_inserted", "sel_replacement", "sel_carried"]
PROMPT_PARTS = ["total", "clean", "errors", "s_corr", "sel", "cost"] + [f"{b}|total" for b in BUCKETS] + \
               [f"{UNCLASSIFIED}|total", "residual|total"]
ROBUST_L1 = ["FRE", "GER", "HIN", "KOR", "SPA", "TEL"]  # flagged in D51
OFFSET_L1 = ["ARA", "TUR"]


def with_corrected(units, corpus):
    corr = pd.read_parquet(data_dir(corpus) / "corrected_units.parquet").reindex(units.index)
    assert np.allclose(corr.total, units.total)
    return units.join(corr[CORR_PARTS + ["c_share", "c_excess"]])


def fce_corr_distortion():
    """D49's FCE distortion (model d minus re-cut gold d, paired) for the corrected-essay parts,
    with D48's resamples (same units, order and generator)."""
    model = pd.read_parquet(data_dir("fce_model") / "units.parquet")
    errant = pd.read_parquet(data_dir("fce_errant") / "units.parquet")
    model, errant = model[model.y.notna()], errant[errant.y.notna()]
    model, errant = with_corrected(model, "fce_model"), with_corrected(errant, "fce_errant")
    l1 = fce_l1().reindex(model.index)
    l1s = sorted(l1.unique())
    cols = ["total"] + CORR_PARTS
    k = len(cols)
    cells = Cells(np.column_stack([model[cols], errant[cols]]), model.group, l1, bands("fce"), l1s)
    shared = [l1s.index(g) for g in SHARED_L1]

    def stats(w):
        n, s, q = cells.sums(w)
        d_model, d_errant = displacement(n, s[..., :k], q)[0], displacement(n, s[..., k:], q)[0]
        return np.stack([d_model, d_errant, d_model - d_errant])[:, shared]

    with np.errstate(invalid="ignore", divide="ignore"):
        point = stats(cells.weights())
        rng = np.random.default_rng(SEED)
        boots = np.array([stats(cells.weights(rng)) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boots[:, 2], [2.5, 97.5], axis=0)
    rows = [{"l1": g, "part": part, "d_model": point[0, i, j], "d_errant": point[1, i, j],
             "distortion": point[2, i, j], "lo": lo[i, j], "hi": hi[i, j], "has_margin": part != "total"}
            for i, g in enumerate(SHARED_L1) for j, part in enumerate(cols)]
    return pd.DataFrame(rows)


def prompt_centred(X, band, prompt):
    """A function of the resampling weights: X with every column centred on its (band, prompt)
    weighted mean, so the prompt means are re-estimated in each draw."""
    cell = pd.Categorical(pd.Series(band).astype(str) + "|" + pd.Series(prompt).astype(str).values).codes
    onehot = sparse.csr_matrix((np.ones(len(cell)), (cell, np.arange(len(cell)))))

    def centre(w):
        mean = (onehot @ (w[:, None] * X)) / np.maximum(onehot @ w, 1e-12)[:, None]
        return X - mean[cell]
    return centre


def d_rows(stat, cols, l1s, margin=None):
    """Rows of d, CI and (if margins are given) whether d clears its margin, per L1 and part."""
    rows = []
    for gi, g in enumerate(l1s):
        for pj, part in enumerate(cols):
            d, lo, hi = stat[0][gi, pj], stat[1][gi, pj], stat[2][gi, pj]
            row = {"l1": g, "part": part, "d": d, "d_lo": lo, "d_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)}
            if margin is not None:
                up, down = (0.0, 0.0) if part == "total" else margin.get((g, part), (np.nan, np.nan))
                row.update({"margin_up": up, "margin_down": down,
                            "clears_margin": None if np.isnan(up) else bool(lo - up > 0 or hi + down < 0)})
            rows.append(row)
    return rows


def run_followup():
    out = results_dir("toefl")
    units, cols = part_columns(load_toefl())
    units = with_corrected(units, "toefl")
    l1s = sorted(units.l1.unique())

    # FCE margins for the new parts (D49's rule), beside D49's own for the others
    dist = fce_corr_distortion()
    dist.to_csv(out / "l1_corr_fce_distortion.csv", index=False)
    margin = {**margins(), **margins(out / "l1_corr_fce_distortion.csv")}

    # step 1a and step 2 on the same draws as D51: same units, order, strata and generator
    parts = cols + CORR_PARTS
    cells = Cells(units[parts], units.group, units.l1, TOEFL_BANDS, l1s)
    X = units[PROMPT_PARTS].to_numpy(float)
    centre = prompt_centred(X, units.group, units.prompt)
    cells_p = Cells(X, units.group, units.l1, TOEFL_BANDS, l1s)

    def stats(w):
        d, bits, d_band = displacement(*cells.sums(w))
        cells_p.X = centre(w)
        d_p = displacement(*cells_p.sums(w))[0]
        return d, d_band, d_p

    point = stats(cells.weights())
    rng = np.random.default_rng(SEED)
    boots = [stats(cells.weights(rng)) for _ in range(N_BOOT)]
    d, d_band, d_p = (boot_ci(p, np.array([b[i] for b in boots])) for i, p in enumerate(point))

    # D51's resamples are reused: its d and CIs come back exactly
    d51 = pd.read_csv(out / "l1_displacement.csv").set_index(["l1", "part"])
    ours = pd.DataFrame(d_rows(d, parts, l1s, margin)).set_index(["l1", "part"])
    gap = (ours.loc[d51.index, ["d", "d_lo", "d_hi"]] - d51[["d", "d_lo", "d_hi"]]).abs().to_numpy().max()
    assert gap < 1e-12, gap
    ours.reset_index().to_csv(out / "l1_corr_displacement.csv", index=False)

    rows = [{"l1": g, "band": b, "part": part, "d": d_band[0][si, gi, pj], "lo": d_band[1][si, gi, pj],
             "hi": d_band[2][si, gi, pj], "n": int(((units.l1 == g) & (units.group == b)).sum())}
            for si, b in enumerate(TOEFL_BANDS) for gi, g in enumerate(l1s)
            for pj, part in enumerate(parts) if part in CORR_PARTS]
    pd.DataFrame(rows).to_csv(out / "l1_corr_displacement_band.csv", index=False)
    mean, lo, hi, n_min = balanced_bands(units[["total"] + CORR_PARTS], units.group, units.l1, l1s,
                                         np.random.default_rng(SEED))
    rows = [{"l1": g, "band": b, "part": part, "d": mean[si, gi, pj], "lo": lo[si, gi, pj], "hi": hi[si, gi, pj],
             "n_per_l1": n_min[si]}
            for si, b in enumerate(TOEFL_BANDS) for gi, g in enumerate(l1s)
            for pj, part in enumerate(["total"] + CORR_PARTS)]
    pd.DataFrame(rows).to_csv(out / "l1_corr_displacement_band_balanced.csv", index=False)
    # the two factors of sel, by L1 and band (descriptive)
    (units.groupby(["l1", "group"])[["c_share", "c_excess", "clean", "s_corr", "sel", "cost"]].mean()
     .reset_index().rename(columns={"group": "band"}).to_csv(out / "l1_corr_factors.csv", index=False))

    # step 2: the prompt
    pd.crosstab([units.l1, units.group], units.prompt).reset_index().rename(columns={"group": "band"}) \
        .to_csv(out / "l1_prompt_counts.csv", index=False)
    (units.groupby(["group", "prompt"])[["total", "clean", "errors", "s_corr"]].agg(["mean", "size"])
     .pipe(lambda t: t.drop(columns=[c for c in t.columns if c[1] == "size" and c[0] != "total"]))
     .set_axis(["total", "n", "clean", "errors", "s_corr"], axis=1).reset_index().rename(columns={"group": "band"})
     .to_csv(out / "l1_prompt_means.csv", index=False))
    prompt = pd.DataFrame(d_rows(d_p, PROMPT_PARTS, l1s))
    prompt = prompt.merge(ours.reset_index()[["l1", "part", "d", "d_lo", "d_hi"]], on=["l1", "part"],
                          suffixes=("", "_uncentred"))
    prompt.to_csv(out / "l1_prompt_displacement.csv", index=False)

    rules(ours, prompt.set_index(["l1", "part"])).to_csv(out / "l1_corr_rules.csv", index=False)
    show = ours.reset_index()
    print(show[show.part.isin(["total", "clean", "errors", "s_corr", "sel", "cost"])]
          .pivot(index="l1", columns="part", values="d").round(3).to_string())
    print(prompt.pivot(index="l1", columns="part", values="d")[["total", "clean", "errors", "s_corr"]].round(3))
    print(pd.read_csv(out / "l1_corr_rules.csv").to_string(index=False))


N_TOP, N_EXAMPLE_TYPES, N_EXAMPLES = 50, 15, 10


def sentence_at(text, i):
    """The sentence (or line) around character i."""
    start = max(text.rfind(c, 0, i) for c in ".?!\n") + 1
    ends = [j for j in (text.find(c, i) for c in ".?!\n") if j >= 0]
    return " ".join(text[start:min(ends) + 1 if ends else len(text)].split())


def run_types():
    """D52 step 3: the clean difference split over word types, for the L1s that pass the rules."""
    out = results_dir("toefl")
    robust = pd.read_csv(out / "l1_corr_rules.csv").query("robust == True").l1.tolist()
    units = load_toefl()
    tok = pd.read_parquet(data_dir("toefl") / "aligned.parquet",
                          columns=["text_id", "idx", "text", "own", "aligned", "s_corr", "unit", "group"])
    tok = tok[tok.own & tok.aligned].assign(type=lambda t: t.text.str.lower())
    tok["l1"] = tok.unit.map(units.l1)
    tok["w"] = 1.0 / tok.groupby("unit").unit.transform("size")  # a unit's tokens weigh 1 in all
    l1s = sorted(units.l1.unique())
    cells = Cells(units[["total", "clean"]], units.group, units.l1, TOEFL_BANDS, l1s)
    n, s, q = cells.sums(cells.weights())
    d_clean = displacement(n, s, q)[0][:, 1]
    n_rest, s_rest, q_rest = n.sum(1, keepdims=True) - n, s.sum(1, keepdims=True) - s, q.sum(1, keepdims=True) - q
    ss = (q - s[..., 0] ** 2 / n) + (q_rest - s_rest[..., 0] ** 2 / n_rest)
    sp = np.sqrt(ss / (n + n_rest - 2))  # (S, G): D47's pooled within-band SD of the total
    essays = pd.read_parquet(data_dir("toefl") / "essays.parquet", columns=["text_id", "text"]).set_index("text_id").text
    rng = np.random.default_rng(SEED)
    rows, examples = [], []
    for g in robust:
        gi = l1s.index(g)
        terms = []
        for si, b in enumerate(TOEFL_BANDS):
            band = tok[tok.group == b]
            focal = band.l1 == g
            by = {}
            for side, rows_of, n_units in [("g", band[focal], n[si, gi]), ("r", band[~focal], n_rest[si, gi])]:
                p = rows_of.w.groupby(rows_of.type).sum() / n_units
                mean_s = (rows_of.w * rows_of.s_corr).groupby(rows_of.type).sum() / (p * n_units)
                by[side] = (p, mean_s)
            types = by["g"][0].index.union(by["r"][0].index)
            p_g, p_r = (by[k][0].reindex(types).fillna(0.0) for k in "gr")
            s_g, s_r = (by[k][1].reindex(types) for k in "gr")
            s_g, s_r = s_g.fillna(s_r), s_r.fillna(s_g)  # a type absent on one side: all frequency
            weight = n[si, gi] / n[:, gi].sum() / sp[si, gi]
            terms.append(pd.DataFrame({"frequency": (p_g - p_r) * (s_g + s_r) / 2 * weight,
                                       "cost": (p_g + p_r) / 2 * (s_g - s_r) * weight,
                                       "p_g": p_g * n[si, gi] / n[:, gi].sum(), "p_r": p_r * n[si, gi] / n[:, gi].sum()}))
        t = pd.concat(terms).groupby(level=0).sum()
        t["d"] = t.frequency + t.cost
        assert abs(t.d.sum() - d_clean[gi]) < 1e-9, (g, t.d.sum(), d_clean[gi])
        focal_tok = tok[tok.l1 == g]
        counts = focal_tok.type.value_counts()
        for direction, top in [("up", t.d.nlargest(N_TOP)), ("down", t.d.nsmallest(N_TOP))]:
            for rank, (w, d) in enumerate(top.items(), 1):
                rows.append({"l1": g, "clean_d": d_clean[gi], "direction": direction, "rank": rank, "type": w,
                             "d": d, "frequency": t.frequency[w], "cost": t.cost[w],
                             "per_1000_g": 1000 * t.p_g[w], "per_1000_rest": 1000 * t.p_r[w],
                             "n_tokens_g": int(counts.get(w, 0))})
                if rank <= N_EXAMPLE_TYPES:
                    hits = focal_tok[focal_tok.type == w]
                    for h in hits.iloc[rng.permutation(len(hits))[:N_EXAMPLES]].itertuples():
                        examples.append({"l1": g, "direction": direction, "rank": rank, "type": w,
                                         "text_id": h.text_id, "band": h.group, "s_corr": h.s_corr,
                                         "sentence": sentence_at(essays[h.text_id], h.idx)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "l1_clean_types.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "l1_clean_type_examples.csv", index=False)  # TOEFL 11 text: local only
    for (g, direction), x in table[table["rank"] <= 20].groupby(["l1", "direction"], sort=False):
        print(g, direction, f"clean d {x.clean_d.iloc[0]:.3f}")
        print(x[["type", "d", "frequency", "cost", "per_1000_g", "per_1000_rest"]].round(4).to_string(index=False))


def rules(disp, prompt):
    """D52's reading rules: robust clean displacement (a-c) for the flagged L1s, and accounting
    offsets for Arabic and Turkish."""
    rows = []
    for g in ROBUST_L1 + OFFSET_L1:
        clean, s_corr, centred = disp.loc[(g, "clean")], disp.loc[(g, "s_corr")], prompt.loc[(g, "clean")]
        errors, sel = disp.loc[(g, "errors")], disp.loc[(g, "sel")]
        sign = np.sign(clean.d)
        a = bool(clean.clears_margin)
        b = bool(np.sign(s_corr.d) == sign and s_corr.clears_margin)
        c = bool(np.sign(centred.d) == sign and centred.ci_excludes_0)
        rows.append({"l1": g, "role": "flagged" if g in ROBUST_L1 else "offset",
                     "clean_d": clean.d, "a_clean_clears": a, "s_corr_d": s_corr.d, "b_s_corr_same_sign_clears": b,
                     "clean_d_prompt_centred": centred.d, "c_centred_same_sign_ci": c,
                     "robust": a and b and c if g in ROBUST_L1 else None,
                     "errors_d": errors.d, "sel_d": sel.d,
                     "accounting": bool(abs(sel.d) >= abs(errors.d) / 2 and np.sign(sel.d) == np.sign(errors.d))
                     if g in OFFSET_L1 else None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# step 1: the corrector by L1 on FCE
# --------------------------------------------------------------------------- #
def fce_l1():
    """FCE script id -> L1 (constant within a script)."""
    rows = [json.loads(line) for path in sorted((RAW_DIRS["fce"] / "json").glob("*.json"))
            for line in path.read_text().splitlines()]
    l1 = pd.DataFrame([(r["id"], r.get("l1")) for r in rows], columns=["unit", "l1"]).drop_duplicates()
    assert l1.unit.is_unique
    return l1.set_index("unit").l1


def text_agreement():
    """TP, FP, FN per FCE answer, as c4_score.py counts them, from the five fold models' edits."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "corrector"))
    from common import final_run
    edits = pd.concat([pd.read_parquet(DATA / "corrector" / "edits" / f"{final_run(k)}__fold{k}.parquet")
                       for k in range(5)])
    key = ["text_id", "para", "o_start", "o_end", "c_str"]
    match = edits[edits.side == "hyp"][key].merge(edits[edits.side == "ref"][key], on=key, how="outer",
                                                  indicator=True)
    return pd.DataFrame({"tp": match[match._merge == "both"].groupby("text_id").size(),
                         "fp": match[match._merge == "left_only"].groupby("text_id").size(),
                         "fn": match[match._merge == "right_only"].groupby("text_id").size()}).fillna(0)


def prf(tp, fp, fn, beta=0.5):
    p, r = tp / (tp + fp), tp / (tp + fn)
    return p, r, (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def run_check():
    model = pd.read_parquet(data_dir("fce_model") / "units.parquet")
    errant = pd.read_parquet(data_dir("fce_errant") / "units.parquet")
    model, errant = model[model.y.notna()], errant[errant.y.notna()]
    assert model.index.equals(errant.index) and np.allclose(model.total, errant.total)
    model, _ = part_columns(model)
    errant, _ = part_columns(errant)
    split = ["total", "clean", "errors"] + SPLIT4[1:]
    l1 = fce_l1().reindex(model.index)
    assert l1.notna().all()
    l1s = sorted(l1.unique())
    shared = [l1s.index(g) for g in SHARED_L1]

    # per script: agreement counts and edit counts, so everything resamples with the same weights
    essays = pd.read_parquet(data_dir("fce_model") / "essays.parquet", columns=["text_id", "unit"])
    agree = text_agreement().join(essays.set_index("text_id").unit).groupby("unit")[["tp", "fp", "fn"]].sum()
    agree = agree.reindex(model.index).fillna(0)
    X = np.column_stack([model[split], errant[split], agree, model.n_edits, errant.n_edits, model.n])
    cells = Cells(X, model.group, l1, bands("fce"), l1s)
    k = len(split)

    def stats(w):
        n, s, q = cells.sums(w)
        d_model = displacement(n, s[..., :k], q)[0]
        d_errant = displacement(n, s[..., k:2 * k], q)[0]
        by_l1 = s.sum(0)  # (G, columns)
        rest = by_l1.sum(0) - by_l1
        tp, fp, fn = (by_l1[:, 2 * k + i] for i in range(3))
        p, r, f = prf(tp, fp, fn)
        p_rest, r_rest, f_rest = prf(*(rest[:, 2 * k + i] for i in range(3)))
        tokens = by_l1[:, -1]
        m_l1 = n.sum(0)
        return np.column_stack([
            p, r, f, r - r_rest, f - f_rest,
            100 * by_l1[:, 2 * k + 3] / tokens, 100 * by_l1[:, 2 * k + 4] / tokens,
            by_l1[:, 2] / m_l1, by_l1[:, k + 2] / m_l1, (by_l1[:, 2] - by_l1[:, k + 2]) / m_l1,
            d_model[:, 0], d_model[:, 2], d_errant[:, 2], d_model[:, 2] - d_errant[:, 2]])[shared]

    names = ["precision", "recall", "f05", "recall_minus_rest", "f05_minus_rest",
             "edits_per_100_model", "edits_per_100_errant", "errors_bits_model", "errors_bits_errant",
             "errors_bits_model_minus_errant", "d_total", "d_errors_model", "d_errors_errant", "distortion"]
    with np.errstate(invalid="ignore", divide="ignore"):  # FCE's two Dutch scripts leave empty cells
        point = stats(cells.weights())
        rng = np.random.default_rng(SEED)
        boots = np.array([stats(cells.weights(rng)) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    rows = []
    for i, g in enumerate(SHARED_L1):
        row = {"l1": g, "toefl_l1": SHARED_L1[g], "n_scripts": int((l1 == g).sum())}
        for j, name in enumerate(names):
            row.update({name: point[i, j], f"{name}_lo": lo[i, j], f"{name}_hi": hi[i, j]})
        row["flag"] = bool((lo[i, -1] > 0 or hi[i, -1] < 0) and abs(point[i, -1]) >= FLAG)
        rows.append(row)
    table = pd.DataFrame(rows)
    out = RESULTS / "validation"
    table.to_csv(out / "l1_check.csv", index=False)

    # the four-part split per L1 under each source (bits per token)
    means = []
    for source, units in [("model", model), ("errant", errant)]:
        m = units.assign(l1=l1).groupby("l1")[split].mean().loc[list(SHARED_L1)]
        means.append(m.assign(source=source).reset_index())
    pd.concat(means).to_csv(out / "l1_split4.csv", index=False)
    part_distortion(model, errant, l1, l1s).to_csv(out / "l1_distortion.csv", index=False)
    show = ["l1", "n_scripts", "recall", "f05", "f05_minus_rest", "edits_per_100_model", "edits_per_100_errant",
            "d_total", "d_errors_model", "d_errors_errant", "distortion", "distortion_lo", "distortion_hi", "flag"]
    print(table[show].round(3).to_string())


def part_distortion(model, errant, l1, l1s):
    """Per shared L1 and part: d under the model and the re-cut gold, and the distortion
    (model minus gold) with paired band-stratified CIs (D49)."""
    cols = list(dict.fromkeys(part_columns(model)[1] + part_columns(errant)[1]))
    k = len(cols)
    cells = Cells(np.column_stack([model[cols], errant[cols]]), model.group, l1, bands("fce"), l1s)
    shared = [l1s.index(g) for g in SHARED_L1]

    def stats(w):
        n, s, q = cells.sums(w)
        d_model, d_errant = displacement(n, s[..., :k], q)[0], displacement(n, s[..., k:], q)[0]
        return np.stack([d_model, d_errant, d_model - d_errant])[:, shared]

    with np.errstate(invalid="ignore", divide="ignore"):
        point = stats(cells.weights())
        rng = np.random.default_rng(SEED)
        boots = np.array([stats(cells.weights(rng)) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boots[:, 2], [2.5, 97.5], axis=0)
    rows = []
    for i, g in enumerate(SHARED_L1):
        for j, part in enumerate(cols):
            cat = part.split("|")[0]
            if cat in GROUPS:  # where n allows: enough corrections of the group under both sources
                n_model, n_errant = (int(u.loc[l1 == g, f"n_{cat}"].sum()) for u in (model, errant))
            else:
                n_model = n_errant = None
            enough = cat not in GROUPS or min(n_model, n_errant) >= MIN_GROUP_CORRECTIONS
            rows.append({"l1": g, "part": part, "d_model": point[0, i, j], "d_errant": point[1, i, j],
                         "distortion": point[2, i, j], "lo": lo[i, j], "hi": hi[i, j],
                         "n_corrections_model": n_model, "n_corrections_errant": n_errant,
                         "has_margin": bool(enough and part != "total")})
    return pd.DataFrame(rows)


# the marks TOEFL 11's tokenizer splits off; it leaves hyphens and slashes in place
TOEFL_SPLIT = re.compile(r"""\s*([.,?!:;()\[\]{}"'`%$])\s*""")


def toefl_format(x: str) -> str:
    return re.sub(r"\s+", " ", TOEFL_SPLIT.sub(r" \1 ", x)).strip()


def run_tokenization():
    texts = pd.read_parquet(DATA / "corrector" / "pairs.parquet", columns=["text_id", "text"]).set_index("text_id").text
    rows = []
    for corpus in ("wi_errant", "fce_errant"):
        E = pd.read_parquet(data_dir(corpus) / "edits.parquet")
        E["erased"] = [toefl_format(t[max(s - 1, 0):e + 1]) == toefl_format(t[max(s - 1, 0):s] + r + t[e:e + 1])
                       for t, s, e, r in zip(texts[E.text_id].values, E.s, E.e, E.repl)]
        rows.append({"corpus": corpus, "level": "all", "part": "all", "n": len(E), "erased": int(E.erased.sum())})
        for level in ("bucket", "group"):
            for part, x in E.groupby(level).erased:
                rows.append({"corpus": corpus, "level": level, "part": part, "n": len(x), "erased": int(x.sum())})
    out = pd.DataFrame(rows).assign(share=lambda d: d.erased / d.n)
    out.to_csv(RESULTS / "validation" / "tokenization_erased.csv", index=False)
    print(out[out.erased > 0].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="step 1: the corrector by L1 on FCE")
    ap.add_argument("--tokenization", action="store_true", help="gold corrections TOEFL 11's tokenization erases")
    ap.add_argument("--followup", action="store_true", help="D52 steps 1a and 2: S_corr, sel, cost; prompt")
    ap.add_argument("--types", action="store_true", help="D52 step 3: the clean difference over word types")
    args = ap.parse_args()
    if args.followup:
        run_followup()
    elif args.types:
        run_types()
    else:
        run_check() if args.check else run_tokenization() if args.tokenization else run_toefl()


if __name__ == "__main__":
    main()
