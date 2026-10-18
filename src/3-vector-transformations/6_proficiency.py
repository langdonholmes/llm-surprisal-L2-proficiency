"""Study 3, stage 6 — essay profiles and the proficiency analyses (RQ2-RQ4).

Profiles. An essay's profile for a representation is the mean NMF loading over
its high-surprisal tokens (the proposal's definition). Essays with no
high-surprisal token get a zero profile.

Prompt and length. Every outcome and predictor is centred on its prompt mean,
with the means estimated on dev only and applied to test, so every R^2 below
is within-prompt. log(n scored tokens) is in every model.

Baseline ladder (RQ3), all elastic net (sklearn ElasticNetCV, standardized
inputs, l1_ratio and alpha tuned by 10-fold CV grouped by prompt within dev):

    B0      mean surprisal + log length
    B1      B0 + surprisal variance + rate of high-surprisal tokens
    R       B0 + LM repair profile (stage 8): per-token rate of each edit
            label + mean positive gain per edit class
    S2+R    S2 + LM repair profile
    B2      B0 + share of each of the 300 most frequent dev lemmas among the
            essay's high-surprisal tokens ("which words were surprising")
    PCA     B0 + mean PCA score of delta, same number of components as M
    M_obs   B0 + NMF profile of -e_obs  (observed word alone)
    M_pred  B0 + NMF profile of e_pred  (prediction alone)
    M       B0 + NMF profile of delta   (primary)
    S2      B0 + surprisal variance + the 13 Study 2 indices
    S2+M    S2 + NMF profile of delta

Two modes. By default everything is estimated on dev by grouped 10-fold
cross-validation (folds by prompt), and test rows are not read. With
--confirmatory the models are fit on all dev essays and scored once on test,
with 2,000 essay-level bootstrap resamples of test for the Delta R^2 CIs. The
confirmatory run refuses to start unless DECISIONS.md has an "Analysis frozen"
entry.

RQ2 and RQ4 (per-component correlations) are computed on whichever set the mode
evaluates: dev in the default mode, test with --confirmatory.

    python src/3-vector-transformations/6_proficiency.py
    python src/3-vector-transformations/6_proficiency.py --confirmatory
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from util.paths import DATA_DIR, ELLIPSE_DIR, RESULTS_DIR

HERE = Path(__file__).resolve().parent
S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"
SEED = 42
N_LEMMAS = 300
N_BOOT = 2000
N_PERM = 10_000
SCORES = ["overall", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
STUDY2 = ["mtld", "lexical_density", "log_mean_token_freq", "mean_word_length",
          "mean_length_of_clause", "complex_nominals_per_clause", "coord_phrases_per_clause",
          "amod_mi", "dobj_mi", "advmod_mi",
          "content_word_overlap", "connective_density", "sentence_similarity"]


def essay_mean(W: np.ndarray, text_ids: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(W, columns=[f"{prefix}{j}" for j in range(W.shape[1])])
    df["text_id"] = text_ids
    return df.groupby("text_id").mean().reindex(index).fillna(0.0)


def build_features(corpus: str = "ellipse") -> tuple[pd.DataFrame, dict]:
    """One row per essay: labels, split, summaries and every profile block."""
    sel = pd.read_parquet(S3 / "tokens_selected.parquet")
    mask = (sel.corpus == corpus).to_numpy()
    s = sel[mask]
    es = pd.read_parquet(S3 / "essay_summary.parquet").query("corpus == @corpus")
    lab = pd.read_parquet(S3 / corpus / "essays.parquet").drop(columns="text")
    df = lab.merge(es.drop(columns="corpus"), on="text_id", how="left").set_index("text_id")
    if corpus == "ellipse":
        df = df.join(pd.read_csv(HERE / "split.csv").set_index("text_id"))
        m = pd.read_csv(ELLIPSE_DIR / "ellipse_metrics.csv").rename(columns={"text_id_kaggle": "text_id"})
        df = df.join(m.set_index("text_id")[STUDY2])
    else:
        df["split"] = "toefl"
    df["log_n"] = np.log(df.n_scored)
    df["rate_high"] = df.n_high / df.n_scored
    tids = s.text_id.to_numpy()

    blocks = {"base": ["mean_surprisal", "log_n"], "dist": ["var_surprisal", "rate_high"]}
    for rep in ["delta", "obs", "pred"]:
        p = S3 / "nmf" / f"W_{rep}.npy"
        if p.exists():
            prof = essay_mean(np.load(p, mmap_mode="r")[mask], tids, df.index, f"{rep}_")
            df = df.join(prof)
            blocks[rep] = list(prof.columns)

    # PCA on delta (fit on dev rows of ELLIPSE), same k as the delta NMF
    if "delta" in blocks:
        pca_path = S3 / "nmf" / "pca.npz"
        k = len(blocks["delta"])
        if not pca_path.exists():
            dev_rows = sel.row_id[sel.split == "dev"].to_numpy()
            rng = np.random.default_rng(SEED)
            fit_rows = np.sort(rng.choice(dev_rows, min(150_000, len(dev_rows)), replace=False))
            pca = PCA(k, random_state=SEED).fit(np.load(S3 / "delta.npy", mmap_mode="r")[fit_rows])
            np.savez(pca_path, components=pca.components_, mean=pca.mean_,
                     evr=pca.explained_variance_ratio_)
        z = np.load(pca_path)
        D = np.load(S3 / "delta.npy", mmap_mode="r")
        rows = np.flatnonzero(mask)
        scores = np.concatenate([(np.asarray(D[rows[a:a + 100_000]]) - z["mean"]) @ z["components"].T
                                 for a in range(0, len(rows), 100_000)])
        prof = essay_mean(scores, tids, df.index, "pca_")
        df = df.join(prof)
        blocks["pca"] = list(prof.columns)

    # LM repair profile (stage 8): per-token rate of each paying edit label, and
    # the mean positive gain per edit class over the high-surprisal tokens
    rp = S3 / "repair_tokens.parquet"
    if rp.exists():
        r = pd.read_parquet(rp).merge(s[["row_id", "text_id"]], on="row_id")
        r = r[r.scored]
        labels = sorted(pd.read_parquet(rp, columns=["label"]).label.unique())
        rates = pd.crosstab(r.text_id, r.label).reindex(index=df.index, columns=labels, fill_value=0)
        rates = rates.drop(columns="NONE", errors="ignore").div(df.n_scored, axis=0)
        rates.columns = [f"rr_{c}" for c in rates.columns]
        gcols = [c for c in r.columns if c.startswith("gain_")]
        gm = r.groupby("text_id")[gcols].mean().reindex(df.index).fillna(0.0)
        gm.columns = [f"rg_{c[5:]}" for c in gcols]
        df = df.join(rates).join(gm)
        blocks["repair"] = list(rates.columns) + list(gm.columns)
        blocks["repair_rates"] = list(rates.columns)

    # lemma bag: top dev lemmas among high-surprisal tokens
    dev_lem = sel[sel.split == "dev"].lemma.str.lower()
    vocab = dev_lem.value_counts().index[:N_LEMMAS]
    lem = s.assign(l=s.lemma.str.lower()).query("l in @vocab")
    bag = pd.crosstab(lem.text_id, lem.l).reindex(index=df.index, columns=vocab, fill_value=0)
    bag = bag.div(df.n_high.clip(lower=1), axis=0)
    bag.columns = [f"lem_{c}" for c in bag.columns]
    df = df.join(bag)
    blocks["lemma"] = list(bag.columns)
    if corpus == "ellipse":
        blocks["study2"] = STUDY2
    return df, blocks


def prompt_center(df: pd.DataFrame, cols: list[str], fit_mask: np.ndarray) -> pd.DataFrame:
    """Subtract per-prompt means estimated on the fit rows; impute NaN by fit median."""
    out = df[cols].copy()
    out = out.fillna(out[fit_mask].median())
    means = out[fit_mask].groupby(df.prompt[fit_mask]).mean()
    return out - means.reindex(df.prompt).to_numpy()


def ladder(blocks: dict) -> dict[str, list[str]]:
    b0 = blocks["base"]
    L = {"B0": b0, "B1": b0 + blocks["dist"], "B2": b0 + blocks["lemma"]}
    for name, key in [("PCA", "pca"), ("M_obs", "obs"), ("M_pred", "pred"), ("M", "delta"),
                      ("R", "repair")]:
        if key in blocks:
            L[name] = b0 + blocks[key]
    if "study2" in blocks:
        L["S2"] = b0 + ["var_surprisal"] + blocks["study2"]
        if "delta" in blocks:
            L["S2+M"] = L["S2"] + blocks["delta"]
        if "repair" in blocks:
            L["S2+R"] = L["S2"] + blocks["repair"]
    return L


def fit_predict(X_fit, y_fit, g_fit, X_eval):
    """Elastic net with alpha and l1_ratio tuned by 10-fold CV grouped by prompt."""
    folds = list(GroupKFold(10).split(X_fit, groups=g_fit))
    m = make_pipeline(StandardScaler(), ElasticNetCV(
        l1_ratio=[.1, .5, .9, 1.0], n_alphas=40, cv=folds, max_iter=20_000,
        random_state=SEED))
    m.fit(X_fit, y_fit)
    return m.predict(X_eval)


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmatory", action="store_true")
    args = ap.parse_args()
    tag = "test" if args.confirmatory else "devcv"
    if args.confirmatory and "Analysis frozen" not in (HERE / "DECISIONS.md").read_text():
        raise SystemExit("DECISIONS.md has no 'Analysis frozen' entry; refusing to read test.")

    df, blocks = build_features("ellipse")
    df.to_parquet(S3 / "profiles_ellipse.parquet")
    keep = ["prompt", "split"] + SCORES + [c for c in df.columns if c.startswith("rr_")]
    df[keep].to_csv(OUT / "profiles_ellipse_repair.csv.gz")  # for results.qmd (no arrow in R)
    dev = (df.split == "dev").to_numpy()
    feats = sorted({c for cols in ladder(blocks).values() for c in cols})
    Xc = prompt_center(df, feats, dev)
    Yc = prompt_center(df, SCORES, dev)
    print(f"{int((df.n_high == 0).sum())} essays with no high-surprisal token")

    # --- RQ3: ladder --------------------------------------------------------
    L = ladder(blocks)
    y = Yc.overall.to_numpy()
    groups = df.prompt.to_numpy()
    preds = {}
    if args.confirmatory:
        ev = ~dev
        for name, cols in L.items():
            preds[name] = fit_predict(Xc[cols][dev], y[dev], groups[dev], Xc[cols][ev])
            print(name, round(r2_score(y[ev], preds[name]), 4), flush=True)
    else:
        ev = dev
        idx = np.flatnonzero(dev)
        for name, cols in L.items():
            p = np.empty(len(idx))
            for tr, te in GroupKFold(10).split(idx, groups=groups[idx]):
                p[te] = fit_predict(Xc[cols].iloc[idx[tr]], y[idx[tr]], groups[idx[tr]],
                                    Xc[cols].iloc[idx[te]])
            preds[name] = p
            print(name, round(r2_score(y[idx], p), 4), flush=True)
    y_ev = y[ev]
    rng = np.random.default_rng([SEED, 1])
    boots = rng.integers(0, len(y_ev), size=(N_BOOT, len(y_ev)))
    rows = []
    for name, p in preds.items():
        r2 = r2_score(y_ev, p)
        d = r2 - r2_score(y_ev, preds["B0"])
        bd = np.array([r2_score(y_ev[b], p[b]) - r2_score(y_ev[b], preds["B0"][b]) for b in boots])
        rows.append({"model": name, "n_predictors": len(L[name]), "r2": r2, "delta_r2_vs_B0": d,
                     "ci_lo": np.percentile(bd, 2.5), "ci_hi": np.percentile(bd, 97.5)})
    for name, ref in [("M", "M_obs"), ("S2+M", "S2"), ("R", "M_obs"), ("R", "M"),
                      ("S2+R", "S2")]:  # increments over the nearest comparison
        if name in preds and ref in preds:
            d = r2_score(y_ev, preds[name]) - r2_score(y_ev, preds[ref])
            bd = np.array([r2_score(y_ev[b], preds[name][b]) - r2_score(y_ev[b], preds[ref][b])
                           for b in boots])
            rows.append({"model": f"{name} - {ref}", "delta_r2_vs_B0": np.nan, "r2": d,
                         "ci_lo": np.percentile(bd, 2.5), "ci_hi": np.percentile(bd, 97.5)})
    lad = pd.DataFrame(rows)
    lad["mode"] = tag
    lad["n_eval"] = len(y_ev)
    lad.to_csv(OUT / f"ladder_{tag}.csv", index=False)
    print(lad.round(4).to_string())

    # --- RQ2 / RQ4: per-component correlations ------------------------------
    if "delta" not in blocks:
        return
    cors = []
    for c in blocks["delta"]:
        x = Xc[c][ev]
        for sc in SCORES:
            rho, p = spearmanr(x, Yc[sc][ev])
            # partial on Overall: residualize both on Overall (linear, fit on eval set)
            if sc != "overall":
                o = Yc.overall[ev].to_numpy()
                rx = x - np.polyval(np.polyfit(o, x, 1), o)
                ry = Yc[sc][ev] - np.polyval(np.polyfit(o, Yc[sc][ev], 1), o)
                prho, pp = spearmanr(rx, ry)
            else:
                prho, pp = np.nan, np.nan
            cors.append({"component": int(c.split("_")[1]), "score": sc, "rho": rho, "p": p,
                         "rho_partial_overall": prho, "p_partial": pp})
    cors = pd.DataFrame(cors)
    ov = cors.score == "overall"
    cors.loc[ov, "q"] = bh(cors.loc[ov, "p"].to_numpy())
    cors["mode"] = tag
    cors.to_csv(OUT / f"component_correlations_{tag}.csv", index=False)

    labels_path = HERE / "component_labels.csv"
    if labels_path.exists():
        labs = pd.read_csv(labels_path)
        r = cors[ov].merge(labs, on="component")
        r["abs_rho"] = r.rho.abs()
        g = r[r.level.isin(["morphosyntactic", "lexical"])]
        is_m = (g.level == "morphosyntactic").to_numpy()
        a = g.abs_rho.to_numpy()
        obs = a[is_m].mean() - a[~is_m].mean()
        prng = np.random.default_rng([SEED, 2])
        null = np.array([(lambda m: a[m].mean() - a[~m].mean())(prng.permutation(is_m))
                         for _ in range(N_PERM)])
        h2 = {"mode": tag, "n_morphosyntactic": int(is_m.sum()), "n_lexical": int((~is_m).sum()),
              "mean_abs_rho_morphosyntactic": float(a[is_m].mean()),
              "mean_abs_rho_lexical": float(a[~is_m].mean()), "difference": float(obs),
              "p_one_sided": float((np.sum(null >= obs) + 1) / (N_PERM + 1))}
        (OUT / f"h2_permutation_{tag}.json").write_text(json.dumps(h2, indent=2))
        print(h2)


if __name__ == "__main__":
    main()
