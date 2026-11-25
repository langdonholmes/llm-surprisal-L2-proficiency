"""Study 3 appendix, stage A7 — sensitivity of the delta decomposition (dev only).

Does any reasonable variation of the delta decomposition give an essay profile
that improves on the written-word decomposition (M_obs)? Each variant changes
one setting of the reference configuration (NMF, k = 50, Frobenius loss, no
penalty, tau = 90th percentile, e_pred averaged over the top 50 predictions):

    center        delta minus the mean delta of its written word (obs_id), so the
                  written word's own embedding is removed exactly: what is left is
                  e_pred - mean(e_pred | written word)
    kmeans        k-means (k = 50) on delta; profile = cluster proportions
    pca           PCA of delta (the ladder's existing PCA row)
    k20, k100     number of components (M_obs refit at the same k)
    kl            Kullback-Leibler loss (multiplicative-update solver)
    l1            L1 penalty on W and H
    tau80, tau95  surprisal threshold at the 80th / 95th percentile, same reference
                  tokens as A0 (M_obs refit on the same tokens)
    top1, top10   e_pred averaged over the top 1 / top 10 predictions

ELLIPSE dev essays only; no test row is read. Every decomposition is fit on all
dev tokens (unsupervised, no outcome), and the essay-level regressions are
cross-validated exactly as in A5's default mode (10 folds grouped by prompt,
elastic net, 2,000 bootstrap resamples of the out-of-fold predictions).

    prepare   dev tokens above the 80th-percentile tau, with their top-50
              predictions; checks that the tau-90 subset reproduces delta.npy
    fit NAME  fit one variant, save its dev loadings
    ladder    regressions for every fitted variant
              -> results/study3/appendix/sensitivity_devcv.csv

    python src/3-surprisal-decomposition/appendix/A7_sensitivity.py prepare
    python src/3-surprisal-decomposition/appendix/A7_sensitivity.py fit center
    python src/3-surprisal-decomposition/appendix/A7_sensitivity.py ladder
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from A3_nmf import NMF_KW, posneg  # noqa: E402
from A5_proficiency import essay_mean, fit_predict, prompt_center  # noqa: E402

from util.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

S3 = DATA_DIR / "study3" / "appendix"
SENS = S3 / "sensitivity"
OUT = RESULTS_DIR / "study3" / "appendix"
SEED = 42
N_BOOT = 2000
TOP_N = 200  # tokens per component for the single-word diagnostic (as in A4)

REFERENCE = dict(rep="delta", method="nmf", k=50, beta="frobenius", l1=False,
                 tau_pct=90, topk=50, center=False)
VARIANTS = {
    "center": dict(center=True),
    "kmeans": dict(method="kmeans"),
    "k20": dict(k=20), "k20_obs": dict(k=20, rep="obs"),
    "k100": dict(k=100), "k100_obs": dict(k=100, rep="obs"),
    "kl": dict(beta="kullback-leibler"),
    "l1": dict(l1=True),
    "tau80": dict(tau_pct=80), "tau80_obs": dict(tau_pct=80, rep="obs"),
    "tau95": dict(tau_pct=95), "tau95_obs": dict(tau_pct=95, rep="obs"),
    "top1": dict(topk=1), "top10": dict(topk=10),
}
# what each delta variant is compared with: the written-word decomposition on the
# same tokens and at the same k
MATCHED_OBS = {"k20": "k20_obs", "k100": "k100_obs", "tau80": "tau80_obs", "tau95": "tau95_obs"}
LABELS = {
    "ref": "Reference (NMF, k = 50, Frobenius, τ = 90th pct, top 50)",
    "center": "δ minus its written word's mean δ",
    "pca": "PCA instead of NMF", "kmeans": "k-means instead of NMF",
    "k20": "k = 20", "k100": "k = 100",
    "kl": "Kullback–Leibler loss", "l1": "L1 sparsity penalty",
    "tau80": "τ = 80th percentile", "tau95": "τ = 95th percentile",
    "top1": "e_pred from the top 1 prediction", "top10": "e_pred from the top 10 predictions",
}


def config(name: str) -> dict:
    return {**REFERENCE, **VARIANTS[name]}


# --- prepare ------------------------------------------------------------------

def taus() -> dict[int, float]:
    """tau at 80/90/95: percentiles of non-space token surprisal in the dev essays
    whose Overall is in the dev top tertile (A0's reference set)."""
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet", columns=["text_id", "overall"])
    lab = essays.merge(pd.read_csv(HERE / "split.csv"), on="text_id")
    dev = lab[lab.split == "dev"]
    ref_ids = set(dev.text_id[dev.overall >= dev.overall.quantile(2 / 3)])
    s = []
    for p in sorted((S3 / "ellipse" / "shards").glob("shard_*.parquet")):
        t = pd.read_parquet(p, columns=["text_id", "surprisal_bits", "is_space"])
        s.append(t.surprisal_bits[t.text_id.isin(ref_ids) & ~t.is_space].to_numpy())
    s = np.concatenate(s)
    return {q: float(np.percentile(s, q)) for q in (80, 90, 95)}


def cmd_prepare(args):
    SENS.mkdir(parents=True, exist_ok=True)
    tau = taus()
    tau_a0 = json.loads((OUT / "threshold.json").read_text())["tau_bits"]
    assert abs(tau[90] - tau_a0) < 1e-9, (tau[90], tau_a0)
    dev_ids = set(pd.read_csv(HERE / "split.csv").query("split == 'dev'").text_id)
    frames, ids, probs = [], [], []
    for p in sorted((S3 / "ellipse" / "shards").glob("shard_*.parquet")):
        t = pd.read_parquet(p, columns=["text_id", "text", "lemma", "obs_id", "surprisal_bits", "is_space"])
        keep = ((t.surprisal_bits >= tau[80]) & ~t.is_space & t.text_id.isin(dev_ids)).to_numpy()
        z = np.load(p.with_suffix(".npz"))
        frames.append(t[keep].drop(columns="is_space"))
        ids.append(z["topk_ids"][keep])
        probs.append(z["topk_probs"][keep])
    tok = pd.concat(frames, ignore_index=True)
    top_ids, top_p = np.concatenate(ids), np.concatenate(probs)
    for q, v in tau.items():
        tok[f"above_{q}"] = tok.surprisal_bits >= v

    # the tau-90 subset must be A0's dev selection, row for row
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["row_id", "split", "obs_id", "text_id"])
    sel_dev = sel[sel.split == "dev"]
    t90 = tok[tok.above_90]
    assert len(t90) == len(sel_dev), (len(t90), len(sel_dev))
    assert (t90.obs_id.to_numpy() == sel_dev.obs_id.to_numpy()).all()
    assert (t90.text_id.to_numpy() == sel_dev.text_id.to_numpy()).all()
    tok["row_id_a0"] = -1
    tok.loc[tok.above_90, "row_id_a0"] = sel_dev.row_id.to_numpy()

    tok.to_parquet(SENS / "tokens_dev.parquet", index=False)
    np.savez(SENS / "topk_dev.npz", ids=top_ids, probs=top_p)
    # delta recomputed here must equal A1's delta for the reference tokens
    rows = np.flatnonzero(tok.above_90)
    check = np.sort(np.random.default_rng(SEED).choice(rows, 5000, replace=False))
    d_here = delta_for(check, topk=50)
    d_a1 = np.load(S3 / "delta.npy", mmap_mode="r")[tok.row_id_a0.to_numpy()[check]]
    max_err = float(np.abs(d_here - d_a1).max())
    assert max_err < 1e-4, max_err
    info = {"tau_bits": tau, "n_rows": {q: int(tok[f"above_{q}"].sum()) for q in tau},
            "delta_check_max_abs_err": max_err}
    (SENS / "prepare.json").write_text(json.dumps(info, indent=2))
    print(info)


# --- representations ------------------------------------------------------------

def load_tokens() -> pd.DataFrame:
    return pd.read_parquet(SENS / "tokens_dev.parquet")


def delta_for(rows: np.ndarray, topk: int) -> np.ndarray:
    """delta = e_pred - e_obs, e_pred over the renormalized top-`topk` predictions
    (A1's arithmetic; float64 weights, float32 output)."""
    z = np.load(SENS / "topk_dev.npz")
    all_ids, all_p = z["ids"][rows, :topk], z["probs"][rows, :topk]
    obs = pd.read_parquet(SENS / "tokens_dev.parquet", columns=["obs_id"]).obs_id.to_numpy()[rows]
    W_U = np.load(S3 / "W_U.npy")
    out = np.empty((len(rows), W_U.shape[1]), dtype=np.float32)
    for a in range(0, len(rows), 1024):
        ids, p = all_ids[a:a + 1024], all_p[a:a + 1024].astype(np.float64)
        p = (p / p.sum(1, keepdims=True)).astype(np.float32)
        e_pred = np.einsum("nk,nkd->nd", p, W_U[ids])
        out[a:a + len(ids)] = e_pred - W_U[obs[a:a + 1024]]
    return out


def representation(cfg: dict, tok: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rows (into tokens_dev) and the real-valued matrix for one configuration."""
    rows = np.flatnonzero(tok[f"above_{cfg['tau_pct']}"])
    if cfg["rep"] == "obs":
        W_U = np.load(S3 / "W_U.npy", mmap_mode="r")
        return rows, -np.asarray(W_U[tok.obs_id.to_numpy()[rows]], dtype=np.float32)
    X = delta_for(rows, cfg["topk"])
    if cfg["center"]:
        obs = tok.obs_id.to_numpy()[rows]
        codes, inv = np.unique(obs, return_inverse=True)
        sums = np.zeros((len(codes), X.shape[1]), dtype=np.float64)
        np.add.at(sums, inv, X)
        X -= (sums / np.bincount(inv)[:, None]).astype(np.float32)[inv]
    return rows, X


# --- fit -------------------------------------------------------------------------

def nmf_model(cfg: dict) -> NMF:
    kw = dict(NMF_KW)
    if cfg["beta"] != "frobenius":
        kw.update(beta_loss=cfg["beta"], solver="mu")
    if cfg["l1"]:
        alpha = json.loads((SENS / "l1_alpha.json").read_text())["alpha"]
        # the penalized fit did not converge within 1,000 iterations on the full data
        kw.update(alpha_W=alpha, alpha_H="same", l1_ratio=1.0, max_iter=5000)
    return NMF(n_components=cfg["k"], random_state=SEED, **kw)


def cmd_fit(args):
    name = args.name
    cfg = config(name)
    tok = load_tokens()
    t0 = time.time()
    rows, X = representation(cfg, tok)
    info = {"name": name, **cfg, "n_rows": int(len(rows)),
            "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    if cfg["method"] == "kmeans":
        km = KMeans(cfg["k"], n_init=3, random_state=SEED).fit(X)
        dist = km.transform(X)[np.arange(len(X)), km.labels_]
        W = np.zeros((len(X), cfg["k"]), dtype=np.float32)
        W[np.arange(len(X)), km.labels_] = 1.0
        # for the single-word diagnostic, rank members by closeness to their centroid
        np.save(SENS / f"rank_{name}.npy", np.where(W > 0, -dist[:, None], -np.inf).astype(np.float32))
        info.update(n_iter=int(km.n_iter_), inertia=float(km.inertia_), converged=True)
    else:
        X = posneg(X)
        m = nmf_model(cfg)
        W = m.fit_transform(X).astype(np.float32)
        max_iter = m.get_params()["max_iter"]
        info.update(n_iter=int(m.n_iter_), converged=bool(m.n_iter_ < max_iter),
                    rel_error=float(m.reconstruction_err_ / np.linalg.norm(X)),
                    share_W_zero=float((W == 0).mean()),
                    n_dead_components=int((W.max(0) == 0).sum()))
        np.save(SENS / f"H_{name}.npy", m.components_.astype(np.float32))
    np.save(SENS / f"W_{name}.npy", W)
    np.save(SENS / f"rows_{name}.npy", rows)
    info["seconds"] = time.time() - t0
    (SENS / f"fit_{name}.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)


def cmd_pilot_l1(args):
    """Choose the L1 penalty on a 30,000-token subsample of the reference input
    (write_l1_choice), recorded before the full fit. sklearn scales the W penalty by n_features and the H penalty by
    n_samples, so the value transfers to the full fit only roughly; the full fit's
    sparsity is reported alongside."""
    tok = load_tokens()
    rows = np.flatnonzero(tok.above_90)
    sub = np.sort(np.random.default_rng(SEED).choice(rows, 30_000, replace=False))
    X = posneg(delta_for(sub, 50))
    out = []
    for alpha in [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        kw = dict(NMF_KW, alpha_W=alpha, alpha_H="same", l1_ratio=1.0)
        m = NMF(n_components=50, random_state=SEED, **kw)
        W = m.fit_transform(X)
        out.append({"alpha": alpha, "share_W_zero": float((W == 0).mean()),
                    "share_H_zero": float((m.components_ == 0).mean()),
                    "n_dead": int((W.max(0) == 0).sum()),
                    "rel_error": float(np.linalg.norm(X - W @ m.components_) / np.linalg.norm(X)),
                    "n_iter": int(m.n_iter_)})
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(SENS / "l1_pilot.csv", index=False)
    write_l1_choice(out)


def write_l1_choice(pilot: list[dict]):
    """The largest alpha below the first one that empties a component or fails to
    converge (a stronger penalty can look 'alive' again once the fit has collapsed)."""
    choice = None
    for r in sorted(pilot, key=lambda r: r["alpha"]):
        if r["n_dead"] > 0 or r["n_iter"] >= NMF_KW["max_iter"]:
            break
        choice = r
    if choice is None or choice["alpha"] == 0:
        raise RuntimeError("no usable L1 penalty on the pilot grid")
    (SENS / "l1_alpha.json").write_text(json.dumps({"alpha": choice["alpha"], "pilot": pilot}, indent=2))
    print("chosen", choice, flush=True)


# --- ladder ----------------------------------------------------------------------

def single_word_share(rank: np.ndarray, lemma: np.ndarray) -> tuple[int, float]:
    """Components whose 200 top tokens are >= 50% one lemma (A4/D5's rule)."""
    shares = []
    for j in range(rank.shape[1]):
        top = np.argpartition(-rank[:, j], TOP_N)[:TOP_N]
        top = top[np.isfinite(rank[top, j])]
        shares.append(pd.Series(lemma[top]).value_counts(normalize=True).iloc[0])
    shares = np.array(shares)
    return int((shares >= .50).sum()), float(np.median(shares))


def essays_frame() -> pd.DataFrame:
    lab = pd.read_parquet(S3 / "ellipse" / "essays.parquet").drop(columns="text")
    es = pd.read_parquet(S3 / "essay_summary.parquet").query("corpus == 'ellipse'")
    df = lab.merge(es.drop(columns="corpus"), on="text_id").set_index("text_id")
    df = df.join(pd.read_csv(HERE / "split.csv").set_index("text_id"))
    df = df[df.split == "dev"].copy()
    df["log_n"] = np.log(df.n_scored)
    return df


def cmd_ladder(args):
    tok = load_tokens()
    lemma = tok.lemma.str.lower().to_numpy()
    df = essays_frame()
    blocks, diag = {"base": ["mean_surprisal", "log_n"]}, []

    def add(name, W, rows):
        prof = essay_mean(W, tok.text_id.to_numpy()[rows], df.index, f"{name}_")
        blocks[name] = list(prof.columns)
        return prof

    # reference: A3's fits and A5's PCA, restricted to dev rows
    t90 = np.flatnonzero(tok.above_90)
    a0_rows = tok.row_id_a0.to_numpy()[t90]
    profs = []
    for name, f in [("ref", "W_delta"), ("ref_obs", "W_obs")]:
        W = np.load(S3 / "nmf" / f"{f}.npy", mmap_mode="r")[a0_rows]
        profs.append(add(name, W, t90))
        n1, med = single_word_share(np.asarray(W), lemma[t90])
        diag.append({"variant": name, "n_components": W.shape[1], "single_word": n1,
                     "median_top_lemma_share": med})
    z = np.load(S3 / "nmf" / "pca.npz")
    D = np.load(S3 / "delta.npy", mmap_mode="r")
    scores = np.concatenate([(np.asarray(D[a0_rows[a:a + 100_000]]) - z["mean"]) @ z["components"].T
                             for a in range(0, len(a0_rows), 100_000)])
    profs.append(add("pca", scores, t90))
    diag.append({"variant": "pca", "n_components": scores.shape[1]})

    fits = {}
    for name in VARIANTS:
        p = SENS / f"W_{name}.npy"
        if not p.exists():
            print("not fitted:", name)
            continue
        W, rows = np.load(p), np.load(SENS / f"rows_{name}.npy")
        fits[name] = json.loads((SENS / f"fit_{name}.json").read_text())
        profs.append(add(name, W, rows))
        rp = SENS / f"rank_{name}.npy"
        n1, med = single_word_share(np.load(rp) if rp.exists() else W, lemma[rows])
        diag.append({"variant": name, "n_components": W.shape[1], "single_word": n1,
                     "median_top_lemma_share": med, "converged": fits[name].get("converged"),
                     "n_iter": fits[name].get("n_iter"), "rel_error": fits[name].get("rel_error")})
    df = df.join(pd.concat(profs, axis=1))

    # models: each representation alone, and added to its matched written-word profile
    b0 = blocks["base"]
    models = {"B0": b0}
    for name in blocks:
        if name == "base":
            continue
        models[name] = b0 + blocks[name]
        if not name.endswith("obs"):
            obs = MATCHED_OBS.get(name, "ref_obs")
            if obs in blocks:
                models[f"{obs}+{name}"] = b0 + blocks[obs] + blocks[name]

    feats = sorted({c for cols in models.values() for c in cols})
    dev = np.ones(len(df), dtype=bool)
    Xc = prompt_center(df, feats, dev)
    y = prompt_center(df, ["overall"], dev).overall.to_numpy()
    groups = df.prompt.to_numpy()
    preds = {}
    for name, cols in models.items():
        p = np.empty(len(y))
        for tr, te in GroupKFold(10).split(Xc, groups=groups):
            p[te] = fit_predict(Xc[cols].iloc[tr], y[tr], groups[tr], Xc[cols].iloc[te])
        preds[name] = p
        print(name, round(r2_score(y, p), 4), flush=True)

    boots = np.random.default_rng([SEED, 1]).integers(0, len(y), size=(N_BOOT, len(y)))

    def diff(a, b):
        d = r2_score(y, preds[a]) - r2_score(y, preds[b])
        bd = np.array([r2_score(y[i], preds[a][i]) - r2_score(y[i], preds[b][i]) for i in boots])
        return d, np.percentile(bd, 2.5), np.percentile(bd, 97.5)

    rows_out = []
    for name in blocks:
        if name == "base":
            continue
        r = {"variant": name, "r2": r2_score(y, preds[name]), "delta_r2_vs_B0": diff(name, "B0")[0]}
        if not name.endswith("obs"):
            obs = MATCHED_OBS.get(name, "ref_obs")
            if obs not in preds:  # matched baseline not fitted yet
                rows_out.append(r)
                continue
            r["compared_with"] = obs
            r["vs_obs"], r["vs_obs_lo"], r["vs_obs_hi"] = diff(name, obs)
            r["added_to_obs"], r["added_to_obs_lo"], r["added_to_obs_hi"] = diff(f"{obs}+{name}", obs)
        rows_out.append(r)
    res = pd.DataFrame(rows_out).merge(pd.DataFrame(diag), on="variant", how="left")
    res["label"] = res.variant.map(LABELS)
    res["n_eval"] = len(y)
    res.to_csv(OUT / "sensitivity_devcv.csv", index=False)
    print(res.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    f = sub.add_parser("fit")
    f.add_argument("name", choices=list(VARIANTS))
    sub.add_parser("pilot_l1")
    sub.add_parser("ladder")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "fit": cmd_fit, "pilot_l1": cmd_pilot_l1, "ladder": cmd_ladder}[args.cmd](args)


if __name__ == "__main__":
    main()
