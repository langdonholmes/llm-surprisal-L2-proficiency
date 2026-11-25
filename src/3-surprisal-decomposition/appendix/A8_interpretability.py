"""Study 3 appendix, stage A8 — are the decompositions' components meaningful error categories?

Every decomposition fitted in A3 and A7 (on ELLIPSE dev, nothing refit) is applied
unchanged to the high-surprisal tokens of W&I+LOCNESS, whose gold corrections give
each token an error label (gold ERR_TOK or ERR_INS) and an error group
(edits.parquet ``group``, from ERRANT). Each variant uses its own tau and its own
representation of the W&I tokens.

Per component, the 200 highest-loading W&I tokens:
    error_share     share that are gold errors (base rate: all selected W&I tokens)
    type_purity     among those errors, share in the most common error group
    top_lemma_share share that are the single most common lemma (the D5 rule)
Per variant:
    AMI between each error token's highest-loading component and its error group,
    with the same AMI for written-word identity (obs_id) as the reference: a
    decomposition that carries error information beyond the word should exceed it.

Loadings: NMF variants are projected with H fixed (sklearn's
non_negative_factorization, update_H=False, the fit's own loss and penalty);
k-means assigns the nearest centroid (centroids recomputed as the mean of each
cluster's ELLIPSE members) and ranks members by closeness; PCA's two poles of each
axis are treated as separate components. "Centering" subtracts each written word's
mean delta over the W&I tokens, since many W&I words do not occur in ELLIPSE dev.

    python src/3-surprisal-decomposition/appendix/A8_interpretability.py
    -> results/study3/appendix/interpretability_{summary,components}.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics import adjusted_mutual_info_score
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from A3_nmf import NMF_KW, posneg  # noqa: E402
from A7_sensitivity import REFERENCE, SENS, VARIANTS, config, load_tokens  # noqa: E402

from util.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

S3 = DATA_DIR / "study3" / "appendix"
WI = DATA_DIR / "study3" / "wi"
OUT = RESULTS_DIR / "study3" / "appendix"
TOP_N = 200
MIN_TOKENS = 20  # a component with fewer loaders on W&I is reported as unused
SEED = 42


# --- W&I tokens -----------------------------------------------------------------------

def wi_tokens(tau_min: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Non-space W&I tokens with surprisal >= tau_min, their gold label and error group,
    and their top-50 predictions."""
    frames, ids, probs = [], [], []
    for p in sorted((S3 / "wi" / "shards").glob("shard_*.parquet")):
        t = pd.read_parquet(p, columns=["text_id", "spacy_idx", "idx", "text", "lemma", "obs_id",
                                        "surprisal_bits", "is_space"])
        keep = ((t.surprisal_bits >= tau_min) & ~t.is_space).to_numpy()
        z = np.load(p.with_suffix(".npz"))
        frames.append(t[keep].drop(columns="is_space"))
        ids.append(z["topk_ids"][keep])
        probs.append(z["topk_probs"][keep])
    tok = pd.concat(frames, ignore_index=True)

    aligned = pd.read_parquet(WI / "aligned.parquet", columns=["text_id", "spacy_idx", "gold", "site_k"])
    tok = tok.merge(aligned, on=["text_id", "spacy_idx"], how="left", validate="one_to_one")
    assert tok.gold.notna().all(), "W&I tokens without a gold label"

    edits = pd.read_parquet(WI / "edits.parquet", columns=["text_id", "k", "s", "e", "group"])
    group_of = edits.set_index(["text_id", "k"]).group
    tok["group"] = pd.Series(list(zip(tok.text_id, tok.site_k))).map(group_of).to_numpy()
    tok.loc[tok.gold != "ERR_TOK", "group"] = None
    # ERR_INS: the insertion point lies between the previous token and this one, so the
    # nearest insertion edit at or before the token's start is its own
    ins = edits[edits.s == edits.e].sort_values("s")
    is_ins = (tok.gold == "ERR_INS").to_numpy()
    q = tok[is_ins][["text_id", "idx"]].reset_index().sort_values("idx")
    m = pd.merge_asof(q, ins[["text_id", "s", "group"]], left_on="idx", right_on="s",
                      by="text_id", direction="backward").set_index("index")
    tok.loc[m.index, "group"] = m.group
    tok["is_error"] = tok.gold.isin(["ERR_TOK", "ERR_INS"])
    return tok, np.concatenate(ids), np.concatenate(probs)


def wi_delta(rows, ids, probs, obs_ids, topk, W_U):
    out = np.empty((len(rows), W_U.shape[1]), dtype=np.float32)
    for a in range(0, len(rows), 1024):
        r = rows[a:a + 1024]
        p = probs[r, :topk].astype(np.float64)
        p = (p / p.sum(1, keepdims=True)).astype(np.float32)
        out[a:a + len(r)] = np.einsum("nk,nkd->nd", p, W_U[ids[r, :topk]]) - W_U[obs_ids[r]]
    return out


def wi_representation(cfg, tok, ids, probs, tau, W_U):
    rows = np.flatnonzero(tok.surprisal_bits >= tau[str(cfg["tau_pct"])])
    obs = tok.obs_id.to_numpy()
    if cfg["rep"] == "obs":
        return rows, -W_U[obs[rows]]
    X = wi_delta(rows, ids, probs, obs, cfg["topk"], W_U)
    if cfg["center"]:
        codes, inv = np.unique(obs[rows], return_inverse=True)
        sums = np.zeros((len(codes), X.shape[1]))
        np.add.at(sums, inv, X)
        X -= (sums / np.bincount(inv)[:, None]).astype(np.float32)[inv]
    return rows, X


# --- loadings ---------------------------------------------------------------------------

def nmf_loadings(cfg, H, X):
    kw = dict(solver=NMF_KW["solver"], beta_loss="frobenius", tol=NMF_KW["tol"], max_iter=NMF_KW["max_iter"])
    if cfg["beta"] != "frobenius":
        kw.update(beta_loss=cfg["beta"], solver="mu")
    if cfg["l1"]:
        alpha = json.loads((SENS / "l1_alpha.json").read_text())["alpha"]
        kw.update(alpha_W=alpha, alpha_H="same", l1_ratio=1.0, max_iter=5000)
    W, _, _ = non_negative_factorization(posneg(X), H=H.astype(np.float32), n_components=H.shape[0],
                                         update_H=False, random_state=SEED, **kw)
    return W


def kmeans_loadings(name, X, el_tok):
    """Nearest centroid; the loading is -distance for the assigned cluster, -inf elsewhere."""
    W_el = np.load(SENS / f"W_{name}.npy")
    rows_el = np.load(SENS / f"rows_{name}.npy")
    a0 = el_tok.row_id_a0.to_numpy()[rows_el]
    D = np.load(S3 / "delta.npy", mmap_mode="r")
    labels = W_el.argmax(1)
    cent = np.zeros((W_el.shape[1], X.shape[1]))
    for a in range(0, len(a0), 50_000):
        np.add.at(cent, labels[a:a + 50_000], np.asarray(D[a0[a:a + 50_000]], dtype=np.float64))
    cent /= np.bincount(labels, minlength=W_el.shape[1])[:, None]
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ cent.T + (cent ** 2).sum(1)[None, :]
    nearest = d2.argmin(1)
    L = np.full(d2.shape, -np.inf, dtype=np.float32)
    L[np.arange(len(X)), nearest] = -np.sqrt(np.maximum(d2[np.arange(len(X)), nearest], 0))
    return L


def pca_loadings(X):
    z = np.load(S3 / "nmf" / "pca.npz")
    s = (X - z["mean"]) @ z["components"].T
    return np.hstack([np.maximum(s, 0), np.maximum(-s, 0)])


# --- metrics --------------------------------------------------------------------------

def describe(L, t: pd.DataFrame, tokenizer, hard: bool) -> pd.DataFrame:
    """hard: k-means, where -inf marks tokens outside the cluster."""
    rows = []
    for j in range(L.shape[1]):
        w = L[:, j]
        used = np.flatnonzero(np.isfinite(w) if hard else w > 0)
        top = used[np.argsort(-w[used])[:TOP_N]]
        r = {"component": j, "n_loaders": int(len(used))}
        if len(top) >= MIN_TOKENS:
            s = t.iloc[top]
            err = s[s.is_error]
            r.update(error_share=float(s.is_error.mean()),
                     top_lemma_share=float(s.lemma.str.lower().value_counts(normalize=True).iloc[0]),
                     type_purity=float(err.group.value_counts(normalize=True).iloc[0]) if len(err) >= 10 else np.nan,
                     top_group=err.group.value_counts().index[0] if len(err) else None,
                     top_observed=", ".join(s.text.str.lower().value_counts().head(6).index),
                     top_predicted=", ".join(pd.Series(tokenizer.convert_ids_to_tokens(
                         s.pred1_id.tolist())).str.replace("Ġ", "_").value_counts().head(6).index))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    prep = json.loads((SENS / "prepare.json").read_text())
    tau = prep["tau_bits"]
    tok, ids, probs = wi_tokens(min(tau.values()))
    tok["pred1_id"] = ids[:, 0]
    W_U = np.load(S3 / "W_U.npy")
    el_tok = load_tokens()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    print(f"{len(tok)} W&I tokens above tau80; error share {tok.is_error.mean():.3f}", flush=True)

    variants = {"ref": dict(REFERENCE), "ref_obs": dict(REFERENCE, rep="obs"), "pca": dict(REFERENCE, method="pca")}
    variants.update({n: config(n) for n in VARIANTS if (SENS / f"W_{n}.npy").exists()})
    summary, comps = [], []
    for name, cfg in variants.items():
        rows, X = wi_representation(cfg, tok, ids, probs, tau, W_U)
        t = tok.iloc[rows].reset_index(drop=True)
        if cfg["method"] == "pca":
            L = pca_loadings(X)
        elif cfg["method"] == "kmeans":
            L = kmeans_loadings(name, X, el_tok)
        else:
            H = np.load(S3 / "nmf" / f"H_{'delta' if name == 'ref' else 'obs'}.npy") if name.startswith("ref") \
                else np.load(SENS / f"H_{name}.npy")
            L = nmf_loadings(cfg, H, X)
        d = describe(L, t, tokenizer, hard=cfg["method"] == "kmeans")
        d.insert(0, "variant", name)
        comps.append(d)

        err = t.is_error.to_numpy() & t.group.notna().to_numpy()
        Le = np.where(np.isfinite(L[err]), L[err], -np.inf)
        has = Le.max(1) > (-np.inf if cfg["method"] == "kmeans" else 0)
        base = float(t.is_error.mean())
        used = d.dropna(subset=["error_share"])
        summary.append({
            "variant": name, "n_tokens": len(t), "n_error_tokens": int(err.sum()),
            "base_error_share": base, "n_components": L.shape[1], "n_used": len(used),
            "median_error_share": float(used.error_share.median()),
            "max_error_share": float(used.error_share.max()),
            "median_type_purity": float(used.type_purity.median()),
            "n_single_word_wi": int((used.top_lemma_share >= .50).sum()),
            "ami_component_group": float(adjusted_mutual_info_score(
                t.group[err][has], Le[has].argmax(1))),
            "ami_word_group": float(adjusted_mutual_info_score(t.group[err], t.obs_id[err])),
        })
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[-1].items()},
              f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(summary).to_csv(OUT / "interpretability_summary.csv", index=False)
        pd.concat(comps).to_csv(OUT / "interpretability_components.csv", index=False)


if __name__ == "__main__":
    main()
