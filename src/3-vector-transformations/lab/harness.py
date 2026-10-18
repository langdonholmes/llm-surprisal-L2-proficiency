"""Lab — context object and the evaluation battery shared by every representation.

A representation is a module ``lab/reps/<name>.py`` exposing

    def featurize(ctx: Ctx) -> np.ndarray    # (n_bench, d), row i == bench inst i

Rows for unscorable instances (``ctx.ok == False``) may hold anything; they are
dropped before evaluation. ``python lab/run.py <name>`` featurizes, caches the
matrix, runs ``evaluate`` and appends a row to the scoreboard.

The battery (all cross-validated; no number is computed on training rows)
-------------------------------------------------------------------------
detect_auc      Mean over error types (SPELL excluded: almost no natural
                matches) of the AUC for error vs. lexeme-matched clean
                control, 5-fold CV grouped by essay. Observed-word identity
                scores .5 here by construction. **The headline metric**: does the
                vector know that *this* "there" is wrong?
type_f1         Macro-F1 over 9 classes (8 error types + CLEAN_SURP),
                5-fold CV grouped by essay.
type_f1_tau     The same, restricted to instances with surprisal >= tau (what
                the Study 3 pipeline actually sees).
lexdis_f1       Macro-F1 over the open-class types (AGR, DET_DEL, NUM, SPELL,
                VFORM, CLEAN_SURP), 5-fold CV grouped by lexeme, so test words
                never appear in training. Generalization beyond the word.
nmf_nmi_type    Unsupervised. NMF (k = 12) on [max(Z,0) | max(-Z,0)] of the
                standardized vectors for above-tau instances, each instance
                assigned to its top component. NMI with error type.
nmf_nmi_word    NMI of the same assignment with the observed lexeme (lower is
                better at a given nmf_nmi_type).
nmf_purity      Share of instances whose component's majority type is theirs.
word_r2         Share of the standardized vectors' total sum of squares lying
                between observed-word groups (lexical leakage; lower is less
                lexical).

Probes are logistic regression on standardized inputs, reduced to at most 128
principal components (fitted inside each fold).
"""

import os
from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from util.paths import DATA_DIR, RESULTS_DIR

LAB = DATA_DIR / "study3" / "lab"
OUT = RESULTS_DIR / "study3" / "lab"
# Which benchmark: "" = the development benchmark, "conf" = the held-out
# confirmation benchmark (different host essays and seed). Set LAB_BENCH=conf.
BENCH = os.environ.get("LAB_BENCH", "")
SUFFIX = f"_{BENCH}" if BENCH else ""
ERR = ["AGR", "DET_DEL", "DET_SUB", "HOMO", "NUM", "PREP", "SPELL", "VFORM"]
OPEN = ["AGR", "DET_DEL", "NUM", "SPELL", "VFORM", "CLEAN_SURP"]
TAU = 11.275990221094187
SEED = 42


class Ctx:
    """Everything a representation may read. Arrays are row-aligned with bench."""

    def __init__(self):
        self.bench = pd.read_parquet(LAB / f"bench{SUFFIX}.parquet")
        self.meta = pd.read_parquet(LAB / f"base_meta{SUFFIX}.parquet")
        self.base = dict(np.load(LAB / f"base{SUFFIX}.npz"))
        self.ok = self.meta.has_target.to_numpy()
        self.n = len(self.bench)

    def get(self, key: str) -> np.ndarray:
        """e.g. ctx.get('t_topk_ids'), ctx.get('n_h_pred_16') (float32)."""
        a = self.base[key]
        return a.astype(np.float32) if a.dtype == np.float16 else a

    @cached_property
    def W_U(self) -> np.ndarray:
        return np.load(DATA_DIR / "study3" / "W_U.npy")

    @cached_property
    def W_E(self) -> np.ndarray:
        import torch
        from transformers import AutoModelForCausalLM
        from lab.lm import HF_ID, REVISION
        m = AutoModelForCausalLM.from_pretrained(HF_ID, revision=REVISION, dtype=torch.float32)
        return m.get_input_embeddings().weight.detach().numpy().copy()

    def e_pred(self, pos: str = "t", k: int = 50, W: np.ndarray | None = None) -> np.ndarray:
        W = self.W_U if W is None else W
        ids, p = self.get(f"{pos}_topk_ids")[:, :k], self.get(f"{pos}_topk_probs")[:, :k].astype(np.float64)
        p = p / p.sum(1, keepdims=True)
        return np.einsum("nk,nkd->nd", p, W[ids]).astype(np.float32)

    def e_obs(self, pos: str = "t", W: np.ndarray | None = None) -> np.ndarray:
        W = self.W_U if W is None else W
        return W[self.get(f"{pos}_target_id")]


def _probe():
    return LogisticRegression(max_iter=3000, C=1.0)


def _cv_predict(X, y, groups, proba=False, n_splits=5):
    out = np.zeros((len(y), len(np.unique(y)))) if proba else np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        nc = min(128, X.shape[1], len(tr) - 1)
        m = make_pipeline(StandardScaler(), PCA(nc, random_state=SEED), _probe())
        m.fit(X[tr], y[tr])
        if proba:
            out[te] = m.predict_proba(X[te])
        else:
            out[te] = m.predict(X[te])
    return out


def between_r2(Z: np.ndarray, groups: np.ndarray) -> float:
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((inv.max() + 1, Z.shape[1]))
    np.add.at(sums, inv, Z)
    n = np.bincount(inv)[:, None]
    g = Z.mean(0)
    return float((n * (sums / n - g) ** 2).sum() / ((Z - g) ** 2).sum())


def evaluate(X: np.ndarray, ctx: Ctx) -> tuple[dict, dict]:
    b = ctx.bench
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    keep = ctx.ok & np.isfinite(X).all(1)
    surpr = ctx.get("t_surprisal")
    word = b.observed.str.lower().to_numpy()
    res, detail = {"dim": X.shape[1]}, {}

    # detect_auc: error vs lexeme-matched control, per type
    aucs = {}
    for t in ERR:
        if t == "SPELL":
            continue
        e = (b.type == t).to_numpy() & keep
        c = ((b.type == "MATCHED") & (b.matched_type == t)).to_numpy() & keep
        idx = np.flatnonzero(e | c)
        y = e[idx].astype(int)
        p = _cv_predict(X[idx], y, b.text_id.to_numpy()[idx], proba=True)[:, 1]
        aucs[t] = roc_auc_score(y, p)
    res["detect_auc"] = float(np.mean(list(aucs.values())))
    detail["detect_auc"] = aucs

    # type_f1 (9 classes), all and above tau
    cls = ERR + ["CLEAN_SURP"]
    m9 = b.type.isin(cls).to_numpy() & keep
    idx = np.flatnonzero(m9)
    y = b.type.to_numpy()[idx]
    pred = _cv_predict(X[idx], y, b.text_id.to_numpy()[idx])
    res["type_f1"] = f1_score(y, pred, average="macro")
    detail["type_f1_per_class"] = dict(zip(cls, f1_score(y, pred, average=None, labels=cls)))
    hi = surpr[idx] >= TAU
    res["type_f1_tau"] = f1_score(y[hi], pred[hi], average="macro")

    # lexeme-disjoint
    mo = b.type.isin(OPEN).to_numpy() & keep
    idx = np.flatnonzero(mo)
    y = b.type.to_numpy()[idx]
    pred = _cv_predict(X[idx], y, b.group.to_numpy()[idx])
    res["lexdis_f1"] = f1_score(y, pred, average="macro")
    detail["lexdis_f1_per_class"] = dict(zip(OPEN, f1_score(y, pred, average=None, labels=OPEN)))

    # unsupervised NMF alignment on the above-tau error + CLEAN_SURP set
    idx = np.flatnonzero(m9 & (surpr >= TAU))
    Z = StandardScaler().fit_transform(X[idx])
    if Z.shape[1] > 256:
        Z = PCA(256, random_state=SEED).fit_transform(Z)
        Z /= Z[:, :1].std()
    Zp = np.hstack([np.maximum(Z, 0), np.maximum(-Z, 0)])
    W = NMF(min(12, Zp.shape[1]), init="nndsvda", max_iter=500, random_state=SEED).fit_transform(Zp)
    lab = W.argmax(1)
    yt = b.type.to_numpy()[idx]
    res["nmf_nmi_type"] = normalized_mutual_info_score(yt, lab)
    res["nmf_nmi_word"] = normalized_mutual_info_score(word[idx], lab)
    ct = pd.crosstab(lab, yt)
    res["nmf_purity"] = float(ct.max(1).sum() / ct.values.sum())
    detail["nmf_crosstab"] = ct.to_dict()

    # lexical leakage
    idx = np.flatnonzero(keep)
    Zs = StandardScaler().fit_transform(X[idx])
    res["word_r2"] = between_r2(Zs, word[idx])
    return res, detail
