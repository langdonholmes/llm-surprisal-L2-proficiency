"""Study 3, stage 10 — split each essay's high surprisal into errors and
acceptable-but-unexpected words (DECISIONS D15).

The acceptability model is `comb_rac` (repair + attestation + context views)
with the fixed gradient boosting of lab/accept.py. It is fit once on the gold
W&I train tokens above τ (error vs OK) and outputs P(OK) for each token above
τ in ELLIPSE and TOEFL. No ELLIPSE or TOEFL score is used. Per essay, with n
the scored non-space tokens:

    high_rate = #(above τ) / n
    err_rate  = Σ (1 − P(OK)) / n
    acc_rate  = Σ P(OK) / n          so high_rate = err_rate + acc_rate

Each token above τ also gets a coarse type (nonword, punctuation, proper noun
or number, function word, and content word split by rare/common and by
whether both bigrams are attested in edited Dolma). The acceptable mass is
broken down by type.

Subcommands:

    feats    comb_rac features for every real bench (one subprocess per bench,
             since the lab harness reads LAB_BENCH at import)
    fit      fit on W&I train; P(OK) per above-τ token for ELLIPSE, TOEFL, and
             W&I (the W&I rows reuse the accept.py CV and confirmation scores)
             -> data/study3/decompose/tokens_<corpus>.parquet
    essays   per-essay rates and the D15 tests
             -> results/study3/decompose_*.csv, decompose_tests.json

    python src/3-vector-transformations/10_decompose.py feats
    python src/3-vector-transformations/10_decompose.py fit
    python src/3-vector-transformations/10_decompose.py essays
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from util.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

S3 = DATA_DIR / "study3"
LAB = S3 / "lab"
DEC = S3 / "decompose"
OUT = RESULTS_DIR / "study3"
REP = "comb_rac"
SEED = 42
BENCHES = {"ellipse": ["real_ellipse_dev", "real_ellipse_test"],
           "toefl11": ["real_toefl11_p0", "real_toefl11_p1"]}
TRAITS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
FUNCTION_POS = ["pos_DET", "pos_ADP", "pos_AUX", "pos_PRON", "pos_CCONJ", "pos_SCONJ", "pos_PART"]
CONTENT_POS = ["pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV"]


def bench_feats(bench: str) -> np.ndarray:
    p = LAB / "reps" / f"{REP}_{bench}.npy"
    if not p.exists():
        subprocess.run([sys.executable, __file__, "_feat"], check=True,
                       env={**os.environ, "LAB_BENCH": bench})
    return np.nan_to_num(np.load(p).astype(np.float64))


def token_type(bench: str, rare_cut: float) -> np.ndarray:
    """Mutually exclusive coarse type per bench row, from the attestation table."""
    f = pd.read_parquet(LAB / "att" / f"feats_{bench}.parquet")
    content = f[CONTENT_POS].sum(1) > 0
    attested = (f.a_unseen_bi_left == 0) & (f.a_unseen_bi_right == 0)
    rare = -f.a_log_uni >= rare_cut
    t = np.select(
        [np.maximum(f.a_unseen_uni, f.b_unseen_uni) > 0, f.is_punct > 0,
         (f.pos_PROPN > 0) | (f.like_num > 0), f[FUNCTION_POS].sum(1) > 0,
         content & rare & attested, content & attested,
         content & rare, content],
        ["nonword", "punct", "propn_num", "function",
         "rare_attested", "common_attested", "rare_unattested", "common_unattested"],
        "other")
    return t


def rare_cutoff() -> float:
    """Upper quartile of rarity among W&I-train content words above τ (fixed from W&I, not ELLIPSE)."""
    f = pd.read_parquet(LAB / "att" / "feats_real_wi.parquet")
    g = wi_gold()
    m = (f[CONTENT_POS].sum(1) > 0).to_numpy() & (g.split == "train").to_numpy()
    return float(np.quantile(-f.a_log_uni[m], 0.75))


def wi_gold() -> pd.DataFrame:
    b = pd.read_parquet(LAB / "bench_real_wi.parquet", columns=["inst", "row_id", "text_id"])
    g = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet",
                        columns=["row_id", "gold", "level", "split", "surprisal_bits", "pos"])
    return b.merge(g, on="row_id", how="left")


def cmd_feat_one(args):
    from lab.harness import Ctx
    from lab.reps import comb_rac
    ctx = Ctx()
    bench = os.environ["LAB_BENCH"]
    np.save(LAB / "reps" / f"{REP}_{bench}.npy", np.asarray(comb_rac.featurize(ctx), dtype=np.float32))
    print(bench, "done", flush=True)


def cmd_feats(args):
    for b in [b for bs in BENCHES.values() for b in bs]:
        bench_feats(b)


def cmd_fit(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    DEC.mkdir(parents=True, exist_ok=True)
    g = wi_gold()
    has = pd.read_parquet(LAB / "base_meta_real_wi.parquet").has_target.to_numpy()
    X = np.nan_to_num(np.load(LAB / "reps" / f"{REP}_real_wi.npy").astype(np.float64))
    fit = (g.split == "train").to_numpy() & has & g.gold.isin(["ERR_TOK", "ERR_INS", "OK"]).to_numpy()
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=SEED)
    m.fit(X[fit], (g.gold == "OK").to_numpy()[fit])
    cut = rare_cutoff()

    for corpus, benches in BENCHES.items():
        if not all((LAB / "ctx" / f"olmo7b_w64_{b}.npz").exists() for b in benches):
            print(corpus, "context views not ready, skipped", flush=True)
            continue
        frames = []
        for b in benches:
            bench = pd.read_parquet(LAB / f"bench_{b}.parquet", columns=["row_id", "text_id"])
            ok = pd.read_parquet(LAB / f"base_meta_{b}.parquet").has_target.to_numpy()
            p = m.predict_proba(bench_feats(b))[:, 1]
            bench["p_ok"] = np.where(ok, p, np.nan)
            bench["ttype"] = token_type(b, cut)
            frames.append(bench)
        t = pd.concat(frames, ignore_index=True)
        t["p_ok"] = t.p_ok.fillna(t.p_ok.mean())  # unaligned rows (<1%): corpus mean
        t.to_parquet(DEC / f"tokens_{corpus}.parquet", index=False)
        print(corpus, len(t), f"mean P(OK) {t.p_ok.mean():.3f}", flush=True)

    # W&I: out-of-fold P(OK) for train essays, the single confirmation fit for dev + LOCNESS
    cv = np.load(LAB / "reps" / f"{REP}_real_wi_p_dev_gbm.npy")
    conf = np.load(LAB / "reps" / f"{REP}_real_wi_p_confirm_gbm.npy")
    g["p_ok"] = np.where(g.split == "train", cv, conf)
    g["p_ok"] = g.p_ok.fillna(g.p_ok.mean())
    g["ttype"] = token_type("real_wi", cut)
    g.to_parquet(DEC / "tokens_wi.parquet", index=False)
    print("wi", len(g), json.dumps({"rare_cut": cut}))


def n_nonspace(corpus: str) -> pd.Series:
    d = S3 / corpus / "shards"
    toks = pd.concat([pd.read_parquet(p, columns=["text_id", "is_space"]) for p in sorted(d.glob("*.parquet"))])
    return toks[~toks.is_space].groupby("text_id").size().rename("n")


PI_TRAIN = 0.637  # OK share among usable W&I-train tokens above tau


def em_adjust(p: pd.Series, essay: pd.Series, iters: int = 50) -> pd.Series:
    """Saerens et al. (2002) prior adjustment of P(OK), separately per essay (D16)."""
    out = p.copy()
    for _, i in p.groupby(essay).groups.items():
        q = p.loc[i].to_numpy()
        pi = q.mean()
        for _ in range(iters):
            a = q * pi / PI_TRAIN
            r = a / (a + (1 - q) * (1 - pi) / (1 - PI_TRAIN))
            pi = r.mean()
        out.loc[i] = r
    return out


def essay_rates(corpus: str) -> pd.DataFrame:
    t = pd.read_parquet(DEC / f"tokens_{corpus}.parquet")
    t["p_ok_em"] = em_adjust(t.p_ok, t.text_id)
    e = n_nonspace(corpus).to_frame()
    e["n_high"] = t.groupby("text_id").size()
    e = e.fillna(0)
    e["high_rate"] = e.n_high / e.n
    for col, sfx in [("p_ok", ""), ("p_ok_em", "_em")]:
        acc = t.groupby("text_id")[col].sum().reindex(e.index).fillna(0)
        rare = t[t.ttype == "rare_attested"].groupby("text_id")[col].sum().reindex(e.index).fillna(0)
        e[f"acc_rate{sfx}"] = acc / e.n
        e[f"err_rate{sfx}"] = (e.n_high - acc) / e.n
        e[f"acc_share{sfx}"] = acc / e.n_high.replace(0, np.nan)
        e[f"acc_rare_attested_rate{sfx}"] = rare / e.n
    return e.reset_index()


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z (rank) partialled out of both."""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def boot(e, fn, n=2000):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(e))
    vals = [fn(e.iloc[rng.choice(idx, len(idx))]) for _ in range(n)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def cv_r2(e, cols):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    y = e.overall - e.groupby("prompt").overall.transform("mean")
    X = e[cols].to_numpy()
    X = X - e.groupby("prompt")[cols].transform("mean").to_numpy()
    p = cross_val_predict(LinearRegression(), X, y, groups=e.prompt, cv=GroupKFold(10))
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ellipse_tests(e, sfx="") -> dict:
    r = {"n_essays": len(e)}
    err, acc = f"err_rate{sfx}", f"acc_rate{sfx}"
    for c in ["high_rate", err, acc, f"acc_share{sfx}", f"acc_rare_attested_rate{sfx}"]:
        r[f"rho_{c}_overall"] = spearman(e[c], e.overall)
        r[f"ci_{c}_overall"] = boot(e, lambda d, c=c: spearman(d[c], d.overall))
    # A1: err_rate more negative than high_rate
    a1 = lambda d: spearman(d[err], d.overall) - spearman(d.high_rate, d.overall)  # noqa: E731
    r["A1_diff"], r["A1_ci"] = a1(e), boot(e, a1)
    # A2: |rho acc| < |rho err|
    a2 = lambda d: abs(spearman(d[acc], d.overall)) - abs(spearman(d[err], d.overall))  # noqa: E731
    r["A2_diff"], r["A2_ci"] = a2(e), boot(e, a2)
    # A3: partial correlations with each analytic trait, controlling Overall
    for c in [err, acc, f"acc_rare_attested_rate{sfx}"]:
        for tr in TRAITS:
            r[f"A3_{c}_{tr}"] = partial_spearman(e[c], e[tr], e.overall)
    # A4: does the split carry information the total does not
    r["A4_r2_high"] = cv_r2(e, ["high_rate"])
    r["A4_r2_split"] = cv_r2(e, [err, acc])
    return r


def cmd_essays(args):
    ess = pd.read_parquet(S3 / "ellipse" / "essays.parquet").drop(columns=["text"])
    split = pd.read_csv(HERE / "split.csv")
    e = essay_rates("ellipse").merge(ess, on="text_id").merge(split, on="text_id")
    e.to_csv(OUT / "decompose_ellipse_essays.csv", index=False)
    tests = {f"{s}{sfx}": ellipse_tests(e[e.split == s].reset_index(drop=True), sfx)
             for s in ["dev", "test"] for sfx in ["", "_em"]}

    tf = pd.read_parquet(S3 / "toefl11" / "essays.parquet").drop(columns=["text"])
    have_toefl = (DEC / "tokens_toefl11.parquet").exists()
    t = essay_rates("toefl11").merge(tf, on="text_id") if have_toefl else None
    if have_toefl:
        _toefl(t, tests)
    _wi(tests)
    _coded_check(e)
    _kinds(ess, split, tf, ["ellipse", "wi"] + (["toefl11"] if have_toefl else []))
    (OUT / "decompose_tests.json").write_text(json.dumps(tests, indent=2, default=float))
    print(json.dumps(tests, indent=1, default=float)[:8000])


def _toefl(t, tests):
    t["level_n"] = t.level.map({"low": 0, "medium": 1, "high": 2})
    t.to_csv(OUT / "decompose_toefl_essays.csv", index=False)
    tests["toefl"] = {f"rho_{c}_level": spearman(t[c], t.level_n)
                      for c in ["high_rate", "err_rate", "acc_rate", "acc_share", "acc_rare_attested_rate",
                                "err_rate_em", "acc_rate_em", "acc_share_em", "acc_rare_attested_rate_em"]}
    tests["toefl_by_level"] = t.groupby("level")[["high_rate", "err_rate", "acc_rate", "acc_share",
                                                     "err_rate_em", "acc_rate_em", "acc_share_em"]].mean().to_dict()


def _coded_check(e):
    """Rates by Overall tertile against the 148 blind-coded dev tokens (3_premises.py)."""
    k = pd.read_csv(HERE / "coding_key.csv").merge(pd.read_csv(HERE / "error_coding_claude.csv"), on="item")
    k = k[k.code != "OTHER"]
    t = pd.read_parquet(DEC / "tokens_ellipse.parquet")
    t["p_ok_em"] = em_adjust(t.p_ok, t.text_id)
    k = k.merge(t, on=["row_id", "text_id"])
    coded = k.groupby("band").agg(n_coded=("code", "size"), coded_ok=("code", lambda c: (c == "OK").mean()),
                                  p_ok=("p_ok", "mean"), p_ok_em=("p_ok_em", "mean"))
    d = e[e.split == "dev"].copy()
    d["band"] = pd.cut(d.overall, [-np.inf, 3, 3.5, np.inf], right=False, labels=["low", "mid", "high"]).astype(str)
    r = d.groupby("band")[["high_rate", "acc_rate", "acc_rate_em", "err_rate", "err_rate_em"]].mean().join(coded)
    r["acc_rate_coded"] = r.high_rate * r.coded_ok
    r["err_rate_coded"] = r.high_rate - r.acc_rate_coded
    r.reset_index().to_csv(OUT / "decompose_coded_check.csv", index=False)


def _wi(tests):
    # A5: W&I by level, model vs gold (gold counts POST as neither)
    w = pd.read_parquet(DEC / "tokens_wi.parquet")
    w["p_err"] = 1 - w.p_ok
    w["p_ok_em"] = em_adjust(w.p_ok, w.text_id)
    w["p_err_em"] = 1 - w.p_ok_em
    w["gold_err"] = w.gold.isin(["ERR_TOK", "ERR_INS"])
    w["gold_ok"] = w.gold == "OK"
    n = n_nonspace("wi")
    lv = w.groupby("text_id").agg(level=("level", "first"), split=("split", "first"),
                                  n_high=("p_ok", "size"), acc=("p_ok", "sum"), err=("p_err", "sum"),
                                  acc_em=("p_ok_em", "sum"), err_em=("p_err_em", "sum"), g_err=("gold_err", "sum"), g_ok=("gold_ok", "sum")).join(n)
    for a in ["acc", "err", "acc_em", "err_em", "g_err", "g_ok", "n_high"]:
        lv[f"{a}_rate"] = lv[a] / lv.n
    lv = lv.reset_index()
    lv.to_csv(OUT / "decompose_wi_essays.csv", index=False)
    held = lv[(lv.split == "dev") | (lv.level == "N")]
    tests["wi_heldout_by_level"] = held.groupby("level")[
        ["n_high_rate", "err_rate", "acc_rate", "err_em_rate", "acc_em_rate", "g_err_rate", "g_ok_rate"]].mean().to_dict()
    lvn = held.level.map({"A": 0, "B": 1, "C": 2})
    learners = held[held.level != "N"]
    tests["wi_heldout_essay"] = {
        **{f"rho_{m}_gold": spearman(held[m], held[g]) for m, g in
           [("err_rate", "g_err_rate"), ("err_em_rate", "g_err_rate"),
            ("acc_rate", "g_ok_rate"), ("acc_em_rate", "g_ok_rate")]},
        **{f"rho_{m}_level": spearman(learners[m], lvn[learners.index])
           for m in ["err_rate", "err_em_rate", "g_err_rate", "acc_rate", "acc_em_rate", "g_ok_rate"]}}


def _kinds(ess, split, tf, corpora):
    # kinds of acceptable surprise: share of the acceptable mass by type
    kinds = []
    for corpus in corpora:
        k = pd.read_parquet(DEC / f"tokens_{corpus}.parquet")
        k["p_ok_em"] = em_adjust(k.p_ok, k.text_id)
        if corpus == "ellipse":
            k = k.merge(split, on="text_id")
            k = k.merge(ess[["text_id", "overall"]], on="text_id")
            k["group"] = pd.cut(k.overall, [0, 2.5, 3.5, 5], labels=["low", "mid", "high"]).astype(str)
        elif corpus == "toefl11":
            k = k.merge(tf[["text_id", "level"]], on="text_id").rename(columns={"level": "group"})
        else:
            k = k[(k.split == "dev") | (k.level == "N")].rename(columns={"level": "group"})
        for grp, d in k.groupby("group"):
            tot, tot_em = d.p_ok.sum(), d.p_ok_em.sum()
            for ty, dd in d.groupby("ttype"):
                kinds.append({"corpus": corpus, "group": grp, "ttype": ty, "n": len(dd),
                              "mean_p_ok": dd.p_ok.mean(), "share_of_acc_mass": dd.p_ok.sum() / tot,
                              "mean_p_ok_em": dd.p_ok_em.mean(), "share_of_acc_mass_em": dd.p_ok_em.sum() / tot_em,
                              "share_of_tokens": len(dd) / len(d)})
    pd.DataFrame(kinds).to_csv(OUT / "decompose_kinds.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ["feats", "fit", "essays", "_feat"]:
        sub.add_parser(c)
    args = ap.parse_args()
    {"feats": cmd_feats, "fit": cmd_fit, "essays": cmd_essays, "_feat": cmd_feat_one}[args.cmd](args)


if __name__ == "__main__":
    main()
