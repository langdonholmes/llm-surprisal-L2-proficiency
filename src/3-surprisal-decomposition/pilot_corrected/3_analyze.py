"""Pilot, step 3 (CPU) — align original and corrected tokens, answer Q1–Q4.

See NOTES.md for the pre-specified statistics. Tables go to
data/study3/pilot_corrected/tables/ (small CSV/JSON; KWIC stays in the data dir).

    python src/3-surprisal-decomposition/pilot_corrected/3_analyze.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import LEVELS, OUT, SEED, tau  # noqa: E402
from lib.bench import CACHE, S3  # noqa: E402
from lib.stats import boot_spearman  # noqa: E402
from util.paths import RESULTS_DIR  # noqa: E402

TAB = OUT / "tables"
N_BOOT = 2000
WIN = 8


def load_gold5():
    spec = importlib.util.spec_from_file_location("gold5", HERE.parent / "5_gold.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def first_subword_windows(tok, text: str, chars: np.ndarray) -> list:
    """The 8-id window ending at the first subword starting at each char (None if none)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids
    start = {}
    for k, (s, e) in enumerate(enc.offset_mapping):
        piece = text[s:e]
        start.setdefault(s + len(piece) - len(piece.lstrip()), k)
    out = []
    for c in chars:
        k = start.get(int(c))
        out.append(None if k is None else tuple(ids[max(0, k - WIN + 1):k + 1]))
    return out


def align(essays, flagged, orig_by, corr, tok) -> pd.DataFrame:
    """One row per scored original token: own, aligned, corrected surprisal, same window."""
    fl = flagged.groupby("text_id")
    corr_by = {t: d for t, d in corr.groupby("text_id")}
    frames = []
    for e in essays.itertuples():
        o = orig_by[e.text_id].copy()
        cmap = np.asarray(e.cmap)
        a = o.idx.to_numpy()
        b = a + o.text.str.len().to_numpy()
        own = np.array([(cmap[x:y] >= 0).all() for x, y in zip(a, b)])
        if e.text_id in fl.groups:
            for s, t in fl.get_group(e.text_id)[["s", "e"]].to_numpy():
                own &= ~((a < max(t, s + 1)) & (b > s))
        mapped = np.where(own, cmap[np.minimum(a, len(cmap) - 1)], -1)
        c = corr_by.get(e.text_id)
        cidx = dict(zip(c.idx, range(len(c)))) if c is not None else {}
        j = np.array([cidx.get(int(m), -1) if m >= 0 else -1 for m in mapped])
        ok = j >= 0
        same_text = np.zeros(len(o), bool)
        same_text[ok] = c.text.to_numpy()[j[ok]] == o.text.to_numpy()[ok]
        aligned = own & ok & same_text
        o["own"] = own
        o["aligned"] = aligned
        o["c_idx"] = np.where(aligned, mapped, -1)
        o["s_corr"] = np.where(aligned, c.surprisal_bits.to_numpy()[np.maximum(j, 0)], np.nan)
        o["c_spacy_idx"] = np.where(aligned, c.spacy_idx.to_numpy()[np.maximum(j, 0)], -1)
        wo = first_subword_windows(tok, e.text, a)
        wc = first_subword_windows(tok, e.corrected, np.where(aligned, mapped, -1))
        o["same_window"] = [aligned[i] and wo[i] is not None and wo[i] == wc[i] for i in range(len(o))]
        o["own_unmatched"] = own & ~aligned
        frames.append(o)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def boot_mean(x, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def level_table(e: pd.DataFrame, cols: list, scale: float = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for lvl in LEVELS:
        d = e[e.level == lvl]
        r = {"level": lvl, "n_essays": len(d)}
        for c in cols:
            bm = boot_mean(d[c], rng) * scale
            r[c] = d[c].mean() * scale
            r[c + "_lo"], r[c + "_hi"] = ci(bm)
        rows.append(r)
    return pd.DataFrame(rows)


def diff(e, col, hi_lvl, lo_lvl, scale=1000, col_ref=None):
    """Mean(hi) - mean(lo) for col, bootstrap CI; with col_ref, the paired
    difference of differences (col - col_ref) over the same resampled essays."""
    rng = np.random.default_rng(SEED)
    H, L = e[e.level == hi_lvl], e[e.level == lo_lvl]
    if col_ref is not None:
        H = H.dropna(subset=[col, col_ref])
        L = L.dropna(subset=[col, col_ref])
    else:
        H = H.dropna(subset=[col])
        L = L.dropna(subset=[col])
    ih = rng.integers(0, len(H), (N_BOOT, len(H)))
    il = rng.integers(0, len(L), (N_BOOT, len(L)))

    hv, lv_ = H[col].to_numpy(), L[col].to_numpy()
    point = hv.mean() - lv_.mean()
    boots = hv[ih].mean(1) - lv_[il].mean(1)
    if col_ref is not None:
        hr, lr = H[col_ref].to_numpy(), L[col_ref].to_numpy()
        point -= hr.mean() - lr.mean()
        boots -= hr[ih].mean(1) - lr[il].mean(1)
    return {"diff": float(point * scale), "ci": [x * scale for x in ci(boots)]}


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    T = tau()
    report = {"tau": T}
    essays = pd.read_parquet(OUT / "essays.parquet")
    edits = pd.read_parquet(OUT / "edits.parquet")
    flagged = pd.read_parquet(OUT / "flagged.parquet")
    orig = pd.read_parquet(OUT / "tokens_orig.parquet")
    corr = pd.read_parquet(OUT / "tokens_corr.parquet")
    gold = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet")
    level = dict(zip(essays.text_id, essays.level))

    # ---- check (a): reproduce stage 1 -------------------------------------
    m = gold[["text_id", "spacy_idx", "surprisal_bits"]].merge(
        orig[["text_id", "spacy_idx", "surprisal_bits"]], on=["text_id", "spacy_idx"],
        how="outer", suffixes=("_stored", "_here"), indicator=True)
    dd = (m.surprisal_bits_stored - m.surprisal_bits_here).abs()
    report["check_reproduce"] = {
        "n_stored": int(len(gold)), "n_here": int(len(orig)),
        "merge": m._merge.value_counts().to_dict(),
        "max_abs_diff_bits": float(dd.max()), "share_exact": float((dd == 0).mean()),
        "share_lt_0.01": float((dd < 0.01).mean()),
        "n_high_flips": int(((m.surprisal_bits_stored >= T) != (m.surprisal_bits_here >= T)).sum())}
    print(json.dumps(report["check_reproduce"], indent=1), flush=True)

    # ---- alignment ----------------------------------------------------------
    o = orig.merge(gold[["text_id", "spacy_idx", "gold", "row_id"]], on=["text_id", "spacy_idx"], how="left")
    o = o[~o.is_space].copy()
    orig_by = {t: d for t, d in o.groupby("text_id")}
    c = corr[~corr.is_space].copy()
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    a = align(essays, flagged, orig_by, c, tok)
    a["level"] = a.text_id.map(level)
    a["high_o"] = a.surprisal_bits >= T
    a["high_c"] = a.s_corr >= T
    a["delta"] = a.surprisal_bits - a.s_corr
    a.drop(columns=["lemma", "tag", "obs_id"]).to_parquet(OUT / "aligned.parquet", index=False)

    # coverage and edge cases
    cov = a.groupby("level").agg(n=("own", "size"), own=("own", "mean"), aligned=("aligned", "mean"),
                                 own_unmatched=("own_unmatched", "sum"),
                                 aligned_of_own=("aligned", lambda s: s[a.loc[s.index, "own"]].mean()),
                                 same_window_of_aligned=("same_window",
                                                         lambda s: s[a.loc[s.index, "aligned"]].mean()))
    cov.loc["all"] = [len(a), a.own.mean(), a.aligned.mean(), a.own_unmatched.sum(),
                      a.aligned[a.own].mean(), a.same_window[a.aligned].mean()]
    cov.reindex(LEVELS + ["all"]).to_csv(TAB / "coverage.csv")
    print(cov.round(4).to_string(), flush=True)
    um = a[a.own_unmatched]
    report["unmatched_own_examples"] = um.text.value_counts().head(25).to_dict()
    c_matched = set(zip(a.text_id[a.aligned], a.c_spacy_idx[a.aligned]))
    c["matched"] = [(t, i) in c_matched for t, i in zip(c.text_id, c.spacy_idx)]
    report["corr_tokens"] = {"n": int(len(c)), "matched_share": float(c.matched.mean())}

    # ---- check (b): identical windows -------------------------------------
    sw = a[a.same_window]
    report["check_same_window"] = {
        "n": int(len(sw)), "max_abs_delta": float(sw.delta.abs().max()),
        "share_abs_lt_0.05": float((sw.delta.abs() < 0.05).mean()),
        "share_exact": float((sw.delta == 0).mean()),
        "high_flips": int((sw.high_o != sw.high_c).sum())}
    print(json.dumps(report["check_same_window"]), flush=True)

    # ---- Q1 contamination -------------------------------------------------
    unc = a[(a.gold == "OK") & a.high_o]  # gold "OK" = uncorrected
    report["q1_uncorr_high_aligned_share"] = float(unc.aligned.mean())
    oka = unc[unc.aligned].copy()
    oka["contaminated"] = ~oka.high_c
    q1 = oka.groupby("level").agg(
        n=("delta", "size"), mean_delta=("delta", "mean"), median_delta=("delta", "median"),
        share_delta_gt1=("delta", lambda s: (s > 1).mean()),
        contaminated=("contaminated", "mean"),
        contaminated_same_window=("contaminated", lambda s: s[oka.loc[s.index, "same_window"]].mean()),
        share_same_window=("same_window", "mean"),
        contaminated_diff_window=("contaminated", lambda s: s[~oka.loc[s.index, "same_window"]].mean()),
        mean_delta_diff_window=("delta", lambda s: s[~oka.loc[s.index, "same_window"]].mean())
    ).reindex(LEVELS)
    q1.to_csv(TAB / "q1_contamination.csv")
    print(q1.round(3).to_string(), flush=True)

    # per essay
    n_scored = a.groupby("text_id").size()
    E = pd.DataFrame({"level": pd.Series(level), "n_orig": n_scored}).dropna(subset=["n_orig"])
    g = oka.groupby("text_id")
    E["n_ok_high_al"] = g.size()
    E["n_contam"] = g.contaminated.sum()
    E = E.fillna({"n_ok_high_al": 0, "n_contam": 0})
    E["contam_share"] = E.n_contam / E.n_ok_high_al.replace(0, np.nan)
    E["uncorr_al_rate"] = E.n_ok_high_al / E.n_orig
    E["contam_rate"] = E.n_contam / E.n_orig
    E["intrinsic_rate"] = (E.n_ok_high_al - E.n_contam) / E.n_orig
    # reverse direction: aligned tokens that rise above tau after correction
    al = a[a.aligned]
    E["n_induced"] = al[~al.high_o & al.high_c].groupby("text_id").size()
    E["n_induced"] = E.n_induced.fillna(0)
    E["induced_rate"] = E.n_induced / E.n_orig

    # ---- Q2 post-correction surprise -------------------------------------
    E["n_corr"] = c.groupby("text_id").size()
    E["n_corr_high"] = c[c.surprisal_bits >= T].groupby("text_id").size()
    E["n_own"] = al.groupby("text_id").size()
    E["n_own_high_c"] = al[al.high_c].groupby("text_id").size()
    E["n_own_high_o"] = al[al.high_o].groupby("text_id").size()
    E = E.fillna({"n_corr_high": 0, "n_own_high_c": 0, "n_own_high_o": 0})
    E["corr_all"] = E.n_corr_high / E.n_corr
    E["corr_own"] = E.n_own_high_c / E.n_own
    E["orig_own"] = E.n_own_high_o / E.n_own
    E["corr_new"] = (E.n_corr_high - E.n_own_high_c) / E.n_corr  # annotator-written tokens
    ge = pd.read_csv(RESULTS_DIR / "study3" / "gold_essays.csv").set_index("text_id")
    E = E.join(ge[["acc_rate", "err_rate", "high_rate"]].rename(columns={"acc_rate": "uncorr_rate"}))
    E.index.name = "text_id"
    E.reset_index().to_csv(TAB / "essays.csv", index=False)

    cols = ["uncorr_rate", "orig_own", "corr_own", "corr_all", "corr_new", "uncorr_al_rate",
            "contam_rate", "intrinsic_rate", "induced_rate"]
    q2 = level_table(E, cols)
    q2["contam_share"] = level_table(E, ["contam_share"], 1)["contam_share"]
    q2.to_csv(TAB / "q2_rates_by_level.csv", index=False)
    print(q2[["level"] + cols + ["contam_share"]].round(2).to_string(index=False), flush=True)

    learners = E[E.level != "N"]
    lv = learners.level.map({"A": 1, "B": 2, "C": 3})
    tests = {"rho_with_level": {}, "C_minus_A": {}, "N_minus_C": {}}
    for col in cols + ["contam_share"]:
        d = learners.dropna(subset=[col])
        rho, lo, hi = boot_spearman(d[col], lv[d.index])
        tests["rho_with_level"][col] = {"rho": rho, "ci": [lo, hi]}
        sc = 1 if col == "contam_share" else 1000
        tests["C_minus_A"][col] = diff(E, col, "C", "A", sc)
        tests["N_minus_C"][col] = diff(E, col, "N", "C", sc)
    tests["primary_dod_corr_own_vs_uncorr"] = diff(E, "corr_own", "C", "A", 1000, col_ref="uncorr_rate")
    tests["dod_corr_own_vs_orig_own"] = diff(E, "corr_own", "C", "A", 1000, col_ref="orig_own")
    tests["dod_orig_own_vs_uncorr"] = diff(E, "orig_own", "C", "A", 1000, col_ref="uncorr_rate")
    tests["dod_corr_all_vs_uncorr"] = diff(E, "corr_all", "C", "A", 1000, col_ref="uncorr_rate")
    tests["N_minus_C_dod_corr_own_vs_uncorr"] = diff(E, "corr_own", "N", "C", 1000, col_ref="uncorr_rate")
    report["q2_tests"] = tests
    print(json.dumps({k: tests[k] for k in tests if k.startswith(("primary", "dod", "N_minus_C_dod"))}, indent=1))

    # ---- Q3 corrections ---------------------------------------------------
    cl = {t: d for t, d in c.groupby("text_id")}
    ol = {t: d for t, d in o.groupby("text_id")}
    rows = []
    for ed in edits.itertuples():
        cd, od = cl.get(ed.text_id), ol.get(ed.text_id)
        if cd is None:
            continue
        r = {"text_id": ed.text_id, "kind": ed.kind, "level": level[ed.text_id]}
        if ed.kind in ("sub", "ins") and ed.ce > ed.cs:
            inside = cd[(cd.idx >= ed.cs) & (cd.idx < ed.ce)]
            r["n_corr_tok"] = len(inside)
            if len(inside):
                f = inside.iloc[0]
                r.update(corr_text=f.text, s_corr_first=f.surprisal_bits, corr_is_punct=f.is_punct)
        else:  # deletion: the next token in the corrected text
            nxt = cd[cd.idx >= ed.cs]
            if len(nxt):
                r.update(corr_text=nxt.iloc[0].text, s_corr_first=nxt.iloc[0].surprisal_bits)
        if ed.e > ed.s:
            oin = od[(od.idx >= ed.s) & (od.idx < ed.e)]
            if len(oin):
                r.update(orig_text=oin.iloc[0].text, s_orig_first=oin.iloc[0].surprisal_bits)
        else:  # insertion: the original token at the insertion point (ERR_INS)
            onx = od[od.idx >= ed.s]
            if len(onx):
                r.update(orig_text=onx.iloc[0].text, s_orig_first=onx.iloc[0].surprisal_bits)
        rows.append(r)
    q3 = pd.DataFrame(rows)
    q3["corr_high"] = q3.s_corr_first >= T
    q3["orig_high"] = q3.s_orig_first >= T
    q3.to_parquet(OUT / "q3_edits.parquet", index=False)
    s3 = q3.dropna(subset=["s_corr_first"])
    t3 = s3.groupby(["kind", "level"]).agg(n=("corr_high", "size"), corr_high=("corr_high", "mean"),
                                           orig_high=("orig_high", "mean"),
                                           corr_high_given_orig_high=(
                                               "corr_high", lambda x: x[s3.loc[x.index, "orig_high"]].mean()),
                                           median_s_corr=("s_corr_first", "median")).reset_index()
    tot = s3.groupby("kind").agg(n=("corr_high", "size"), corr_high=("corr_high", "mean"),
                                 orig_high=("orig_high", "mean"),
                                 corr_high_given_orig_high=(
                                     "corr_high", lambda x: x[s3.loc[x.index, "orig_high"]].mean()),
                                 median_s_corr=("s_corr_first", "median")).reset_index().assign(level="all")
    t3 = pd.concat([t3, tot])
    t3.to_csv(TAB / "q3_corrections.csv", index=False)
    print(t3.round(3).to_string(index=False), flush=True)
    report["q3_base_rate_corr_high"] = float((c.surprisal_bits >= T).mean())
    report["q3_corrections_high_top_words"] = s3[s3.corr_high & (s3.kind != "del")].corr_text.value_counts().head(20).to_dict()

    # ---- Q4 what intrinsic uncorrected surprise is ---------------------------
    g5 = load_gold5()
    bench = pd.read_parquet(CACHE / "bench_wi.parquet", columns=["row_id"])
    att = pd.read_parquet(CACHE / "attestation_wi.parquet")
    bench["ttype"] = g5.token_types(att)
    bench["a_log_uni"] = att.a_log_uni.to_numpy()
    import polars as pl
    total = pl.scan_parquet(str(g5.FREQ / "corpus_a" / "3grams.parquet")).select(pl.col("count").sum()).collect().item()
    bench["freq_pm"] = np.expm1(bench.a_log_uni) / total * 1e6
    q4 = oka.merge(bench, on="row_id", how="left")
    # earlier mention: lowercased word occurs among the essay's earlier tokens
    o_sorted = o.sort_values(["text_id", "spacy_idx"])
    o_sorted["low"] = o_sorted.text.str.lower()
    o_sorted["earlier"] = o_sorted.groupby(["text_id", "low"]).cumcount() > 0
    q4 = q4.merge(o_sorted[["text_id", "spacy_idx", "earlier"]], on=["text_id", "spacy_idx"], how="left")
    q4["cls"] = np.where(q4.contaminated, "contaminated", "intrinsic")
    q4["content"] = q4.pos.isin(g5.CONTENT_POS)
    q4["function"] = q4.pos.isin(g5.FUNCTION_POS)
    q4["propn_num"] = (q4.pos == "PROPN") | q4.like_num
    t4 = q4.groupby(["cls", "level"]).agg(
        n=("delta", "size"), content=("content", "mean"), function=("function", "mean"),
        punct=("is_punct", "mean"), propn_num=("propn_num", "mean"),
        median_freq_pm=("freq_pm", "median"), rare_lt1pm=("freq_pm", lambda s: (s < 1).mean()),
        earlier_mention=("earlier", "mean")).reset_index()
    t4.to_csv(TAB / "q4_kinds.csv", index=False)
    print(t4.round(3).to_string(index=False), flush=True)
    tt = pd.crosstab([q4.cls, q4.level], q4.ttype, normalize="index")
    tt.to_csv(TAB / "q4_ttype.csv")
    pos = pd.crosstab([q4.cls, q4.level], q4.pos, normalize="index")
    pos.to_csv(TAB / "q4_pos.csv")
    q4.drop(columns=["lemma", "tag"], errors="ignore").to_parquet(OUT / "q4_tokens.parquet", index=False)

    (TAB / "report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
