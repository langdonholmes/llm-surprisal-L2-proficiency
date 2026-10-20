"""Association pilot, step 3 (CPU) — token-level LLM association measures and their D24 split.

Design as revised (NOTES.md, revision R1). Per original token (W&I learners, FCE
scripts with score > 0), from the pilot_corrected / pilot_fce alignment:

    s_o, s_c          first-subword surprisal, original / corrected essay (bits)
    id_o, id_c        the first-subword id scored in each version
    real              alphabetic, not number-like, not PROPN, form attested in corpus a
    sw                real and a single subword (decoded id_o == token text): the primary set
    content           sw and NOUN/VERB/ADJ/ADV
    att1, att2        left bigram (previous spaCy token, w) attested in corpus a / both
                      left and right bigrams attested (previous and next token, any kind,
                      whitespace tokens included, as the 3-gram tables count them)
    lowS              s_o <= tau (the surprisal cutoff; selects on the outcome)
    logf              log10 f(w), f = count_a(w) / N_a (lowercased form)

  primary (denominator f(w)):
    pmi_o/_c          log2 p(w|ctx) - log2 f(w)
    dp_o/_c           p(w|ctx) - f(w)
    fadj_o/_c         pmi minus its OLS prediction from logf (fit on original sw tokens)
    ge3_o/_c          pmi >= 3
    hi_o/_c           s > tau
  sensitivity (denominator over first-subword ids):
    pmim_o/_c         model marginal p̄, corpus-a contexts (all pbar_a*.npz runs pooled)
    pmimb_o/_c        model marginal p̄, corpus-b contexts
    pmis_o/_c         observed subword relative frequency at the same corpus-a positions
                      (add-0.5 smoothing)

Then the exact D24 split (pilot_corrected/5_decompose.split) of every token-mean
measure over its token subset, per unit.

Writes data/study3/pilot_assoc/:
    tokens_<corpus>.parquet          per-token table
    split_<corpus>.parquet           per unit x measure: total, clean, omission, context, error_tokens
    pbar_summary.json                p̄ precision and agreement of the three denominators

    python src/3-surprisal-decomposition/pilot_assoc/3_tokens.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assoc_common import CONTENT, FREQ, OUT, S3_SRC, TOKDIR, load, tau  # noqa: E402

sys.path.insert(0, str(S3_SRC / "pilot_corrected"))
sys.path.insert(0, str(S3_SRC))
D = load("pc_5_decompose", S3_SRC / "pilot_corrected" / "5_decompose.py")
from lib.lm import HF_ID, REVISION  # noqa: E402

PAD = "#"
# measure -> (orig column, corrected column, token subset)
TOKEN_MEASURES = {
    "surprisal": ("s_o", "s_c", "all"),
    "surprisal_sw": ("s_o", "s_c", "sw"),
    "llm_pmi": ("pmi_o", "pmi_c", "sw"),
    "llm_dp": ("dp_o", "dp_c", "sw"),
    "llm_pmi_content": ("pmi_o", "pmi_c", "content"),
    "llm_pmi_fadj": ("fadj_o", "fadj_c", "sw"),
    "llm_pmi_ge3": ("ge3_o", "ge3_c", "sw"),
    "high_s_content": ("hi_o", "hi_c", "content"),
    "llm_pmi_att1": ("pmi_o", "pmi_c", "sw_att1"),
    "llm_pmi_att2": ("pmi_o", "pmi_c", "sw_att2"),
    "llm_dp_att1": ("dp_o", "dp_c", "sw_att1"),
    "llm_dp_att2": ("dp_o", "dp_c", "sw_att2"),
    "surprisal_att2": ("s_o", "s_c", "sw_att2"),
    "llm_pmi_lowS": ("pmi_o", "pmi_c", "sw_lowS"),
    "surprisal_lowS": ("s_o", "s_c", "lowS"),
    "llm_pmi_pbar": ("pmim_o", "pmim_c", "sw"),
    "llm_pmi_pbar_b": ("pmimb_o", "pmimb_c", "sw"),
    "llm_pmi_subfreq": ("pmis_o", "pmis_c", "sw"),
}


def pbar_runs(prefix):
    """Pool the pbar runs of a corpus; also return every half (for precision)."""
    files = sorted(OUT.glob(f"pbar_{prefix}*.npz")) if prefix == "a" else [OUT / "pbar_b.npz"]
    files = [f for f in files if f.name.startswith(f"pbar_{prefix}")]
    S, n, obs, halves, metas = 0, 0, 0, [], []
    for f in files:
        z = np.load(f)
        S = S + z["sum_p_0"] + z["sum_p_1"]
        n += int(z["n_0"]) + int(z["n_1"])
        obs = obs + z["obs"]
        halves += [z["sum_p_0"] / z["n_0"], z["sum_p_1"] / z["n_1"]]
        metas.append(json.loads(str(z["meta"])))
    return S / n, obs, halves, metas, n


def bigram_attestation(a, corpus):
    """Left / right bigram attestation in corpus a for each original token."""
    it = pd.read_parquet(OUT / f"items_tok_{corpus}_orig.parquet", columns=["text_id", "spacy_idx", "lower"])
    it = it.sort_values(["text_id", "spacy_idx"])
    g = it.groupby("text_id").lower
    it["prev"] = g.shift(1).fillna(PAD)
    it["next"] = g.shift(-1).fillna(PAD)
    keys = pl.concat([pl.from_pandas(it[["prev", "lower"]].rename(columns={"prev": "x", "lower": "y"})),
                      pl.from_pandas(it[["lower", "next"]].rename(columns={"lower": "x", "next": "y"}))]).unique()
    lf = pl.scan_parquet(str(FREQ / "corpus_a" / "3grams.parquet"))
    b = (lf.join(keys.lazy(), left_on=["token_1", "token_2"], right_on=["x", "y"], how="semi")
         .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    B = set(zip(b["token_1"].to_list(), b["token_2"].to_list()))
    it["att_left"] = [k in B for k in zip(it.prev, it.lower)]
    it["att_right"] = [k in B for k in zip(it.lower, it.next)]
    return a.merge(it[["text_id", "spacy_idx", "att_left", "att_right"]], on=["text_id", "spacy_idx"], how="left")


def tokens(corpus, den, tok, t, Na):
    d = TOKDIR[corpus]
    a = pd.read_parquet(d / "aligned.parquet")
    a = a[~a.is_space].copy()
    if corpus == "wi":
        a = a[a.level != "N"]
        a["y"] = a.level.map({"A": 1, "B": 2, "C": 3}).astype(float)
        a["unit"] = a.text_id
    else:
        a["y"] = pd.to_numeric(a.script_s, errors="coerce")
        a = a[a.y > 0]
        a["unit"] = a.script_id
    o = pd.read_parquet(d / "tokens_orig.parquet", columns=["text_id", "spacy_idx", "obs_id"])
    a = a.merge(o.rename(columns={"obs_id": "id_o"}), on=["text_id", "spacy_idx"], how="left")
    k = pd.read_parquet(d / "tokens_corr.parquet", columns=["text_id", "spacy_idx", "obs_id"])
    a = a.merge(k.rename(columns={"spacy_idx": "c_spacy_idx", "obs_id": "id_c"}),
                on=["text_id", "c_spacy_idx"], how="left")
    oc = a.own & a.aligned
    a.loc[~oc, "id_c"] = 0
    it = pd.read_parquet(OUT / f"items_tok_{corpus}_orig.parquet", columns=["text_id", "spacy_idx", "lower", "cnt_a"])
    a = a.merge(it, on=["text_id", "spacy_idx"], how="left")
    a = bigram_attestation(a, corpus)
    assert a.cnt_a.notna().all() and a.id_o.notna().all() and a.att_left.notna().all()
    oc = (a.own & a.aligned).to_numpy()
    io, ic = a.id_o.to_numpy().astype(int), a.id_c.to_numpy().astype(int)

    a["s_o"], a["s_c"] = a.surprisal_bits, np.where(oc, a.s_corr, np.nan)
    a["real"] = a.is_alpha & ~a.like_num & (a.pos != "PROPN") & (a.cnt_a > 0)
    dec = {i: tok.decode([i]) for i in np.unique(io)}
    a["sw"] = a.real & np.array([dec[i].strip() == w for i, w in zip(io, a.text)])
    a["content"] = a.sw & a.pos.isin(CONTENT)
    a["att1"] = a.att_left.astype(bool)
    a["att2"] = a.att1 & a.att_right.astype(bool)
    a["lowS"] = a.s_o <= t
    a["sw_att1"], a["sw_att2"], a["sw_lowS"] = a.sw & a.att1, a.sw & a.att2, a.sw & a.lowS
    f = a.cnt_a.clip(lower=1) / Na
    a["logf"] = np.log10(f)
    a["pmi_o"] = -a.s_o - np.log2(f)
    a["pmi_c"] = -a.s_c - np.log2(f)
    a["dp_o"] = 2.0 ** -a.s_o - f
    a["dp_c"] = 2.0 ** -a.s_c - f
    r = a[a.sw]
    b1, b0 = np.polyfit(r.logf, r.pmi_o, 1)
    a["fadj_o"] = a.pmi_o - (b0 + b1 * a.logf)
    a["fadj_c"] = a.pmi_c - (b0 + b1 * a.logf)
    a["ge3_o"] = (a.pmi_o >= 3).astype(float)
    a["ge3_c"] = np.where(oc, (a.pmi_c >= 3).astype(float), np.nan)
    a["hi_o"] = (a.s_o > t).astype(float)
    a["hi_c"] = np.where(oc, (a.s_c > t).astype(float), np.nan)
    for tag, p in [("m", den["pbar_a"]), ("mb", den["pbar_b"]), ("s", den["subfreq"])]:
        lp = np.log2(p)
        a[f"pmi{tag}_o"] = -a.s_o - lp[io]
        a[f"pmi{tag}_c"] = np.where(oc, -a.s_c - lp[ic], np.nan)
    a["same_id"] = oc & (io == ic)
    info = {"fadj_slope_per_log10": float(b1), "fadj_intercept": float(b0),
            "share_real": float(a.real.mean()), "share_sw": float(a.sw.mean()),
            "share_sw_of_real": float(a.sw.sum() / a.real.sum()), "share_content": float(a.content.mean()),
            "share_att1_of_sw": float(a.sw_att1.sum() / a.sw.sum()), "share_att2_of_sw": float(a.sw_att2.sum() / a.sw.sum()),
            "own_aligned_same_id": float(a.same_id.sum() / oc.sum()),
            "rho_pmi_logf_sw": float(spearmanr(r.pmi_o, r.logf)[0]),
            "rho_s_logf_sw": float(spearmanr(r.s_o, r.logf)[0]),
            "rho_pmi_s_sw": float(spearmanr(r.pmi_o, r.s_o)[0]),
            "mean_pmi_sw": float(r.pmi_o.mean())}
    # agreement of the three denominators on the essay's sw tokens (log2, bits)
    lf = np.log2(f[a.sw].to_numpy())
    agree = {}
    for name, p in den.items():
        lq = np.log2(p[io[a.sw.to_numpy()]])
        agree[f"log2 f vs log2 {name}"] = {"pearson": float(np.corrcoef(lf, lq)[0, 1]),
                                           "spearman": float(spearmanr(lf, lq)[0]),
                                           "median_abs_diff_bits": float(np.median(np.abs(lf - lq))),
                                           "mean_diff_bits": float(np.mean(lq - lf))}
    lpa, lps = np.log2(den["pbar_a"][io[a.sw]]), np.log2(den["subfreq"][io[a.sw]])
    agree["log2 pbar_a vs log2 subfreq"] = {"pearson": float(np.corrcoef(lpa, lps)[0, 1]),
                                            "median_abs_diff_bits": float(np.median(np.abs(lpa - lps)))}
    info["denominator_agreement_tokens"] = agree
    keep = ["unit", "text_id", "spacy_idx", "c_spacy_idx", "y", "gold", "own", "aligned", "text", "pos", "is_alpha",
            "is_punct", "like_num", "lower", "cnt_a", "real", "sw", "content", "att1", "att2", "lowS", "sw_att1",
            "sw_att2", "sw_lowS", "id_o", "id_c", "s_o", "s_c", "logf", "pmi_o", "pmi_c", "dp_o", "dp_c",
            "fadj_o", "fadj_c", "ge3_o", "ge3_c", "hi_o", "hi_c", "pmim_o", "pmim_c", "pmimb_o", "pmimb_c",
            "pmis_o", "pmis_c"]
    return a[keep], info


def splits(a):
    """D24 split per unit for every token-mean measure."""
    b = a.rename(columns={"text_id": "answer_id", "unit": "text_id", "y": "level"})
    rows = []
    for name, (so, sc, s) in TOKEN_MEASURES.items():
        x = b if s == "all" else b[b[s].to_numpy()]
        x = x.assign(**{sc: x[sc].fillna(0.0)})  # only read on own & aligned rows
        e = D.split(x, so, sc)
        e["measure"] = name
        rows.append(e.reset_index())
    return pd.concat(rows, ignore_index=True).rename(columns={"text_id": "unit", "level": "y"})


def main():
    na = OUT / "N_a.json"
    if not na.exists():
        N = int(pl.scan_parquet(str(FREQ / "corpus_a" / "3grams.parquet")).select(pl.col("count").sum()).collect().item())
        na.write_text(json.dumps({"N_a": N}))
    Na = json.loads(na.read_text())["N_a"]
    pa, obs, halves, metas_a, n_a = pbar_runs("a")
    pb, _, halves_b, metas_b, n_b = pbar_runs("b")
    V = len(pa)
    sub = (obs + 0.5) / (obs.sum() + 0.5 * V)
    den = {"pbar_a": pa, "pbar_b": pb, "subfreq": sub}
    summ = {"pbar": {"n_contexts_a": n_a, "n_contexts_b": n_b, "runs_a": metas_a, "runs_b": metas_b,
                     "obs_positions_a": int(obs.sum())}}
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    t = tau()
    H = np.log2(np.stack(halves))  # (k halves, V)
    for corpus in ["wi", "fce"]:
        a, info = tokens(corpus, den, tok, t, Na)
        # precision of log2 p̄_a for the ids the essays use (token-weighted): SE from the k halves
        ids = a.id_o[a.sw].to_numpy().astype(int)
        se = H[:, ids].std(0, ddof=1) / np.sqrt(H.shape[0])
        info["pbar_a_se_bits_tokens"] = {"median": float(np.median(se)), "p95": float(np.percentile(se, 95)),
                                         "max": float(se.max()), "k_halves": int(H.shape[0])}
        ab = np.log2(pa[ids]) - np.log2(pb[ids])
        info["pbar_a_vs_b_tokens"] = {"median_abs_diff_bits": float(np.median(np.abs(ab))),
                                      "p95_abs_diff_bits": float(np.percentile(np.abs(ab), 95))}
        info["subfreq_ids_unseen_share_tokens"] = float((obs[ids] == 0).mean())
        a.to_parquet(OUT / f"tokens_{corpus}.parquet", index=False)
        s = splits(a)
        s.to_parquet(OUT / f"split_{corpus}.parquet", index=False)
        summ[corpus] = info
        print(corpus, json.dumps(info, indent=1), flush=True)
        print(s.groupby("measure")[["total", "clean", "omission", "context", "error_tokens", "n"]].mean().round(4))
    (OUT / "pbar_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
