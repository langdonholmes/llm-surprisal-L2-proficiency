"""Study 3, stage 2 — split, threshold, select high-surprisal tokens, compute deltas.

CPU/GPU-light. Reads the stage 1 shards for both corpora and writes:

    src/3-vector-transformations/split.csv      ELLIPSE dev/test split (committed)
    data/study3/tokens_selected.parquet         one row per selected token, keyed by row_id
    data/study3/delta.npy                       [n_selected, 2048] float32, row i == row_id i
    data/study3/essay_summary.parquet           per essay: n scored, mean/var surprisal, n_high
    results/study3/threshold.json               tau, tertile cut, counts, check results

Definitions
-----------
* tau = 90th percentile of token surprisal (bits) among **dev** ELLIPSE essays in
  the top Overall tertile (Overall >= the dev 2/3 quantile). The same tau is
  applied to every ELLIPSE and TOEFL 11 essay.
* Excluded before thresholding: whitespace tokens (spaCy ``is_space``). The
  essay's first token is never scored under generative scoring. Punctuation is
  kept.
* p~ = the top-K (K = 50) predicted probabilities renormalized to sum to one.
  e_pred = sum_i p~_i W_U[i];  e_obs = W_U[obs first subword];  delta = e_pred - e_obs.
  W_U is OLMo-2's output (unembedding) matrix, which is untied from W_E. Only
  delta is stored, since e_obs is a row lookup and e_pred = delta + e_obs.

    python src/3-vector-transformations/2_deltas.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.paths import DATA_DIR, PREDICTABILITY_DIR, RESULTS_DIR

HERE = Path(__file__).resolve().parent
S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"
SPLIT_PATH = HERE / "split.csv"
SEED = 42
DEV_FRAC = 0.70
TAU_PCT = 90
TOP_K = 50
CHUNK = 4096


def load_corpus(corpus: str):
    d = S3 / corpus / "shards"
    stems = sorted(p.with_suffix("") for p in d.glob("shard_*.npz"))
    toks = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    zs = [np.load(s.with_suffix(".npz")) for s in stems]
    ids = np.concatenate([z["topk_ids"] for z in zs])
    probs = np.concatenate([z["topk_probs"] for z in zs])
    assert len(toks) == len(ids) == len(probs)
    toks["corpus"] = corpus
    toks["pos_in_corpus"] = np.arange(len(toks))
    essays = pd.read_parquet(S3 / corpus / "essays.parquet")
    return toks, ids, probs, essays


def check_against_study1(corpus: str, toks: pd.DataFrame) -> dict:
    s1 = pd.read_parquet(PREDICTABILITY_DIR / corpus / "olmo2-1b" / "surprisal.parquet")
    s1 = s1[s1.window == "8"].set_index("text_id")
    m = toks.groupby("text_id").surprisal_bits.agg(["mean", "size"]).join(
        s1[["mean_surprisal_bits", "n_tokens_scored"]], how="outer")
    return {
        "n_essays_study1": int(s1.shape[0]),
        "n_essays_here": int(toks.text_id.nunique()),
        "max_abs_diff_mean_bits": float((m["mean"] - m.mean_surprisal_bits).abs().max()),
        "max_abs_diff_n_tokens": int((m["size"] - m.n_tokens_scored).abs().max()),
    }


def make_split(essays: pd.DataFrame) -> pd.DataFrame:
    """70/30 dev/test, stratified by prompt x Overall tertile (whole-corpus cuts)."""
    if SPLIT_PATH.exists():
        return pd.read_csv(SPLIT_PATH, dtype={"text_id": str})
    e = essays[["text_id", "prompt", "overall"]].copy()
    e["band"] = pd.qcut(e.overall.rank(method="first"), 3, labels=False)
    strata = e.prompt + "|" + e.band.astype(str)
    dev, _ = train_test_split(e.text_id, train_size=DEV_FRAC, stratify=strata, random_state=SEED)
    split = pd.DataFrame({"text_id": e.text_id,
                          "split": np.where(e.text_id.isin(set(dev)), "dev", "test")})
    split.to_csv(SPLIT_PATH, index=False)
    return split


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"top_k": TOP_K, "tau_percentile": TAU_PCT, "checks": {}}

    # A corpus is used once its stage 1 manifest exists (the pass is complete).
    # ELLIPSE comes first, so its row_ids do not change when TOEFL 11 is added.
    corpora = {c: load_corpus(c) for c in ["ellipse", "toefl11"]
               if (S3 / c / "manifest.json").exists()}
    report["corpora"] = list(corpora)
    for c, (toks, *_rest) in corpora.items():
        report["checks"][c] = check_against_study1(c, toks)
    print(json.dumps(report["checks"], indent=2))

    # --- split + threshold (ELLIPSE dev only) --------------------------------
    el_toks, _, _, el_essays = corpora["ellipse"]
    split = make_split(el_essays)
    lab = el_essays[["text_id", "overall"]].merge(split, on="text_id")
    dev_lab = lab[lab.split == "dev"]
    top_cut = float(dev_lab.overall.quantile(2 / 3))
    ref_ids = set(dev_lab.text_id[dev_lab.overall >= top_cut])
    ref = el_toks[el_toks.text_id.isin(ref_ids) & ~el_toks.is_space]
    tau = float(np.percentile(ref.surprisal_bits, TAU_PCT))
    report.update({
        "dev_top_tertile_overall_cut": top_cut,
        "n_reference_essays": len(ref_ids),
        "share_dev_essays_in_reference": len(ref_ids) / len(dev_lab),
        "n_reference_tokens": int(len(ref)),
        "tau_bits": tau,
        "n_split": split.split.value_counts().to_dict(),
    })
    print(f"tau = {tau:.3f} bits  (top tertile Overall >= {top_cut}, {len(ref_ids)} essays)")

    # --- select ---------------------------------------------------------------
    sel_frames, sel_ids, sel_probs, summaries = [], [], [], []
    for c, (toks, ids, probs, _) in corpora.items():
        toks["is_high"] = (toks.surprisal_bits >= tau) & ~toks.is_space
        nonspace = toks[~toks.is_space]
        summaries.append(nonspace.groupby("text_id").agg(
            n_scored=("surprisal_bits", "size"),
            mean_surprisal=("surprisal_bits", "mean"),
            var_surprisal=("surprisal_bits", "var"),
            n_high=("is_high", "sum"),
        ).reset_index().assign(corpus=c))
        keep = toks.is_high.to_numpy()
        report.setdefault("exclusions", {})[c] = {
            "n_scored": int(len(toks)), "n_space": int(toks.is_space.sum()),
            "n_selected": int(keep.sum())}
        sel_frames.append(toks[keep])
        sel_ids.append(ids[keep])
        sel_probs.append(probs[keep])

    sel = pd.concat(sel_frames, ignore_index=True)
    top_ids = np.concatenate(sel_ids)
    top_p = np.concatenate(sel_probs).astype(np.float64)
    sel["row_id"] = np.arange(len(sel))
    sel = sel.merge(split, on="text_id", how="left")
    sel.loc[sel.corpus == "toefl11", "split"] = "toefl"
    assert sel.row_id.is_monotonic_increasing

    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    obs = sel.obs_id.to_numpy()
    in_top = top_ids == obs[:, None]
    sel["p_obs"] = 2.0 ** (-sel.surprisal_bits)
    sel["topk_mass"] = top_p.sum(1)
    sel["obs_rank_in_topk"] = np.where(in_top.any(1), in_top.argmax(1), -1)
    for j in range(3):
        sel[f"pred{j + 1}"] = tok.convert_ids_to_tokens(top_ids[:, j].tolist())
        sel[f"pred{j + 1}_p"] = top_p[:, j]
    sel["pred1_id"] = top_ids[:, 0]
    sel["obs_subword"] = tok.convert_ids_to_tokens(obs.tolist())
    sel = sel.drop(columns=["pos_in_corpus"])

    # --- deltas -------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-2-0425-1B", revision="a1847dff35000b4271fa70afc5db10fd29fedbdf",
        torch_dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach()
    assert W_U.data_ptr() != model.get_input_embeddings().weight.data_ptr()  # untied
    np.save(S3 / "W_U.npy", W_U.numpy())  # stage 3 needs e_obs = W_U[obs_id]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W_U = W_U.to(dev)
    del model

    delta = np.lib.format.open_memmap(S3 / "delta.npy", mode="w+", dtype=np.float32,
                                      shape=(len(sel), W_U.shape[1]))
    p_tilde = top_p / top_p.sum(1, keepdims=True)
    for a in range(0, len(sel), CHUNK):
        b = min(a + CHUNK, len(sel))
        ids_t = torch.from_numpy(top_ids[a:b].astype(np.int64)).to(dev)
        p_t = torch.from_numpy(p_tilde[a:b]).to(dev, torch.float32)
        e_pred = torch.einsum("nk,nkd->nd", p_t, W_U[ids_t])
        e_obs = W_U[torch.from_numpy(obs[a:b].astype(np.int64)).to(dev)]
        delta[a:b] = (e_pred - e_obs).cpu().numpy()
    delta.flush()
    # one-hot sanity: if p~ were one-hot on the observed id, delta would be 0
    report["delta_shape"] = list(delta.shape)

    sel.to_parquet(S3 / "tokens_selected.parquet", index=False)
    pd.concat(summaries, ignore_index=True).to_parquet(S3 / "essay_summary.parquet", index=False)
    report["n_selected_by_split"] = sel.split.value_counts().to_dict()
    report["share_obs_in_topk"] = float((sel.obs_rank_in_topk >= 0).mean())
    report["mean_topk_mass"] = float(sel.topk_mass.mean())
    (OUT / "threshold.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2, default=float))


if __name__ == "__main__":
    main()
