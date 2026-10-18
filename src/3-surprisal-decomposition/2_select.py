"""Study 3, stage 2 — split, threshold, select the high-surprisal tokens, attach gold.

CPU. Reads the stage 1 shards and writes:

    src/3-surprisal-decomposition/split.csv      ELLIPSE dev/test split (committed)
    data/study3/tokens_selected.parquet         ELLIPSE (and TOEFL 11) tokens above tau, keyed by row_id
    data/study3/essay_summary.parquet           per essay: n scored, mean/var surprisal, n_high
    data/study3/wi/tokens_gold.parquet          every scored W&I+LOCNESS token, with its gold category
    data/study3/repair/bench_<name>.parquet     the tokens above tau, one bench per corpus part
    results/study3/threshold.json               tau, counts, the Study 1 reproduction check

Definitions
-----------
* tau = 90th percentile of token surprisal (bits) among **dev** ELLIPSE essays in
  the top Overall tertile (Overall >= the dev 2/3 quantile). The same tau is
  applied to every corpus.
* Excluded before thresholding: whitespace tokens (spaCy ``is_space``). The
  essay's first token is never scored under generative scoring. Punctuation is
  kept.
* W&I gold, from the first annotator's edits (character offsets), per token:
    ERR_TOK  the token overlaps an edited span (substitution or deletion)
    ERR_INS  an insertion point lies between the previous token and this one,
             i.e. a word is missing just before it
    POST     the previous token is ERR_TOK (it may inherit surprisal)
    OK       none of the above

    python src/3-surprisal-decomposition/2_select.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.bench import CACHE, S3  # noqa: E402
from util.paths import PREDICTABILITY_DIR, RESULTS_DIR, WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

OUT = RESULTS_DIR / "study3"
SPLIT_PATH = HERE / "split.csv"
SEED = 42
DEV_FRAC = 0.70
TAU_PCT = 90
TOEFL_PARTS = 2  # TOEFL 11 benches are split by essay so the GPU passes can run in parallel


def load_corpus(corpus: str):
    d = S3 / corpus / "shards"
    stems = sorted(p.with_suffix("") for p in d.glob("shard_*.npz"))
    toks = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    zs = [np.load(s.with_suffix(".npz")) for s in stems]
    ids = np.concatenate([z["topk_ids"] for z in zs])
    probs = np.concatenate([z["topk_probs"] for z in zs])
    assert len(toks) == len(ids) == len(probs)
    toks["corpus"] = corpus
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


def next_words(toks: pd.DataFrame) -> pd.DataFrame:
    """The next scored token of the same essay (where an insertion would go)."""
    toks = toks.sort_values(["text_id", "spacy_idx"])
    same = toks.text_id.to_numpy()[1:] == toks.text_id.to_numpy()[:-1]
    toks["next_char"] = np.append(np.where(same, toks.idx.to_numpy()[1:], -1), -1).astype(float)
    toks["next_word"] = np.append(np.where(same, toks.text.to_numpy()[1:], None), None)
    toks.loc[toks.next_char < 0, ["next_char", "next_word"]] = [np.nan, None]
    return toks


def write_bench(name: str, high: pd.DataFrame, texts: dict) -> int:
    """High tokens (sorted by row_id) -> data/study3/repair/bench_<name>.parquet."""
    h = high.sort_values("row_id").reset_index(drop=True)
    pd.DataFrame({
        "inst": np.arange(len(h)), "row_id": h.row_id.to_numpy(), "text_id": h.text_id,
        "observed": h.text, "pos": h.pos, "target_char": h.idx,
        "next_char": h.next_char, "next_word": h.next_word, "text": h.text_id.map(texts),
    }).to_parquet(CACHE / f"bench_{name}.parquet", index=False)
    return len(h)


def gold_for_doc(doc, edits):
    """Gold category per spaCy token index."""
    spans = [(s, e) for s, e, _ in edits if e > s]
    ins = [s for s, e, _ in edits if e == s]
    cat = {}
    prev_end = 0
    err_prev = False
    for t in doc:
        a, b = t.idx, t.idx + len(t.text)
        if any(s < b and e > a for s, e in spans):
            c = "ERR_TOK"
        elif any(prev_end <= p <= a for p in ins):
            c = "ERR_INS"
        elif err_prev:
            c = "POST"
        else:
            c = "OK"
        cat[t.i] = c
        if not t.is_space:
            err_prev = c == "ERR_TOK"
            prev_end = b
    return cat


def select_wi(tau: float) -> dict:
    toks = pd.concat([pd.read_parquet(p) for p in sorted((S3 / "wi" / "shards").glob("*.parquet"))],
                     ignore_index=True)
    meta, gold = {}, []
    for doc in load_all_docbins(WI_DOCBINS_DIR):
        m = doc.user_data["meta"]
        meta[m["text_id"]] = (m, doc.text)
        for i, c in gold_for_doc(doc, json.loads(m["edits"])).items():
            gold.append((m["text_id"], i, c))
    g = pd.DataFrame(gold, columns=["text_id", "spacy_idx", "gold"])
    toks = toks.merge(g, on=["text_id", "spacy_idx"], how="left")
    toks["level"] = toks.text_id.map(lambda t: meta[t][0]["level"])
    toks["high"] = (toks.surprisal_bits >= tau) & ~toks.is_space
    toks = next_words(toks).reset_index(drop=True)
    toks["row_id"] = np.arange(len(toks))
    toks.drop(columns=["obs_id"]).to_parquet(S3 / "wi" / "tokens_gold.parquet", index=False)
    n = write_bench("wi", toks[toks.high], {t: v[1] for t, v in meta.items()})
    return {"n_scored": int(len(toks)), "n_space": int(toks.is_space.sum()), "n_selected": n}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {"tau_percentile": TAU_PCT, "checks": {}}

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
    sel.to_parquet(S3 / "tokens_selected.parquet", index=False)
    pd.concat(summaries, ignore_index=True).to_parquet(S3 / "essay_summary.parquet", index=False)
    report["n_selected_by_split"] = sel.split.value_counts().to_dict()

    # --- benches for the repair pass -----------------------------------------
    report["benches"] = {}
    for c, (toks, _, _, essays) in corpora.items():
        nxt = next_words(toks)[["text_id", "spacy_idx", "next_char", "next_word"]]
        s = sel[sel.corpus == c].merge(nxt, on=["text_id", "spacy_idx"])
        texts = dict(zip(essays.text_id, essays.text))
        if c == "ellipse":
            parts = {f"ellipse_{p}": s[s.split == p] for p in ["dev", "test"]}
        else:
            ids = np.sort(s.text_id.unique())
            parts = {f"{c}_p{i}": s[s.text_id.isin(ids[i::TOEFL_PARTS])] for i in range(TOEFL_PARTS)}
        for name, part in parts.items():
            report["benches"][name] = write_bench(name, part, texts)

    # --- W&I+LOCNESS: gold labels and its bench ------------------------------
    if (S3 / "wi" / "manifest.json").exists():
        report["exclusions"]["wi"] = select_wi(tau)
        report["benches"]["wi"] = report["exclusions"]["wi"]["n_selected"]

    (OUT / "threshold.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2, default=float))


if __name__ == "__main__":
    main()
