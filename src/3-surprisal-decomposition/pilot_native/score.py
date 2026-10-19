"""Pilot (native calibration), GPU — score documents with the stage 1 instrument.

Same model (OLMo-2 1B base @ a1847dff), window (8 subwords ending at the
word's first subword), batch size (32), dtype (bf16 forward, fp32 softmax) and
``features.predictability.Predictor`` path as ``1_extract.py``; spaCy parse as
in ``src/essays/ingest.py`` (en_core_web_lg, NER disabled).

    score.py docs   IN.parquet OUT.parquet      text_id, group, text -> token rows
    score.py verify                              rescore W&I + ELLIPSE essays from raw text

``verify`` re-parses VERIFY_N essays per corpus from ``essays.parquet`` (the raw
text, not the stored DocBins, so the whole path is tested) and compares
token-level surprisal with the stage 1 shards, and the essay mean with the
Study 1 table (ELLIPSE, window 8).
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1-predictability-benchmark"))
sys.path.insert(0, str(HERE.parents[1]))
from models import MODEL_REGISTRY  # noqa: E402
from run_surprisal import load_predictor, window_int  # noqa: E402
from util.paths import DATA_DIR  # noqa: E402

LOG2E = 1.0 / math.log(2.0)
MODEL_KEY, WINDOW, BATCH_SIZE = "olmo2-1b", "8", 32
REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
VERIFY_N = 25
OUT = DATA_DIR / "study3" / "pilot_native"


class Recorder:
    def __init__(self):
        self.rows = []

    def __call__(self, spacy_idx, target_id, pred_logits):
        logp = torch.log_softmax(pred_logits.float(), dim=-1)
        self.rows.append((spacy_idx, target_id, -logp[target_id].item() * LOG2E))


def setup():
    spec = MODEL_REGISTRY[MODEL_KEY]
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    predictor, commit = load_predictor(spec, torch.bfloat16, "cuda", BATCH_SIZE)
    assert commit == REVISION, f"model revision {commit} != {REVISION}"
    rec = Recorder()
    predictor.on_score = rec
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    return predictor, rec, window_int(spec, WINDOW), nlp, commit


def score_texts(predictor, rec, w, nlp, ids, texts) -> pd.DataFrame:
    out = []
    for tid, doc in zip(ids, nlp.pipe(texts, batch_size=64)):
        rec.rows = []
        predictor(doc, window_size=w)
        for si, oid, s in rec.rows:
            t = doc[si]
            out.append((tid, si, t.idx, t.text, t.is_punct, t.is_space, oid, s))
    return pd.DataFrame(out, columns=["text_id", "spacy_idx", "idx", "text", "is_punct",
                                      "is_space", "obs_id", "surprisal_bits"])


def verify(predictor, rec, w, nlp):
    rng = np.random.default_rng(42)
    report = {}
    for corpus in ["wi", "ellipse"]:
        ess = pd.read_parquet(DATA_DIR / "study3" / corpus / "essays.parquet")
        ess = ess.iloc[np.sort(rng.choice(len(ess), VERIFY_N, replace=False))]
        mine = score_texts(predictor, rec, w, nlp, ess.text_id.tolist(), ess.text.tolist())
        keep = set(ess.text_id)
        stored = pd.concat([pd.read_parquet(f, columns=["text_id", "spacy_idx", "surprisal_bits",
                                                        "is_space"],
                                            filters=[("text_id", "in", list(keep))])
                            for f in sorted(glob.glob(str(DATA_DIR / "study3" / corpus / "shards" / "*.parquet")))])
        m = stored.merge(mine, on=["text_id", "spacy_idx"], how="outer", suffixes=("_st", "_me"),
                         indicator=True)
        both = m[m._merge == "both"]
        d = (both.surprisal_bits_st - both.surprisal_bits_me).abs()
        r = {"n_essays": VERIFY_N, "n_stored": int(len(stored)), "n_mine": int(len(mine)),
             "n_matched": int(len(both)), "max_abs_diff": float(d.max()),
             "share_lt_0.01": float((d < 0.01).mean()), "share_exact": float((d == 0).mean())}
        em = mine.groupby("text_id").surprisal_bits.mean()
        es = stored.groupby("text_id").surprisal_bits.mean()
        r["essay_mean_max_abs_diff"] = float((em - es).abs().max())
        if corpus == "ellipse":
            s1 = pd.read_parquet(DATA_DIR / "predictability" / "ellipse" / MODEL_KEY / "surprisal.parquet")
            s1 = s1[s1.window.astype(str) == WINDOW].set_index("text_id").mean_surprisal_bits
            r["study1_mean_max_abs_diff"] = float((em - s1.reindex(em.index)).abs().max())
            ns = mine[~mine.is_space].groupby("text_id").surprisal_bits.mean()
            r["study1_vs_nonspace_mean_max_abs_diff"] = float((ns - s1.reindex(ns.index)).abs().max())
            r["share_space_tokens_scored"] = float(mine.is_space.mean())
        report[corpus] = r
        print(corpus, r, flush=True)
    (OUT / "verify.json").write_text(json.dumps(report, indent=2))


def main():
    predictor, rec, w, nlp, commit = setup()
    if sys.argv[1] == "verify":
        verify(predictor, rec, w, nlp)
        return
    _, _, src, dst = sys.argv
    docs = pd.read_parquet(src)
    df = score_texts(predictor, rec, w, nlp, docs.text_id.tolist(), docs.text.tolist())
    df.to_parquet(dst, index=False)
    print("scored", len(docs), "docs", len(df), "tokens ->", dst, "revision", commit)


if __name__ == "__main__":
    main()
