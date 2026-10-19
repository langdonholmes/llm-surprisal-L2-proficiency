"""Stratified sample of W&I+LOCNESS high tokens, with gold, baselines and sentence start.

    .venv/bin/python src/3-surprisal-decomposition/pilot_lens/sample.py
Writes data/study3/pilot_lens/sample.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from lib.bench import CACHE, S3  # noqa: E402
from util.paths import WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

OUT = S3 / "pilot_lens"
SEED = 42
PER_LEVEL = 3000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = pd.read_parquet(CACHE / "bench_wi.parquet")
    meta = pd.read_parquet(CACHE / "base_meta_wi.parquet")
    att = pd.read_parquet(CACHE / "attestation_wi.parquet")
    b = pd.concat([b, meta[["has_target"]], att[["a_log_uni", "a_unseen_uni", "is_punct"]]], axis=1)
    g = pd.read_parquet(S3 / "wi" / "tokens_gold.parquet",
                        columns=["row_id", "level", "gold", "surprisal_bits", "entropy_bits", "spacy_idx"])
    b = b.merge(g, on="row_id", how="left")
    b = b[b.has_target]
    rng = np.random.default_rng(SEED)
    parts = []
    for lvl, d in b.groupby("level"):
        n = len(d) if lvl == "N" else PER_LEVEL
        parts.append(d.iloc[np.sort(rng.choice(len(d), min(n, len(d)), replace=False))])
    s = pd.concat(parts).reset_index(drop=True)

    # sentence start (char) of each sampled token, from the spaCy docs
    want = set(s.text_id)
    sent_start = {}
    for doc in load_all_docbins(WI_DOCBINS_DIR):
        tid = doc.user_data["meta"]["text_id"]
        if tid not in want:
            continue
        for sent in doc.sents:
            for t in sent:
                sent_start[(tid, t.idx)] = (sent.start_char, sent.end_char)
    se = [sent_start.get((t, int(c)), (-1, -1)) for t, c in zip(s.text_id, s.target_char)]
    s["sent_char"] = [a for a, _ in se]
    s["sent_end_char"] = [b for _, b in se]
    assert (s.sent_char >= 0).mean() > 0.99, (s.sent_char >= 0).mean()
    s["err"] = s.gold.isin(["ERR_TOK", "ERR_INS"])
    s.to_parquet(OUT / "sample.parquet", index=False)
    print(pd.crosstab(s.level, s.gold), len(s), s.text_id.nunique(), "essays")


if __name__ == "__main__":
    main()
