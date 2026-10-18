"""Lab — turn the real selected tokens into a lab "bench", so any lab
representation can be featurized on real essays unchanged.

Writes data/study3/lab/bench_real_<corpus>.parquet with the bench columns
(inst, text_id, type="REAL", orig, observed, group, target_char, next_char,
next_word, text) plus ``row_id``, the key into the main pipeline's
tokens_selected.parquet. Then run the base pass and the featurizers with
LAB_BENCH=real_<corpus>:

    python src/3-vector-transformations/lab/real.py --corpus ellipse [--split dev]
    LAB_BENCH=real_ellipse LAB_LAYERS= CUDA_VISIBLE_DEVICES=0 python src/3-vector-transformations/lab/extract.py

``LAB_LAYERS=`` (empty) skips the hidden states, which would be ~12 GB per
corpus and no finalist needs them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.paths import DATA_DIR  # noqa: E402

S3 = DATA_DIR / "study3"
LAB = S3 / "lab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["ellipse", "toefl11"])
    ap.add_argument("--split", default=None, help="restrict to one split (dev/test)")
    ap.add_argument("--part", default=None, help="i/n: keep essay-level part i of n (parallel runs)")
    args = ap.parse_args()

    sel = pd.read_parquet(S3 / "tokens_selected.parquet",
                          columns=["row_id", "corpus", "split", "text_id", "spacy_idx", "idx", "text", "lemma"])
    sel = sel[sel.corpus == args.corpus]
    if args.split:
        sel = sel[sel.split == args.split]
    if args.part:
        i, n = map(int, args.part.split("/"))
        ids = np.sort(sel.text_id.unique())
        sel = sel[sel.text_id.isin(ids[i::n])]

    # next word = next scored token of the same essay (stage 1 shards hold every scored token)
    toks = pd.concat([pd.read_parquet(p, columns=["text_id", "spacy_idx", "idx", "text"])
                      for p in sorted((S3 / args.corpus / "shards").glob("*.parquet"))])
    toks = toks.sort_values(["text_id", "spacy_idx"])
    same = toks.text_id.to_numpy()[1:] == toks.text_id.to_numpy()[:-1]
    toks["next_char"] = np.append(np.where(same, toks.idx.to_numpy()[1:], -1), -1)
    toks["next_word"] = np.append(np.where(same, toks.text.to_numpy()[1:], None), None)
    sel = sel.merge(toks[["text_id", "spacy_idx", "next_char", "next_word"]], on=["text_id", "spacy_idx"])
    sel.loc[sel.next_char < 0, ["next_char", "next_word"]] = [np.nan, None]

    essays = pd.read_parquet(S3 / args.corpus / "essays.parquet", columns=["text_id", "text"])
    texts = dict(zip(essays.text_id, essays.text))
    sel = sel.sort_values("row_id").reset_index(drop=True)
    b = pd.DataFrame({
        "inst": np.arange(len(sel)), "row_id": sel.row_id.to_numpy(), "text_id": sel.text_id,
        "type": "REAL", "orig": sel.text, "observed": sel.text, "group": sel.lemma.str.lower(),
        "target_char": sel.idx, "next_char": sel.next_char, "next_word": sel.next_word,
        "text": sel.text_id.map(texts), "matched_to": np.nan, "matched_type": None,
    })
    name = f"real_{args.corpus}" + (f"_{args.split}" if args.split else "") \
        + (f"_p{args.part.split('/')[0]}" if args.part else "")
    b.to_parquet(LAB / f"bench_{name}.parquet", index=False)
    print(name, len(b))


if __name__ == "__main__":
    main()
