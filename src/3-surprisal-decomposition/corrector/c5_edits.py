"""Corrector, stage c5 (CPU) — character edits with ERRANT types, for stage 1.

Turns every (original, corrected) pair of one source into ERRANT's edits on
the characters of the original (``common.char_edits``), so that stage 1 can
apply them, and stage 4 apply any subset of them, as it does the annotators'
json edits. Applied together, a text's edits give the corrected text exactly
(checked for every text). For the model sources, the correction first takes
the original's leading and trailing whitespace, which generation does not
reproduce (a dropped final newline is not a correction).

    --source model     the five cross-fitted fold models' corrections of W&I/FCE
    --source gold      the first annotator's corrected texts of W&I/FCE, re-segmented
                       by ERRANT in the same way (so that model and gold differ only
                       in the corrections, not in how edits are cut)
    --source ellipse   the final model's corrections of ELLIPSE
    --source toefl     the final model's corrections of TOEFL 11. The corrector saw the
                       essays detokenized; the edits are made on that text and moved onto
                       the tokenized original, which stages 1-5 use (D50)

Writes data/study3/corrector/char_edits_<source>.parquet:
    text_id, k (index in the text's edit list), s, e, orig, repl, errant
char_edits_<source>_texts.parquet (text_id, corrected: every text of the
source, the check for stage 1) and char_edits_<source>.json (counts).

    uv run --project src/3-surprisal-decomposition/corrector \
        python src/3-surprisal-decomposition/corrector/c5_edits.py --source model
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, char_edits, detokenize, final_run, map_edits, splice  # noqa: E402

from util.paths import DATA_DIR  # noqa: E402


def texts(source: str) -> pd.DataFrame:
    """text_id, text, corrected."""
    if source == "gold":
        return pd.read_parquet(DATA / "pairs.parquet", columns=["text_id", "text", "corrected"])
    if source == "model":
        f = [DATA / "out" / f"{final_run(k)}__fold{k}.parquet" for k in range(5)]
        return pd.concat([pd.read_parquet(x, columns=["text_id", "text", "output"]) for x in f]
                         ).rename(columns={"output": "corrected"})
    if source in ("ellipse", "toefl"):
        return pd.read_parquet(DATA / "out" / f"{final_run('all')}__{source}.parquet",
                               columns=["text_id", "text", "output"]).rename(columns={"output": "corrected"})
    raise ValueError(source)


def keep_boundary_whitespace(text: str, cor: str) -> str:
    """The original's leading and trailing whitespace on the correction: generation drops
    a trailing newline, which is not a correction."""
    lead, trail = text[:len(text) - len(text.lstrip())], text[len(text.rstrip()):]
    return lead + cor.strip() + trail


def one(args):
    tid, text, cor, original = args
    eds = char_edits(text, cor)
    assert splice(text, eds) == cor, tid
    if original is not None:  # TOEFL 11: from the detokenized text back to the tokenized one
        text, eds = original, map_edits(original, detokenize(original)[1], eds)
    return [(tid, k, s, e, text[s:e], r, t) for k, (s, e, r, t) in enumerate(eds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["model", "gold", "ellipse", "toefl"], required=True)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    T = texts(args.source)
    assert T.text_id.is_unique
    if args.source != "gold":
        T["corrected"] = [keep_boundary_whitespace(t, c) for t, c in zip(T.text, T.corrected)]
    T["original"] = None
    if args.source == "toefl":
        tok = pd.read_parquet(DATA_DIR / "study3" / "appendix" / "toefl11" / "essays.parquet",
                              columns=["text_id", "text"]).set_index("text_id").text
        T["original"] = T.text_id.map(tok)
        assert all(detokenize(o)[0] == t for o, t in zip(T.original, T.text))
    with Pool(args.procs) as pool:
        rows = pool.map(one, zip(T.text_id, T.text, T.corrected, T.original), chunksize=8)
    if args.source == "toefl":  # the tokenized original with the edits, for stage 1's check
        T["text"] = T.original
        T["corrected"] = [splice(t, [(x[2], x[3], x[5]) for x in r]) for t, r in zip(T.text, rows)]
    E = pd.DataFrame([r for x in rows for r in x], columns=["text_id", "k", "s", "e", "orig", "repl", "errant"])
    E.to_parquet(DATA / f"char_edits_{args.source}.parquet", index=False)
    T[["text_id", "corrected"]].to_parquet(DATA / f"char_edits_{args.source}_texts.parquet", index=False)
    rep = {"source": args.source, "n_texts": len(T), "n_unchanged": int((T.text == T.corrected).sum()),
           "n_edits": len(E), "edits_per_text": len(E) / len(T),
           "whitespace_only": int((E.orig.str.strip().eq("") & E.repl.str.strip().eq("")).sum()),
           "with_newline": int((E.orig.str.contains("\n") | E.repl.str.contains("\n")).sum())}
    (DATA / f"char_edits_{args.source}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
