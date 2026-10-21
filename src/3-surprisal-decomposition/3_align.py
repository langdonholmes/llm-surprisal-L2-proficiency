"""Study 3, stage 3 (CPU) — align the writer's own tokens between the two versions.

An original token is *own* when none of its characters lies in an applied or
flagged edit span. It is *aligned* when it is own and the corrected essay has a
scored token that starts at its mapped offset with identical text. Aligned
tokens carry their surprisal in the corrected essay (s_corr). Every token also
carries its gold category (ERR_TOK / ERR_INS / POST / OK) and, if it lies in a
correction, the index k of the first such edit (``site_k``, -1 otherwise),
which gives its ERRANT type through edits.parquet.

Check: aligned tokens whose 8-subword window is identical in both versions
should have the same surprisal (``same_window``).

Writes data/study3/<corpus>/aligned.parquet (one row per non-space scored
original token) and results/study3/align_<corpus>.json.

    python src/3-surprisal-decomposition/3_align.py --corpus wi
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import RES, out  # noqa: E402
from lib.lm import HF_ID, REVISION  # noqa: E402

WIN = 8


def first_subword_windows(tok, text: str, chars) -> list:
    """The 8-id window ending at the first subword starting at each char (None if none)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start = {}
    for k, (s, e) in enumerate(enc.offset_mapping):
        piece = text[s:e]
        start.setdefault(s + len(piece) - len(piece.lstrip()), k)
    ids = enc.input_ids
    return [None if (k := start.get(int(c))) is None else tuple(ids[max(0, k - WIN + 1):k + 1]) for c in chars]


def align_essay(e, o, c, flagged, corr_edits, tok) -> pd.DataFrame:
    cmap = np.asarray(e.cmap)
    a = o.idx.to_numpy()
    b = a + o.text.str.len().to_numpy()
    own = np.array([(cmap[x:y] >= 0).all() for x, y in zip(a, b)])
    for s, t in flagged:
        own &= ~((a < max(t, s + 1)) & (b > s))
    mapped = np.where(own, cmap[np.minimum(a, len(cmap) - 1)], -1)
    cidx = dict(zip(c.idx, range(len(c))))
    j = np.array([cidx.get(int(m), -1) if m >= 0 else -1 for m in mapped])
    ok = j >= 0
    same_text = np.zeros(len(o), bool)
    same_text[ok] = c.text.to_numpy()[j[ok]] == o.text.to_numpy()[ok]
    aligned = own & ok & same_text
    o = o.assign(own=own, aligned=aligned)
    o["s_corr"] = np.where(aligned, c.surprisal_bits.to_numpy()[np.maximum(j, 0)], np.nan)
    site = np.full(len(o), -1)
    for k, s, t in corr_edits:  # json order; the first overlapping edit wins
        hit = (site < 0) & (a < t) & (b > s)
        site[hit] = k
    o["site_k"] = site
    wo = first_subword_windows(tok, e.text, a)
    wc = first_subword_windows(tok, e.corrected, np.where(aligned, mapped, -1))
    o["same_window"] = [bool(aligned[i] and wo[i] is not None and wo[i] == wc[i]) for i in range(len(o))]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["wi", "fce"], required=True)
    args = ap.parse_args()
    O = out(args.corpus)
    essays = pd.read_parquet(O / "essays.parquet")
    edits = pd.read_parquet(O / "edits.parquet")
    gold = pd.read_parquet(O / "gold.parquet")
    orig = pd.read_parquet(O / "tokens_orig.parquet")
    corr = pd.read_parquet(O / "tokens_corr.parquet")
    orig = orig.merge(gold, on=["text_id", "spacy_idx"], how="left", validate="one_to_one")
    assert orig.gold.notna().all()
    o_by = {t: d for t, d in orig[~orig.is_space].groupby("text_id")}
    c_by = {t: d for t, d in corr[~corr.is_space].groupby("text_id")}
    fl = {t: list(zip(d.s, d.e)) for t, d in edits[edits.kind == "flagged"].groupby("text_id")}
    sp = edits[(edits.kind != "flagged") & (edits.e > edits.s)].sort_values(["text_id", "s", "k"])
    ce = {t: list(zip(d.k, d.s, d.e)) for t, d in sp.groupby("text_id")}
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    frames = [align_essay(e, o_by[e.text_id], c_by[e.text_id], fl.get(e.text_id, []),
                          ce.get(e.text_id, []), tok)
              for e in essays.itertuples() if e.text_id in o_by]
    A = pd.concat(frames, ignore_index=True).drop(columns="is_space")
    A = A.merge(essays[["text_id", "unit", "group", "y"]], on="text_id")
    A.to_parquet(O / "aligned.parquet", index=False)

    sw = A[A.same_window]
    d = (sw.surprisal_bits - sw.s_corr).abs()
    rep = {"n_tokens": len(A), "own": float(A.own.mean()), "aligned": float(A.aligned.mean()),
           "aligned_of_own": float(A.aligned[A.own].mean()),
           "corr_tokens_matched": float(A.aligned.sum() / (~corr.is_space).sum()),
           "same_window_of_aligned": float(A.same_window[A.aligned].mean()),
           "same_window_abs_delta_lt_0.05": float((d < 0.05).mean()),
           "same_window_max_abs_delta": float(d.max())}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"align_{args.corpus}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
