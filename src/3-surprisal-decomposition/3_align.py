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
from lib.corpora import ALL_IDS, RESULTS, data_dir  # noqa: E402
from lib.lm import HF_ID, REVISION  # noqa: E402

WIN = 8


def first_subword_windows(tok, text: str, chars) -> list:
    """The 8-id window ending at the first subword starting at each char (None if none)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    # character where each subword's text starts (leading space skipped) -> the first such subword
    subword_at = {}
    for k, (s, e) in enumerate(enc.offset_mapping):
        piece = text[s:e]
        subword_at.setdefault(s + len(piece) - len(piece.lstrip()), k)
    ids = enc.input_ids
    return [None if (k := subword_at.get(int(c))) is None else tuple(ids[max(0, k - WIN + 1):k + 1]) for c in chars]


def align_essay(essay, orig, corr, flagged, corr_edits, tok) -> pd.DataFrame:
    """One essay's original tokens with own, aligned, s_corr, site_k and same_window."""
    cmap = np.asarray(essay.cmap)
    starts = orig.idx.to_numpy()
    ends = starts + orig.text.str.len().to_numpy()
    # own: every character survives correction, and the token does not touch a flagged span
    own = np.array([(cmap[s:e] >= 0).all() for s, e in zip(starts, ends)])
    for s, e in flagged:
        own &= ~((starts < max(e, s + 1)) & (ends > s))
    # aligned: the corrected essay has a token with the same text at the mapped position
    mapped = np.where(own, cmap[np.minimum(starts, len(cmap) - 1)], -1)
    corr_row_at = dict(zip(corr.idx, range(len(corr))))
    corr_row = np.array([corr_row_at.get(int(m), -1) if m >= 0 else -1 for m in mapped])
    found = corr_row >= 0
    same_text = np.zeros(len(orig), bool)
    same_text[found] = corr.text.to_numpy()[corr_row[found]] == orig.text.to_numpy()[found]
    aligned = own & found & same_text
    orig = orig.assign(own=own, aligned=aligned)
    orig["s_corr"] = np.where(aligned, corr.surprisal_bits.to_numpy()[np.maximum(corr_row, 0)], np.nan)
    # site_k: the correction a non-own token lies in
    site = np.full(len(orig), -1)
    for k, s, e in corr_edits:  # json order; the first overlapping edit wins
        hit = (site < 0) & (starts < e) & (ends > s)
        site[hit] = k
    orig["site_k"] = site
    # same_window: the 8-subword window is identical in both versions (a check: same surprisal expected)
    win_orig = first_subword_windows(tok, essay.text, starts)
    win_corr = first_subword_windows(tok, essay.corrected, np.where(aligned, mapped, -1))
    orig["same_window"] = [bool(aligned[i] and win_orig[i] is not None and win_orig[i] == win_corr[i])
                           for i in range(len(orig))]
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=ALL_IDS, required=True)
    args = ap.parse_args()
    out_dir = data_dir(args.corpus)
    essays = pd.read_parquet(out_dir / "essays.parquet")
    edits = pd.read_parquet(out_dir / "edits.parquet")
    gold = pd.read_parquet(out_dir / "gold.parquet")
    orig = pd.read_parquet(out_dir / "tokens_orig.parquet")
    corr = pd.read_parquet(out_dir / "tokens_corr.parquet")
    orig = orig.merge(gold, on=["text_id", "spacy_idx"], how="left", validate="one_to_one")
    assert orig.gold.notna().all()
    orig_by_text = {t: d for t, d in orig[~orig.is_space].groupby("text_id")}
    corr_by_text = {t: d for t, d in corr[~corr.is_space].groupby("text_id")}
    flagged = {t: list(zip(d.s, d.e)) for t, d in edits[edits.kind == "flagged"].groupby("text_id")}
    spans = edits[(edits.kind != "flagged") & (edits.e > edits.s)].sort_values(["text_id", "s", "k"])
    corr_edits = {t: list(zip(d.k, d.s, d.e)) for t, d in spans.groupby("text_id")}
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
    frames = [align_essay(e, orig_by_text[e.text_id], corr_by_text[e.text_id], flagged.get(e.text_id, []),
                          corr_edits.get(e.text_id, []), tok)
              for e in essays.itertuples() if e.text_id in orig_by_text]
    aligned = pd.concat(frames, ignore_index=True).drop(columns="is_space")
    aligned = aligned.merge(essays[["text_id", "unit", "group", "y"]], on="text_id")
    aligned.to_parquet(out_dir / "aligned.parquet", index=False)

    same = aligned[aligned.same_window]
    delta = (same.surprisal_bits - same.s_corr).abs()
    rep = {"n_tokens": len(aligned), "own": float(aligned.own.mean()), "aligned": float(aligned.aligned.mean()),
           "aligned_of_own": float(aligned.aligned[aligned.own].mean()),
           "corr_tokens_matched": float(aligned.aligned.sum() / (~corr.is_space).sum()),
           "same_window_of_aligned": float(aligned.same_window[aligned.aligned].mean()),
           "same_window_abs_delta_lt_0.05": float((delta < 0.05).mean()),
           "same_window_max_abs_delta": float(delta.max())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"align_{args.corpus}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
