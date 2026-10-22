"""Study 3, stage 4 (GPU) — split each own word's Δ among the errors in its window.

For every aligned own token, Delta = s_orig - s_corr is split among the gold
edits in its window by edit-level Shapley with the correction as baseline
(``lib.shapley``). Players are the non-null edits that lie before the target
and end after the start of the original or the fully corrected window (an
insertion counts when its point is at or after that start). Targets with more
than 7 players are skipped (``too_many``) and later split Δ equally. ``gap_k``
names insertions right before the target (a word is missing there), and
``dist`` is the distance of each player from the target in words.

Writes data/study3/<corpus>/spill.parquet
    text_id, spacy_idx, status (ok / no_player / too_many / no_window), edit_k,
    gap_k, dist, v_none, v_all, psi
and results/study3/<corpus>/spill.json (status counts, timing).

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/4_spill.py --corpus wi
"""

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# batches of 4,096 windows fragment the reserved memory (wi_model ran out with 13 GiB of it free)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import ALL_IDS, data_dir, results_dir  # noqa: E402
from lib.corrections import apply_edits  # noqa: E402
from lib.shapley import BACK, load_model, shapley_matrix, subset_masks, surprisal, variants  # noqa: E402

MAXP = 7


def build_essay(job):
    """All targets of one essay, with the 2^p windows of each."""
    text, edits = job["text"], job["edits"]
    rows = []
    for text_id, spacy_idx, idx, prev_end in job["targets"]:
        # the segment: from a space at least BACK chars before the target to the end of the next word
        seg_start = text.rfind(" ", 0, max(0, idx - BACK)) + 1 if idx > BACK else 0
        next_word = re.search(r"\S*\s+\S*", text[idx:])
        seg_end = idx + (next_word.end() if next_word else len(text) - idx)
        segment, target_char = text[seg_start:seg_end], idx - seg_start
        # candidate players: edits in the segment before the target (insertions up to it)
        candidates = [(k, s, e, r) for k, s, e, r in edits if s >= seg_start and e <= idx and (s < idx or s == e)]
        row = {"text_id": text_id, "spacy_idx": spacy_idx, "edit_k": [], "dist": [], "wins": None,
               "gap_k": [k for k, s, e, r in candidates if s == e and prev_end <= s <= idx]}
        if not candidates:
            rows.append(row | {"status": "no_player"})
            continue
        seg_edits = [(s - seg_start, e - seg_start, r) for k, s, e, r in candidates]
        (orig_win, orig_start), (corr_win, corr_start) = variants(segment, target_char, [[], seg_edits],
                                                                  seg_start == 0)
        if orig_win is None or corr_win is None:
            rows.append(row | {"status": "no_window", "edit_k": [k for k, *_ in candidates]})
            continue
        # start of the corrected window, mapped back to the original segment
        cmap = apply_edits(segment, [list(e) for e in seg_edits])["cmap"]
        before = np.flatnonzero((cmap >= 0) & (cmap <= corr_start))
        window_start = min(orig_start, int(before.max()) if len(before) else 0)
        # players: the candidates that reach into the original or the corrected window
        keep = [i for i, (s, e, _) in enumerate(seg_edits) if e > window_start or (s == e and s >= window_start)]
        if not keep:
            rows.append(row | {"status": "no_player"})
            continue
        edit_ks = [candidates[i][0] for i in keep]
        players = [seg_edits[i] for i in keep]
        dist = [len(segment[s:target_char].split()) for s, _, _ in players]
        if len(keep) > MAXP:
            rows.append(row | {"status": "too_many", "edit_k": edit_ks, "dist": dist})
            continue
        # one window per subset of players, in subset_masks order
        subsets = [[p for p, on in zip(players, mask) if on] for mask in subset_masks(len(players))]
        wins = [w for w, _ in variants(segment, target_char, subsets, seg_start == 0)]
        if any(w is None for w in wins):
            rows.append(row | {"status": "no_window", "edit_k": edit_ks, "dist": dist})
            continue
        rows.append(row | {"status": "ok", "edit_k": edit_ks, "dist": dist, "wins": wins})
    return rows


def jobs_for(out_dir):
    """One job per essay: its text, its corrections and its aligned own tokens (the targets)."""
    tokens = pd.read_parquet(out_dir / "aligned.parquet", columns=["text_id", "spacy_idx", "idx", "text", "aligned"])
    tokens = tokens.sort_values(["text_id", "spacy_idx"])
    tokens["prev_end"] = (tokens.idx + tokens.text.str.len()).groupby(tokens.text_id).shift(1).fillna(0).astype(int)
    edits = pd.read_parquet(out_dir / "edits.parquet")
    edits = edits[edits.kind != "flagged"].sort_values(["text_id", "s", "e"])
    edits_by_text = {t: list(zip(g.k, g.s, g.e, g.repl)) for t, g in edits.groupby("text_id")}
    text = pd.read_parquet(out_dir / "essays.parquet", columns=["text_id", "text"]).set_index("text_id").text
    tokens = tokens[tokens.aligned]
    return [{"text": text[t], "edits": edits_by_text.get(t, []),
             "targets": list(zip(g.text_id, g.spacy_idx, g.idx, g.prev_end))} for t, g in tokens.groupby("text_id")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=ALL_IDS, required=True)
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()
    out_dir = data_dir(args.corpus)
    t0 = time.time()
    with Pool(args.procs) as pool:
        rows = [r for essay_rows in pool.imap(build_essay, jobs_for(out_dir), chunksize=4) for r in essay_rows]
    t_build = time.time() - t0
    status = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    print(f"built {len(rows)} targets in {t_build:.0f} s: {status}", flush=True)

    # score every window at once, then hand each target its slice
    model = load_model()
    windows = [w for r in rows if r["status"] == "ok" for w in r["wins"]]
    bits = surprisal(model, windows)
    pos = 0
    for r in rows:
        wins = r.pop("wins")
        r.update(v_none=np.nan, v_all=np.nan, psi=None)
        if wins is not None:
            v = bits[pos:pos + len(wins)]
            pos += len(wins)
            # Shapley values of u(S) = v(none) - v(S)
            r.update(v_none=v[0], v_all=v[-1], psi=shapley_matrix(len(r["edit_k"])) @ (v[0] - v))
    pd.DataFrame(rows).to_parquet(out_dir / "spill.parquet", index=False)
    secs = time.time() - t0
    (results_dir(args.corpus) / "spill.json").write_text(json.dumps(
        {"seconds": secs, "build_seconds": t_build, "gpu_seconds": secs - t_build, "n_targets": len(rows),
         "windows": len(windows), "status": status, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    print(f"done {len(rows)} targets, {len(windows)} windows, {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main()
