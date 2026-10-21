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
and results/study3/spill_<corpus>.json (status counts, timing).

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/4_spill.py --corpus wi
"""

import argparse
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import RES, out  # noqa: E402
from lib.corrections import apply_edits  # noqa: E402
from lib.shapley import BACK, load_model, shapley_matrix, subset_masks, surprisal, variants  # noqa: E402

MAXP = 7


def build_essay(job):
    """All targets of one essay, with the 2^p windows of each."""
    text, edits = job["text"], job["edits"]
    rows = []
    for tid, sidx, idx, prev_end in job["targets"]:
        a = text.rfind(" ", 0, max(0, idx - BACK)) + 1 if idx > BACK else 0
        m_ = re.search(r"\S*\s+\S*", text[idx:])
        z = idx + (m_.end() if m_ else len(text) - idx)
        seg, tgt = text[a:z], idx - a
        cand = [(k, s, e, r) for k, s, e, r in edits if s >= a and e <= idx and (s < idx or s == e)]
        base = {"text_id": tid, "spacy_idx": sidx, "edit_k": [], "dist": [], "wins": None,
                "gap_k": [k for k, s, e, r in cand if s == e and prev_end <= s <= idx]}
        if not cand:
            rows.append(base | {"status": "no_player"})
            continue
        all_e = [(s - a, e - a, r) for k, s, e, r in cand]
        (w0, st0), (w1, st1) = variants(seg, tgt, [[], all_e], a == 0)
        if w0 is None or w1 is None:
            rows.append(base | {"status": "no_window", "edit_k": [k for k, *_ in cand]})
            continue
        # start of the corrected window, mapped back to the original segment
        cm = apply_edits(seg, [list(e) for e in all_e])["cmap"]
        back = np.flatnonzero((cm >= 0) & (cm <= st1))
        lo = min(st0, int(back.max()) if len(back) else 0)
        keep = [i for i, (es, ee, _) in enumerate(all_e) if ee > lo or (es == ee and es >= lo)]
        if not keep:
            rows.append(base | {"status": "no_player"})
            continue
        ks = [cand[i][0] for i in keep]
        pl = [all_e[i] for i in keep]
        dist = [len(seg[s:tgt].split()) for s, _, _ in pl]
        if len(keep) > MAXP:
            rows.append(base | {"status": "too_many", "edit_k": ks, "dist": dist})
            continue
        sets = [[p for p, on in zip(pl, m) if on] for m in subset_masks(len(pl))]
        wins = [w for w, _ in variants(seg, tgt, sets, a == 0)]
        if any(w is None for w in wins):
            rows.append(base | {"status": "no_window", "edit_k": ks, "dist": dist})
            continue
        rows.append(base | {"status": "ok", "edit_k": ks, "dist": dist, "wins": wins})
    return rows


def jobs_for(O):
    al = pd.read_parquet(O / "aligned.parquet", columns=["text_id", "spacy_idx", "idx", "text", "aligned"])
    al = al.sort_values(["text_id", "spacy_idx"])
    al["prev_end"] = (al.idx + al.text.str.len()).groupby(al.text_id).shift(1).fillna(0).astype(int)
    ed = pd.read_parquet(O / "edits.parquet")
    ed = ed[ed.kind != "flagged"].sort_values(["text_id", "s", "e"])
    ebt = {t: list(zip(g.k, g.s, g.e, g.repl)) for t, g in ed.groupby("text_id")}
    text = pd.read_parquet(O / "essays.parquet", columns=["text_id", "text"]).set_index("text_id").text
    al = al[al.aligned]
    return [{"text": text[t], "edits": ebt.get(t, []),
             "targets": list(zip(g.text_id, g.spacy_idx, g.idx, g.prev_end))} for t, g in al.groupby("text_id")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["wi", "fce"], required=True)
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()
    O = out(args.corpus)
    t0 = time.time()
    with Pool(args.procs) as pool:
        rows = [r for rr in pool.imap(build_essay, jobs_for(O), chunksize=4) for r in rr]
    t_build = time.time() - t0
    st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    print(f"built {len(rows)} targets in {t_build:.0f} s: {st}", flush=True)

    model = load_model()
    W = [w for r in rows if r["status"] == "ok" for w in r["wins"]]
    v = surprisal(model, W)
    a = 0
    for r in rows:
        wins = r.pop("wins")
        r.update(v_none=np.nan, v_all=np.nan, psi=None)
        if wins is not None:
            vv = v[a:a + len(wins)]
            a += len(wins)
            r.update(v_none=vv[0], v_all=vv[-1], psi=shapley_matrix(len(r["edit_k"])) @ (vv[0] - vv))
    pd.DataFrame(rows).to_parquet(O / "spill.parquet", index=False)
    secs = time.time() - t0
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"spill_{args.corpus}.json").write_text(json.dumps(
        {"seconds": secs, "build_seconds": t_build, "gpu_seconds": secs - t_build, "n_targets": len(rows),
         "windows": len(W), "status": st, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    print(f"done {len(rows)} targets, {len(W)} windows, {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main()
