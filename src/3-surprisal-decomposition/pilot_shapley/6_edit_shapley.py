"""Pilot step 6 (GPU, post hoc) — Shapley values over the *edits* in the window.

Added after the step-4 test run showed that removal-based Shapley answers
"word vs no word", not "error vs correction" (see NOTES.md). Here the players
are the gold edits that touch the target's window, and the two states of each
player are *as written* and *corrected*:

    u(S) = s_orig - v(S),   v(S) = surprisal of the target when only the
                                   edits in S are applied

so u(empty) = 0, u(all) = s_orig - s_corr = Delta, and the Shapley values split
the gold counterfactual Delta exactly over the individual errors. With one
edit in the window the value is Delta itself; the method only adds information
when a window holds several edits.

Window: the text from >= 300 characters before the target (cut at a space)
through the end of the target's word is
re-tokenised with the edits in S applied (``pilot_corrected/common.apply_edits``,
same whitespace repairs), and the 8 subwords ending at the target's first
subword are scored (fp32, delete scheme's full window). Players: non-null
first-annotator edits that start before the target and end after the start of
the original window or of the fully corrected window, whichever is earlier.
Targets with more than 7 such edits are skipped (counted).

Output: data/study3/pilot_shapley/edit_shapley.parquet
    text_id, spacy_idx, n_edits, ok_none / ok_all (the no-edit window equals the
    stored window / the all-edit window equals the corrected essay's), edit_k (json edit indices), groups, kinds,
    dist (words back from the target), v (2^n list), psi (Shapley of u),
    loo_fix (s_orig - v({i})), loo_keep (v(all minus i) - v(all)), v_none, v_all

    CUDA_VISIBLE_DEVICES=0 python .../6_edit_shapley.py
"""

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, PC, WIN, WI_RAW  # noqa: E402
from engine import HF_ID, REV, load, shapley_matrix, subset_masks, surprisal_batch  # noqa: E402

spec = importlib.util.spec_from_file_location("pc_common", HERE.parent / "pilot_corrected" / "common.py")
pcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcc)

torch.backends.cuda.matmul.allow_tf32 = False
MAXP = 7
BACK = 300


def window_at(tok, text, char):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    for k, (s, e) in enumerate(enc.offset_mapping):
        piece = text[s:e]
        if s + len(piece) - len(piece.lstrip()) == char:
            if k < WIN - 1:
                return None, None
            return np.array(enc.input_ids[k - WIN + 1:k + 1]), enc.offset_mapping[k - WIN + 1][0]
    return None, None


def variant(tok, seg, seg_edits, tgt):
    """Window for the segment with ``seg_edits`` applied; tgt is the target's char in seg."""
    r = pcc.apply_edits(seg, [list(e) for e in seg_edits])
    c = int(r["cmap"][tgt])
    w, start = window_at(tok, r["corrected"], c)
    return w, start, r


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REV)
    s = pd.read_parquet(OUT / "sample.parquet")
    cand = s[s.set == "cand"]
    al = pd.read_parquet(PC / "aligned.parquet", columns=["text_id", "spacy_idx", "idx", "c_idx"])
    cand = cand.merge(al, on=["text_id", "spacy_idx"])
    ess = pd.read_parquet(PC / "essays.parquet", columns=["text_id", "text", "corrected"]).set_index("text_id")
    texts, corrected = ess.text, ess.corrected
    et = pd.read_parquet(OUT / "edit_types.parquet")
    et = et[et.kind != "flagged"]
    ebt = {k: g for k, g in et.groupby("text_id")}
    jed = {}
    for f in sorted((WI_RAW / "json").glob("*.json")):
        for line in open(f):
            d = json.loads(line)
            jed[d["id"]] = [x for c, x in d["edits"] if c == 0][0]

    jobs, skipped = [], {"too_many": 0, "no_window": 0, "no_edit": 0}
    for r in cand.itertuples():
        text = texts[r.text_id]
        a = text.rfind(" ", 0, max(0, r.idx - BACK)) + 1 if r.idx > BACK else 0
        m_ = re.search(r"\s", text[r.idx:])
        z = r.idx + (m_.start() if m_ else len(text) - r.idx)
        seg = text[a:z]  # through the end of the target's whitespace-delimited word
        tgt = r.idx - a
        g = ebt.get(r.text_id)
        g = g[(g.s >= a) & (g.e <= r.idx) & (g.s < r.idx)] if g is not None else g
        if g is None or not len(g):
            skipped["no_edit"] += 1
            continue
        all_e = [(e.s - a, e.e - a, e.repl) for e in g.itertuples()]
        w0, st0 = window_at(tok, seg, tgt)
        w1, st1, r1 = variant(tok, seg, all_e, tgt)
        if w0 is None or w1 is None:
            skipped["no_window"] += 1
            continue
        # start of the corrected window, mapped back to the original segment
        cm = r1["cmap"]
        back = np.flatnonzero((cm >= 0) & (cm <= st1))
        st1o = int(back.max()) if len(back) else 0
        lo = min(st0, st1o)
        keep = [i for i, (es, ee, _) in enumerate(all_e) if ee > lo or (es == ee and es >= lo)]
        if not keep:
            skipped["no_edit"] += 1
            continue
        if len(keep) > MAXP:
            skipped["too_many"] += 1
            continue
        sub = g.iloc[keep]
        pl = [all_e[i] for i in keep]
        wins = []
        for m in subset_masks(len(pl)):
            w, _, _ = variant(tok, seg, [p for p, on in zip(pl, m) if on], tgt)
            wins.append(w)
        if any(w is None for w in wins):
            skipped["no_window"] += 1
            continue
        # distance in words: count non-space words between edit start and target in the original
        dist = [len(seg[e[0]:tgt].split()) for e in pl]
        wc, _ = window_at(tok, corrected[r.text_id], int(r.c_idx))
        jobs.append({"text_id": r.text_id, "spacy_idx": r.spacy_idx, "n_edits": len(pl),
                     "ok_none": list(wins[0]) == list(r.window),
                     "ok_all": wc is not None and list(wins[-1]) == list(wc),
                     "edit_k": sub.k.tolist(), "groups": sub.group.tolist(), "kinds": sub.kind.tolist(),
                     "dist": dist, "wins": wins})
    print(f"built {len(jobs)} targets in {time.time() - t0:.0f} s; skipped {skipped}", flush=True)

    model = load(dtype=torch.float32)
    W = np.stack([w for j in jobs for w in j["wins"]])
    P = np.tile(np.arange(WIN), (len(W), 1))
    v = np.concatenate([surprisal_batch(model, W[i:i + 2048], P[i:i + 2048]) for i in range(0, len(W), 2048)])
    rows, a = [], 0
    for j in jobs:
        n = j["n_edits"]
        vv = v[a:a + 2 ** n]
        a += 2 ** n
        u = vv[0] - vv
        full = 2 ** n - 1
        rows.append({k: j[k] for k in ["text_id", "spacy_idx", "n_edits", "ok_none", "ok_all", "edit_k", "groups", "kinds", "dist"]}
                    | {"v": vv, "psi": shapley_matrix(n) @ u,
                       "loo_fix": np.array([vv[0] - vv[1 << i] for i in range(n)]),
                       "loo_keep": np.array([vv[full ^ (1 << i)] - vv[full] for i in range(n)]),
                       "v_none": vv[0], "v_all": vv[full]})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "edit_shapley.parquet", index=False)
    secs = time.time() - t0
    (OUT / "edit_shapley.time.json").write_text(json.dumps({"seconds": secs, "n": len(out),
                                                            "sequences": int(len(W)), "skipped": skipped}))
    print(f"done {len(out)} targets, {len(W)} windows, {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main()
