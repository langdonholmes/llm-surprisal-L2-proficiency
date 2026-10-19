"""Step 2 (GPU) — edit-level Shapley with the correction as baseline, every token.

The D27 method (``pilot_shapley/6_edit_shapley.py``) run on every scored token
of every essay instead of a sample. Two kinds of target:

  own   the writer's own aligned tokens (``aligned``). Players are the gold edits
        in the token's window; u(S) = s_orig - v(S), with v(S) the surprisal of
        the token when only the edits in S are applied, so the Shapley values
        split Delta = s_orig - s_corr exactly among the edits. Includes ERR_INS
        tokens (a word is missing right before them); the gap insertion is then
        one of the players, and ``gap_k`` names it.
  err   the corrected tokens themselves (not own, inside a non-null edit). The
        players are the *earlier* edits in the window, never the token's own
        edit. v(none) is the token as written, v(all) the token with every
        earlier error in its window corrected. Used only to check how much of
        an error token's surprisal is an earlier edit's spill.

Same windowing as D27: the text from >= 300 characters before the target (cut
at a space) to the end of the word after the target's word (D27 stopped at the
end of the target's word, which re-tokenises a subword such as ".\\n"), re-tokenised with the edits in S
applied (``pilot_corrected/common.apply_edits``), 8 subwords ending at the
target's first subword, fp32 with TF32 off. Two extensions, needed to cover
every token: a target in the first 8 subwords of the essay gets the shorter
window the instrument uses there (``Predictor._process_causal`` phase 1), and a
spaCy token that starts inside a subword is scored at that subword. Players
are the non-null first-annotator edits that lie before the target and end
after the start of the original or the fully corrected window (an insertion
counts when its point is at or after that start). Targets with more than 7
players are skipped and counted.

Output: data/study3/pilot_types/shapley_<corpus>.parquet
    text_id, spacy_idx, role, status (ok / no_player / too_many / no_window),
    edit_k (json edit indices of the players), gap_k (gap insertions, own
    ERR_INS targets), v_none, v_all, psi (Shapley of u), n_edits
and shapley_<corpus>.time.json.

    CUDA_VISIBLE_DEVICES=0 python .../2_shapley.py --corpus wi
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from types_common import CORPORA, OUT, S3_SRC, SH, WIN, edit_table, load  # noqa: E402

pcc = load("pc_common", S3_SRC / "pilot_corrected" / "common.py")
engine = load("sh_engine", SH / "engine.py")

MAXP = 7
BACK = 300
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(engine.HF_ID, revision=engine.REV)
    return _TOK


def window_of(enc_ids, offsets, text, char, from_start):
    """8-id window ending at the subword where ``char`` starts; (ids, start char)."""
    k = None
    for j, (s, e) in enumerate(offsets):
        piece = text[s:e]
        if s + len(piece) - len(piece.lstrip()) == char:
            k = j
            break
    if k is None:  # spaCy token starts inside a subword
        for j, (s, e) in enumerate(offsets):
            if s <= char < e:
                k = j
                break
    if k is None or k == 0 or (k < WIN - 1 and not from_start):
        return None, None
    a = max(0, k - WIN + 1)
    return tuple(enc_ids[a:k + 1]), offsets[a][0]


def variants(seg, tgt, edit_sets, from_start):
    """Windows for ``seg`` with each list of edits applied."""
    texts, chars = [], []
    for es in edit_sets:
        r = pcc.apply_edits(seg, [list(e) for e in es])
        texts.append(r["corrected"])
        chars.append(int(r["cmap"][tgt]))
    enc = tokenizer()(texts, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for t, c, ids, off in zip(texts, chars, enc.input_ids, enc.offset_mapping):
        out.append(window_of(ids, off, t, c, from_start) if c >= 0 else (None, None))
    return out, texts


def build_essay(job):
    """All targets of one essay -> list of dicts with windows."""
    text, edits, targets = job["text"], job["edits"], job["targets"]
    rows = []
    for tid, sidx, idx, role, prev_end in targets:
        a = text.rfind(" ", 0, max(0, idx - BACK)) + 1 if idx > BACK else 0
        # through the end of the *next* word, so that a subword that runs into the
        # following whitespace (".\n") is tokenised as in the essay
        m_ = re.search(r"\S*\s+\S*", text[idx:])
        z = idx + (m_.end() if m_ else len(text) - idx)
        seg, tgt = text[a:z], idx - a
        cand = [(k, s, e, r) for k, s, e, r in edits
                if s >= a and e <= idx and (s < idx or s == e)]
        gap = [k for k, s, e, r in cand if s == e and prev_end <= s <= idx] if role == "own" else []
        base = {"text_id": tid, "spacy_idx": sidx, "role": role, "gap_k": gap}
        if not cand:
            rows.append(base | {"status": "no_player", "edit_k": [], "wins": None})
            continue
        all_e = [(s - a, e - a, r) for k, s, e, r in cand]
        (w0, st0), (w1, st1) = variants(seg, tgt, [[], all_e], a == 0)[0]
        if w0 is None or w1 is None:
            rows.append(base | {"status": "no_window", "edit_k": [k for k, *_ in cand], "wins": None})
            continue
        r1 = pcc.apply_edits(seg, [list(e) for e in all_e])
        cm = r1["cmap"]
        back = np.flatnonzero((cm >= 0) & (cm <= st1))
        lo = min(st0, int(back.max()) if len(back) else 0)
        keep = [i for i, (es, ee, _) in enumerate(all_e) if ee > lo or (es == ee and es >= lo)]
        if not keep:
            rows.append(base | {"status": "no_player", "edit_k": [], "wins": None})
            continue
        ks = [cand[i][0] for i in keep]
        if len(keep) > MAXP:
            rows.append(base | {"status": "too_many", "edit_k": ks, "wins": None})
            continue
        pl = [all_e[i] for i in keep]
        sets = [[p for p, on in zip(pl, m) if on] for m in engine.subset_masks(len(pl))]
        wins = [w for w, _ in variants(seg, tgt, sets, a == 0)[0]]
        if any(w is None for w in wins):
            rows.append(base | {"status": "no_window", "edit_k": ks, "wins": None})
            continue
        rows.append(base | {"status": "ok", "edit_k": ks, "wins": wins})
    return rows


def jobs_for(corpus):
    d = CORPORA[corpus]["data"]
    al = pd.read_parquet(d / "aligned.parquet",
                         columns=["text_id", "spacy_idx", "idx", "text", "own", "aligned", "is_space"])
    al = al[~al.is_space].sort_values(["text_id", "spacy_idx"])
    al["end"] = al.idx + al.text.str.len()
    al["prev_end"] = al.groupby("text_id").end.shift(1).fillna(0).astype(int)
    et = edit_table(corpus)
    # err targets: not own and inside a non-null edit (flagged-only tokens excluded)
    ebt = {t: list(zip(g.k, g.s, g.e, g.repl)) for t, g in et.sort_values(["text_id", "s", "e"]).groupby("text_id")}
    ess = pd.read_parquet(d / "essays.parquet", columns=["text_id", "text"]).set_index("text_id").text
    jobs = []
    for tid, g in al.groupby("text_id"):
        eds = ebt.get(tid, [])
        spans = [(s, e) for k, s, e, r in eds if e > s]
        tg = []
        for r in g.itertuples():
            if r.aligned:
                tg.append((tid, r.spacy_idx, r.idx, "own", r.prev_end))
            elif not r.own and any(s < r.end and e > r.idx for s, e in spans):
                tg.append((tid, r.spacy_idx, r.idx, "err", r.prev_end))
        jobs.append({"text": ess[tid], "edits": eds, "targets": tg})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), required=True)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="first N essays only (testing)")
    args = ap.parse_args()
    t0 = time.time()
    jobs = jobs_for(args.corpus)
    if args.limit:
        jobs = jobs[:args.limit]
    with Pool(args.procs) as pool:
        rows = [r for rr in pool.imap(build_essay, jobs, chunksize=4) for r in rr]
    t_build = time.time() - t0
    st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    print(f"built {len(rows)} targets in {t_build:.0f} s: {st}", flush=True)

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    model = engine.load(dtype=torch.float32)
    W = [w for r in rows if r["status"] == "ok" for w in r["wins"]]
    lens = np.array([len(w) for w in W])
    v = np.empty(len(W))
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        ids = np.array([W[i] for i in idx])
        pos = np.tile(np.arange(L), (len(idx), 1))
        for a in range(0, len(idx), 4096):
            v[idx[a:a + 4096]] = engine.surprisal_batch(model, ids[a:a + 4096], pos[a:a + 4096])
    a = 0
    out = []
    for r in rows:
        o = {k: r[k] for k in ["text_id", "spacy_idx", "role", "status", "edit_k", "gap_k"]}
        o["n_edits"] = len(r["edit_k"])
        if r["status"] == "ok":
            n = len(r["edit_k"])
            vv = v[a:a + 2 ** n]
            a += 2 ** n
            o.update(v_none=vv[0], v_all=vv[-1], psi=engine.shapley_matrix(n) @ (vv[0] - vv), v=vv)
        else:
            o.update(v_none=np.nan, v_all=np.nan, psi=None, v=None)
        out.append(o)
    df = pd.DataFrame(out)
    df.to_parquet(OUT / f"shapley_{args.corpus}.parquet", index=False)
    secs = time.time() - t0
    (OUT / f"shapley_{args.corpus}.time.json").write_text(json.dumps(
        {"seconds": secs, "build_seconds": t_build, "gpu_seconds": secs - t_build, "n_targets": len(df),
         "windows": int(len(W)), "status": st, "gpu": torch.cuda.get_device_name(0)}))
    print(f"done {len(df)} targets, {len(W)} windows, {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main()
