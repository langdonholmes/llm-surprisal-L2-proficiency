"""Repair lab — build and score the candidate edits for every bench instance.

One row per (instance, candidate): op class, the candidate word, and the
log2-probability of the edited local continuation. Cached at
data/study3/lab/repair/cands{SUFFIX}.parquet; featurizers read that table.

    CUDA_VISIBLE_DEVICES=0 python src/3-vector-transformations/lab/repair_build.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import LAB, SUFFIX  # noqa: E402
from lab.repair_core import (N_INS, N_PRED, RIGHT, Vocab, match_case,  # noqa: E402
                             score_continuations)

REPAIR = LAB / "repair"
WIN = 7  # left context: Study 1's 8-token window minus the target itself


def cands_path():
    return REPAIR / f"cands{SUFFIX}.parquet"


def build(ctx, lm=None):
    """Return the candidate table, computing it (GPU) if it is not cached."""
    p = cands_path()
    if p.exists():
        return pd.read_parquet(p)
    if lm is None:
        from lab.lm import LM
        lm = LM(layers=())
    REPAIR.mkdir(parents=True, exist_ok=True)
    tok = lm.tok
    vocab = Vocab(tok)
    b = ctx.bench
    top = ctx.get("t_topk_ids")
    rows, seqs = [], []
    t0 = time.time()
    enc_cache = {}

    def enc(s):
        if s not in enc_cache:
            enc_cache[s] = tok(s, add_special_tokens=False).input_ids
        return enc_cache[s]

    for r in b.itertuples():
        if not ctx.ok[r.inst]:
            continue
        e = tok(r.text, add_special_tokens=False, return_offsets_mapping=True)
        ids = e.input_ids
        starts = [s + len(r.text[s:t]) - len(r.text[s:t].lstrip()) for s, t in e.offset_mapping]
        k = starts.index(r.target_char)
        end_char = r.target_char + len(r.observed)
        e_idx = next((j for j in range(k + 1, len(ids)) if starts[j] >= end_char), len(ids))
        L = ids[max(0, k - WIN):k]
        X = ids[k:e_idx]
        R = ids[e_idx:e_idx + RIGHT]
        first = tok.convert_ids_to_tokens(ids[k])
        sp = " " if first.startswith("Ġ") else ""
        obs = r.observed
        low = obs.lower()

        def add(cls, word, x):
            rows.append((r.inst, cls, word, len(x)))
            seqs.append((L, list(x), R))

        add("ORIG", obs, X)
        seen = {low}
        morph = sorted(vocab.inflections(low), key=lambda w: vocab.rank[w])[:8]
        for w in morph:
            seen.add(w)
            add("MORPH", w, enc(sp + match_case(w, obs)))
        for w, _d in vocab.similar(low):
            if w in seen:
                continue
            seen.add(w)
            add("FORM", w, enc(sp + match_case(w, obs)))
        for pid in top[r.inst, :N_PRED]:
            w = tok.decode([int(pid)])
            if w.strip().lower() in seen or (len(X) == 1 and int(pid) == X[0]):
                continue
            seen.add(w.strip().lower())
            add("PRED", w, [int(pid)])
        add("DEL", "", [])
        for pid in top[r.inst, :N_INS]:
            add("INS", tok.decode([int(pid)]), [int(pid)] + list(X))
    t1 = time.time()
    print(f"{len(seqs)} candidate sequences for {int(ctx.ok.sum())} instances "
          f"({len(seqs) / ctx.ok.sum():.1f}/inst), built in {t1 - t0:.0f}s", flush=True)
    lx, lr = score_continuations(lm.model, seqs)
    t2 = time.time()
    print(f"scored in {t2 - t1:.0f}s = {1000 * (t2 - t1) / ctx.ok.sum():.2f} ms/instance GPU", flush=True)
    df = pd.DataFrame(rows, columns=["inst", "cls", "word", "x_len"])
    df["logp_x"] = lx
    df["logp_r"] = lr
    df["logp"] = lx + lr
    df["len_R"] = [len(s[2]) for s in seqs]
    df.to_parquet(p, index=False)
    (REPAIR / f"timing{SUFFIX}.txt").write_text(
        f"build_s={t1 - t0:.1f} score_s={t2 - t1:.1f} n_inst={int(ctx.ok.sum())} n_seq={len(seqs)}\n")
    return df


if __name__ == "__main__":
    from lab.harness import Ctx
    build(Ctx())
