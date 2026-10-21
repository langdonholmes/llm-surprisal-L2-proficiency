"""Corrector, stage c4 (CPU) — ERRANT agreement with the first annotator.

For every text in a c3 output file, ERRANT (Bryant et al. 2017) extracts the
edits from the original to the model's correction (hypothesis) and from the
original to the first annotator's correction (reference), with the same ERRANT
version and spaCy model for both. Paragraphs are aligned on newlines; when the
model or the annotator changed the number of paragraphs, the whole text is
aligned as one paragraph on both sides (counted as n_para_mismatch).

Agreement is span-based correction, as in ERRANT's default scorer: an edit
matches when its original span and its correction string are the same. Noop
edits are ignored. Reported per corrector: TP, FP, FN, precision, recall, F0.5
(pooled over texts, with a bootstrap CI over texts), edits per 100 original
tokens for hypothesis and reference, and per-type counts.

Writes results/study3/corrector/score_<name>.json and
data/study3/corrector/edits/<name>.parquet (one row per edit: text_id, side
(hyp/ref), para, o_start, o_end, o_str, c_str, type).

    uv run --project src/3-surprisal-decomposition/corrector \
        python src/3-surprisal-decomposition/corrector/c4_score.py data/study3/corrector/out/<file>.parquet
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES  # noqa: E402

SEED = 42
N_BOOT = 1000
BETA = 0.5
_ann = None


def annotator():
    global _ann
    if _ann is None:
        import errant
        _ann = errant.load("en")
    return _ann


def split(orig: str, hyp: str, ref: str):
    """Paragraph pairs per side; whole texts when any side changed the paragraph count."""
    orig, hyp, ref = (s.rstrip("\n") for s in (orig, hyp, ref))  # trailing newlines carry no edits
    po, ph, pr = orig.split("\n"), hyp.split("\n"), ref.split("\n")
    if len(po) == len(ph) == len(pr):
        return {"hyp": list(zip(po, ph)), "ref": list(zip(po, pr))}, False
    flat = [s.replace("\n", " ") for s in (orig, hyp, ref)]
    return {"hyp": [(flat[0], flat[1])], "ref": [(flat[0], flat[2])]}, True


def text_edits(args):
    text_id, orig, hyp, ref = args
    ann = annotator()
    rows, n_tok = [], 0
    sides, mm = split(orig, hyp, ref)
    for side, pars in sides.items():
        for k, (o, c) in enumerate(pars):
            if not o.strip() and not c.strip():
                continue
            po, pc = ann.parse(o, tokenise=True), ann.parse(c, tokenise=True)
            if side == "ref":
                n_tok += len(po)
            for e in ann.annotate(po, pc):
                if e.type == "noop":
                    continue
                rows.append({"text_id": text_id, "side": side, "para": -1 if mm else k, "o_start": e.o_start,
                             "o_end": e.o_end, "o_str": e.o_str, "c_str": e.c_str, "type": e.type})
    return rows, n_tok, mm


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = (1 + BETA**2) * p * r / (BETA**2 * p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_file")
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    f = Path(args.out_file)
    name = f.stem
    out = pd.read_parquet(f)
    gold = pd.read_parquet(DATA / "pairs.parquet", columns=["text_id", "corrected"])
    df = out.merge(gold, on="text_id", how="left")
    if df.corrected.isna().any():
        raise SystemExit("texts without a gold correction (ELLIPSE?) cannot be scored")

    jobs = list(zip(df.text_id, df.text, df.output, df.corrected))
    with Pool(args.procs) as pool:
        res = pool.map(text_edits, jobs, chunksize=8)
    edits = pd.DataFrame([r for rows, _, _ in res for r in rows])
    n_tok = pd.Series([n for _, n, _ in res], index=df.text_id)
    n_mm = sum(m for _, _, m in res)

    key = ["text_id", "para", "o_start", "o_end", "c_str"]
    h = edits[edits.side == "hyp"][key + ["type"]]
    r = edits[edits.side == "ref"][key + ["type"]]
    m = h.merge(r[key], on=key, how="outer", indicator=True)
    per = pd.DataFrame({"tp": m[m._merge == "both"].groupby("text_id").size(),
                        "fp": m[m._merge == "left_only"].groupby("text_id").size(),
                        "fn": m[m._merge == "right_only"].groupby("text_id").size()}).reindex(df.text_id).fillna(0)
    tp, fp, fn = per.sum()
    p, rec, f05 = prf(tp, fp, fn)
    rng = np.random.default_rng(SEED)
    A = per.to_numpy()
    boots = [prf(*A[rng.integers(0, len(A), len(A))].sum(0))[2] for _ in range(N_BOOT)]

    def rate(side):
        return 100 * (edits.side == side).sum() / n_tok.sum()

    hm = h.merge(r[key], on=key, how="left", indicator=True)
    by_type = pd.DataFrame({
        "hyp": h.type.value_counts(), "ref": r.type.value_counts(),
        "tp": hm[hm._merge == "both"].type.value_counts()}).fillna(0).astype(int).sort_values("ref", ascending=False)
    rep = {"name": name, "n_texts": len(df), "n_para_mismatch": int(n_mm),
           "n_hit_limit": int(out.hit_limit.sum()),
           "tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": rec, "f05": f05,
           "f05_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "edits_per_100_tokens_hyp": rate("hyp"), "edits_per_100_tokens_ref": rate("ref"),
           "unchanged_texts_hyp": int((df.output == df.text).sum()),
           "by_type": by_type.reset_index(names="type").to_dict("records")}
    RES.mkdir(parents=True, exist_ok=True)
    (DATA / "edits").mkdir(parents=True, exist_ok=True)
    edits.to_parquet(DATA / "edits" / f"{name}.parquet", index=False)
    (RES / f"score_{name}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "by_type"}, indent=2))


if __name__ == "__main__":
    main()
