"""Study 3, stage 3 — LM repairs for every high-surprisal token of a bench.

For each token above tau, about 23 minimal edits are proposed (another
inflection, a near spelling, one of the model's top predictions, deletion,
insertion of a predicted word before it) and scored by how much they raise the
log-probability of the word plus its next five subwords, under the same model
and window as the surprisal (lib/repair_core.py). Two readouts per token:

    label    the best edit after a cost of 1 bit per character changed, by
             subclass (e.g. INS_DET, MORPH_S_V, FORM_NONWORD), or NONE when no
             edit pays for itself
    gain_*   the best gain (bits, positive part, no cost) in each of 29
             subclasses: the vector behind essay profiles

Steps, per bench (ellipse_dev, ellipse_test, wi, toefl11_p0, toefl11_p1):

    base     GPU. Top-50 prediction, surprisal and entropy at the token and the
             next word (needed for the PRED and INS candidates)
    cands    GPU, ~13 ms per token. Build and score the candidate edits
    labels   CPU. -> data/study3/repair/labels_<bench>.parquet
             (row_id, label, net_gain, best_word, scored, gain_<subclass> x 29)

    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/3_repair.py base ellipse_dev
    CUDA_VISIBLE_DEVICES=0 python src/3-surprisal-decomposition/3_repair.py cands ellipse_dev
    python src/3-surprisal-decomposition/3_repair.py labels ellipse_dev ellipse_test wi
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bench import Bench  # noqa: E402

BATCH = 256


def cmd_base(name: str):
    from lib.lm import LM
    b = Bench(name, base=False)
    lm = LM()
    wins = {"t": [], "n": []}
    meta = []
    # Words that do not start on a subword boundary are unscored, as in Study 1.
    for r in b.tokens.itertuples():
        try:
            ids, k = lm.locate(r.text, r.target_char)
        except ValueError:
            wins["t"].append(None)
            wins["n"].append(None)
            meta.append({"inst": r.inst, "has_target": False, "has_next": False, "k": -1})
            continue
        wins["t"].append(lm.window(ids, k) if k > 0 else None)
        has_next = not pd.isna(r.next_char)
        if has_next:
            try:
                _, kn = lm.locate(r.text, int(r.next_char))
                wins["n"].append(lm.window(ids, kn))
            except ValueError:
                has_next = False
                wins["n"].append(None)
        else:
            wins["n"].append(None)
        meta.append({"inst": r.inst, "has_target": k > 0, "has_next": has_next, "k": k})
    out = {}
    for pos, ws in wins.items():
        idx = [i for i, w in enumerate(ws) if w is not None]
        chunks = [lm.score([ws[i] for i in idx[a:a + BATCH]]) for a in range(0, len(idx), BATCH)]
        for key in chunks[0]:
            full = np.concatenate([c[key] for c in chunks])
            arr = np.zeros((len(ws),) + full.shape[1:], dtype=full.dtype)
            arr[idx] = full
            out[f"{pos}_{key}"] = arr
        print(pos, "done", flush=True)
    np.savez(b.path("base", "npz"), **out)
    pd.DataFrame(meta).to_parquet(b.path("base_meta"), index=False)


def cmd_cands(name: str):
    from lib.repair_candidates import build
    build(Bench(name))


def cmd_labels(name: str):
    from lib import repair_labels as RL
    b = Bench(name)
    df = RL.labels(b)
    gains = pd.DataFrame(RL.sub_matrix(b), columns=[f"gain_{s}" for s in RL.subclass_names()])
    df = pd.concat([df, gains], axis=1)
    df.insert(0, "row_id", b.tokens.row_id.to_numpy())
    df["scored"] = b.ok
    df.to_parquet(b.path("labels"), index=False)
    print(name, df[df.scored].label.value_counts().head(10).to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["base", "cands", "labels"])
    ap.add_argument("benches", nargs="+")
    args = ap.parse_args()
    for name in args.benches:
        {"base": cmd_base, "cands": cmd_cands, "labels": cmd_labels}[args.step](name)


if __name__ == "__main__":
    main()
