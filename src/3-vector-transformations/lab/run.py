"""Lab — featurize + evaluate one or more representations, append to the scoreboard.

    python src/3-vector-transformations/lab/run.py delta e_obs surprisal
    python src/3-vector-transformations/lab/run.py --list

Each name is a module in lab/reps/. Matrices are cached under
data/study3/lab/reps/<name>.npy (delete the file, or pass --refresh, to
recompute). Results: results/study3/lab/scoreboard.csv (one row per run,
newest last) and results/study3/lab/details/<name>.json.
"""

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import LAB, OUT, SUFFIX, Ctx, evaluate  # noqa: E402

REPS = Path(__file__).resolve().parent / "reps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(sorted(p.stem for p in REPS.glob("*.py") if p.stem != "__init__"))
        return
    ctx = Ctx()
    (LAB / "reps").mkdir(exist_ok=True)
    (OUT / "details").mkdir(parents=True, exist_ok=True)
    for name in args.names:
        mod = importlib.import_module(f"lab.reps.{name}")
        cache = LAB / "reps" / f"{name}{SUFFIX}.npy"
        t0 = time.time()
        if cache.exists() and not args.refresh:
            X = np.load(cache)
        else:
            X = np.asarray(mod.featurize(ctx), dtype=np.float32)
            assert X.shape[0] == ctx.n, f"{name}: {X.shape[0]} rows, bench has {ctx.n}"
            np.save(cache, X)
        t1 = time.time()
        res, detail = evaluate(X, ctx)
        row = {"rep": name, "when": datetime.now().isoformat(timespec="seconds"),
               "desc": (mod.__doc__ or "").strip().splitlines()[0],
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
               "featurize_s": round(t1 - t0, 1), "eval_s": round(time.time() - t1, 1)}
        sb = OUT / f"scoreboard{SUFFIX}.csv"
        pd.concat([pd.read_csv(sb) if sb.exists() else pd.DataFrame(), pd.DataFrame([row])]) \
            .to_csv(sb, index=False)
        (OUT / "details" / f"{name}{SUFFIX}.json").write_text(json.dumps(detail, indent=2, default=str))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
