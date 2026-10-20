"""Association pilot, step 1 (GPU) — the model's own marginal subword distribution p̄.

p̄(v) = E_context[ p(v | context) ], the mean next-subword distribution of
OLMo-2 1B (rev a1847dff) over contexts of 7 subwords, i.e. exactly the context a
first subword gets under the Study 1 8-subword window (``lib/lm.py``). The
contexts are drawn from the local Dolma v1.7 sample (``data/dolma/corpus_{a,b}``;
public text, nothing leaves the machine):

  * documents are read in file order and kept with probability 1/4 (seeded);
  * each kept document is tokenized (no special tokens) and cut at 2,048 subwords;
  * every 4th position k >= 7 is a context ids[k-7:k];
  * a source's quota is its share of the corpus (gzip size) of N contexts.

Accumulated over the full vocabulary in float64: sum of softmax (two halves, by
batch parity, for a split-half check) and counts of the observed next subword
(the empirical subword frequency at the same positions).

Writes data/study3/pilot_assoc/pbar_<corpus>.npz:
    sum_p_0, sum_p_1   (V,) float64, sums of p(v | ctx) over the two halves
    n_0, n_1           number of contexts in each half
    obs                (V,) int64 observed next-subword counts
    per-source n, seconds, settings

    CUDA_VISIBLE_DEVICES=0 python .../1_pbar.py --corpus a --n 10000000
    CUDA_VISIBLE_DEVICES=1 python .../1_pbar.py --corpus b --n 5000000
    CUDA_VISIBLE_DEVICES=1 python .../1_pbar.py --corpus a --n 5000000 --seed 7 --offset 2 --tag _2
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from assoc_common import OUT, SEED  # noqa: E402
from lib.lm import LM  # noqa: E402
from util.paths import DATA_DIR  # noqa: E402

CTX = 7
STRIDE = 4
KEEP = 0.25
MAX_SUB = 2048
BATCH = 4096
SOURCES = {"a": ["wiki", "pes2o", "cc_news"], "b": ["reddit", "stackexchange", "cc_blogs"]}


def contexts(path: Path, quota: int, tok, rng, offset: int = 0):
    """Yield arrays of shape (m, CTX) of contexts until ``quota`` is reached."""
    got = 0
    buf = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if rng.random() >= KEEP:
                continue
            buf.append(json.loads(line)["text"])
            if len(buf) < 256:
                continue
            for ids in tok(buf, add_special_tokens=False).input_ids:
                ids = np.asarray(ids[:MAX_SUB], dtype=np.int64)
                if len(ids) <= CTX:
                    continue
                ks = np.arange(CTX + offset, len(ids), STRIDE)
                win = np.lib.stride_tricks.sliding_window_view(ids, CTX)[ks - CTX]
                win = win[: quota - got]
                got += len(win)
                yield win, ids[ks[: len(win)]]
                if got >= quota:
                    return
            buf = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["a", "b"], required=True)
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--offset", type=int, default=0, help="first position k = 7 + offset (0..3)")
    ap.add_argument("--tag", default="", help="output suffix, for extra runs pooled in step 3")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    lm = LM()
    V = lm.model.config.vocab_size
    dev = lm.device
    sum_p = [torch.zeros(V, dtype=torch.float64, device=dev) for _ in range(2)]
    n_half = [0, 0]
    obs = torch.zeros(V, dtype=torch.int64, device=dev)
    files = {s: DATA_DIR / "dolma" / f"corpus_{args.corpus}" / f"{s}.jsonl.gz" for s in SOURCES[args.corpus]}
    size = {s: p.stat().st_size for s, p in files.items()}
    quota = {s: int(round(args.n * size[s] / sum(size.values()))) for s in files}
    rng = np.random.default_rng(args.seed)
    meta = {"corpus": args.corpus, "n_target": args.n, "ctx": CTX, "stride": STRIDE, "keep": KEEP,
            "max_sub": MAX_SUB, "seed": args.seed, "offset": args.offset, "quota": quota, "per_source": {}}
    b = 0
    t_all = time.time()
    for s, path in files.items():
        t0 = time.time()
        pend_x, pend_y, ns = [], [], 0

        def flush(x, y):
            nonlocal b
            xx = torch.as_tensor(x, device=dev)
            with torch.no_grad():
                logits = lm.model(input_ids=xx).logits[:, -1, :V].float()
                p = torch.softmax(logits, -1)
            sum_p[b % 2] += p.sum(0, dtype=torch.float64)
            n_half[b % 2] += len(x)
            obs.index_add_(0, torch.as_tensor(y, device=dev), torch.ones(len(y), dtype=torch.int64, device=dev))
            b += 1

        for win, nxt in contexts(path, quota[s], lm.tok, rng, args.offset):
            pend_x.append(win)
            pend_y.append(nxt)
            ns += len(win)
            if sum(len(w) for w in pend_x) >= BATCH:
                X, Y = np.concatenate(pend_x), np.concatenate(pend_y)
                for i in range(0, len(X) - BATCH + 1, BATCH):
                    flush(X[i:i + BATCH], Y[i:i + BATCH])
                r = len(X) % BATCH
                pend_x, pend_y = ([X[len(X) - r:]], [Y[len(Y) - r:]]) if r else ([], [])
                if b % 200 == 0:
                    el = time.time() - t_all
                    print(f"{s} {ns:,}/{quota[s]:,}  total {sum(n_half):,}  {el:.0f}s "
                          f"({sum(n_half) / el:,.0f} ctx/s)", flush=True)
        if pend_x:
            flush(np.concatenate(pend_x), np.concatenate(pend_y))
        meta["per_source"][s] = {"n": ns, "seconds": round(time.time() - t0, 1)}
        print(s, meta["per_source"][s], flush=True)
    meta["seconds"] = round(time.time() - t_all, 1)
    meta["n"] = sum(n_half)
    meta["gpu"] = torch.cuda.get_device_name(0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"pbar_{args.corpus}{args.tag}.npz", sum_p_0=sum_p[0].cpu().numpy(), sum_p_1=sum_p[1].cpu().numpy(),
             n_0=n_half[0], n_1=n_half[1], obs=obs.cpu().numpy(), meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
