"""Pilot (native calibration), contamination spot check via the infini-gram API.

For N_DOCS documents per group, SPANS_PER_DOC random spans of SPAN_WORDS
consecutive whitespace words (within one paragraph) are counted in the
OLMo-mix-1124 index (``v4_olmo-mix-1124_llama``: OLMo 2's pretraining mix,
Llama-2 tokenizer). A 15-word span of free prose essentially never recurs by
chance, so count >= 1 means that text (or a copy of it) is in the mix.

    python -I contam.py DOCS.parquet OUT.csv [N_DOCS]

DOCS.parquet: group, text_id, text. Public, self-downloaded text only (PERSUADE,
Wikipedia): never send the licensed learner corpora (W&I+LOCNESS, FCE, ELLIPSE,
TOEFL) to the API.
"""

import json
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

INDEX = "v4_olmo-mix-1124_llama"
URL = "https://api.infini-gram.io/"
N_DOCS, SPANS_PER_DOC, SPAN_WORDS = 30, 3, 15
SEED = 42


def count(q: str) -> int | None:
    body = json.dumps({"index": INDEX, "query_type": "count", "query": q}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(d["error"])
            return int(d["count"])
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def spans(text: str, rng) -> list[str]:
    paras = [p.split() for p in re.split(r"\n\s*\n|\n", text)]
    paras = [p for p in paras if len(p) >= SPAN_WORDS]
    out = []
    for _ in range(SPANS_PER_DOC):
        if not paras:
            break
        p = paras[rng.integers(len(paras))]
        s = rng.integers(len(p) - SPAN_WORDS + 1)
        out.append(" ".join(p[s:s + SPAN_WORDS]))
    return out


def main():
    docs = pd.read_parquet(sys.argv[1])
    n_docs = int(sys.argv[3]) if len(sys.argv) > 3 else N_DOCS
    rng = np.random.default_rng(SEED)
    rows = []
    for g, d in docs.groupby("group", sort=False):
        d = d.sample(n=min(n_docs, len(d)), random_state=SEED)
        for tid, text in zip(d.text_id, d.text):
            for q in spans(text, rng):
                rows.append({"group": g, "text_id": tid, "span": q, "count": count(q)})
                time.sleep(0.1)
        r = pd.DataFrame([x for x in rows if x["group"] == g])
        print(g, "spans", len(r), "hit", (r["count"] > 0).mean().round(3), flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
