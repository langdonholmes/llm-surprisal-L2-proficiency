"""Per-token surprisal for the worked example behind the H2 self-conditioning conjecture.

Study 1 finds that concurrent validity *falls* as the context window grows, which
is the opposite of what a discourse-sensitivity account predicts. The conjecture
offered for that reversal is that a long window conditions the model on the
writer's own earlier production, so a non-standard form the writer has already
used becomes cheap the next time it appears — and the measure drifts from
conventionality toward self-consistency.

This script produces the token-level evidence for one essay. It scores the same
essay at every window with the selected Study 1 configuration (OLMo-2 1B base)
and writes one row per token per window, flagging the repeated non-standard form
and, where the essay also uses the standard counterpart, that form too. The
figure and the numbers are built from this table in
`window-self-conditioning.qmd`.

    python src/1-predictability-benchmark/self_conditioning_example.py

Output: results/predictability/self_conditioning_tokens.csv
"""

import argparse
import math

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from features.predictability import Predictor
from util.paths import RESULTS_DIR, TOEFL_DOCBINS_DIR
from util.process_docs import load_all_docbins

# Sibling module whose directory name starts with a digit (see run_surprisal.py).
from models import MODEL_REGISTRY

LOG2E = 1.0 / math.log(2.0)  # nats -> bits
WINDOWS = ["8", "64", "full"]
MODEL_KEY = "olmo2-1b"  # the configuration selected for Studies 2-3

# The essays carrying a repeated non-standard form, found by scanning TOEFL 11
# for a single form repeated three or more times within one response.
# `standard` is that form's conventional counterpart where the same essay also
# uses it, which is what makes the discrimination gap measurable within a text.
EXAMPLES = {
    # 12x "peoples" alongside 3x correct "people" — the only essay in the scan
    # that supplies the minimal pair inside a single response.
    "704896.txt": {"target": "peoples", "standard": "people"},
    # 9x "peoples", no standard counterpart, but the cleanest occurrence trend
    # and a first occurrence in essay-initial position (no prior context in
    # either condition, so the two windows have to agree there).
    "1885778.txt": {"target": "peoples", "standard": None},
    # 6x "informations" in otherwise fluent prose by a high-proficiency writer,
    # with the correct singular at the essay's last content position.
    "594970.txt": {"target": "informations", "standard": "information"},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-ids", nargs="+", default=list(EXAMPLES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    wanted = set(args.text_ids)
    docs = {}
    for doc in load_all_docbins(TOEFL_DOCBINS_DIR):
        tid = str(doc.user_data["meta"]["text_id"])
        if tid in wanted:
            docs[tid] = doc
            if len(docs) == len(wanted):
                break
    missing = wanted - set(docs)
    if missing:
        raise SystemExit(f"not found in the TOEFL DocBins: {sorted(missing)}")

    spec = MODEL_REGISTRY[MODEL_KEY]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
    )
    predictor = Predictor(
        tokenizer=tok, model=model, model_type=spec.arch,
        batch_size=args.batch_size, device=args.device,
    )

    rows = []
    for tid in args.text_ids:
        doc = docs[tid]
        meta = doc.user_data["meta"]
        target = EXAMPLES[tid]["target"]
        standard = EXAMPLES[tid]["standard"]

        # Occurrence index over the target form, in reading order, so the figure
        # can plot surprisal against "how many times has this appeared already".
        occurrence = {}
        for t in doc:
            if t.text.lower() == target:
                occurrence[t.i] = len(occurrence) + 1

        for window in WINDOWS:
            w = spec.max_ctx if window == "full" else int(window)
            dp = predictor(doc, window_size=w)
            for t in dp:
                low = doc[t.spacy_idx].text.lower()
                rows.append({
                    "text_id": tid,
                    "l1": meta["l1"],
                    "level": meta["level"],
                    "prompt": meta["prompt"],
                    "window": window,
                    "spacy_idx": t.spacy_idx,
                    "token": t.text,
                    "space_after": bool(doc[t.spacy_idx].whitespace_),
                    "surprisal_bits": t.mean_loss * LOG2E,
                    "form": ("target" if low == target
                             else "standard" if standard and low == standard
                             else "other"),
                    "occurrence": occurrence.get(t.spacy_idx, pd.NA),
                })
            print(f"{tid}  window {window:<4}  {len(dp)} tokens scored  "
                  f"mean {dp.mean_loss * LOG2E:.2f} bits", flush=True)

    out = RESULTS_DIR / "predictability" / "self_conditioning_tokens.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
