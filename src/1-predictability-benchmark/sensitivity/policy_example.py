"""One sentence scored under every aggregation policy, for the worked example.

`subword-policy.qmd` reports corpus-level differences between the policies. This
makes the mechanism concrete on a single sentence containing the three cases that
matter: words the tokenizer leaves whole, a misspelling, and a long derivation.

Runs on CPU in seconds. Writes results/predictability/subword_policy_example.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from util.paths import RESULTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subword_policy import (  # noqa: E402
    LOG2E, WINDOW, causal_losses, masked_word_losses, word_spans,
)

SENTENCE = ("The students misstake the rules for unpredictability every single day.")
MODELS = [("olmo2-1b", "allenai/OLMo-2-0425-1B", "causal"),
          ("bert-base", "google-bert/bert-base-uncased", "masked")]
OUT = RESULTS_DIR / "predictability" / "subword_policy_example.csv"


def main() -> None:
    nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "lemmatizer"])
    nlp.add_pipe("sentencizer")
    doc = nlp(SENTENCE)

    rows = []
    for key, hf_id, arch in MODELS:
        tok = AutoTokenizer.from_pretrained(hf_id)
        cls = AutoModelForMaskedLM if arch == "masked" else AutoModelForCausalLM
        model = cls.from_pretrained(hf_id, dtype=torch.float32).eval()
        spans, ids, _ = word_spans(doc, tok)

        if arch == "causal":
            targets = sorted({t for _i, s, e in spans for t in range(s, e)})
            loss = causal_losses(model, ids, targets, WINDOW, "cpu", 64)
            for spacy_idx, s, e in spans:
                vals = [loss[t] for t in range(s, e)]
                if not np.isfinite(vals).all():
                    continue
                rows.append({
                    "model": key, "arch": arch, "word": doc[spacy_idx].text,
                    "position": spacy_idx, "k": e - s,
                    "pieces": "|".join(tok.decode([ids[i]]).strip() for i in range(s, e)),
                    "first": vals[0] * LOG2E,
                    "joint": float(np.sum(vals)) * LOG2E,
                    "density": float(np.mean(vals)) * LOG2E,
                    "joint_indep": np.nan,
                })
        else:
            per_word = masked_word_losses(model, tok, ids, spans, WINDOW, "cpu", 64)
            for spacy_idx, s, e in spans:
                indep, chain = per_word[spacy_idx]
                rows.append({
                    "model": key, "arch": arch, "word": doc[spacy_idx].text,
                    "position": spacy_idx, "k": len(indep),
                    "pieces": "|".join(tok.decode([ids[i]]).strip() for i in range(s, e)),
                    "first": indep[0] * LOG2E,
                    "joint": float(np.sum(chain)) * LOG2E,
                    "density": float(np.mean(indep)) * LOG2E,
                    "joint_indep": float(np.sum(indep)) * LOG2E,
                })

    df = pd.DataFrame(rows).sort_values(["model", "position"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  {df.shape}\n")
    with pd.option_context("display.width", 200):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
