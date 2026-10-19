"""Pilot, step 1 (CPU) — the corrected version of every W&I+LOCNESS essay.

Applies the first annotator's gold edits to each essay (``common.apply_edits``)
and parses the corrected text with the same spaCy model and settings as
``src/essays/ingest.py`` (en_core_web_lg, NER disabled).

Outputs, under data/study3/pilot_corrected/:
    docbins_corrected/processed_docs_00.spacy   corrected docs, meta as in the originals
    essays.parquet      text_id, level, original and corrected text, cmap (list), fixes
    edits.parquet       one row per applied edit: orig [s, e), corrected [cs, ce), kind
    flagged.parquet     null-replacement spans (flagged, not corrected)

    python src/3-surprisal-decomposition/pilot_corrected/1_correct.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, apply_edits  # noqa: E402
from util.paths import WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, edits, flagged, metas, texts = [], [], [], [], []
    for doc in load_all_docbins(WI_DOCBINS_DIR):
        m = doc.user_data["meta"]
        r = apply_edits(doc.text, json.loads(m["edits"]))
        tid = m["text_id"]
        rows.append({"text_id": tid, "level": m["level"], "text": doc.text,
                     "corrected": r["corrected"], "cmap": r["cmap"].tolist(), **r["fixes"]})
        edits += [{"text_id": tid, **s} for s in r["spans"]]
        flagged += [{"text_id": tid, "s": s, "e": e} for s, e in r["flagged"]]
        metas.append(m)
        texts.append(r["corrected"])
    e = pd.DataFrame(rows)
    e.to_parquet(OUT / "essays.parquet", index=False)
    pd.DataFrame(edits).to_parquet(OUT / "edits.parquet", index=False)
    pd.DataFrame(flagged).to_parquet(OUT / "flagged.parquet", index=False)
    print(e.drop(columns=["text_id", "level", "text", "corrected", "cmap"]).sum().to_string())
    print(f"{len(edits)} applied edits, {len(flagged)} flagged spans")

    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    db = DocBin(store_user_data=True)
    for doc, meta in nlp.pipe(zip(texts, metas), as_tuples=True, n_process=16, batch_size=256):
        doc.user_data["meta"] = meta
        db.add(doc)
    (OUT / "docbins_corrected").mkdir(exist_ok=True)
    db.to_disk(OUT / "docbins_corrected" / "processed_docs_00.spacy")
    print("parsed", len(texts))


if __name__ == "__main__":
    main()
