"""Study 3, stage 1 (CPU) — the corrected version of every essay, and the typed edits.

Applies the first annotator's gold edits to each text (``lib.corrections``),
gives every edit its ERRANT type, group and bucket (``lib.errant``), records
the gold category of every original token, and parses the corrected texts with
the spaCy model and settings of ``src/essays/ingest.py`` (en_core_web_lg, NER
disabled).

Writes, under data/study3/<corpus>/:
    essays.parquet      text_id, unit, group, y, text, corrected, cmap
    edits.parquet       text_id, k (index in the json edit list), s, e, repl, kind,
                        errant, group, bucket, op
    gold.parquet        text_id, spacy_idx, gold (ERR_TOK / ERR_INS / POST / OK)
    docbins_corrected/processed_docs_00.spacy

    python src/3-surprisal-decomposition/1_edits.py --corpus wi
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import DOCBINS, LEVEL_Y, fce_band, out  # noqa: E402
from lib.corrections import apply_edits, gold_for_doc  # noqa: E402
from lib.errant import edit_types  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["wi", "fce"], required=True)
    args = ap.parse_args()
    O = out(args.corpus)
    essays, edits, gold, metas, texts = [], [], [], [], []
    fixes = pd.Series(0, index=["del_swallow_space", "del_space_before_punct", "strip_lead",
                                "strip_trail", "ins_add_space", "skipped_overlap"])
    for doc in load_all_docbins(DOCBINS[args.corpus]):
        m = doc.user_data["meta"]
        tid = m["text_id"]
        ed = json.loads(m["edits"])
        r = apply_edits(doc.text, ed)
        fixes += pd.Series(r["fixes"])
        if args.corpus == "wi":
            unit, group, y = tid, m["level"], LEVEL_Y.get(m["level"])
        else:
            score = pd.to_numeric(m["script_s"], errors="coerce")
            unit, group = m["script_id"], fce_band(score)
            y = score if score > 0 else None
        essays.append({"text_id": tid, "unit": unit, "group": group, "y": y, "text": doc.text,
                       "corrected": r["corrected"], "cmap": r["cmap"].tolist()})
        edits += [(tid, k, s, e, rep, "flagged" if rep is None else
                   ("ins" if s == e else ("del" if rep == "" else "sub"))) for k, (s, e, rep) in enumerate(ed)]
        gold += [(tid, i, c) for i, c in gold_for_doc(doc, ed).items()]
        metas.append(m)
        texts.append(r["corrected"])

    E = pd.DataFrame(edits, columns=["text_id", "k", "s", "e", "repl", "kind"])
    T = edit_types(args.corpus)
    E = E.merge(T, on=["text_id", "k"], how="left", suffixes=("", "_m2"), validate="one_to_one")
    assert E.errant.notna().all() and (E.s == E.s_m2).all() and (E.e == E.e_m2).all()
    E = E.drop(columns=["s_m2", "e_m2"])
    pd.DataFrame(essays).to_parquet(O / "essays.parquet", index=False)
    E.to_parquet(O / "edits.parquet", index=False)
    pd.DataFrame(gold, columns=["text_id", "spacy_idx", "gold"]).to_parquet(O / "gold.parquet", index=False)
    print(fixes.to_string())
    print(E.kind.value_counts().to_string())
    print("unmapped share of corrections:", round(float((E.errant[E.kind != "flagged"] == "unmapped").mean()), 4))

    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    db = DocBin(store_user_data=True)
    for doc, meta in nlp.pipe(zip(texts, metas), as_tuples=True, n_process=16, batch_size=256):
        doc.user_data["meta"] = meta
        db.add(doc)
    (O / "docbins_corrected").mkdir(exist_ok=True)
    db.to_disk(O / "docbins_corrected" / "processed_docs_00.spacy")
    print("parsed", len(texts))


if __name__ == "__main__":
    main()
