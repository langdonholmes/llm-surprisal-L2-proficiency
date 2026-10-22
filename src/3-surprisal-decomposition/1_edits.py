"""Study 3, stage 1 (CPU) — the corrected version of every essay, and the typed edits.

For ``wi`` and ``fce``, applies the first annotator's gold edits to each text
(``lib.corrections``) and gives every edit its ERRANT type from the BEA-2019 M2
files (``lib.errant``). For the other corpus ids (``lib.corpora.VARIANTS``:
the corrector's corrections of W&I, FCE and ELLIPSE, and the annotator's
corrected texts re-cut by ERRANT), the edits and their ERRANT 3 types come
from ``corrector/c5_edits.py``. They carry their own whitespace, so they are
applied as they are, and the result is checked against c5's corrected text.

Then, for every corpus id: each edit's group and bucket, the gold category of
every original token, and a parse of the corrected texts with the spaCy model
and settings of ``src/essays/ingest.py`` (en_core_web_lg, NER disabled).
("Gold" in gold.parquet means the category of a token relative to the edits
in use, whatever their source.)

Writes, under data/study3/<corpus id>/:
    essays.parquet      text_id, unit, group, y, text, corrected, cmap (TOEFL 11: + l1, prompt)
    edits.parquet       text_id, k (index in the text's edit list), s, e, repl, kind,
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
from lib.corpora import (  # noqa: E402
    ALL_IDS, CEFR_SCORE, DATA, DOCBINS, TOEFL_BANDS, VARIANTS, base_corpus, data_dir, ellipse_band, fce_band)
from lib.corrections import apply_edits, gold_for_doc  # noqa: E402
from lib.errant import bucket_of, edit_types, group_of, op_of  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

C5 = DATA / "corrector"


def unit_group_y(corpus: str, meta: dict):
    """The analysis unit, level group and criterion of one text."""
    text_id = meta["text_id"]
    if corpus == "wi":
        return text_id, meta["level"], CEFR_SCORE.get(meta["level"])
    if corpus == "fce":
        score = pd.to_numeric(meta["script_s"], errors="coerce")
        return meta["script_id"], fce_band(score), (score if score > 0 else None)
    if corpus == "toefl":  # TOEFL 11: group is the band, y its rank
        return text_id, meta["level"], float(TOEFL_BANDS.index(meta["level"]) + 1)
    y = float(meta["overall"])  # ELLIPSE: group is the score band
    return text_id, ellipse_band(y), y


def c5_edits(source: str):
    """{text_id: [(s, e, repl, errant), ...]} and {text_id: corrected} from corrector/c5_edits.py."""
    edits = pd.read_parquet(C5 / f"char_edits_{source}.parquet").sort_values(["text_id", "k"])
    texts = pd.read_parquet(C5 / f"char_edits_{source}_texts.parquet")
    by_text = {t: list(zip(d.s, d.e, d.repl, d.errant)) for t, d in edits.groupby("text_id")}
    return by_text, dict(zip(texts.text_id, texts.corrected))


def edit_kind(s, e, repl):
    return "flagged" if repl is None else ("ins" if s == e else ("del" if repl == "" else "sub"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=ALL_IDS, required=True)
    args = ap.parse_args()
    corpus, base = args.corpus, base_corpus(args.corpus)
    out_dir = data_dir(corpus)
    if corpus in VARIANTS:
        typed, target = c5_edits(VARIANTS[corpus][1])
    essays, edits, gold, metas, texts = [], [], [], [], []
    fixes = pd.Series(0, index=["del_swallow_space", "del_space_before_punct", "strip_lead",
                                "strip_trail", "ins_add_space", "skipped_overlap"])
    for doc in load_all_docbins(DOCBINS[base]):
        meta = doc.user_data["meta"]
        text_id = meta["text_id"]
        if corpus in VARIANTS:
            # ERRANT's typed edits; applied as they are, they must give the c5 correction exactly
            assert text_id in target, f"{text_id} has no correction in the c5 source"
            typed_edits = typed.get(text_id, [])
            edit_list = [[s, e, repl] for s, e, repl, _ in typed_edits]
            applied = apply_edits(doc.text, edit_list, repair=False)
            assert applied["corrected"] == target[text_id], text_id
            edits += [(text_id, k, s, e, repl, edit_kind(s, e, repl), etype, group_of(etype), bucket_of(etype),
                       op_of(etype, s, e, repl)) for k, (s, e, repl, etype) in enumerate(typed_edits)]
        else:
            # the annotator's json edits; their ERRANT types are joined below
            edit_list = json.loads(meta["edits"])
            applied = apply_edits(doc.text, edit_list)
            edits += [(text_id, k, s, e, repl, edit_kind(s, e, repl)) for k, (s, e, repl) in enumerate(edit_list)]
        fixes += pd.Series(applied["fixes"])
        unit, group, y = unit_group_y(base, meta)
        essays.append({"text_id": text_id, "unit": unit, "group": group, "y": y, "text": doc.text,
                       "corrected": applied["corrected"], "cmap": applied["cmap"].tolist(),
                       **({"l1": meta["l1"], "prompt": meta["prompt"]} if base == "toefl" else {})})
        gold += [(text_id, i, cat) for i, cat in gold_for_doc(doc, edit_list).items()]
        metas.append(meta)
        texts.append(applied["corrected"])

    cols = ["text_id", "k", "s", "e", "repl", "kind"]
    if corpus in VARIANTS:
        edit_table = pd.DataFrame(edits, columns=cols + ["errant", "group", "bucket", "op"])
    else:
        edit_table = pd.DataFrame(edits, columns=cols)
        types = edit_types(corpus)
        edit_table = edit_table.merge(types, on=["text_id", "k"], how="left", suffixes=("", "_m2"),
                                      validate="one_to_one")
        assert edit_table.errant.notna().all()
        assert (edit_table.s == edit_table.s_m2).all() and (edit_table.e == edit_table.e_m2).all()
        edit_table = edit_table.drop(columns=["s_m2", "e_m2"])
    pd.DataFrame(essays).to_parquet(out_dir / "essays.parquet", index=False)
    edit_table.to_parquet(out_dir / "edits.parquet", index=False)
    pd.DataFrame(gold, columns=["text_id", "spacy_idx", "gold"]).to_parquet(out_dir / "gold.parquet", index=False)
    print(fixes.to_string())
    print(edit_table.kind.value_counts().to_string())
    corrections = edit_table[edit_table.kind != "flagged"]
    print("unmapped share of corrections:", round(float((corrections.errant == "unmapped").mean()), 4))

    # parse the corrected essays, for stage 2
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    db = DocBin(store_user_data=True)
    for doc, meta in nlp.pipe(zip(texts, metas), as_tuples=True, n_process=16, batch_size=256):
        doc.user_data["meta"] = meta
        db.add(doc)
    (out_dir / "docbins_corrected").mkdir(exist_ok=True)
    db.to_disk(out_dir / "docbins_corrected" / "processed_docs_00.spacy")
    print("parsed", len(texts))


if __name__ == "__main__":
    main()
