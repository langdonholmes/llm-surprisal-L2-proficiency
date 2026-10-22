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
    essays.parquet      text_id, unit, group, y, text, corrected, cmap
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
from lib.corpora import ALL_IDS, DOCBINS, LEVEL_Y, S3, VARIANTS, base, fce_band, out  # noqa: E402
from lib.corrections import apply_edits, gold_for_doc  # noqa: E402
from lib.errant import bucket_of, edit_types, group_of, op_of  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

C5 = S3 / "corrector"


def unit_group_y(corpus: str, m: dict):
    tid = m["text_id"]
    if corpus == "wi":
        return tid, m["level"], LEVEL_Y.get(m["level"])
    if corpus == "fce":
        score = pd.to_numeric(m["script_s"], errors="coerce")
        return m["script_id"], fce_band(score), (score if score > 0 else None)
    y = float(m["overall"])  # ELLIPSE: group is the overall score as written
    return tid, f"{y:.1f}", y


def c5_edits(source: str):
    """{text_id: [(s, e, repl, errant), ...]} and {text_id: corrected} from corrector/c5_edits.py."""
    E = pd.read_parquet(C5 / f"char_edits_{source}.parquet").sort_values(["text_id", "k"])
    T = pd.read_parquet(C5 / f"char_edits_{source}_texts.parquet")
    by = {t: list(zip(d.s, d.e, d.repl, d.errant)) for t, d in E.groupby("text_id")}
    return by, dict(zip(T.text_id, T.corrected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=ALL_IDS, required=True)
    args = ap.parse_args()
    corpus, bc = args.corpus, base(args.corpus)
    O = out(corpus)
    if corpus in VARIANTS:
        typed, target = c5_edits(VARIANTS[corpus][1])
    essays, edits, gold, metas, texts = [], [], [], [], []
    fixes = pd.Series(0, index=["del_swallow_space", "del_space_before_punct", "strip_lead",
                                "strip_trail", "ins_add_space", "skipped_overlap"])
    for doc in load_all_docbins(DOCBINS[bc]):
        m = doc.user_data["meta"]
        tid = m["text_id"]
        if corpus in VARIANTS:
            assert tid in target, f"{tid} has no correction in the c5 source"
            te = typed.get(tid, [])
            ed = [[s, e, r] for s, e, r, _ in te]
            r = apply_edits(doc.text, ed, repair=False)
            assert r["corrected"] == target[tid], tid
            edits += [(tid, k, s, e, rep, "ins" if s == e else ("del" if rep == "" else "sub"),
                       t, group_of(t), bucket_of(t), op_of(t, s, e, rep)) for k, (s, e, rep, t) in enumerate(te)]
        else:
            ed = json.loads(m["edits"])
            r = apply_edits(doc.text, ed)
            edits += [(tid, k, s, e, rep, "flagged" if rep is None else
                       ("ins" if s == e else ("del" if rep == "" else "sub"))) for k, (s, e, rep) in enumerate(ed)]
        fixes += pd.Series(r["fixes"])
        unit, group, y = unit_group_y(bc, m)
        essays.append({"text_id": tid, "unit": unit, "group": group, "y": y, "text": doc.text,
                       "corrected": r["corrected"], "cmap": r["cmap"].tolist()})
        gold += [(tid, i, c) for i, c in gold_for_doc(doc, ed).items()]
        metas.append(m)
        texts.append(r["corrected"])

    cols = ["text_id", "k", "s", "e", "repl", "kind"]
    if corpus in VARIANTS:
        E = pd.DataFrame(edits, columns=cols + ["errant", "group", "bucket", "op"])
    else:
        E = pd.DataFrame(edits, columns=cols)
        T = edit_types(corpus)
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
