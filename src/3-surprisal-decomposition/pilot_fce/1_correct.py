"""FCE step 1 (CPU) — corrected answers, gold categories, edit-offset check.

Same as ``pilot_corrected/1_correct.py`` (its ``apply_edits``, the same spaCy
model and settings) on the FCE DocBins from ``src/essays/ingest.py --fce``.
Also writes the stage 2 gold category of every original token
(``2_select.gold_for_doc``: ERR_TOK / ERR_INS / POST / OK) and checks the json
character edits against the released M2 files.

Outputs, under data/study3/pilot_fce/:
    docbins_corrected/processed_docs_00.spacy
    essays.parquet   one row per answer: ids, scores, l1, original and corrected text, cmap, fixes
    edits.parquet, flagged.parquet, gold.parquet
    tables/check_m2.json

    python src/3-surprisal-decomposition/pilot_fce/1_correct.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fce_common import FCE_DOCBINS_DIR, FCE_M2_DIR, OUT, TAB, pc, select  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

apply_edits = pc("common").apply_edits
gold_for_doc = select().gold_for_doc
NS = re.compile(r"\s")


def squash(t: str) -> str:
    """Whitespace removed; the M2 files normalise a backtick apostrophe to "'"."""
    return NS.sub("", t).replace("`", "'")


def m2_sentences(path: Path) -> list:
    """(original, corrected) per M2 sentence, annotator 0, whitespace removed."""
    out = []
    for block in path.read_text().strip().split("\n\n"):
        lines = block.split("\n")
        toks = lines[0][2:].split(" ")
        eds = []
        for ln in lines[1:]:
            f = ln[2:].split("|||")
            s, e = map(int, f[0].split())
            if f[1] == "noop" or f[-1] != "0":
                continue
            eds.append((s, e, "" if f[2] == "-NONE-" else f[2]))
        c = list(toks)
        for s, e, r in sorted(eds, key=lambda x: (x[0], x[1]), reverse=True):
            c[s:e] = r.split()
        out.append(("".join(toks), "".join(c)))
    return out


def check_m2(essays: pd.DataFrame) -> dict:
    """Original text must match M2 exactly (no whitespace); corrected text is compared per answer."""
    import difflib
    res = {}
    for split, d in essays.groupby("split", sort=False):
        sents = m2_sentences(FCE_M2_DIR / f"fce.{split}.gold.bea19.m2")
        mo = "".join(o for o, _ in sents)
        jo = "".join(squash(t) for t in d.text)
        r = {"n_answers": len(d), "n_m2_sentences": len(sents), "original_identical": mo == jo}
        # segment M2 sentences into answers by cumulative original length
        k, n_eq, n_chars, n_diff, worst = 0, 0, 0, 0, []
        for tid, t, c in zip(d.text_id, d.text, d.corrected):
            need, got_o, got_c = len(squash(t)), "", ""
            while len(got_o) < need:
                got_o += sents[k][0]
                got_c += sents[k][1]
                k += 1
            assert got_o == squash(t), tid
            mine = squash(c)
            n_chars += len(mine)
            if mine == got_c:
                n_eq += 1
            else:
                sm = difflib.SequenceMatcher(None, mine, got_c, autojunk=False)
                dd = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
                n_diff += dd
                worst.append((dd, tid))
        assert k == len(sents)
        r.update(corrected_identical_share=n_eq / len(d), corrected_diff_chars=n_diff,
                 corrected_chars=n_chars, corrected_diff_share=n_diff / n_chars,
                 worst=sorted(worst, reverse=True)[:5])
        res[split] = r
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rows, edits, flagged, gold, metas, texts = [], [], [], [], [], []
    for doc in load_all_docbins(FCE_DOCBINS_DIR):
        m = doc.user_data["meta"]
        ed = json.loads(m["edits"])
        r = apply_edits(doc.text, ed)
        tid = m["text_id"]
        rows.append({"text_id": tid, "script_id": m["script_id"], "q": m["q"], "split": m["split"],
                     "l1": m["l1"], "age": m["age"], "script_s": m["script_s"],
                     "answer_s": m["answer_s"], "text": doc.text, "corrected": r["corrected"],
                     "cmap": r["cmap"].tolist(), **r["fixes"]})
        edits += [{"text_id": tid, **s} for s in r["spans"]]
        flagged += [{"text_id": tid, "s": s, "e": e} for s, e in r["flagged"]]
        gold += [(tid, i, c) for i, c in gold_for_doc(doc, ed).items()]
        metas.append(m)
        texts.append(r["corrected"])
    e = pd.DataFrame(rows)
    e.to_parquet(OUT / "essays.parquet", index=False)
    pd.DataFrame(edits).to_parquet(OUT / "edits.parquet", index=False)
    pd.DataFrame(flagged).to_parquet(OUT / "flagged.parquet", index=False)
    pd.DataFrame(gold, columns=["text_id", "spacy_idx", "gold"]).to_parquet(OUT / "gold.parquet", index=False)
    fixes = e.drop(columns=[c for c in e.columns if c not in
                            ("del_swallow_space", "del_space_before_punct", "strip_lead",
                             "strip_trail", "ins_add_space", "skipped_overlap")]).sum()
    print(fixes.to_string())
    print(f"{len(edits)} applied edits, {len(flagged)} flagged spans")
    chk = {"fixes": {k: int(v) for k, v in fixes.items()}, "n_applied": len(edits),
           "n_flagged": len(flagged), "m2": check_m2(e)}
    (TAB / "check_m2.json").write_text(json.dumps(chk, indent=2))
    print(json.dumps(chk, indent=1))

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
