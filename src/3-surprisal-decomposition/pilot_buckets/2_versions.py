"""Bucket pilot, step 2 (CPU) — partially corrected versions, parsed, classic items.

For every unit (W&I learner essay; FCE script with score > 0, both answers) the
players are the correction buckets present among its non-null gold edits
(mechanical, function word, content-word choice, multi-word / order, and
unclassified as a fifth player). For every subset A of the present players, each
answer of the unit gets the version with only the edits of buckets in A applied
(``pilot_corrected/common.apply_edits``; flagged null spans are never applied,
so they stay in every version). Identical texts are parsed once.

Parsing: en_core_web_lg with NER disabled, as ``pilot_corrected/1_correct.py``
and ``src/essays/ingest.py``. Items: ``pilot_assoc/2_classic.doc_items``.

Writes data/study3/pilot_buckets/:
    versions_<corpus>.parquet   text_id, unit, mask (bitmask over PLAYERS of applied buckets), uid,
                                requested/applied edit counts, apply_edits fixes
    texts_<corpus>.parquet      uid, text (unique version texts)
    items_<kind>_<corpus>.parquet   doc_items output keyed by uid (text_id column = uid)
    tables/versions_<corpus>.json   counts and parse time

    python src/3-surprisal-decomposition/pilot_buckets/2_versions.py [--corpus wi|fce] [--procs 24]
"""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bucket_common import OUT, PLAYERS, S3_SRC, TAB, bucket_of, load  # noqa: E402
from util.paths import DATA_DIR, FCE_DOCBINS_DIR, WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

apply_edits = load("pc_common", S3_SRC / "pilot_corrected" / "common.py").apply_edits
classic = load("assoc_classic", S3_SRC / "pilot_assoc" / "2_classic.py")

EDIT_TYPES = {"wi": DATA_DIR / "study3" / "pilot_shapley" / "edit_types.parquet",
              "fce": DATA_DIR / "study3" / "pilot_types" / "edit_types_fce.parquet"}
SRC = {"wi": WI_DOCBINS_DIR, "fce": FCE_DOCBINS_DIR}
FIXES = ["del_swallow_space", "del_space_before_punct", "strip_lead", "strip_trail", "ins_add_space",
         "skipped_overlap"]


def subsets(players):
    for r in range(len(players) + 1):
        yield from combinations(players, r)


def build(corpus):
    et = pd.read_parquet(EDIT_TYPES[corpus])
    bk = {(t, k): bucket_of(e) for t, k, e, kind in zip(et.text_id, et.k, et.errant, et.kind) if kind != "flagged"}
    pos = {(t, k): (s, e) for t, k, s, e in zip(et.text_id, et.k, et.s, et.e)}
    docs = []
    for doc in load_all_docbins(SRC[corpus]):
        m = doc.user_data["meta"]
        if corpus == "wi":
            if m["level"] == "N":
                continue
            unit = m["text_id"]
        else:
            s = pd.to_numeric(m["script_s"], errors="coerce")
            if not s > 0:
                continue
            unit = m["script_id"]
        docs.append((m["text_id"], unit, doc.text, json.loads(m["edits"])))
    # players present per unit
    pres, mismatch, missing = {}, 0, 0
    for tid, unit, _, ed in docs:
        for k, (s, e, r) in enumerate(ed):
            if r is None:
                continue
            if (tid, k) not in bk:
                missing += 1
                continue
            mismatch += pos[(tid, k)] != (s, e)
            pres.setdefault(unit, set()).add(bk[(tid, k)])
    rows, texts, seen = [], [], {}
    for tid, unit, text, ed in docs:
        P = [p for p in PLAYERS if p in pres.get(unit, set())]
        for A in subsets(P):
            mask = sum(1 << PLAYERS.index(p) for p in A)
            sub = [x for k, x in enumerate(ed) if x[2] is None or bk.get((tid, k), "unclassified") in A]
            req = sum(x[2] is not None for x in sub)
            r = apply_edits(text, sub)
            c = r["corrected"]
            if c not in seen:
                seen[c] = f"{corpus}{len(seen)}"
                texts.append((seen[c], c))
            rows.append({"text_id": tid, "unit": unit, "mask": mask, "n_players": len(P), "uid": seen[c],
                         "requested": req, "applied": len(r["spans"]), **r["fixes"]})
    V = pd.DataFrame(rows)
    T = pd.DataFrame(texts, columns=["uid", "text"])
    info = {"docs": len(docs), "units": int(V.unit.nunique()), "versions": len(V), "unique_texts": len(T),
            "edit_type_missing": missing, "edit_offset_mismatch": int(mismatch),
            "players_per_unit": V.groupby("unit").n_players.first().value_counts().sort_index().to_dict(),
            "units_with_player": {p: int(sum(p in s for u, s in pres.items() if u in set(V.unit))) for p in PLAYERS}}
    return V, T, info


def parse(T, procs):
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    out = {"tok": [], "bigram": [], "dep": []}
    batch, t0 = [], time.time()
    stream = nlp.pipe(((t, {"text_id": u}) for u, t in zip(T.uid, T.text)), as_tuples=True,
                      n_process=procs, batch_size=64)
    n = 0
    for doc, meta in stream:
        doc.user_data["meta"] = meta
        batch.append(doc)
        n += 1
        if len(batch) == 4000:
            for k, x in zip(out, classic.doc_items(batch)):
                out[k].append(x)
            batch = []
            print(f"  parsed {n}/{len(T)} {time.time() - t0:.0f}s", flush=True)
    if batch:
        for k, x in zip(out, classic.doc_items(batch)):
            out[k].append(x)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["wi", "fce"], nargs="*", default=["wi", "fce"])
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    for corpus in args.corpus:
        t0 = time.time()
        V, T, info = build(corpus)
        info["build_s"] = time.time() - t0
        print(corpus, json.dumps(info, default=str), flush=True)
        V.to_parquet(OUT / f"versions_{corpus}.parquet", index=False)
        T.to_parquet(OUT / f"texts_{corpus}.parquet", index=False)
        items, secs = parse(T, args.procs)
        info["parse_s"] = secs
        info["parse_docs_per_s"] = len(T) / secs
        for k, d in items.items():
            d.to_parquet(OUT / f"items_{k}_{corpus}.parquet", index=False)
        (TAB / f"versions_{corpus}.json").write_text(json.dumps(info, indent=1, default=str))
        print(corpus, "done", f"{secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
