"""Step 1 (CPU) — an ERRANT type for every FCE json edit.

``pilot_shapley/1_errant.py`` applied to BEA-2019 FCE v2.1 unchanged: the M2
files were made from the json files by the same ``json_to_m2.py`` (the two
copies are identical), so the same walk-back of M2 tokens onto the json
characters and the same matching rule give each first-annotator json edit the
type of the M2 edit it overlaps. text_id is "<script id>_<q>", as in
``src/essays/ingest.py --fce``.

Output: data/study3/pilot_types/edit_types_fce.parquet (same columns as
pilot_shapley/edit_types.parquet) and tables/errant_coverage_fce.csv.

    python src/3-surprisal-decomposition/pilot_types/1_errant_fce.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from types_common import FCE_RAW, OUT, SH, TAB, group_of, load  # noqa: E402

er = load("sh_errant", SH / "1_errant.py")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rows, essay_ok = [], []
    for split in ["train", "dev", "test"]:
        sents = er.read_m2(FCE_RAW / "m2" / f"fce.{split}.gold.bea19.m2")
        for line in open(FCE_RAW / "json" / f"fce.{split}.json"):
            d = json.loads(line)
            text = d["text"]
            m2, ok = er.essay_m2_spans(text, sents)
            essay_ok.append(ok)
            edits = [x for c, x in d["edits"] if c == 0][0]
            for k, ed in enumerate(edits):
                s, e, r = ed[0], ed[1], ed[2]
                if r is None:
                    typ, n = "flagged", 0
                else:
                    h = er.match(ed, m2, text) if ok else []
                    typ, n = (h[0] if h else "unmapped"), len(h)
                kind = "flagged" if r is None else ("ins" if s == e else ("del" if r == "" else "sub"))
                rows.append({"text_id": f"{d['id']}_{d['q']}", "file": split, "k": k, "s": s, "e": e,
                             "repl": r, "kind": kind, "errant": typ, "n_m2": n, "fce_code": ed[3]})
        rest = sum(1 for _ in sents)
        print(f"{split}: M2 sentences left over {rest}", flush=True)
    df = pd.DataFrame(rows)
    df["group"] = [("flagged" if t == "flagged" else group_of(t)) for t in df.errant]
    df.to_parquet(OUT / "edit_types_fce.parquet", index=False)
    c = df[df.kind != "flagged"]
    cov = pd.DataFrame({
        "n_edits": c.groupby("kind").size(),
        "mapped": c.groupby("kind").apply(lambda x: (x.errant != "unmapped").mean()),
        "multi_m2": c.groupby("kind").apply(lambda x: (x.n_m2 > 1).mean())})
    cov.loc["all"] = [len(c), (c.errant != "unmapped").mean(), (c.n_m2 > 1).mean()]
    cov.to_csv(TAB / "errant_coverage_fce.csv")
    print(f"answers aligned to M2 without a token mismatch: {sum(essay_ok)}/{len(essay_ok)}")
    print(cov.round(4).to_string())
    print(c.group.value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
