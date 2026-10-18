"""Study 3, stage 9 — gold-standard validation on BEA-2019 W&I+LOCNESS.

W&I (learner essays, CEFR A/B/C) and LOCNESS (50 native student essays, N)
carry expert error edits as character offsets. Each scored token gets a gold
category from the first annotator's edits:

    ERR_TOK  the token overlaps an edited span (substitution or deletion)
    ERR_INS  an insertion point lies between the previous token and this one,
             i.e. a word is missing just before it
    POST     the previous token is ERR_TOK (the token may inherit surprisal)
    OK       none of the above

Subcommands:

    select   gold labels for every scored token, and a lab bench of the tokens
             above the ELLIPSE tau (so lab featurizers run on them unchanged)
             -> data/study3/wi/tokens_gold.parquet
                data/study3/lab/bench_real_wi.parquet (+ obs_pos for repair)
    base     how often surprisal above tau is an error, by CEFR level and split
             -> results/study3/wi_base_rates.csv

The W&I train split is for development; dev (A/B/C) and N are held out.
License: non-commercial research only, excerpts under 100 words, cite
Yannakoudakis et al. (2018), credit CECL for LOCNESS (DECISIONS D13).

    python src/3-vector-transformations/9_wi.py select
    python src/3-vector-transformations/9_wi.py base
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1-predictability-benchmark"))
from util.paths import DATA_DIR, RESULTS_DIR, WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

S3 = DATA_DIR / "study3"
WI = S3 / "wi"
OUT = RESULTS_DIR / "study3"
TAU = json.loads((OUT / "threshold.json").read_text())["tau_bits"]


def gold_for_doc(doc, edits):
    """Gold category per spaCy token index."""
    spans = [(s, e) for s, e, _ in edits if e > s]
    ins = [s for s, e, _ in edits if e == s]
    cat = {}
    prev_end = 0
    err_prev = False
    for t in doc:
        a, b = t.idx, t.idx + len(t.text)
        if any(s < b and e > a for s, e in spans):
            c = "ERR_TOK"
        elif any(prev_end <= p <= a for p in ins):
            c = "ERR_INS"
        elif err_prev:
            c = "POST"
        else:
            c = "OK"
        cat[t.i] = c
        if not t.is_space:
            err_prev = c == "ERR_TOK"
            prev_end = b
    return cat


def cmd_select(args):
    WI.mkdir(parents=True, exist_ok=True)
    toks = pd.concat([pd.read_parquet(p) for p in sorted((S3 / "wi" / "shards").glob("*.parquet"))],
                     ignore_index=True)
    meta, gold = {}, []
    for doc in load_all_docbins(WI_DOCBINS_DIR):
        m = doc.user_data["meta"]
        meta[m["text_id"]] = (m, doc.text)
        for i, c in gold_for_doc(doc, json.loads(m["edits"])).items():
            gold.append((m["text_id"], i, c))
    g = pd.DataFrame(gold, columns=["text_id", "spacy_idx", "gold"])
    toks = toks.merge(g, on=["text_id", "spacy_idx"], how="left")
    toks["level"] = toks.text_id.map(lambda t: meta[t][0]["level"])
    toks["cefr"] = toks.text_id.map(lambda t: meta[t][0]["cefr"])
    toks["split"] = toks.text_id.map(lambda t: meta[t][0]["split"])
    toks["high"] = (toks.surprisal_bits >= TAU) & ~toks.is_space
    toks = toks.sort_values(["text_id", "spacy_idx"]).reset_index(drop=True)
    same = toks.text_id.to_numpy()[1:] == toks.text_id.to_numpy()[:-1]
    toks["next_char"] = np.append(np.where(same, toks.idx.to_numpy()[1:], -1), -1).astype(float)
    toks["next_word"] = np.append(np.where(same, toks.text.to_numpy()[1:], None), None)
    toks.loc[toks.next_char < 0, ["next_char", "next_word"]] = [np.nan, None]
    toks["row_id"] = np.arange(len(toks))
    toks.drop(columns=["obs_id"]).to_parquet(WI / "tokens_gold.parquet", index=False)

    h = toks[toks.high].reset_index(drop=True)
    b = pd.DataFrame({
        "inst": np.arange(len(h)), "row_id": h.row_id, "text_id": h.text_id, "type": "REAL",
        "orig": h.text, "observed": h.text, "group": h.lemma.str.lower(),
        "target_char": h.idx, "next_char": h.next_char, "next_word": h.next_word,
        "text": h.text_id.map(lambda t: meta[t][1]), "matched_to": np.nan, "matched_type": None,
    })
    b.to_parquet(S3 / "lab" / "bench_real_wi.parquet", index=False)
    np.save(S3 / "lab" / "repair" / "obs_pos_real_wi.npy", h.pos.to_numpy().astype(object))
    print(f"{len(toks):,} scored tokens, {len(h):,} above tau")
    print(pd.crosstab(h.level, h.gold, normalize="index").round(3))


def cmd_base(args):
    t = pd.read_parquet(WI / "tokens_gold.parquet")
    t = t[~t.is_space]
    rows = []
    for (lvl, split), d in t.groupby(["level", "split"]):
        h = d[d.high]
        rows.append({"level": lvl, "split": split, "n_tokens": len(d), "n_high": len(h),
                     "rate_high": len(h) / len(d),
                     **{f"share_high_{c}": (h.gold == c).mean() for c in ["ERR_TOK", "ERR_INS", "POST", "OK"]},
                     "share_all_err": d.gold.isin(["ERR_TOK", "ERR_INS"]).mean(),
                     "p_high_given_err_tok": d[d.gold == "ERR_TOK"].high.mean(),
                     "p_high_given_ok": d[d.gold == "OK"].high.mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "wi_base_rates.csv", index=False)
    print(r.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    sub.add_parser("base")
    args = ap.parse_args()
    {"select": cmd_select, "base": cmd_base}[args.cmd](args)


if __name__ == "__main__":
    main()
