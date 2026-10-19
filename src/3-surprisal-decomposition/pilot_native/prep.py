"""Pilot (native calibration), prep — sample and length-normalize the external documents.

    python -I prep.py PERSUADE_CSV WIKI_DIR OUT.parquet

Groups (N_PER_GROUP each, seed 42):
    persuade_native     PERSUADE 2.0 train, ell_status == "No", independent task
    persuade_native_hi  same, holistic score >= 5 (of 6)
    persuade_ell        PERSUADE 2.0 train, ell_status == "Yes", independent task
    wiki_old            Wikipedia, created 2018-21, text as of 2022-12-31
    wiki_new            Wikipedia, created Jan-Sep 2026, current text

Length rule (all groups): keep documents with >= MIN_WORDS whitespace words;
truncate longer than MAX_WORDS at the last sentence end (., !, ? followed by
whitespace) that keeps <= MAX_WORDS words (hard cut if there is none).
PERSUADE text: CRLF -> LF only, otherwise as written.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

SEED = 42
N_PER_GROUP = 300
MIN_WORDS, MAX_WORDS = 200, 600
SENT_END = re.compile(r"[.!?][\"')\]]*(?=\s)")


def truncate(text: str) -> str:
    text = text.strip()
    words = list(re.finditer(r"\S+", text))
    if len(words) <= MAX_WORDS:
        return text
    limit = words[MAX_WORDS - 1].end()
    ends = [m.end() for m in SENT_END.finditer(text, 0, limit + 1) if m.end() <= limit]
    cut = ends[-1] if ends and len(text[:ends[-1]].split()) >= MIN_WORDS else limit
    return text[:cut]


def finish(df: pd.DataFrame, group: str) -> pd.DataFrame:
    df = df.copy()
    df["n_words_raw"] = df.text.str.split().str.len()
    df = df[df.n_words_raw >= MIN_WORDS]
    df = df.sample(n=min(N_PER_GROUP, len(df)), random_state=SEED)
    df["text"] = df.text.map(truncate)
    df["n_words"] = df.text.str.split().str.len()
    df["truncated"] = df.n_words < df.n_words_raw
    df["group"] = group
    return df


def main():
    persuade_csv, wiki_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    p = pd.read_csv(persuade_csv, usecols=["essay_id_comp", "full_text", "holistic_essay_score",
                                           "prompt_name", "task", "grade_level", "ell_status"])
    p = p.drop_duplicates("essay_id_comp")
    p = p[p.task == "Independent"].rename(columns={"essay_id_comp": "text_id", "full_text": "text",
                                                    "prompt_name": "prompt",
                                                    "holistic_essay_score": "score"})
    p["text"] = p.text.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
    frames = [
        finish(p[p.ell_status == "No"], "persuade_native"),
        finish(p[(p.ell_status == "No") & (p.score >= 5)], "persuade_native_hi"),
        finish(p[p.ell_status == "Yes"], "persuade_ell"),
    ]
    for name in ["wiki_old", "wiki_new"]:
        if not (wiki_dir / f"{name}.jsonl").exists():
            print("missing", name)
            continue
        rows = [json.loads(line) for line in (wiki_dir / f"{name}.jsonl").read_text().splitlines()]
        w = pd.DataFrame(rows)
        w["text_id"] = name + "-" + w.pageid.astype(str)
        w["prompt"] = w.title
        frames.append(finish(w, name))
    cols = ["group", "text_id", "prompt", "score", "grade_level", "created", "rev_timestamp",
            "n_words_raw", "n_words", "truncated", "text"]
    df = pd.concat(frames, ignore_index=True).reindex(columns=cols)
    df.to_parquet(out, index=False)
    print(df.groupby("group").agg(n=("text_id", "size"), words=("n_words", "median"),
                                  trunc=("truncated", "mean")))


if __name__ == "__main__":
    main()
