"""Study 3, stage 9 — where native and published writing fall on mean surprisal (D26).

Public documents, downloaded once into their own directories under
data/study3/calibration/ (dl_persuade_train/, dl_wiki/):
    persuade_native     PERSUADE 2.0 train, independent task, ell_status "No"
    persuade_native_hi  the same with holistic score >= 5 (of 6)
    persuade_ell        PERSUADE 2.0, ell_status "Yes"
    wiki_old            English Wikipedia articles created 2018-21 (text as of 2022-12-31; in the training data)
    wiki_new            articles created Jan-Sep 2026 (after the model's training data)
compared with W&I A/B/C, LOCNESS and the ELLIPSE Overall bands. Every document
is cut to 200-600 words (fewer than 200 words: dropped; more than 600: cut at
the last sentence end that keeps at most 600 words); 300 per external group,
seed 42. The learner essays are cut the same way on their stored tokens (the
8-subword window means the cut changes which tokens count, not their scores).
Per document: mean surprisal over scored tokens, as in Study 1. Per group: the
mean of the document means with a 2,000-resample bootstrap CI.

    python -I src/3-surprisal-decomposition/9_calibration.py prep      # reads the downloads
    CUDA_VISIBLE_DEVICES=1 python src/3-surprisal-decomposition/9_calibration.py score
    python src/3-surprisal-decomposition/9_calibration.py table

Writes data/study3/calibration/{docs,tokens}.parquet and
results/study3/calibration/calibration.csv and contrasts.csv.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.corpora import DATA, ELLIPSE_APPENDIX, N_BOOT, RESULTS, SEED  # noqa: E402

CAL = DATA / "calibration"
N_PER_GROUP = 300
MIN_WORDS, MAX_WORDS = 200, 600
SENT_END = re.compile(r"[.!?][\"')\]]*(?=\s)")
ORDER = ["W&I A", "W&I B", "W&I C", "LOCNESS", "ELLIPSE <= 2.5", "ELLIPSE 3-3.5", "ELLIPSE >= 4",
         "persuade_ell", "persuade_native", "persuade_native_hi", "wiki_old", "wiki_new"]
CONTRASTS = [("persuade_native", "W&I C"), ("persuade_native", "ELLIPSE >= 4"),
             ("persuade_native_hi", "ELLIPSE >= 4"), ("persuade_native", "persuade_ell"),
             ("LOCNESS", "W&I C"), ("wiki_new", "wiki_old")]


def truncate(text: str) -> str:
    """Cut at the last sentence end within MAX_WORDS words (or at MAX_WORDS, if that leaves under MIN_WORDS)."""
    text = text.strip()
    words = list(re.finditer(r"\S+", text))
    if len(words) <= MAX_WORDS:
        return text
    limit = words[MAX_WORDS - 1].end()
    ends = [m.end() for m in SENT_END.finditer(text, 0, limit + 1) if m.end() <= limit]
    return text[:ends[-1] if ends and len(text[:ends[-1]].split()) >= MIN_WORDS else limit]


def sample(docs: pd.DataFrame, group: str) -> pd.DataFrame:
    docs = docs[docs.text.str.split().str.len() >= MIN_WORDS]
    docs = docs.sample(n=min(N_PER_GROUP, len(docs)), random_state=SEED)
    return docs.assign(text=docs.text.map(truncate), group=group)


def prep():
    persuade = pd.read_csv(CAL / "dl_persuade_train" / "persuade_train.bin",
                           usecols=["essay_id_comp", "full_text", "holistic_essay_score", "task", "ell_status"])
    persuade = persuade.drop_duplicates("essay_id_comp")
    persuade = persuade[persuade.task == "Independent"].rename(
        columns={"essay_id_comp": "text_id", "full_text": "text", "holistic_essay_score": "score"})
    persuade["text"] = persuade.text.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
    native = persuade.ell_status == "No"
    frames = [sample(persuade[native], "persuade_native"),
              sample(persuade[native & (persuade.score >= 5)], "persuade_native_hi"),
              sample(persuade[persuade.ell_status == "Yes"], "persuade_ell")]
    for name in ["wiki_old", "wiki_new"]:
        wiki = pd.DataFrame([json.loads(x) for x in (CAL / "dl_wiki" / f"{name}.jsonl").read_text().splitlines()])
        frames.append(sample(wiki.assign(text_id=name + "-" + wiki.pageid.astype(str)), name))
    docs = pd.concat(frames, ignore_index=True)[["group", "text_id", "text"]]
    docs.to_parquet(CAL / "docs.parquet", index=False)
    print(docs.groupby("group").size().to_string())


def score():
    import spacy
    from lib.lm import instrument
    docs = pd.read_parquet(CAL / "docs.parquet")
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000
    tokens = instrument()(nlp.pipe(docs.text.tolist(), batch_size=64), docs.text_id.tolist())
    tokens.to_parquet(CAL / "tokens.parquet", index=False)
    print("scored", len(docs), "docs,", len(tokens), "tokens")


def learner(essays: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """Per-essay mean over the tokens of the cut text; essays under MIN_WORDS dropped."""
    essays = essays[essays.text.str.split().str.len() >= MIN_WORDS]
    lead = essays.text.str.len() - essays.text.str.lstrip().str.len()  # truncate() strips; offsets do not
    cut = pd.Series((lead + essays.text.map(lambda t: len(truncate(t)))).to_numpy(), index=essays.text_id)
    tokens = tokens[tokens.text_id.isin(cut.index)]
    tokens = tokens[tokens.idx < tokens.text_id.map(cut)]
    return tokens.groupby("text_id").surprisal_bits.mean().rename("mean_s").to_frame().join(
        essays.set_index("text_id").group)


def table():
    docs = pd.read_parquet(CAL / "docs.parquet")
    external = pd.read_parquet(CAL / "tokens.parquet").groupby("text_id").surprisal_bits.mean().rename("mean_s")
    external = external.to_frame().join(docs.set_index("text_id").group)
    wi = pd.read_parquet(DATA / "wi" / "essays.parquet", columns=["text_id", "group", "text"])
    wi["group"] = wi.group.map({"A": "W&I A", "B": "W&I B", "C": "W&I C", "N": "LOCNESS"})
    wi = learner(wi, pd.read_parquet(DATA / "wi" / "tokens_orig.parquet", columns=["text_id", "idx", "surprisal_bits"]))
    ellipse = pd.read_parquet(ELLIPSE_APPENDIX / "essays.parquet", columns=["text_id", "overall", "text"])
    ellipse["group"] = pd.cut(ellipse.overall, [0, 2.5, 3.5, 5], labels=ORDER[4:7]).astype(str)
    ellipse_tokens = pd.concat([pd.read_parquet(p, columns=["text_id", "idx", "surprisal_bits"])
                                for p in sorted((ELLIPSE_APPENDIX / "shards").glob("shard_*.parquet"))])
    all_docs = pd.concat([external, wi, learner(ellipse, ellipse_tokens)])
    rng = np.random.default_rng(SEED)
    rows, boots = [], {}
    for g in ORDER:
        x = all_docs.mean_s[all_docs.group == g].to_numpy()
        boots[g] = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
        rows.append({"group": g, "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
                     "lo": np.percentile(boots[g], 2.5), "hi": np.percentile(boots[g], 97.5)})
    groups = pd.DataFrame(rows)
    (RESULTS / "calibration").mkdir(parents=True, exist_ok=True)
    groups.to_csv(RESULTS / "calibration" / "calibration.csv", index=False)
    means = groups.set_index("group")["mean"]
    contrasts = pd.DataFrame([{"a": a, "b": b, "diff": means[a] - means[b],
                               "lo": np.percentile(boots[a] - boots[b], 2.5),
                               "hi": np.percentile(boots[a] - boots[b], 97.5)} for a, b in CONTRASTS])
    contrasts.to_csv(RESULTS / "calibration" / "contrasts.csv", index=False)
    print(groups.round(3).to_string(index=False))
    print(contrasts.round(3).to_string(index=False))


if __name__ == "__main__":
    {"prep": prep, "score": score, "table": table}[sys.argv[1]]()
