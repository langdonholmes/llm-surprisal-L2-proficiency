"""FCE step 3 (CPU) — align original and corrected tokens (D22 alignment, unchanged).

Uses ``pilot_corrected/3_analyze.align`` as is. Writes aligned.parquet (one row
per non-space original token: gold, own, aligned, s_corr, same_window) and the
coverage / identical-window checks.

    python src/3-surprisal-decomposition/pilot_fce/3_align.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fce_common import OUT, TAB, pc  # noqa: E402

A = pc("3_analyze")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    essays = pd.read_parquet(OUT / "essays.parquet")
    flagged = pd.read_parquet(OUT / "flagged.parquet")
    gold = pd.read_parquet(OUT / "gold.parquet")
    orig = pd.read_parquet(OUT / "tokens_orig.parquet")
    corr = pd.read_parquet(OUT / "tokens_corr.parquet")

    o = orig.merge(gold, on=["text_id", "spacy_idx"], how="left", validate="one_to_one")
    assert o.gold.notna().all()
    o = o[~o.is_space].copy()
    c = corr[~corr.is_space].copy()
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    a = A.align(essays, flagged, {t: d for t, d in o.groupby("text_id")}, c, tok)
    meta = essays.set_index("text_id")[["script_id", "split", "l1", "script_s", "answer_s"]]
    a = a.join(meta, on="text_id")
    a["delta"] = a.surprisal_bits - a.s_corr
    a.drop(columns=["lemma", "tag", "obs_id"]).to_parquet(OUT / "aligned.parquet", index=False)

    sw = a[a.same_window]
    c_matched = set(zip(a.text_id[a.aligned], a.c_spacy_idx[a.aligned]))
    rep = {
        "n_orig_scored": int(len(a)), "n_corr_scored": int(len(c)),
        "own_share": float(a.own.mean()), "aligned_share": float(a.aligned.mean()),
        "aligned_of_own": float(a.aligned[a.own].mean()), "own_unmatched": int(a.own_unmatched.sum()),
        "own_vs_not_errtok_disagree": int((a.own != (a.gold != "ERR_TOK")).sum()),
        "gold_counts": a.gold.value_counts().to_dict(),
        "corr_matched_share": float(pd.Series([(t, i) in c_matched for t, i in
                                               zip(c.text_id, c.spacy_idx)]).mean()),
        "unmatched_own_top": a[a.own_unmatched].text.value_counts().head(10).to_dict(),
        "same_window": {"n": int(len(sw)), "share_of_aligned": float(a.same_window[a.aligned].mean()),
                        "max_abs_delta": float(sw.delta.abs().max()),
                        "share_exact": float((sw.delta == 0).mean()),
                        "share_abs_lt_0.05": float((sw.delta.abs() < 0.05).mean())},
    }
    (TAB / "check_align.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
