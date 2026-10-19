"""Native-vs-C pilot, step 3 (CPU) — what the residual N excess is, plus post hoc checks.

Pre-declared (descriptive): among `own` tokens kept under C5, the rate of high
tokens per 1,000 kept tokens by POS and by Dolma frequency band, N vs C, and a
KWIC sample (40 N, 40 C residual high tokens, seed 42) written to the data dir
only (W&I licence: excerpts well under 100 words, never in NOTES.md).

Post hoc: N/C ratios for the main contrasts, and C1 vs C2 CEFR sublevels.

    python src/3-surprisal-decomposition/pilot_native_c/3_describe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from util.paths import DATA_DIR  # noqa: E402

c2 = __import__("2_contrasts")
OUT, TAB, T = c2.OUT, c2.TAB, c2.tau()
WI_JSON = DATA_DIR / "wi_locness" / "raw" / "wi+locness" / "json"
BANDS = [-1, 1, 10, 100, np.inf]
BAND_LABELS = ["<1", "1-10", "10-100", ">=100"]
POS_GROUPS = {"NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJ", "ADV": "ADV", "AUX": "function", "ADP": "function",
              "DET": "function", "PRON": "function", "CCONJ": "function", "SCONJ": "function", "PART": "function",
              "PUNCT": "PUNCT"}


def cat_rates(k: pd.DataFrame, col: str) -> pd.DataFrame:
    """High tokens of each category per 1,000 kept tokens (pooled within group), N vs C."""
    rows = []
    for lv in ["C", "N"]:
        d = k[k.level == lv]
        n = len(d)
        h = d[d.high]
        rows.append(pd.DataFrame({"level": lv, col: h[col].value_counts().index,
                                  "per_1000_kept": h[col].value_counts().to_numpy() / n * 1000,
                                  "share_of_high": h[col].value_counts(normalize=True).to_numpy(),
                                  "share_of_kept": d[col].value_counts(normalize=True).reindex(h[col].value_counts().index).to_numpy(),
                                  "p_high": (h[col].value_counts() / d[col].value_counts()).reindex(h[col].value_counts().index).to_numpy()}))
    R = pd.concat(rows).pivot(index=col, columns="level")
    R.columns = [f"{a}_{b}" for a, b in R.columns]
    R["excess_N_minus_C"] = R.per_1000_kept_N - R.per_1000_kept_C
    return R.sort_values("excess_N_minus_C", ascending=False)


def kwic(text: str, idx: int, tok: str, left=12, right=6) -> str:
    lw = text[:idx].split()[-left:]
    rw = text[idx + len(tok):].split()[:right]
    return (" ".join(lw) + " [[" + tok + "]] " + " ".join(rw)).replace("\n", " ")


def main():
    tok = pd.read_parquet(OUT / "tokens.parquet")
    ess = pd.read_parquet(OUT / "essays.parquet").set_index("text_id")
    t = tok[tok.own & tok.level.isin(["C", "N"])].copy()
    excl = c2.controls(t)["C5"]
    k = t[~excl].copy()
    k["high"] = k.surprisal_bits >= T
    k["pos_g"] = k.pos.map(POS_GROUPS).fillna("other")
    k["band"] = pd.cut(k.freq_pm, BANDS, labels=BAND_LABELS, right=False).astype(str)
    pos = cat_rates(k, "pos_g")
    band = cat_rates(k, "band")
    pos.to_csv(TAB / "residual_pos.csv")
    band.to_csv(TAB / "residual_freqband.csv")
    pd.set_option("display.width", 200)
    print(pos.round(3).to_string())
    print(band.round(3).to_string())
    # medians among residual high tokens
    print(k[k.high].groupby("level").freq_pm.describe(percentiles=[.25, .5, .75]).round(1).to_string())
    # within-band: is N's excess a frequency effect? share of kept tokens in each band
    print(k.groupby("level").band.value_counts(normalize=True).unstack().round(3).to_string())

    # KWIC sample (local file only)
    texts = pd.read_parquet(DATA_DIR / "study3" / "pilot_corrected" / "essays.parquet",
                            columns=["text_id", "corrected"]).set_index("text_id").corrected
    rng = np.random.default_rng(42)
    with open(OUT / "kwic_residual.tsv", "w") as f:
        f.write("id\tlevel\ttext_id\ttoken\tpos\tfreq_pm\tbits\tcontext\n")
        for lv in ["N", "C"]:
            h = k[(k.level == lv) & k.high]
            s = h.iloc[rng.choice(len(h), 40, replace=False)]
            for j, r in enumerate(s.itertuples()):
                f.write(f"{lv}{j:02d}\t{lv}\t{r.text_id}\t{r.text}\t{r.pos}\t{r.freq_pm:.1f}\t{r.surprisal_bits:.1f}\t"
                        f"{kwic(texts[r.text_id], r.idx, r.text)}\n")

    # ---- post hoc -----------------------------------------------------------
    R = pd.read_csv(TAB / "contrasts.csv")
    R["ratio_N_over_C"] = R.N / R.C
    R.to_csv(TAB / "contrasts.csv", index=False)
    cefr = {}
    for p in WI_JSON.glob("C.*.json"):
        for line in open(p):
            r = json.loads(line)
            cefr[r["id"]] = r["cefr"][:2]
    t2 = tok[tok.own & (tok.level == "C")]
    for cname in ["C0", "C5"]:
        E = c2.per_essay(t2, c2.controls(t2)[cname], T)
        E["cefr"] = E.index.map(cefr)
        E["c6c"] = E.index.map(ess.c6_register)
        print(cname, E.groupby("cefr")[["rate", "mean_s"]].agg(["mean", "size"]).round(2).to_string())
        print(cname, "register-matched share by sublevel", E.groupby("cefr").c6c.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
