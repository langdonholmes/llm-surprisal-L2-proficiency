"""Study 3 appendix, stage A4 — describe each NMF component (dev rows only).

For every component of a fitted representation, writes one summary row and one
human-readable card. The cards are what the components are labelled from, so
they use dev essays only and show no test-set or outcome information beyond
the dev tertile mix.

Per component:
    share of total loading, mean surprisal of its loaders (loading-weighted),
    top observed words and top predicted words among its 200 highest-loading
    dev tokens, POS mix, lexical concentration (share of those 200 tokens that
    are the single most common lowercased observed lemma, and the number of
    lemmas needed to cover half of them), and the loading-weighted Overall
    tertile mix compared with the base rate of all dev high-surprisal tokens.

    results/study3/components_<rep>.csv
    results/study3/components_<rep>.md     cards with KWIC lines

    python src/3-surprisal-decomposition/appendix/A4_components.py --rep delta
"""

import argparse

import numpy as np
import pandas as pd

from util.paths import DATA_DIR, RESULTS_DIR

S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"
TOP_N = 200
N_KWIC_TOP = 12
N_KWIC_RANDOM = 12
CTX = 55


def kwic(text: str, idx: int, word: str) -> str:
    left = text[max(0, idx - CTX):idx].replace("\n", " ")
    right = text[idx + len(word):idx + len(word) + CTX].replace("\n", " ")
    return f"{left:>{CTX}} [[{word}]] {right}"


def top_counts(s: pd.Series, n: int = 8) -> str:
    vc = s.value_counts()
    return ", ".join(f"{k} ({v})" for k, v in vc.head(n).items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="delta")
    args = ap.parse_args()

    sel = pd.read_parquet(S3 / "tokens_selected.parquet")
    W = np.load(S3 / "nmf" / f"W_{args.rep}.npy")
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet")
    texts = dict(zip(essays.text_id, essays.text))
    split = pd.read_csv(DATA_DIR.parent / "src/3-surprisal-decomposition/split.csv")
    dev_e = essays.merge(split).query("split == 'dev'")
    cuts = dev_e.overall.quantile([1 / 3, 2 / 3]).to_list()
    band_of = dict(zip(dev_e.text_id, pd.cut(dev_e.overall, [-np.inf, *cuts, np.inf],
                                                right=False, labels=["low", "mid", "high"])))

    dev_mask = (sel.split == "dev").to_numpy()
    dev = sel[dev_mask].reset_index(drop=True)
    Wd = W[dev_mask]
    dev["band"] = dev.text_id.map(band_of)
    dev["obs_lc"] = dev.text.str.lower()
    dev["lemma_lc"] = dev.lemma.str.lower()
    base = dev.band.value_counts(normalize=True)
    rng = np.random.default_rng(42)

    rows, cards = [], []
    total = Wd.sum()
    for j in range(W.shape[1]):
        w = Wd[:, j]
        top = np.argsort(-w)[:TOP_N]
        t = dev.iloc[top]
        lem = t.lemma_lc.value_counts()
        cover = int(np.searchsorted(lem.cumsum().to_numpy(), TOP_N / 2) + 1)
        mix = pd.Series(w).groupby(dev.band.to_numpy()).sum() / w.sum()
        row = {
            "component": j,
            "loading_share": float(w.sum() / total),
            "n_nonzero_dev": int((w > 0).sum()),
            "mean_surprisal_w": float((w * dev.surprisal_bits).sum() / w.sum()),
            "top_lemma": lem.index[0],
            "top_lemma_share": float(lem.iloc[0] / TOP_N),
            "lemmas_to_cover_half": cover,
            "top_observed": top_counts(t.obs_lc),
            "top_predicted": top_counts(t.pred1.str.replace("Ġ", "", regex=False)),
            "top_pos": top_counts(t.pos, 4),
            "share_punct": float(t.is_punct.mean()),
            **{f"mix_{b}": float(mix.get(b, 0)) for b in ["low", "mid", "high"]},
            **{f"mix_{b}_minus_base": float(mix.get(b, 0) - base[b]) for b in ["low", "mid", "high"]},
        }
        rows.append(row)

        lines = [f"## Component {j}",
                 f"loading share {row['loading_share']:.3f} | weighted surprisal "
                 f"{row['mean_surprisal_w']:.1f} bits | top-lemma share "
                 f"{row['top_lemma_share']:.2f} ({row['top_lemma']}) | lemmas to cover half "
                 f"{cover} | tertile mix L/M/H {row['mix_low']:.2f}/{row['mix_mid']:.2f}/"
                 f"{row['mix_high']:.2f} (base {base['low']:.2f}/{base['mid']:.2f}/{base['high']:.2f})",
                 f"- observed: {row['top_observed']}",
                 f"- predicted: {row['top_predicted']}",
                 f"- POS: {row['top_pos']}", "", "```"]
        picks = list(top[:N_KWIC_TOP]) + list(rng.choice(top[N_KWIC_TOP:], N_KWIC_RANDOM, replace=False))
        for i in picks:
            r = dev.iloc[i]
            preds = "/".join(str(r[f"pred{k}"]).replace("Ġ", "_") for k in (1, 2, 3))
            lines.append(f"{kwic(texts[r.text_id], int(r.idx), r.text)}   <- {preds}")
        lines += ["```", ""]
        cards.append("\n".join(lines))

    pd.DataFrame(rows).to_csv(OUT / f"components_{args.rep}.csv", index=False)
    (OUT / f"components_{args.rep}.md").write_text(
        f"# Component cards: {args.rep}\n\nDev ELLIPSE rows only. First {N_KWIC_TOP} lines "
        f"are the highest loaders; the rest are random draws from the top {TOP_N}. "
        f"Predictions after the arrow are the model's top three (_ = leading space).\n\n"
        + "\n".join(cards))


if __name__ == "__main__":
    main()
