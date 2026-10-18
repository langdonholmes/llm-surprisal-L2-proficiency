"""Study 3, stage 7 — TOEFL 11 (RQ5). Out of sample throughout.

TOEFL 11 tokens are thresholded at the ELLIPSE tau and projected onto the
ELLIPSE-dev H (stage 4). Nothing here is fit on TOEFL except the replication NMF.

1. Validity transfer. Per component, Spearman between the essay profile and
   level (low < medium < high), and the same for mean surprisal. Then the
   ELLIPSE-dev elastic nets for B0 and M (stage 6 definitions) are applied
   unchanged to TOEFL, and their predictions are correlated with level.
   Profiles and predictors are centred on TOEFL prompt means for this step,
   since TOEFL prompts are not ELLIPSE prompts.
2. L1. Study 1's fairness procedure, per component: within each level, Cohen's
   d between the focal L1 and all other learners, standardized by the pooled
   within-level SD, averaged across levels weighted by focal-group size.
   |d| >= .20 is flagged.
3. Replication. A separate NMF on TOEFL deltas at the same k, matched to the
   ELLIPSE components by Hungarian assignment on the cosine of H rows.

    results/study3/toefl_validity.csv, toefl_l1_d.csv, toefl_replication.csv,
    toefl_transfer.json
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from util.paths import DATA_DIR, RESULTS_DIR

HERE = Path(__file__).resolve().parent
S3 = DATA_DIR / "study3"
OUT = RESULTS_DIR / "study3"
SEED = 42
D_FLAG = 0.20
LEVELS = {"low": 1, "medium": 2, "high": 3}


def load_stage(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weighted_d(x: np.ndarray, l1: np.ndarray, level: np.ndarray) -> dict:
    out = {}
    for g in np.unique(l1):
        num = wsum = 0.0
        for s in LEVELS:
            foc = (level == s) & (l1 == g)
            rst = (level == s) & (l1 != g)
            n1, n2 = foc.sum(), rst.sum()
            if n1 < 2 or n2 < 2:
                continue
            sp = np.sqrt(((n1 - 1) * x[foc].var(ddof=1) + (n2 - 1) * x[rst].var(ddof=1))
                         / (n1 + n2 - 2))
            num += n1 * (x[foc].mean() - x[rst].mean()) / sp
            wsum += n1
        out[g] = num / wsum
    return out


def main():
    prof = load_stage("6_proficiency")
    nmf = load_stage("4_nmf")

    df, blocks = prof.build_features("toefl11")
    df.to_parquet(S3 / "profiles_toefl11.parquet")
    lvl = df.level.map(LEVELS).to_numpy()
    comp = blocks["delta"]

    # --- 1. validity ------------------------------------------------------------
    rows = []
    for c in ["mean_surprisal", "rate_high"] + comp:
        rho, p = spearmanr(df[c], lvl)
        rows.append({"feature": c, "rho_level": rho, "p": p})
    val = pd.DataFrame(rows)
    val["q"] = prof.bh(val.p.to_numpy())
    val.to_csv(OUT / "toefl_validity.csv", index=False)

    # ELLIPSE-dev models applied to TOEFL
    el = pd.read_parquet(S3 / "profiles_ellipse.parquet")
    dev = (el.split == "dev").to_numpy()
    L = prof.ladder(blocks | {"study2": []})
    transfer = {}
    for name in ["B0", "B1", "M_obs", "M", "R"]:
        cols = L[name]
        Xe = prof.prompt_center(el, cols, dev)[dev]
        ye = prof.prompt_center(el, ["overall"], dev).overall[dev]
        Xt = prof.prompt_center(df, cols, np.ones(len(df), bool))
        pred = prof.fit_predict(Xe, ye.to_numpy(), el.prompt[dev].to_numpy(), Xt)
        transfer[name] = float(spearmanr(pred, lvl)[0])
    print(transfer)

    # --- 2. L1 ------------------------------------------------------------------
    d_rows = []
    for c in ["mean_surprisal", "rate_high"] + comp:
        for g, d in weighted_d(df[c].to_numpy(), df.l1.to_numpy(), df.level.to_numpy()).items():
            d_rows.append({"feature": c, "l1": g, "d": d, "flag": abs(d) >= D_FLAG})
    pd.DataFrame(d_rows).to_csv(OUT / "toefl_l1_d.csv", index=False)

    # --- 3. replication ---------------------------------------------------------
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["row_id", "split"])
    trows = sel.row_id[sel.split == "toefl"].to_numpy()
    H_el = np.load(S3 / "nmf" / "H_delta.npy")
    m = nmf.fit_nmf(nmf.posneg(nmf.load_rep("delta", trows)), H_el.shape[0])
    A = H_el / np.linalg.norm(H_el, axis=1, keepdims=True)
    B = m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True)
    C = A @ B.T
    from scipy.optimize import linear_sum_assignment
    r, cidx = linear_sum_assignment(-C)
    pd.DataFrame({"component": r, "toefl_component": cidx, "cosine": C[r, cidx]}) \
        .to_csv(OUT / "toefl_replication.csv", index=False)

    info = json.loads((S3 / "nmf" / "fit_delta.json").read_text())
    (OUT / "toefl_transfer.json").write_text(json.dumps({
        "spearman_prediction_vs_level": transfer,
        "rel_error_by_split": info["rel_error_by_split_transform"],
        "median_replication_cosine": float(np.median(C[r, cidx])),
        "n_essays": len(df), "n_zero_high": int((df.n_high == 0).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
