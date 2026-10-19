"""Pilot step 7 (CPU, post hoc) — the edit-level Shapley split of Delta.

Reads edit_shapley.parquet (step 6). Reports checks, how often the split is
trivial (one edit in the window), Shapley vs the two leave-one-out variants in
multi-edit windows, the spillover profile by ERRANT group and distance, and the
level question (spill per error) on the same footing as Q3/Q4 of step 4.

    python src/3-surprisal-decomposition/pilot_shapley/7_edit_analyze.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import GROUPS, LEVELS, OUT, PC, TAB  # noqa: E402

spec = importlib.util.spec_from_file_location("an", HERE / "4_analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def main():
    res = {}
    x = pd.read_parquet(OUT / "edit_shapley.parquet")
    s = pd.read_parquet(OUT / "sample.parquet")
    main_ = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("shapley_main_*.parquet"))])
    x = x.merge(s[["text_id", "spacy_idx", "level", "surprisal_bits", "s_corr", "delta", "n_edited",
                   "player_edited", "n_players"]], on=["text_id", "spacy_idx"])
    x = x.merge(main_[["text_id", "spacy_idx", "v_full", "phi"]], on=["text_id", "spacy_idx"])
    n_cand = int((s.set == "cand").sum())
    ok = x.ok_none & x.ok_all
    eff = np.array([p.sum() - (a - b) for p, a, b in zip(x.psi, x.v_none, x.v_all)])
    res["checks"] = {
        "n_cand": n_cand, "n_scored": len(x), "share_windows_match": float(ok.mean()),
        "ok_none": float(x.ok_none.mean()), "ok_all": float(x.ok_all.mean()),
        "efficiency_max_abs": float(np.abs(eff).max()),
        "v_none_vs_main_vfull_max": float((x.v_none - x.v_full)[x.ok_none].abs().max()),
        "v_all_vs_s_corr_median": float((x.v_all - x.s_corr)[x.ok_all].abs().median()),
        "v_all_vs_s_corr_p99": float((x.v_all - x.s_corr)[x.ok_all].abs().quantile(.99)),
        "sum_psi_vs_delta_pearson": float(np.corrcoef([p.sum() for p in x.psi[ok]], x.delta[ok])[0, 1]),
    }
    x = x[ok].reset_index(drop=True)
    res["n_edits_dist"] = x.n_edits.value_counts(normalize=True).sort_index().round(4).to_dict()
    res["share_single_edit"] = float((x.n_edits == 1).mean())
    # share of |Delta| in multi-edit windows
    res["share_abs_delta_multi"] = float(x.delta.abs()[x.n_edits > 1].sum() / x.delta.abs().sum())
    res["share_delta_multi"] = float(x.delta[x.n_edits > 1].sum() / x.delta.sum())

    # ---- multi-edit windows: Shapley vs LOO -----------------------------------
    m = x[x.n_edits > 1]
    P = np.concatenate(m.psi.to_numpy())
    F = np.concatenate(m.loo_fix.to_numpy())
    K = np.concatenate(m.loo_keep.to_numpy())
    inter = np.array([d - f.sum() for d, f in zip((m.v_none - m.v_all), m.loo_fix)])
    res["multi"] = {
        "n_targets": len(m),
        "pearson_psi_loo_fix": float(np.corrcoef(P, F)[0, 1]),
        "pearson_psi_loo_keep": float(np.corrcoef(P, K)[0, 1]),
        "spearman_psi_loo_fix": float(stats.spearmanr(P, F)[0]),
        "top_edit_same_fix": float(np.mean([np.argmax(a) == np.argmax(b) for a, b in zip(m.psi, m.loo_fix)])),
        "top_edit_same_keep": float(np.mean([np.argmax(a) == np.argmax(b) for a, b in zip(m.psi, m.loo_keep)])),
        "mean_abs_interaction_fix": float(np.abs(inter).mean()),
        "mean_abs_delta": float((m.v_none - m.v_all).abs().mean()),
        "share_abs_delta_not_split_by_loo_fix": float(np.abs(inter).sum() / (m.v_none - m.v_all).abs().sum()),
    }

    # ---- pairs: profile by group and distance -----------------------------------
    rows = []
    for r in x.itertuples():
        for k, g, kd, d, p, f in zip(r.edit_k, r.groups, r.kinds, r.dist, r.psi, r.loo_fix):
            rows.append({"text_id": r.text_id, "level": r.level, "err": f"{r.text_id}:{k}", "group": g,
                         "kind": kd, "d": min(int(d), 5), "psi": p, "loo_fix": f, "one": 1.0})
    p = pd.DataFrame(rows)
    sampled = s[s.set != "ctrl"][["text_id", "level"]].drop_duplicates("text_id")
    boot = an.Boot(sampled)
    prof = []
    for g in GROUPS + ["all"]:
        gg = p if g == "all" else p[p.group == g]
        for d in range(1, 6):
            y = gg[gg.d == d]
            if len(y) < 30:
                continue
            mm = boot.ratio(y, "psi", "one")
            prof.append({"group": g, "d": d, "n": len(y), "mean_psi": mm[0], "lo": mm[1][0], "hi": mm[1][1]})
    prof = pd.DataFrame(prof)
    prof.to_csv(TAB / "edit_profile.csv", index=False)
    pe = p.groupby(["err", "text_id", "level", "group", "kind"]).agg(spill=("psi", "sum"),
                                                                     n_targets=("psi", "size")).reset_index()
    pe["one"] = 1.0
    per = []
    for g in GROUPS + ["all"]:
        y = pe if g == "all" else pe[pe.group == g]
        if len(y) < 20:
            continue
        mm = boot.ratio(y, "spill", "one")
        per.append({"group": g, "n_errors": len(y), "spill_per_error": mm[0], "lo": mm[1][0], "hi": mm[1][1],
                    "share_of_total": y.spill.sum() / pe.spill.sum(), "share_of_errors": len(y) / len(pe)})
    per = pd.DataFrame(per)
    per.to_csv(TAB / "edit_per_error.csv", index=False)
    kind = pe.groupby("kind").spill.agg(["mean", "size"])
    res["per_error_by_kind"] = kind.round(4).reset_index().to_dict("records")
    reach = {}
    for g in GROUPS + ["all"]:
        y = prof[(prof.group == g) & (prof.mean_psi >= .1) & (prof.lo > 0)]
        reach[g] = int(y.d.max()) if len(y) else 0
    res["reach"] = reach
    # compare to the removal-Shapley per-error ranking
    rem = pd.read_csv(TAB / "q3_per_error_shapley.csv").set_index("group")
    common_g = [g for g in GROUPS if g in rem.index and g in set(per.group)]
    pr = per.set_index("group")
    res["rank_rho_edit_vs_removal_spill"] = float(stats.spearmanr(pr.loc[common_g, "spill_per_error"],
                                                                  rem.loc[common_g, "spill_per_error"])[0])

    # ---- level: spill per error, shift-share -------------------------------------------
    al = pd.read_parquet(PC / "aligned.parquet", columns=["text_id", "is_space"])
    ntok = al[~al.is_space].groupby("text_id").size()
    e = pd.DataFrame({"n": ntok[sampled.text_id].to_numpy(), "level": sampled.level.to_numpy()},
                     index=sampled.text_id.to_numpy())
    e["val"] = p.groupby("text_id").psi.sum()
    e["n_err"] = pe.groupby("text_id").size()
    e = e.fillna({"val": 0.0, "n_err": 0.0})
    e.index.name = "text_id"
    res["level"] = an.q4_stats(e)
    sp = pe[pe.group.isin(GROUPS)].groupby(["level", "group"]).spill.mean().unstack()
    w = pe[pe.group.isin(GROUPS)].groupby(["level", "group"]).size().unstack()
    w = w.div(w.sum(1), axis=0)
    res["level_mix"] = {"spill_A": float((sp.loc["A"] * w.loc["A"]).sum()),
                        "spill_C": float((sp.loc["C"] * w.loc["C"]).sum()),
                        "spill_C_with_A_mix": float((sp.loc["C"] * w.loc["A"]).sum()),
                        "spill_A_with_C_mix": float((sp.loc["A"] * w.loc["C"]).sum())}
    sp.T.to_csv(TAB / "edit_spill_by_group_level.csv")

    (TAB / "edit_results.json").write_text(json.dumps(res, indent=1, default=an.json_default))
    print(json.dumps(res, indent=1, default=an.json_default))
    print(per.round(3).to_string())
    print(prof.pivot(index="group", columns="d", values="mean_psi").round(3).to_string())


if __name__ == "__main__":
    main()
