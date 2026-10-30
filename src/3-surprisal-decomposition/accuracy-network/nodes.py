"""Accuracy network (D54): the correction-rate nodes, and their validity against gold.

A node is an arm's corrections (or, for the detector, flagged tokens) in some
ERRANT groups, divided by the essay's tokens. The four nodes together account for
every edit. Determiner/preposition is its own node, and its corrections are not
classified any further.

Writes data/study3/ellipse/accuracy_network/nodes_<arm>.csv (one row per essay,
keyed by text_id_kaggle) and results/study3/ellipse/accuracy_network/validity_<arm>.csv
(on W&I and FCE, the arm's rates against gold cut the same way: Spearman rho, the
ratio of mean rates, and each one's rho with the level).

    python src/3-surprisal-decomposition/accuracy-network/nodes.py --arm corrector
    python src/3-surprisal-decomposition/accuracy-network/nodes.py --arm electra

The electra arm (D56) also writes validity_electra_by_split.csv, the same agreement
by release split as a check on what the checkpoint saw, and validity_languagetool.csv,
LanguageTool's flags against gold where its categories allow the comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.corpora import DATA, RAW_DIRS, results_dir  # noqa: E402
from lib.errant import FILES  # noqa: E402

NODE_GROUPS = {
    "correction_mechanics": ["spelling/orth", "punctuation"],
    "correction_grammar": ["verb form/SVA/tense", "noun number"],
    "correction_det_prep": ["determiner", "preposition"],
    "correction_lexical": ["other lexical", "word order", "other"],
}
# The units table each arm's corrections are counted in (ELLIPSE, then the validation corpora)
ARMS = {"corrector": {"ellipse": "ellipse", "wi": "wi_model", "fce": "fce_model"},
        "electra": {"ellipse": "ellipse", "wi": "wi", "fce": "fce"}}
# The columns of the node tables: what each arm's rate counts
COLUMN_PREFIX = {"corrector": "correction", "electra": "flag"}
# LanguageTool's rule categories that are mechanics in the sense of the mechanics node
LANGUAGETOOL_MECHANICS = ["n_TYPOS", "n_PUNCTUATION", "n_CASING", "n_TYPOGRAPHY", "n_COMPOUNDING"]


def read_units(corpus_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / corpus_id / "units.parquet")


def read_arm(arm: str, corpus_id: str) -> pd.DataFrame:
    """The arm's counts per unit, with the unit's tokens (n) from the same table for every arm."""
    units = read_units(corpus_id)
    if arm == "corrector":
        return units
    flagged = pd.read_parquet(DATA / corpus_id / "detector_electra.parquet").reindex(units.index)
    assert flagged.n_read.notna().all()
    return flagged.assign(n=units.n, y=units.y)


def correction_rates(units: pd.DataFrame) -> pd.DataFrame:
    """Each node's corrections per token, plus the total."""
    rates = pd.DataFrame({node: units[[f"n_{group}" for group in groups]].sum(axis=1) / units.n
                          for node, groups in NODE_GROUPS.items()})
    rates["correction_total"] = rates.sum(axis=1)
    return rates


def validity(arm: str) -> pd.DataFrame:
    rows = []
    for corpus in ["wi", "fce"]:
        arm_units, gold_units = read_arm(arm, ARMS[arm][corpus]), read_units(f"{corpus}_errant")
        shared = arm_units[arm_units.y.notna()].index.intersection(gold_units.index)  # rated learners only
        arm_rates, gold_rates = correction_rates(arm_units.loc[shared]), correction_rates(gold_units.loc[shared])
        level = arm_units.loc[shared, "y"]
        for node in arm_rates:
            rows.append({"corpus": corpus, "n_units": len(shared), "node": node,
                         "rho_with_gold": spearmanr(arm_rates[node], gold_rates[node])[0],
                         "mean_ratio_to_gold": arm_rates[node].mean() / gold_rates[node].mean(),
                         "rho_level_arm": spearmanr(arm_rates[node], level)[0],
                         "rho_level_gold": spearmanr(gold_rates[node], level)[0]})
    return pd.DataFrame(rows)


def release_split(corpus: str) -> pd.Series:
    """Each unit's split in the BEA-2019 release (train, dev, test)."""
    unit_of_text = pd.read_parquet(DATA / corpus / "essays.parquet", columns=["text_id", "unit"]).set_index("text_id").unit
    split_of_unit = {}
    for json_name, _ in FILES[corpus]:
        split = next(name for name in ["train", "dev", "test"] if name in json_name)
        for line in open(RAW_DIRS[corpus] / json_name):
            record = json.loads(line)
            text_id = record["id"] if corpus == "wi" else f"{record['id']}_{record['q']}"
            if text_id in unit_of_text.index:
                split_of_unit[unit_of_text[text_id]] = split
    return pd.Series(split_of_unit)


def validity_by_split(arm: str) -> pd.DataFrame:
    """Agreement with gold in each release split: a split the model trained on should agree better."""
    rows = []
    for corpus in ["wi", "fce"]:
        arm_units, gold_units = read_arm(arm, ARMS[arm][corpus]), read_units(f"{corpus}_errant")
        shared = arm_units[arm_units.y.notna()].index.intersection(gold_units.index)
        arm_rates, gold_rates = correction_rates(arm_units.loc[shared]), correction_rates(gold_units.loc[shared])
        splits = release_split(corpus).reindex(shared)
        for split, units_in_split in splits.groupby(splits).groups.items():
            for node in arm_rates:
                rows.append({"corpus": corpus, "split": split, "n_units": len(units_in_split), "node": node,
                             "rho_with_gold": spearmanr(arm_rates.loc[units_in_split, node], gold_rates.loc[units_in_split, node])[0],
                             "mean_ratio_to_gold": arm_rates.loc[units_in_split, node].mean() / gold_rates.loc[units_in_split, node].mean()})
    return pd.DataFrame(rows)


def validity_languagetool() -> pd.DataFrame:
    """LanguageTool against gold: its total and mechanics, and its GRAMMAR category against the grammar node."""
    rows = []
    for corpus in ["wi", "fce"]:
        gold_units = read_units(f"{corpus}_errant")
        gold_units = gold_units[gold_units.y.notna()]
        flags = pd.read_parquet(DATA / corpus / "detector_languagetool.parquet").reindex(gold_units.index).fillna(0)
        gold_rates = correction_rates(gold_units)
        compared = {"correction_total": flags.n_flagged, "correction_mechanics": flags[LANGUAGETOOL_MECHANICS].sum(axis=1),
                    "correction_grammar": flags.n_GRAMMAR}
        for node, flagged in compared.items():
            rate = flagged / gold_units.n
            rows.append({"corpus": corpus, "n_units": len(gold_units), "node": node,
                         "rho_with_gold": spearmanr(rate, gold_rates[node])[0],
                         "mean_ratio_to_gold": rate.mean() / gold_rates[node].mean(),
                         "rho_level_arm": spearmanr(rate, gold_units.y)[0],
                         "rho_level_gold": spearmanr(gold_rates[node], gold_units.y)[0]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    arm = parser.parse_args().arm
    units = read_arm(arm, ARMS[arm]["ellipse"])
    rates = correction_rates(units)
    counted = units.n_edits if arm == "corrector" else units.n_flagged
    assert len(rates) == 6482 and (rates.correction_total * units.n).round().eq(counted).all()  # every edit is in a node
    rates.columns = rates.columns.str.replace("correction", COLUMN_PREFIX[arm])
    node_dir = DATA / "ellipse" / "accuracy_network"
    node_dir.mkdir(exist_ok=True)
    rates.rename_axis("text_id_kaggle").reset_index().to_csv(node_dir / f"nodes_{arm}.csv", index=False)
    out = results_dir("ellipse") / "accuracy_network"
    out.mkdir(exist_ok=True)
    table = validity(arm)
    table.to_csv(out / f"validity_{arm}.csv", index=False)
    print(table.round(3).to_string(index=False))
    if arm == "electra":
        validity_by_split(arm).to_csv(out / "validity_electra_by_split.csv", index=False)
        validity_languagetool().to_csv(out / "validity_languagetool.csv", index=False)


if __name__ == "__main__":
    main()
