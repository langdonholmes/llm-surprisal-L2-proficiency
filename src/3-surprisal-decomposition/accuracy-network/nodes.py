"""Accuracy network (D54): the correction-rate nodes, and their validity against gold.

A node is an arm's corrections in some ERRANT groups, divided by the essay's
tokens. The four nodes together account for every edit. Determiner/preposition
is its own node, and its corrections are not classified any further.

Writes data/study3/ellipse/accuracy_network/nodes_<arm>.csv (one row per essay,
keyed by text_id_kaggle) and results/study3/ellipse/accuracy_network/validity_<arm>.csv
(on W&I and FCE, the arm's rates against gold cut the same way: Spearman rho, the
ratio of mean rates, and each one's rho with the level).

    python src/3-surprisal-decomposition/accuracy-network/nodes.py --arm corrector
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.corpora import DATA, results_dir  # noqa: E402

NODE_GROUPS = {
    "correction_mechanics": ["spelling/orth", "punctuation"],
    "correction_grammar": ["verb form/SVA/tense", "noun number"],
    "correction_det_prep": ["determiner", "preposition"],
    "correction_lexical": ["other lexical", "word order", "other"],
}
# The units table each arm's corrections are counted in (ELLIPSE, then the validation corpora)
ARMS = {"corrector": {"ellipse": "ellipse", "wi": "wi_model", "fce": "fce_model"}}


def read_units(corpus_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / corpus_id / "units.parquet")


def correction_rates(units: pd.DataFrame) -> pd.DataFrame:
    """Each node's corrections per token, plus the total."""
    rates = pd.DataFrame({node: units[[f"n_{group}" for group in groups]].sum(axis=1) / units.n
                          for node, groups in NODE_GROUPS.items()})
    rates["correction_total"] = rates.sum(axis=1)
    return rates


def validity(arm: str) -> pd.DataFrame:
    rows = []
    for corpus in ["wi", "fce"]:
        arm_units, gold_units = read_units(ARMS[arm][corpus]), read_units(f"{corpus}_errant")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    arm = parser.parse_args().arm
    units = read_units(ARMS[arm]["ellipse"])
    rates = correction_rates(units)
    assert len(rates) == 6482 and (rates.correction_total * units.n).round().eq(units.n_edits).all()  # every edit is in a node
    node_dir = DATA / "ellipse" / "accuracy_network"
    node_dir.mkdir(exist_ok=True)
    rates.rename_axis("text_id_kaggle").reset_index().to_csv(node_dir / f"nodes_{arm}.csv", index=False)
    out = results_dir("ellipse") / "accuracy_network"
    out.mkdir(exist_ok=True)
    table = validity(arm)
    table.to_csv(out / f"validity_{arm}.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
