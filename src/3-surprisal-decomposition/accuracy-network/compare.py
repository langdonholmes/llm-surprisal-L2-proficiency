"""Accuracy network (D54): an arm's readouts against the approved 15-node network.

Communities are compared by membership, because their numbers are arbitrary.
Writes, in results/study3/ellipse/accuracy_network/<arm>/:
    compare_membership.csv  each node's community members in both networks
    compare_edges.csv       every edge among the approved nodes in both networks, with CIs
    accuracy_edges.csv      every edge touching an accuracy node

    python src/3-surprisal-decomposition/accuracy-network/compare.py --arm corrector
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.corpora import results_dir  # noqa: E402


def community_partners(membership: pd.DataFrame) -> pd.Series:
    """Per node, the sorted labels of the other nodes in its community."""
    return membership.set_index("node").apply(
        lambda row: ", ".join(sorted(membership.label[(membership.community == row.community) & (membership.label != row.label)])),
        axis=1)


def undirected(edges: pd.DataFrame) -> pd.DataFrame:
    """Edges keyed by the sorted pair of nodes, so that two runs line up."""
    pair = edges[["node1", "node2"]].apply(sorted, axis=1, result_type="expand")
    return edges.assign(node1=pair[0], node2=pair[1]).set_index(["node1", "node2"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    arm = parser.parse_args().arm
    base = results_dir("ellipse") / "accuracy_network"
    approved_dir, arm_dir = base / "approved", base / arm

    approved_members = pd.read_csv(approved_dir / "membership.csv")
    arm_members = pd.read_csv(arm_dir / "membership.csv")
    compared = arm_members.set_index("node")[["label", "construct", "community", "item_stability"]].join(
        approved_members.set_index("node")[["community", "item_stability"]], rsuffix="_approved")
    compared["partners"] = community_partners(arm_members)
    compared["partners_approved"] = community_partners(approved_members)
    compared.to_csv(arm_dir / "compare_membership.csv")

    columns = ["sample", "q2.5", "q97.5", "prop0"]
    approved_edges = undirected(pd.read_csv(approved_dir / "edges.csv"))[columns]
    arm_edges = undirected(pd.read_csv(arm_dir / "edges.csv"))[columns]
    arm_edges.join(approved_edges, rsuffix="_approved", how="inner").reset_index().to_csv(
        arm_dir / "compare_edges.csv", index=False)
    arm_edges[~arm_edges.index.isin(approved_edges.index)].reset_index().to_csv(arm_dir / "accuracy_edges.csv", index=False)


if __name__ == "__main__":
    main()
