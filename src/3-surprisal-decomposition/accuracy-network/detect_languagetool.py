"""Accuracy network, detector arm (D55): LanguageTool on W&I and FCE, for validation only.

Sends every essay to a local LanguageTool server (no text leaves this machine) and
counts its flags per unit, by LanguageTool's own rule category. LanguageTool's
categories do not map onto the four nodes, so only mechanics and the total are
compared with gold.

Start the server first:
    java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081 --allow-origin

Writes data/study3/<corpus>/detector_languagetool.parquet, one row per unit: n_<category>
and n_flagged.

    python src/3-surprisal-decomposition/accuracy-network/detect_languagetool.py --corpus wi fce
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.corpora import DATA, data_dir  # noqa: E402

SERVER = "http://localhost:8081/v2/check"
LANGUAGE = "en-GB"  # assumed for W&I and FCE, not checked; en-US moves ρ by less than .02 on the total and mechanics (D56)


def categories_flagged(text: str) -> Counter:
    """How many flags of each rule category LanguageTool gives one essay."""
    body = urllib.parse.urlencode({"text": text, "language": LANGUAGE}).encode()
    with urllib.request.urlopen(SERVER, data=body, timeout=300) as response:
        return Counter(match["rule"]["category"]["id"] for match in json.load(response)["matches"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    for corpus in args.corpus:
        essays = pd.read_parquet(DATA / corpus / "essays.parquet", columns=["unit", "text"])
        with ThreadPool(args.threads) as pool:
            counts = pool.map(categories_flagged, essays.text, chunksize=4)
        table = pd.DataFrame(counts).fillna(0).astype(int).add_prefix("n_")
        table["n_flagged"] = table.sum(axis=1)
        table = table.groupby(essays.unit.values).sum().rename_axis("unit")  # an FCE script pools its two answers
        table.to_parquet(data_dir(corpus) / "detector_languagetool.parquet")
        print(corpus, len(table), table.sum().to_dict(), flush=True)


if __name__ == "__main__":
    main()
