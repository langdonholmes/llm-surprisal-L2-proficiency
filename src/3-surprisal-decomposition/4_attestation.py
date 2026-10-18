"""Study 3, stage 4 — attestation of every high-surprisal token in Dolma.

For each token above tau: is the word attested at all, and are its bigrams with
the previous and the next word attested, in the Study 2 reference corpora
(a = edited prose, b = unedited prose); plus its rarity and part of speech.
CPU, about 20 min per million tokens. See lib/attestation.py.

    -> data/study3/repair/attestation_<bench>.parquet

    python src/3-surprisal-decomposition/4_attestation.py wi
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.attestation import build  # noqa: E402
from lib.bench import Bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benches", nargs="+")
    for name in ap.parse_args().benches:
        build(Bench(name, base=False))


if __name__ == "__main__":
    main()
