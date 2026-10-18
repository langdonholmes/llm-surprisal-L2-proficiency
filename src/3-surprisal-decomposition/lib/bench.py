"""A bench: the high-surprisal tokens of one corpus (or one part of it), with
their base LM pass. Every later step (repair candidates, labels, attestation)
reads and writes caches keyed by the bench name.

Bench names: ``ellipse_dev``, ``ellipse_test``, ``wi``, ``toefl11_p0``,
``toefl11_p1``. Files, under ``data/study3/repair/``:

    bench_<name>.parquet      one row per token: inst, row_id, text_id,
                              observed, target_char, next_char, next_word, text
    base_<name>.npz           target and next-word top-50, surprisal, entropy
    base_meta_<name>.parquet  inst, has_target, has_next, k
"""

import numpy as np
import pandas as pd

from util.paths import DATA_DIR

S3 = DATA_DIR / "study3"
CACHE = S3 / "repair"


def corpus_benches(corpus: str) -> list[str]:
    """Bench names for a corpus, in the order the stage scripts create them."""
    return {"ellipse": ["ellipse_dev", "ellipse_test"], "wi": ["wi"],
            "toefl11": ["toefl11_p0", "toefl11_p1"]}[corpus]


class Bench:
    """Arrays are row-aligned with ``tokens`` (row i == inst i)."""

    def __init__(self, name: str, base: bool = True):
        self.name = name
        self.tokens = pd.read_parquet(CACHE / f"bench_{name}.parquet")
        self.n = len(self.tokens)
        if base:
            self.meta = pd.read_parquet(CACHE / f"base_meta_{name}.parquet")
            self.base = dict(np.load(CACHE / f"base_{name}.npz"))
            # Words that do not start on a subword boundary are unscored, as in Study 1.
            self.ok = self.meta.has_target.to_numpy()

    def path(self, stem: str, ext: str = "parquet"):
        return CACHE / f"{stem}_{self.name}.{ext}"

    def get(self, key: str) -> np.ndarray:
        """e.g. bench.get('t_topk_ids'), bench.get('n_surprisal')."""
        return self.base[key]
