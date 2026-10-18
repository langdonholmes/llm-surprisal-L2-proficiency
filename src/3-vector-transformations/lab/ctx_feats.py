"""Context & capacity features from the ctx_score.py caches (no gold labels).

Every view's surprisal is of the same observed first subword, so a difference
is "how much less surprising the word becomes" under that view:

    gain_ctx   = s(1B, 8)  - s(1B, 64)    more left context
    gain_cap   = s(1B, 8)  - s(7B, 8)     more capacity
    gain_both  = s(1B, 8)  - s(7B, 64)    both
    gain_bidir = s(1B, 8)  - s(MLM)       right context (bidirectional, masked)
"""

import numpy as np

from lab.harness import LAB, SUFFIX

VIEWS = ["olmo1b_w8", "olmo1b_w64", "olmo7b_w8", "olmo7b_w64", "mlm"]


def load(view: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(LAB / "ctx" / f"{view}{SUFFIX}.npz")
    s, e = z["surprisal"].astype(np.float64), z["entropy"].astype(np.float64)
    # unscorable rows: fill with the view's median (a neutral value)
    return np.where(np.isfinite(s), s, np.nanmedian(s)), np.where(np.isfinite(e), e, np.nanmedian(e))


def available() -> list[str]:
    return [v for v in VIEWS if (LAB / "ctx" / f"{v}{SUFFIX}.npz").exists()]


def gains() -> dict[str, np.ndarray]:
    base, _ = load("olmo1b_w8")
    out = {}
    for name, view in [("gain_ctx", "olmo1b_w64"), ("gain_cap", "olmo7b_w8"),
                       ("gain_both", "olmo7b_w64"), ("gain_bidir", "mlm")]:
        if view in available():
            out[name] = base - load(view)[0]
    return out
