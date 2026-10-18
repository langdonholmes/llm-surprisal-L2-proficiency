"""Study 3 appendix, stage A1 — delta vectors for the selected tokens.

The proposal's representation. For every token in tokens_selected.parquet
(stage 2), in row_id order:

    p~     = the top-K (K = 50) predicted probabilities renormalized to sum to one
    e_pred = sum_i p~_i W_U[i];   e_obs = W_U[obs first subword]
    delta  = e_pred - e_obs

W_U is OLMo-2's output (unembedding) matrix, which is untied from W_E. Only
delta is stored, since e_obs is a row lookup and e_pred = delta + e_obs.

    data/study3/delta.npy    [n_selected, 2048] float32, row i == row_id i
    data/study3/W_U.npy      the output matrix (A2 needs e_obs = W_U[obs_id])

    python src/3-surprisal-decomposition/appendix/A1_deltas.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.bench import S3  # noqa: E402
from util.paths import RESULTS_DIR  # noqa: E402

CHUNK = 4096


def selected_topk() -> tuple[np.ndarray, np.ndarray]:
    """Top-K ids and probabilities of the selected tokens, in row_id order
    (the same corpora, order and mask as 2_select.py)."""
    tau = json.loads((RESULTS_DIR / "study3" / "threshold.json").read_text())["tau_bits"]
    ids, probs = [], []
    for corpus in ["ellipse", "toefl11"]:
        d = S3 / corpus / "shards"
        if not (S3 / corpus / "manifest.json").exists():
            continue
        for s in sorted(p.with_suffix("") for p in d.glob("shard_*.npz")):
            t = pd.read_parquet(s.with_suffix(".parquet"), columns=["surprisal_bits", "is_space"])
            keep = ((t.surprisal_bits >= tau) & ~t.is_space).to_numpy()
            z = np.load(s.with_suffix(".npz"))
            ids.append(z["topk_ids"][keep])
            probs.append(z["topk_probs"][keep])
    return np.concatenate(ids), np.concatenate(probs).astype(np.float64)


def main():
    sel = pd.read_parquet(S3 / "tokens_selected.parquet", columns=["row_id", "obs_id"])
    top_ids, top_p = selected_topk()
    assert len(top_ids) == len(sel) and sel.row_id.is_monotonic_increasing

    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-2-0425-1B", revision="a1847dff35000b4271fa70afc5db10fd29fedbdf",
        torch_dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach()
    assert W_U.data_ptr() != model.get_input_embeddings().weight.data_ptr()  # untied
    np.save(S3 / "W_U.npy", W_U.numpy())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W_U = W_U.to(dev)
    del model

    obs = sel.obs_id.to_numpy()
    delta = np.lib.format.open_memmap(S3 / "delta.npy", mode="w+", dtype=np.float32,
                                      shape=(len(sel), W_U.shape[1]))
    p_tilde = top_p / top_p.sum(1, keepdims=True)
    for a in range(0, len(sel), CHUNK):
        b = min(a + CHUNK, len(sel))
        ids_t = torch.from_numpy(top_ids[a:b].astype(np.int64)).to(dev)
        p_t = torch.from_numpy(p_tilde[a:b]).to(dev, torch.float32)
        e_pred = torch.einsum("nk,nkd->nd", p_t, W_U[ids_t])
        e_obs = W_U[torch.from_numpy(obs[a:b].astype(np.int64)).to(dev)]
        delta[a:b] = (e_pred - e_obs).cpu().numpy()
    delta.flush()
    print("delta", delta.shape)


if __name__ == "__main__":
    main()
