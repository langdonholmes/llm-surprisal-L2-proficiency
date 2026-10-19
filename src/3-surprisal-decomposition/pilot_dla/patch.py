"""Pilot DLA: is the linear attribution causally meaningful? (post hoc check)

Mean-ablates one component at the prediction position (a head's z block before
o_proj, or an MLP's post-norm write) and compares the actual change in
log p(obs) with the DLA prediction, contrib(mean) - contrib(actual), which
freezes every norm scale and ignores downstream layers. Components: the top
suppressors in the main (logit-gap) attribution.

Output: src/3-surprisal-decomposition/pilot_dla/out/patch.csv

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/3-surprisal-decomposition/pilot_dla/patch.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from dla import BATCH, COMPONENTS, HEAD_DIM, OUT, WINDOW, DLA  # noqa: E402
from lib.bench import CACHE  # noqa: E402

TARGETS = ["L15H10", "L14H5", "L14MLP", "L13MLP", "L12MLP"]


def main():
    s = pd.read_parquet(OUT / "sample_stats.parquet")
    bench = pd.read_parquet(CACHE / "bench_wi.parquet", columns=["row_id", "text"])
    s = s.merge(bench, on="row_id", how="left")
    C = np.load(OUT / "contrib.npy")
    dla = DLA()
    wins = []
    for r in s.itertuples():
        ids, k = dla.locate(r.text, r.target_char)
        wins.append(ids[max(0, k - WINDOW + 1):k + 1])
    batches = [wins[a:a + BATCH] for a in range(0, len(wins), BATCH)]

    # clean pass: capture z and MLP writes, keep log p(obs)
    caps = {t: [] for t in TARGETS}
    lp0 = []
    with torch.no_grad():
        for w in batches:
            obs, logits = dla.forward(w)
            lp0.append(torch.log_softmax(logits, -1).gather(1, obs[:, None])[:, 0].cpu())
            for t in TARGETS:
                l = int(t[1:].split("H")[0].split("M")[0])
                if t.endswith("MLP"):
                    caps[t].append(dla.cap[f"m{l}"].cpu())
                else:
                    h = int(t.split("H")[1])
                    caps[t].append(dla.cap[f"z{l}"][:, h * HEAD_DIM:(h + 1) * HEAD_DIM].cpu())
    lp0 = torch.cat(lp0).numpy()
    means = {t: torch.cat(v).mean(0).cuda() for t, v in caps.items()}

    rows = []
    layers = dla.model.model.layers
    for t in TARGETS:
        l = int(t[1:].split("H")[0].split("M")[0])
        if t.endswith("MLP"):
            def hook(mod, i, o, t=t):
                o = o.clone(); o[:, -2] = means[t]; return o
            hd = layers[l].post_feedforward_layernorm.register_forward_hook(hook)
        else:
            h = int(t.split("H")[1])
            def hook(mod, i, t=t, h=h):
                x = i[0].clone(); x[:, -2, h * HEAD_DIM:(h + 1) * HEAD_DIM] = means[t]; return (x,)
            hd = layers[l].self_attn.o_proj.register_forward_pre_hook(hook)
        # predicted: contribution of the mean write minus the actual one, frozen scales
        lp1, pred = [], []
        with torch.no_grad():
            for w in batches:
                obs, logits = dla.forward(w)
                lp1.append(torch.log_softmax(logits, -1).gather(1, obs[:, None])[:, 0].cpu())
        hd.remove()
        # predicted change needs the clean-run g and u; recompute them with a clean pass
        with torch.no_grad():
            for bi, w in enumerate(batches):
                obs, logits = dla.forward(w)
                p = torch.softmax(logits, -1)
                hL = dla.cap["hL"]
                rL = torch.sqrt(hL.pow(2).mean(-1, keepdim=True) + dla.eps)
                g = (dla.W_U[obs] - p @ dla.W_U) * dla.w_f / rL
                if t.endswith("MLP"):
                    m = dla.cap[f"m{l}"]
                    pred.append(((g * (means[t] - m)).sum(-1)).cpu())
                else:
                    ln = layers[l].post_attention_layernorm
                    ain = dla.cap[f"ain{l}"]
                    ra = torch.sqrt(ain.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon)
                    u = g * ln.weight / ra
                    W = layers[l].self_attn.o_proj.weight[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    z = dla.cap[f"z{l}"][:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    pred.append(((u @ W) * (means[t] - z)).sum(-1).cpu())
        actual = torch.cat(lp1).numpy() - lp0
        pred = torch.cat(pred).numpy()
        ci = COMPONENTS.index(t)
        for name, m in [("all", np.ones(len(s), bool)), ("ERR", s.err.to_numpy()), ("OK", ~s.err.to_numpy()),
                        ("top-suppressed by it", C[:, ci] < np.percentile(C[:, ci], 10))]:
            rows.append({"component": t, "subset": name, "n": m.sum(),
                         "mean_actual_dlogp": actual[m].mean(), "mean_pred_dlogp": pred[m].mean(),
                         "pearson": pearsonr(pred[m], actual[m])[0], "spearman": spearmanr(pred[m], actual[m])[0],
                         "slope": np.polyfit(pred[m], actual[m], 1)[0]})
        print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "out" / "patch.csv", index=False)


if __name__ == "__main__":
    main()
