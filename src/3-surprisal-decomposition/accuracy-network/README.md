# Accuracy network

A post hoc, exploratory network on ELLIPSE: Study 2's 15 nodes plus four
correction rates (DECISIONS D54, D55). It runs Study 2's
`_network-body.qmd` with `extra_nodes` set, so the model and the code are the
approved ones.

| File | What it does |
|---|---|
| `nodes.py --arm <arm>` | The four correction rates per essay → `data/study3/ellipse/accuracy_network/nodes_<arm>.csv`, and their validity against gold on W&I and FCE → `results/study3/ellipse/accuracy_network/validity_<arm>.csv` |
| `accuracy-network-corrector.qmd` | The network with the corrector's rates → `results/study3/ellipse/accuracy_network/corrector/` (tables, `figures/`) |
| `approved_readouts.R` | The approved 15-node network's readouts in the same tables → `.../accuracy_network/approved/`. Writes nothing to `fig/` |
| `compare.py --arm <arm>` | The arm against the approved network: membership, shared edges, edges touching an accuracy node |

```bash
A=src/3-surprisal-decomposition/accuracy-network
python $A/nodes.py --arm corrector
Rscript $A/approved_readouts.R                         # about 20 min
quarto render $A/accuracy-network-corrector.qmd        # about 20 min
python $A/compare.py --arm corrector
```
