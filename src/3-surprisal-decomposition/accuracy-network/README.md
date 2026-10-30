# Accuracy network

A post hoc, exploratory network on ELLIPSE: Study 2's 15 nodes plus four
correction rates (DECISIONS D54, D55), and the same network with an off-the-shelf
detector's flag rates (D56). It runs Study 2's
`_network-body.qmd` with `extra_nodes` set, so the model and the code are the
approved ones.

| File | What it does |
|---|---|
| `nodes.py --arm <arm>` | The four correction rates per essay → `data/study3/ellipse/accuracy_network/nodes_<arm>.csv`, and their validity against gold on W&I and FCE → `results/study3/ellipse/accuracy_network/validity_<arm>.csv` |
| `accuracy-network-corrector.qmd` | The network with the corrector's rates → `results/study3/ellipse/accuracy_network/corrector/` (tables, `figures/`) |
| `detect_electra.py --corpus <ids>` (GPU) | ELECTRA's flagged tokens per unit, by ERRANT group → `data/study3/<corpus>/detector_electra.parquet` |
| `detect_languagetool.py --corpus wi fce` | LanguageTool's flags per unit, for validation only. Needs a local server and Java 17 |
| `accuracy-network-electra.qmd` | The network with ELECTRA's rates → `results/study3/ellipse/accuracy_network/electra/` |
| `approved_readouts.R` | The approved 15-node network's readouts in the same tables → `.../accuracy_network/approved/`. Writes nothing to `fig/` |
| `compare.py --arm <arm>` | The arm against the approved network: membership, shared edges, edges touching an accuracy node |

```bash
A=src/3-surprisal-decomposition/accuracy-network
python $A/nodes.py --arm corrector
Rscript $A/approved_readouts.R                         # about 20 min
quarto render $A/accuracy-network-corrector.qmd        # about 20 min
python $A/compare.py --arm corrector

# detector arm
CUDA_VISIBLE_DEVICES=0 python $A/detect_electra.py --corpus wi fce ellipse   # about 5 min
python $A/detect_languagetool.py --corpus wi fce       # after starting the server
python $A/nodes.py --arm electra
quarto render $A/accuracy-network-electra.qmd          # about 20 min
python $A/compare.py --arm electra
```
