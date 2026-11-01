# 03 · Saliency U-Net and experiments

| Script | What it does | Notebook cells |
|---|---|---|
| `train.py --experiment A…F` | trains one saliency model; checkpoint names match Drive | 124–131, 144–165 |
| `evaluate.py` | per-dataset test Dice / IoU (409 images) | 132–135, 246–249 |
| `analysis.py thresholds|errors|refine` | threshold sweep, Kvasir error analysis, Experiment G refinement | 168–172 |
| `visualize.py` | prediction grid + gradient saliency | 133, 140–141, 167 |

Selected model: **Experiment A** (U-Net, BCE + Dice, best val Dice 0.7545 at epoch 7).
The first test run reported Dice 0.0 on ORIGA/HAM; that was a mask-encoding bug (0..1 masks thresholded after /255), fixed in `SANCMedicalDataset` (`mask > 0`).
