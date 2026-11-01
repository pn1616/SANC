# 09 · Leave-one-dataset-out runs

> **Generated.** The uploaded notebook does not contain the LODO code. `run_lodo.py` implements the protocol with exactly the same models, losses, schedule and sampler as the main pipeline. If your LODO runs were done in another notebook, drop that code here instead (or adjust this script to match).

Three folds (hold out Kvasir-SEG, ORIGA or HAM10000). Per fold:

1. train the saliency U-Net (Experiment A recipe) on the other two datasets
2. train the reverse network with SANC noise on the same pool
3. evaluate only on the held-out test split: saliency Dice/IoU and PSNR/SSIM for low-light input, SANC baseline and +A+B+C (`--all-arms` for all eight)

Checkpoints: `$SANC_ROOT/models/lodo/<dataset>/`. Results: `results/09_leave_one_dataset_out/` + `lodo_summary.csv`.
