# Results

Everything in this folder (except `drive/`) was extracted from the **executed** notebook: tables were parsed from printed outputs, logs are verbatim, figures are the embedded plots. Running a pipeline writes new files into the matching sub-folder (`$SANC_RESULTS`, default `$SANC_ROOT/results`).

| Folder | Contents |
|---|---|
| `01_dataset_integration/` | split counts, integration/audit logs |
| `02_low_light_preprocessing/` | low-light generation + verification logs |
| `03_saliency_unet/` | `experiments_summary.csv` (A–G), per-epoch `history_experiment_*.csv`, `saliency_test_results_experiment_A.csv`, Experiment G, logs |
| `04_sanc_noise_coloring/` | spectral test (HF/LF 0.473 → 1.753), logs |
| `05_diffusion/` | reverse-network training loss per epoch, sampling logs |
| `06_consistency_audit/` | schedule / posterior / SANC / checkpoint audit logs |
| `07_extensions_abc/` | audits A1, A2, A3; noise stats per arm; Component C band shares |
| `08_ablation/` | full 409-image summary (seed 42), PSNR by dataset, quick run, smoke test, t0 sensitivity |
| `figures/` | 23 figures, named `<pipeline>_<content>_cell<N>.png` |
| `drive/` | **put the Drive `SANC/results` folder here** (e.g. `member4/ablation_test_seed42_per_image.csv`) |

## Headline numbers

**Saliency U-Net (Experiment A), test set, 409 images**

| Dataset | N | Dice | IoU |
|---|---|---|---|
| HAM10000 | 130 | 0.8891 | 0.8180 |
| Kvasir-SEG | 149 | 0.7565 | 0.6423 |
| ORIGA | 130 | 0.9368 | 0.8848 |

**Saliency experiments, best validation Dice:** A 0.7545 · B 0.7507 · C 0.7502 · D 0.7199 · E 0.7315 · F no gain · G −0.0022 on Kvasir test.

**Eight-arm ablation, 409 test images, seed 42 (DDIM t0 = 99, 25 steps)**

| Arm | PSNR (dB) | SSIM |
|---|---|---|
| Low-light input | 10.0434 | 0.4974 |
| SANC (baseline) | 12.4460 | 0.5069 |
| +C | 12.4473 | 0.5006 |
| +B | 12.4474 | 0.5060 |
| +B+C | 12.4486 | 0.4997 |
| +A | 12.4541 | 0.5035 |
| +A+C | 12.4564 | 0.4967 |
| +A+B | 12.4551 | 0.5028 |
| +A+B+C | 12.4571 | 0.4960 |

**Audit A3:** λ_t = 0.05 keeps every forward-chain increment PSD (worst +0.0877 β); λ_t = 0.10 already fails (−0.0014 β).

## Not in the notebook
* **Leave-one-dataset-out results** - the LODO runs are not in this notebook. Add their outputs under `09_leave_one_dataset_out/` (or `drive/`), or produce them with `pipelines/09_leave_one_dataset_out/run_lodo.py`.
* The first test evaluation (Dice 0.0 on ORIGA/HAM, cells 132/134 in `03_saliency_unet/logs_saliency.txt`) predates the mask-encoding fix and should not be reported.
