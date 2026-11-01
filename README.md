# SANC — Saliency-Adaptive Noise Coloring for Low-Light Medical Image Enhancement

Conditional diffusion for low-light medical images in which the **spectrum** of the diffusion noise is shaped per pixel by a learned saliency map: structurally important regions receive high-frequency noise, background regions low-frequency noise. Evaluated on **Kvasir-SEG** (endoscopy), **ORIGA** (fundus) and **HAM10000** (dermoscopy).

```
low-light image ──► Saliency U-Net ──► M ∈ [0,1]
                                       │
z ~ N(0,I) ──► FFT bands (low/mid/high) ──► per-pixel blend w(M) ──► standardise ──► ε_SANC
                                                                                       │
x0 ──► x_t = √ᾱ_t x0 + √(1−ᾱ_t) ε_SANC ──► Dual-branch reverse U-Net(x_t, low-light, M, t) ──► ε̂
```

## Project status

| Stage | Pipeline |
|---|---|
| ✓ Dataset integration and audit | `pipelines/01_dataset_integration` |
| ✓ Low-light generation, preprocessing | `pipelines/02_low_light_preprocessing` |
| ✓ Saliency U-Net and experiments (A–G) | `pipelines/03_saliency_unet` |
| ✓ SANC formulation, FFT noise coloring | `pipelines/04_sanc_noise_coloring` |
| ✓ Forward diffusion, reverse network training | `pipelines/05_diffusion` |
| ✓ Mathematical consistency audit | `pipelines/06_consistency_audit` |
| ✓ Extensions A, B, C implemented and audited | `pipelines/07_extensions_abc` |
| ✓ Eight-arm ablation (409 images) | `pipelines/08_ablation` |
| ✓ Leave-one-dataset-out runs | `pipelines/09_leave_one_dataset_out` |

## Repository layout

```
SANC/
├── sanc/                     shared library — every class defined once
│   ├── config.py             paths (SANC_ROOT) and all hyper-parameters
│   ├── data/                 metadata merge, SANCMedicalDataset, augmentation
│   ├── models/               U-Net, Attention U-Net, UNet++, DualBranchReverseUNet
│   ├── noise/                SANC (FFT), extensions A/B/C, ablation arms, blue noise
│   ├── diffusion/            linear schedule, q_sample, DDPM and DDIM samplers
│   ├── losses.py · metrics.py · training.py · preprocessing.py · runtime.py
├── pipelines/01 … 09/        one folder per stage, runnable scripts + README
├── results/                  numbers, logs and figures extracted from the notebook
├── notebooks/                original notebook (outputs stripped)
├── requirements.txt
└── .gitignore
```

Each pipeline README maps its scripts to the notebook cells they came from.

## Setup

```bash
git clone <repo-url> && cd SANC
pip install -r requirements.txt
export SANC_ROOT=/path/to/SANC        # folder containing datasets/ and models/
```

In Colab:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <repo-url> && cd SANC && pip install -q -r requirements.txt
# SANC_ROOT defaults to /content/drive/MyDrive/SANC
```

Expected data layout (same as the shared Drive folder):

```
$SANC_ROOT/
├── datasets/
│   ├── kvasir/{original,low_light,masks}/, kvasir_metadata.csv
│   ├── metadata/kvasir_{train,val,test}.csv
│   ├── ORIGA_processed/integration/{images,low_light,masks}/, ORIGA_common_metadata.csv
│   └── ham/HAM preprocessing/{images/images,low_light/low_light,masks/masks}/, metadata.csv
└── models/   best_unet_saliency.pth, reverse_diffusion_model.pth, …
```

## Running the pipeline

```bash
python pipelines/01_dataset_integration/build_metadata.py
python pipelines/01_dataset_integration/audit_dataset.py
python pipelines/02_low_light_preprocessing/generate_low_light.py --input <images> --output <out>
python pipelines/03_saliency_unet/train.py --experiment A
python pipelines/03_saliency_unet/evaluate.py
python pipelines/04_sanc_noise_coloring/sanc_tests.py spectral
python pipelines/05_diffusion/train_reverse.py
python pipelines/05_diffusion/sample.py --sampler ddim
python pipelines/06_consistency_audit/audit.py
python pipelines/07_extensions_abc/audit_extensions.py
python pipelines/08_ablation/run_ablation.py full --seeds 42
python pipelines/08_ablation/summarize_ablation.py results/08_ablation/ablation_full_seed42_per_image.csv
python pipelines/09_leave_one_dataset_out/run_lodo.py
```

Scripts that only evaluate (03 evaluate/analysis, 04–08) load the existing checkpoints from `$SANC_ROOT/models`, so nothing has to be retrained to reproduce the reported numbers. Outputs go to `$SANC_ROOT/results/<pipeline>/` (override with `SANC_RESULTS`).

## Key results

See [`results/README.md`](results/README.md) for all tables.

* **Saliency U-Net (test, 409 images):** Dice 0.8891 HAM10000 · 0.7565 Kvasir-SEG · 0.9368 ORIGA
* **Ablation (409 images, seed 42):** low-light 10.04 dB → SANC 12.45 dB → +A+B+C 12.46 dB; all arms within ±0.012 dB of the baseline, SSIM 0.507 → 0.496 when C is on
* **Audit A3:** λ_t = 0.05 is the largest tested value that keeps the timestep-dependent forward chain valid

## Implementation notes

* **Mask encoding.** Kvasir masks are 0–255, ORIGA/HAM masks 0–1. Masks are binarised with `mask > 0`; the earlier `/255 > 0.5` logic produced Dice 0.0 on ORIGA/HAM in the first test run.
* **Excluded image.** `cju89y9h0puti0818i5yw29e6.jpg` (Kvasir test) has a 0-byte mask, so the test set is 409 images, not 410.
* **Two U-Net heads, one checkpoint.** `UNet(return_probs=False)` gives logits for training; `return_probs=True` gives the saliency map M used by SANC.
* **Two SANC instances.** Training used normalisation ε = 1e-8, the early tests ε = 1e-6 (`SANCNoiseColoring(norm_eps=…)`); they agree to 1.6e-5 (audit A1).
* **Schedule.** ᾱ_T = 0.364 for T = 100, so x_T still contains 60 % signal; samplers therefore start from the low-light image.
* **Drive writes.** Generating thousands of files directly into a mounted Drive folder dropped files; `generate_low_light.py` writes locally, verifies, then zips to Drive.

## Generated code (not in the original notebook)

* `pipelines/09_leave_one_dataset_out/run_lodo.py` — the LODO runs were not in this notebook; this script re-implements them with the same recipe.
* `sanc/noise/baselines.py: blue_colour` — defined in Member 3's evaluation notebook; this is a standard stand-in.
* `pipelines/02_low_light_preprocessing/resize_and_binarize.py` — 128 × 128 resizing done before this notebook.
* `pipelines/08_ablation/summarize_ablation.py` — paired-difference summary helper.

Replace these with the team's original versions where they exist.

## Model checkpoints

`.pth` files are excluded by `.gitignore`: the U-Net checkpoint is above GitHub's 100 MB file limit. They remain in the shared Drive folder `SANC/models/`. To version them in the repo, use Git LFS (`git lfs track "*.pth"`) or attach them to a GitHub Release.

## Smoke test without a GPU

```bash
SANC_ROOT=/tmp/fake/SANC SANC_IMAGE_SIZE=64 python pipelines/06_consistency_audit/audit.py
```

`SANC_IMAGE_SIZE` exists only to make CPU smoke tests fast; leave it unset for real runs (128).
