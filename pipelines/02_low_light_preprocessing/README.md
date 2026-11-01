# 02 · Low-light generation and preprocessing

| Script | What it does | Notebook cells |
|---|---|---|
| `generate_low_light.py` | `simulate_low_light` (×0.40 brightness, γ = 1.5, σ = 0.01 noise, seed 42) on a whole folder; local write → verify → zip → Drive | 26–53 |
| `resize_and_binarize.py` | **generated** – 128×128 resize + mask binarisation (done before this notebook for Kvasir/HAM) | – |
| `visualize_samples.py` | original / low-light / mask grid per dataset | 11, 78 |

The low-light function itself lives in `sanc/preprocessing.py`.
Kvasir-SEG and HAM10000 low-light folders were produced outside this notebook (they arrive as zips on Drive); ORIGA was generated here. If teammates used different parameters for Kvasir/HAM, pass them via `--brightness/--gamma/--noise-std`.
