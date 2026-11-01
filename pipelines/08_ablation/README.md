# 08 · Eight-arm ablation (409 test images)

Arms = all on/off combinations of A, B, C (baseline = plain SANC). Sampler frozen: Member-3 DDIM, t0 = 99, 25 steps, low-light start, identical z across arms.

| Script | What it does | Notebook cells |
|---|---|---|
| `run_ablation.py smoke|t0|quick|full` | runs the arms, writes a per-image CSV (never overwrites) | 332–336 |
| `summarize_ablation.py <csv>` | per-arm means, per-dataset table, paired differences vs baseline | – (new helper) |

Reported (seed 42, 409 images): low-light 10.043 dB → SANC 12.446 dB → +A+B+C 12.457 dB. Every arm stays within ±0.012 dB of the baseline in PSNR, while SSIM drops slightly whenever C is on (0.5069 → 0.4960). Per-image CSV: `SANC/results/member4/ablation_test_seed42_per_image.csv` on Drive.
