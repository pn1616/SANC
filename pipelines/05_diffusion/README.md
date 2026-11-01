# 05 · Forward diffusion and reverse network training

| Script | What it does | Notebook cells |
|---|---|---|
| `forward_diffusion.py` | SANC-coloured forward process on a real image, trajectory figure | 184–189, 205–206 |
| `train_reverse.py` | trains `DualBranchReverseUNet` (7-ch input, illumination + detail decoders, attention fusion) on SANC noise | 207–237 |
| `sample.py` | DDPM (β or posterior variance, noise/low-light start) and DDIM (Member 3) sampling | 238, 256–272, 332 |

Schedule: linear β 1e-4 → 0.02, T = 100 (ᾱ_T = 0.3636, so x_T keeps 60 % of the signal — this is why sampling starts from the low-light image).
Model: AdamW 1e-4, wd 1e-4, 10 epochs, train MSE 0.194 → 0.023.
