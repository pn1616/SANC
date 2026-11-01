# 04 · SANC formulation and FFT noise colouring

The module itself is `sanc/noise/sanc.py` (`SANCNoiseColoring`).

**Formulation.** z ~ N(0, I) → FFT → three smooth radial bands (centres 0.15 / 0.45 / 0.80, width 0.20, partition of unity) → per-pixel blend with weights
w_low = 0.55 − 0.35 M, w_mid = 0.30, w_high = 0.15 + 0.35 M → per-channel standardisation to zero mean / unit variance.

| Script | What it does | Notebook cells |
|---|---|---|
| `sanc_tests.py synthetic` | mean/std/NaN check + spectrum figure | 190–192 |
| `sanc_tests.py spectral` | HF/LF energy for M = 0 vs M = 1 (0.473 vs 1.753) | 193 |
| `sanc_tests.py real` | real image → U-Net saliency → local HF energy check | 194–204 |
| `spatial_noise_exploration.py` | first amplitude-based prototype (design history) | 173–183 |
