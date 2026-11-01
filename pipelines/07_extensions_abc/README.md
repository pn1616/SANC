# 07 · Extensions A, B, C

Implementation: `sanc/noise/extensions.py`.

| | Extension | Formula | λ |
|---|---|---|---|
| A | Structural fusion | S = (1 − λ_s) M + λ_s E, E = multiscale Sobel (scales 1, 3, 7) | λ_s = 0.30 |
| B | Uncertainty attenuation | S_u = S (1 − λ_u U), U = binary entropy of M / ln 2 | λ_u = 0.50 |
| C | Timestep spectral shift | w_low += λ_t τ, w_high −= λ_t τ, τ = t/(T−1), renormalised | λ_t = 0.05 |

The reverse model is always conditioned on M; only the noise colouring changes.

| Script | What it does | Notebook cells |
|---|---|---|
| `extension_checks.py maps|timestep|arms|regions|prototype` | implementation checks + figures | 297–331 |
| `audit_extensions.py` | A1 equivalence, A2 frozen-model compatibility, A3 forward-chain validity (λ_t = 0.05 valid, 0.10 invalid) | 325–327 |
