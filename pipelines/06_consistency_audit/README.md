# 06 · Mathematical consistency audit

`audit.py` runs read-only checks and writes `audit_report.json`; it exits non-zero if any check fails, so it can be used in CI.

| Section | Checks | Notebook cells |
|---|---|---|
| schedule | β ∈ (0,1), α = 1 − β, ᾱ = ∏α, ᾱ_T = 0.3636 | 263–264, 274, 276 |
| posterior | posterior variance finite, 0 at t = 0, ≤ β | 268–269 |
| sanc | output mean 0 / std 1 / finite, M = 0 vs 1 differ, two SANC instances agree (≤ 1.6e-5) | 275, 296, 325 |
| models | saliency ∈ [0,1]; reverse checkpoint strict load, shape, MSE, DDPM finite | 244–245, 255–267, 293–310 |

Extension-specific audits (A1 equivalence, A2 frozen-model compatibility, A3 forward-chain validity) live in `07_extensions_abc/audit_extensions.py`.
