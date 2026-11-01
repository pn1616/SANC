# 01 · Dataset integration and audit

| Script | What it does | Notebook cells |
|---|---|---|
| `build_metadata.py` | Kvasir 70/15/15 split (seed 42), ORIGA common metadata, merged 2,300-row dataframe | 3–20, 48–71, 87–123, 221, 243, 279 |
| `audit_dataset.py` | duplicates, missing/empty files, mask encoding, corrupt-mask report | 6, 52, 71, 91, 118, 142–143, 243, 247, 251–252 |

Final composition: Kvasir-SEG 1000 (700/150/150), ORIGA 650 (520/–/130), HAM10000 650 (520/–/130) → train 1740, val 150, test 410 (409 after excluding the empty Kvasir mask).
