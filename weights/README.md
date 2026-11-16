# Released weights

Checkpoints are published separately (GitHub Releases / Hugging Face) because of their size.
This directory is where they should be placed after download.

| File | Model | Dataset | Note |
|---|---|---|---|
| `retainseg_pst900.pth` | RetainSeg Stage-2R | PST900 | main model, 60 ep |
| `retainseg_mfnet.pth` | RetainSeg Stage-2R | MFNet | main model, 60 ep |
| `retainseg_fmb.pth` | RetainSeg Stage-2R | FMB | main model |
| `stage2a_pst900.pth` | Stage-2A dense baseline | PST900 | warm start for Stage-2R |
| `stage2a_fmb.pth` | Stage-2A dense baseline | FMB | warm start for Stage-2R |
| `teacher_rgb_pst900.pth` | RGB-only reference | PST900 | retention-ratio denominator |
| `teacher_thermal_pst900.pth` | Thermal-only reference | PST900 | retention-ratio denominator |
| `teacher_rgb_fmb.pth` | RGB-only reference | FMB | retention-ratio denominator |
| `teacher_thermal_fmb.pth` | Thermal-only reference | FMB | retention-ratio denominator |

Expected layout:

```
weights/
├── retainseg_pst900.pth
├── retainseg_mfnet.pth
├── retainseg_fmb.pth
├── stage2a_pst900.pth
├── stage2a_fmb.pth
├── teacher_rgb_pst900.pth
├── teacher_thermal_pst900.pth
├── teacher_rgb_fmb.pth
└── teacher_thermal_fmb.pth
```

Each release asset is accompanied by its SHA-256 in the release notes. Load them through the
`--checkpoint` flag of `tools/bench_v4.py` or `--cfg-options load_from=...` when training.
