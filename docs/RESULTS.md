# RetainSeg — full results

Everything below is produced by one evaluator (`tools/bench_v4.py` for RetainSeg,
`tools/baselines/` for the baselines) over the same RGBT-C protocol. Machine-readable
versions of every table: [`../results/csv/`](../results/csv/).

## Protocol

| Item | Value |
|---|---|
| Severity levels | 2, 3, 5 |
| Corrupted modality | RGB, thermal, or both (sampled per case) |
| Spatial scope | `global` (whole image), `scatter` (small disjoint rectangles) |
| Corruption pool (ID) | all operators except the three held out |
| Corruption pool (OOD) | `fog`, `stripe_noise`, `t_quantization` |
| Cases per image | 15 = 12 degraded cells (3 levels × 2 scopes × 2 pools) + clean + 2 missing-modality cases |
| Sampling | one operator per cell, deterministic per (image, cell, seed) |
| Missing-modality cases | RGB zeroed, thermal zeroed |
| Metric | mIoU from a test-set confusion matrix accumulated before averaging |
| Retention | `ρ−R = 100·mIoU(no RGB)/mIoU(thermal-only teacher)`, `ρ−T = 100·mIoU(no T)/mIoU(RGB-only teacher)` |

`ID` = mean of the 6 in-distribution cells, `OOD` = mean of the 6 held-out cells (the 3 held-out
operators × 3 levels).

## Teachers (retention denominators, clean input, own modality only)

| Dataset | classes | RGB-only mIoU | Thermal-only mIoU |
|---|---|---:|---:|
| PST900 | 5 | 85.73 | 58.84 |
| MFNet | 9 | 59.71 | 57.67 |
| FMB | 13 | 73.36 | 62.61 |

## Per-cell breakdown

### PST900 (5 classes, 288 images) — baselines at 720×1280, RetainSeg at 360×640

| Method | Clean | ID g2 | ID g3 | ID g5 | ID s2 | ID s3 | ID s5 | OOD g2 | OOD g3 | OOD g5 | OOD s2 | OOD s3 | OOD s5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StitchFusion | 84.30 | 81.53 | 79.15 | 74.63 | 83.65 | 82.86 | 82.60 | 84.54 | 84.19 | 83.29 | 83.49 | 83.51 | 83.18 |
| Keep the Balance | 88.14 | 85.28 | 83.25 | 78.18 | 87.21 | 87.02 | 86.31 | 87.78 | 87.14 | 85.97 | 87.57 | 87.47 | 86.81 |
| MRFS | 86.87 | 81.77 | 79.34 | 70.43 | 85.10 | 84.96 | 82.85 | 85.74 | 84.86 | 83.54 | 83.36 | 82.59 | 83.42 |
| EAEFNet | 79.34 | 72.58 | 69.65 | 58.45 | 74.27 | 71.51 | 67.04 | 72.69 | 71.23 | 65.70 | 72.75 | 72.87 | 70.85 |
| AMDANet | 88.57 | 84.70 | 82.67 | 73.06 | 87.53 | 86.48 | 85.30 | 85.62 | 85.40 | 81.79 | 86.01 | 85.88 | 86.42 |
| PEAFusion (released) | 89.63 | 86.02 | 79.96 | 68.27 | 88.61 | 88.04 | 87.66 | 89.26 | 89.01 | 88.04 | 89.16 | 89.06 | 89.07 |
| AMDANet + RGBT-C | 87.15 | 84.03 | 82.85 | 79.18 | 85.63 | 84.73 | 83.40 | 83.88 | 83.41 | 81.88 | 84.76 | 84.72 | 84.30 |
| PEAFusion + RGBT-C | 89.10 | 85.60 | 81.48 | 78.84 | 88.24 | 87.86 | 87.56 | 88.43 | 88.45 | 87.00 | 88.64 | 88.55 | 88.26 |
| **RetainSeg** | 86.54 | 84.24 | 80.12 | 76.91 | 85.09 | 84.71 | 81.77 | 85.85 | 85.09 | 82.86 | 83.65 | 84.12 | 83.11 |

### MFNet (9 classes, 393 images, 480×640)

| Method | Clean | ID g2 | ID g3 | ID g5 | ID s2 | ID s3 | ID s5 | OOD g2 | OOD g3 | OOD g5 | OOD s2 | OOD s3 | OOD s5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CMNeXt | 60.10 | 54.85 | 52.87 | 50.56 | 58.11 | 56.54 | 55.41 | 58.67 | 58.10 | 52.82 | 57.74 | 57.61 | 55.67 |
| StitchFusion | 57.99 | 54.69 | 52.08 | 49.33 | 56.15 | 55.90 | 54.66 | 56.31 | 56.00 | 51.11 | 56.00 | 56.01 | 53.86 |
| Keep the Balance | 59.14 | 54.52 | 53.28 | 50.89 | 57.60 | 57.07 | 56.87 | 56.38 | 55.82 | 51.86 | 57.72 | 57.37 | 56.18 |
| MRFS | 59.06 | 55.29 | 53.22 | 49.84 | 56.81 | 55.93 | 54.46 | 56.65 | 56.66 | 49.19 | 54.70 | 54.94 | 49.26 |
| EAEFNet | 58.94 | 50.91 | 47.22 | 39.72 | 55.24 | 53.74 | 44.36 | 54.63 | 50.35 | 38.70 | 51.98 | 52.93 | 39.32 |
| PEAFusion (released) | 61.01 | 52.98 | 50.87 | 44.92 | 59.04 | 58.32 | 58.27 | 59.28 | 54.99 | 51.96 | 59.31 | 59.50 | 58.92 |
| AMDANet + RGBT-C | 62.88 | 61.61 | 60.67 | 58.61 | 61.32 | 61.91 | 60.38 | 62.04 | 62.13 | 59.84 | 61.00 | 60.11 | 58.83 |
| PEAFusion + RGBT-C | 58.75 | 57.13 | 55.82 | 53.52 | 57.77 | 57.08 | 58.39 | 58.07 | 54.83 | 51.60 | 58.35 | 58.17 | 57.67 |
| **RetainSeg** | 62.01 | 59.38 | 59.01 | 58.41 | 61.39 | 60.98 | 61.36 | 60.99 | 59.45 | 60.47 | 60.93 | 62.13 | 61.74 |

### FMB (280 images, 600×800) — RetainSeg / AMDANet / PEAFusion+ at 13 classes

| Method | Clean | ID g2 | ID g3 | ID g5 | ID s2 | ID s3 | ID s5 | OOD g2 | OOD g3 | OOD g5 | OOD s2 | OOD s3 | OOD s5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StitchFusion †12 | 70.37 | 64.57 | 62.35 | 58.27 | 68.46 | 67.75 | 65.18 | 69.42 | 68.13 | 66.74 | 65.63 | 65.29 | 64.52 |
| MRFS †12 | 67.21 | 63.95 | 62.25 | 57.16 | 63.38 | 62.82 | 60.98 | 63.08 | 60.74 | 54.72 | 57.03 | 54.64 | 50.84 |
| SHIFNet †12 | 73.13 | 67.00 | 63.96 | 60.25 | 69.58 | 67.97 | 64.66 | 70.65 | 67.03 | 66.82 | 69.15 | 67.70 | 63.47 |
| PEAFusion (released) | 69.87 | 61.05 | 57.02 | 44.27 | 65.43 | 65.18 | 62.43 | 58.15 | 55.08 | 51.84 | 63.62 | 62.41 | 61.46 |
| AMDANet + RGBT-C | 69.78 | 68.27 | 66.66 | 64.82 | 67.52 | 66.37 | 65.35 | 68.74 | 66.36 | 62.13 | 64.76 | 61.83 | 57.06 |
| PEAFusion + RGBT-C | 69.17 | 65.89 | 62.43 | 60.43 | 67.68 | 66.64 | 65.64 | 62.34 | 59.22 | 56.65 | 65.99 | 65.45 | 63.82 |
| **RetainSeg** | 75.45 | 74.04 | 73.37 | 71.80 | 74.16 | 74.37 | 74.15 | 74.24 | 74.22 | 71.38 | 74.87 | 74.78 | 73.60 |

†12 — 12-class profile (raw labels 1–12); the other rows in this table are 13-class.

## Retention, both directions

| Dataset | Method | RGB loss | ρ−R | T loss | ρ−T |
|---|---|---:|---:|---:|---:|
| PST900 | StitchFusion | 19.44 | 33.0 | 83.10 | 96.9 |
| PST900 | Keep the Balance | 38.15 | 64.8 | 87.48 | 102.0 |
| PST900 | MRFS | 19.44 | 33.0 | 85.90 | 100.2 |
| PST900 | EAEFNet | 19.64 | 33.4 | 74.87 | 87.3 |
| PST900 | AMDANet | 19.44 | 33.0 | 85.43 | 99.7 |
| PST900 | PEAFusion (released) | 19.44 | 33.0 | 89.49 | 104.4 |
| PST900 | AMDANet + RGBT-C | 25.75 | 43.8 | 84.59 | 98.7 |
| PST900 | PEAFusion + RGBT-C | 36.30 | 61.7 | 88.55 | 103.3 |
| PST900 | **RetainSeg** | **57.45** | **97.6** | 83.96 | 97.9 |
| MFNet | CMNeXt | 38.65 | 67.0 | 53.71 | 90.0 |
| MFNet | StitchFusion | 42.35 | 73.4 | 49.52 | 82.9 |
| MFNet | Keep the Balance | 50.43 | 87.4 | 55.30 | 92.6 |
| MFNet | MRFS | 44.41 | 77.0 | 48.51 | 81.2 |
| MFNet | EAEFNet | 41.72 | 72.3 | 35.23 | 59.0 |
| MFNet | PEAFusion (released) | 40.47 | 70.2 | 53.81 | 90.1 |
| MFNet | AMDANet + RGBT-C | 53.92 | 93.5 | 57.83 | 96.8 |
| MFNet | PEAFusion + RGBT-C | 49.48 | 85.8 | 54.42 | 91.1 |
| MFNet | **RetainSeg** | **55.92** | **97.0** | **57.99** | **97.1** |
| FMB | StitchFusion | 39.72 | 63.4 | 61.94 | 84.4 |
| FMB | MRFS | 45.02 | 71.9 | 58.85 | 80.2 |
| FMB | SHIFNet | 42.95 | 68.6 | 70.61 | 96.3 |
| FMB | PEAFusion (released) | 11.62 | 18.6 | 58.16 | 79.3 |
| FMB | AMDANet + RGBT-C | 53.27 | 85.1 | 64.31 | 87.7 |
| FMB | PEAFusion + RGBT-C | 51.25 | 81.9 | 62.62 | 85.4 |
| FMB | **RetainSeg** | **67.59** | **108.0** | **72.63** | **99.0** |

## Protocol asymmetries (read before quoting a single number)

| Item | Status |
|---|---|
| Degraded pixels | identical for every method (shared `rgbt_c`, same seeds) |
| PST900 resolution | baselines 720×1280; RetainSeg 360×640 |
| MFNet resolution | all 480×640 |
| FMB resolution | 600×800, except AMDANet 480×640 |
| FMB label profile | rows marked †12 are 12-class; AMDANet / PEAFusion+ / RetainSeg are 13-class |
| PST900 / MFNet classes | 5 / 9 for every method |
| PEAFusion variant | replays corruption through its own pipeline and filters excluded-class predictions |
| OGP-Net rows | quoted from the original papers, not re-evaluated here |

The resolution and label-profile differences are properties of the baselines' released
checkpoints, not of the corruption protocol: a fixed-resolution checkpoint cannot be evaluated
on another grid without retraining. `RGB loss` — the column where a model must fall back on its
surviving sensor — is where RetainSeg's margin is largest and where the comparison is least
sensitive to that asymmetry.

## Reproducing

```bash
# RetainSeg
python tools/bench_v4.py --config configs/dino_ts/stage2r_reliable_pst900.py \
    --checkpoint weights/retainseg_pst900.pth --levels 2,3,5 --scopes global,scatter --ood \
    --out results/pst900_v4.json

# baselines
python tools/baselines/eval_amd_pst_v4.py --out results/pst900_amdanet_v4.json
python tools/baselines/eval_pea_v4.py --out results/pst900_peafusion_v4.json

# teachers
python tools/teacher_clean.py --config configs/dino_ts/baseline_thermal_only_pst900.py \
    --checkpoint weights/teacher_thermal_pst900.pth --num-classes 5 \
    --out results/pst900_thermal_teacher.json
```
