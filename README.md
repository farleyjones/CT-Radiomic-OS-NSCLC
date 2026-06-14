# CT-Radiomic-OS-NSCLC

CT radiomic and deep learning features for overall survival prediction in non-small cell lung cancer receiving radical radiotherapy.

## Overview

This repository contains the production pipeline code for an MSc thesis project (University of Edinburgh, 2026) evaluating whether CT radiomic features — and a 3D deep learning peri-tumoral encoder — can predict 12-month overall survival in NSCLC patients receiving radical radiotherapy.

**Dataset:** NSCLC-Radiomics (LUNG1), n = 411 patients, publicly available via [TCIA](https://www.cancerimagingarchive.net/).

**Primary result:** LASSO-selected logistic regression; held-out test AUROC 0.790 (CV AUROC 0.623 ± 0.076). 37/45 selected features (82%) were wavelet-derived.

**Deep learning result:** 3D ResNet-18 peri-tumoral encoder (pretrained on LIDC-IDRI) collapsed during pretraining (7/30 passes; Dice ≈ 0.034); DL features contributed no signal — documented as a training failure.

---

## Repository Structure

```
CT-Radiomic-OS-NSCLC/
├── Production/
│   └── scripts/
│       ├── 00_build_master_cohort.py       # Build master_cohort.csv (clinical + scanner metadata)
│       ├── run_radiomics_nsclc.py          # Run PyRadiomics feature extraction (parallelised)
│       └── radiomics_pipeline/             # Core package
│           ├── config.py                   # RadiomicsConfig dataclass
│           ├── features/
│           │   └── radiomics.py            # RadiomicsExtractor (parallel batch extraction)
│           └── ...
└── README.md
```

---

## Prerequisites

- Python 3.10+
- NAS or local directory containing per-patient NIfTI files (`ct.nii.gz`, `Segmentation_300.nii.gz`)
- NSCLC-Radiomics-Lung1 clinical CSV (available from TCIA)
- Deep feature CSV (`nsclc_deep_features.csv`) from the pretrained encoder step

### Dependencies

| Package | Purpose |
|---|---|
| `pyradiomics` | Radiomic feature extraction |
| `SimpleITK` | Image I/O and preprocessing |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | LASSO, classifiers, cross-validation |
| `scikit-survival` | Cox PH, Random Survival Forest |
| `lifelines` | Kaplan-Meier, C-index |
| `neuroCombat` | Multi-scanner harmonisation |
| `xgboost` | XGBoost classifier |
| `MONAI` | 3D ResNet-18 encoder |

Install via:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Build master cohort

```bash
# Mount NAS (or set NIFTI_ROOT to a local path)
export NIFTI_ROOT=/path/to/nifti_output/NSCLC-Radiomics

python Production/scripts/00_build_master_cohort.py
# → master_cohort.csv
```

### Step 2 — Extract radiomic features

```bash
python Production/scripts/run_radiomics_nsclc.py --workers 6
# → nsclc_radiomics_features_v2.csv

# Dry run (first 5 patients)
python Production/scripts/run_radiomics_nsclc.py --dry-run 5

# Interrupt and resume safely — checkpoint saved incrementally
```

Extraction uses Original + LoG (σ = 2, 3) + Wavelet image types, yielding ~806 features per patient before LASSO selection.

### Step 3 — Feature integration and modelling

See `Feature_Integration_Example.ipynb` (not included — contains patient-derived outputs).

Inputs: `nsclc_radiomics_features_v2.csv` + `nsclc_deep_features.csv`  
Pipeline: `FeatureIntegrator → ComBat (batch_column='manufacturer') → LASSO → classifiers`

---

## Data Availability

Patient imaging data and clinical outcomes are **not included** in this repository.

- **NSCLC-Radiomics (LUNG1):** Publicly available via TCIA under the TCIA Data Use Agreement — https://www.cancerimagingarchive.net/collection/nsclc-radiomics/
- **LIDC-IDRI** (encoder pretraining): https://www.cancerimagingarchive.net/collection/lidc-idri/

---

## Citation

If you use this code, please cite:

```
Jones D. CT Radiomic Signature for Overall Survival Prediction in Non-Small Cell Lung Cancer
Receiving Radical Radiotherapy. MSc Thesis, University of Edinburgh; 2026.
```

---

## Licence

MIT — see [LICENSE](LICENSE).
