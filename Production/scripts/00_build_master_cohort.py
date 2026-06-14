"""
Build master cohort CSV for NSCLC-Radiomics pipeline.

Merges:
  1. Scanner metadata from dcm2niix JSON sidecars (batch variable for ComBat)
  2. Clinical endpoints from NSCLC-Radiomics-Lung1 CSV
  3. Patient list reconciliation against deep feature extraction cohort (417 pts)

Outputs:
  master_cohort.csv  — one row per patient, ready for pipeline ingestion
  exclusion_log.csv  — documents every patient excluded and why
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

# Set NIFTI_ROOT to the directory containing per-patient NIfTI files,
# or set the NIFTI_ROOT environment variable before running.
NIFTI_ROOT    = Path(os.environ.get("NIFTI_ROOT", "/Volumes/AIR-NSCLC/nifti_output/NSCLC-Radiomics"))
CLINICAL_CSV  = (HERE / "../01. Data Curation & Preprocessing/Datasets"
                 "/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv").resolve()
DEEP_FEAT_CSV = HERE / "nsclc_deep_features.csv"
OUTPUT_CSV    = HERE / "master_cohort.csv"
EXCLUSION_LOG = HERE / "exclusion_log.csv"

# Manufacturer normalisation map (DICOM values are inconsistent across scanners)
MANUFACTURER_MAP = {
    "Siemens":                  "Siemens",
    "SIEMENS":                  "Siemens",
    "Philips":                  "Philips",
    "Philips Medical Systems":  "Philips",
    "GE MEDICAL SYSTEMS":       "GE",
    "GE Healthcare":            "GE",
    "GE":                       "GE",
}

# ManufacturersModelName → Manufacturer for patients where Manufacturer field is absent.
# XiO is a Philips CT system used at MAASTRO Clinic.
MODEL_TO_MANUFACTURER = {
    "XiO": "Philips",
}


# ── Step 1: extract scanner metadata ─────────────────────────────────────────

def extract_scanner_metadata(nifti_root: Path) -> pd.DataFrame:
    """Read Manufacturer, model, and kernel from each patient's ct.json sidecar."""
    records = []

    patient_dirs = sorted(d for d in nifti_root.iterdir() if d.is_dir())
    logger.info(f"Scanning {len(patient_dirs)} patient directories for metadata...")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        json_path  = patient_dir / "ct.json"

        row = {"patient_id": patient_id, "manufacturer": None,
               "scanner_model": None, "convolution_kernel": None}

        if not json_path.exists():
            logger.warning(f"  {patient_id}: ct.json not found")
        else:
            with open(json_path) as f:
                meta = json.load(f)

            model = meta.get("ManufacturersModelName")

            # Some patients lack the Manufacturer field — infer from model name.
            # Patients where ct.json was contaminated by dcmqi SEG metadata
            # (ManufacturersModelName = dcmqi URL) are flagged as unknown.
            if meta.get("Manufacturer"):
                mfr = meta["Manufacturer"]
            elif model and not model.startswith("http"):
                mfr = MODEL_TO_MANUFACTURER.get(model)
                if mfr is None:
                    logger.warning(f"  {patient_id}: unknown model '{model}', manufacturer set to None")
            else:
                if model and model.startswith("http"):
                    logger.warning(f"  {patient_id}: ct.json has dcmqi SEG metadata, not CT — manufacturer unknown")
                mfr = None

            row["manufacturer"]       = mfr
            row["scanner_model"]      = model
            row["convolution_kernel"] = meta.get("ConvolutionKernel")

        records.append(row)

    df = pd.DataFrame(records)

    # Normalise manufacturer labels
    df["manufacturer"] = df["manufacturer"].map(
        lambda x: MANUFACTURER_MAP.get(x, x) if pd.notna(x) else x
    )

    counts = df["manufacturer"].value_counts(dropna=False).to_dict()
    logger.info(f"Manufacturer breakdown: {counts}")
    return df


# ── Step 2: load and standardise clinical CSV ─────────────────────────────────

def load_clinical(clinical_csv: Path) -> pd.DataFrame:
    """Load NSCLC-Radiomics-Lung1 clinical CSV and standardise column names."""
    df = pd.read_csv(clinical_csv)

    df = df.rename(columns={
        "PatientID":          "patient_id",
        "Survival.time":      "survival_days",
        "deadstatus.event":   "event_occurred",
        "age":                "age",
        "gender":             "gender",
        "clinical.T.Stage":   "t_stage",
        "Clinical.N.Stage":   "n_stage",
        "Clinical.M.Stage":   "m_stage",
        "Overall.Stage":      "overall_stage",
        "Histology":          "histology",
    })

    logger.info(f"Clinical CSV loaded: {len(df)} patients")
    return df


# ── Step 3: reconcile against deep feature cohort ────────────────────────────

def reconcile_cohorts(
    clinical: pd.DataFrame,
    scanner: pd.DataFrame,
    deep_feat_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (final_cohort, exclusion_log).

    Final cohort = patients present in BOTH clinical CSV and deep features CSV.
    Exclusion log = all patients dropped, with reasons.
    """
    deep_ids = set(
        pd.read_csv(deep_feat_csv, usecols=["patient_id"])["patient_id"]
    )
    logger.info(f"Deep feature patients: {len(deep_ids)}")

    # Merge clinical + scanner on patient_id
    merged = clinical.merge(scanner, on="patient_id", how="outer", indicator=True)

    # Identify exclusions
    exclusion_rows = []

    # Patients in clinical CSV but no deep features (the 5 missing)
    no_deep = set(clinical["patient_id"]) - deep_ids
    for pid in sorted(no_deep):
        exclusion_rows.append({"patient_id": pid,
                               "reason": "no_deep_features",
                               "detail": "Failed during LIDC pretraining feature extraction"})

    # Patients with deep features but missing clinical data (shouldn't happen, flag anyway)
    no_clinical = deep_ids - set(clinical["patient_id"])
    for pid in sorted(no_clinical):
        exclusion_rows.append({"patient_id": pid,
                               "reason": "no_clinical_data",
                               "detail": "Patient ID not found in clinical CSV"})

    # Patients missing NIfTI files on NAS (can't run radiomics)
    missing_nifti = []
    for pid in deep_ids:
        ct   = NIFTI_ROOT / pid / "ct.nii.gz"
        mask = NIFTI_ROOT / pid / "Segmentation_300.nii.gz"
        if not ct.exists() or not mask.exists():
            missing_nifti.append(pid)
            exclusion_rows.append({"patient_id": pid,
                                   "reason": "missing_nifti",
                                   "detail": f"ct.nii.gz or Segmentation_300.nii.gz absent"})

    excl_df = pd.DataFrame(exclusion_rows)
    logger.info(f"Total exclusions: {len(excl_df)}")

    # Build final cohort
    final_ids = deep_ids - no_clinical - set(missing_nifti)
    final = merged[merged["patient_id"].isin(final_ids)].drop(columns=["_merge"])

    return final.reset_index(drop=True), excl_df


# ── Step 4: clinical cleaning ─────────────────────────────────────────────────

def clean_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and encode covariates needed downstream."""

    # Missing histology (10% of cohort) — fill as 'unknown' rather than drop,
    # preserving sample size. ComBat will receive histology as a covariate.
    n_missing_histo = df["histology"].isna().sum()
    df["histology"] = df["histology"].fillna("unknown")
    if n_missing_histo:
        logger.info(f"Histology missing → filled 'unknown': {n_missing_histo} patients")

    # Missing manufacturer — log but don't exclude; ComBat will skip these rows
    n_missing_mfr = df["manufacturer"].isna().sum()
    if n_missing_mfr:
        logger.warning(
            f"{n_missing_mfr} patients missing manufacturer. "
            "ComBat will exclude them from harmonisation."
        )

    # Validate survival data
    neg_surv = (df["survival_days"] <= 0).sum()
    if neg_surv:
        logger.warning(f"{neg_surv} patients with survival_days ≤ 0")

    missing_surv = df["survival_days"].isna().sum()
    if missing_surv:
        logger.warning(f"{missing_surv} patients with missing survival_days")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    scanner  = extract_scanner_metadata(NIFTI_ROOT)
    clinical = load_clinical(CLINICAL_CSV)
    final, excl = reconcile_cohorts(clinical, scanner, DEEP_FEAT_CSV)
    final = clean_clinical(final)

    # Save
    final.to_csv(OUTPUT_CSV, index=False)
    excl.to_csv(EXCLUSION_LOG, index=False)

    logger.info(f"master_cohort.csv  → {len(final)} patients")
    logger.info(f"exclusion_log.csv  → {len(excl)} patients excluded")

    # Summary
    print("\n── Master Cohort Summary ─────────────────────────────────────")
    print(f"  N patients            : {len(final)}")
    print(f"  Events (deaths)       : {int(final['event_occurred'].sum())} "
          f"({final['event_occurred'].mean():.1%})")
    print(f"  Median survival (days): {final['survival_days'].median():.0f}")
    print(f"\n  Manufacturer breakdown:")
    for mfr, n in final["manufacturer"].value_counts(dropna=False).items():
        print(f"    {str(mfr):10s}: {n}")
    print(f"\n  Overall stage:")
    for stage, n in final["overall_stage"].value_counts().items():
        print(f"    {stage:6s}: {n}")
    print("──────────────────────────────────────────────────────────────\n")

    return final


if __name__ == "__main__":
    main()
