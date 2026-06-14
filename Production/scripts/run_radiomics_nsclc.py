"""
Run radiomics feature extraction for NSCLC-Radiomics (417 patients).

Prerequisites:
  1. Run 00_build_master_cohort.py first → master_cohort.csv
  2. NAS mounted at /Volumes/AIR-NSCLC

Outputs:
  nsclc_radiomics_features.csv    — ~107 features per patient
  radiomics_checkpoint.csv        — incremental save (safe to interrupt/resume)

Usage:
  python run_radiomics_nsclc.py                   # default 6 workers
  python run_radiomics_nsclc.py --workers 10      # adjust to core count
  python run_radiomics_nsclc.py --dry-run 5       # test on first 5 patients
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from radiomics_pipeline.config import RadiomicsConfig
from radiomics_pipeline.features.radiomics import RadiomicsExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Set NIFTI_ROOT to the directory containing per-patient NIfTI files,
# or pass --data-root at the command line.
HERE          = Path(__file__).parent
NIFTI_ROOT    = Path(os.environ.get("NIFTI_ROOT", "/Volumes/AIR-NSCLC/nifti_output/NSCLC-Radiomics"))
MASTER_CSV    = HERE / "master_cohort.csv"
CHECKPOINT    = HERE / "radiomics_checkpoint_v2.csv"
OUTPUT_CSV    = HERE / "nsclc_radiomics_features_v2.csv"

# Image types: Original + LoG (σ=2, σ=3) + Wavelet-LLL
# Adds ~260 features over original-only, increasing pre-LASSO pool from 102 → ~360
IMAGE_TYPES   = ["Original", "LoG", "Wavelet"]
LOG_SIGMA     = [2.0, 3.0]   # used by PyRadiomics LoG filter


def build_sample_list(master: pd.DataFrame) -> list:
    """
    Build (image_path, mask_path, patient_id) tuples from master cohort.
    Logs any patients whose NIfTI files are unexpectedly absent.
    """
    samples = []
    missing = []

    for _, row in master.iterrows():
        pid       = row["patient_id"]
        ct_path   = NIFTI_ROOT / pid / "ct.nii.gz"
        mask_path = NIFTI_ROOT / pid / "Segmentation_300.nii.gz"

        if not ct_path.exists() or not mask_path.exists():
            missing.append(pid)
            continue

        samples.append((ct_path, mask_path, pid))

    if missing:
        logger.warning(f"{len(missing)} patients missing NIfTI files, skipping: {missing}")

    logger.info(f"Sample list: {len(samples)} patients ready for extraction")
    return samples


def run(n_workers: int = 6, dry_run: int = 0) -> pd.DataFrame:
    # ── Preflight ────────────────────────────────────────────────────────────
    if not NIFTI_ROOT.exists():
        logger.error(f"NAS not mounted at {NIFTI_ROOT}. Mount first.")
        sys.exit(1)

    if not MASTER_CSV.exists():
        logger.error(f"master_cohort.csv not found. Run 00_build_master_cohort.py first.")
        sys.exit(1)

    master = pd.read_csv(MASTER_CSV)
    logger.info(f"Master cohort: {len(master)} patients")

    samples = build_sample_list(master)

    if dry_run:
        samples = samples[:dry_run]
        logger.info(f"Dry run: limiting to {dry_run} patients")

    # ── Extraction ───────────────────────────────────────────────────────────
    # Image types: Original + LoG (sigma 2.0, 3.0) + Wavelet-LLL
    # LoG and wavelet-LLL add ~260 features, increasing the pre-LASSO pool
    # from 102 to ~360 radiomics features for richer downstream selection.
    config = RadiomicsConfig(
        bin_width=25,
        feature_classes=["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm"],
        normalize=True,
        normalize_scale=100,
        force_2d=False,
        resample=False,
    )

    extractor = RadiomicsExtractor(config=config)

    logger.info(f"Starting extraction: {len(samples)} patients, {n_workers} workers")
    logger.info(f"Checkpoint: {CHECKPOINT} (safe to interrupt and resume)")

    features_df = extractor.extract_batch_from_paths(
        samples=samples,
        n_workers=n_workers,
        checkpoint_path=CHECKPOINT,
        image_types=IMAGE_TYPES,
    )

    # ── Post-processing ──────────────────────────────────────────────────────
    # Flag extraction failures
    if "extraction_error" in features_df.columns:
        errors = features_df[features_df["extraction_error"].notna()]
        if len(errors):
            logger.warning(f"{len(errors)} extraction errors:")
            for _, row in errors.iterrows():
                logger.warning(f"  {row['patient_id']}: {row['extraction_error']}")
        features_df = features_df.drop(columns=["extraction_error"])

    # Merge manufacturer/clinical metadata so the features CSV is self-contained
    meta_cols = ["patient_id", "survival_days", "event_occurred",
                 "manufacturer", "age", "gender", "overall_stage", "histology"]
    available = [c for c in meta_cols if c in master.columns]
    features_df = features_df.merge(master[available], on="patient_id", how="left")

    # Report NaN rate (high NaN on a feature may indicate extraction issues)
    feat_cols  = [c for c in features_df.columns if c not in meta_cols]
    nan_counts = features_df[feat_cols].isna().sum()
    high_nan   = nan_counts[nan_counts > 0]
    if len(high_nan):
        logger.warning(
            f"{len(high_nan)} features have NaN values "
            f"(max: {high_nan.max()} patients). "
            "These will be imputed during integration."
        )

    features_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved: {OUTPUT_CSV} — {len(features_df)} patients, "
                f"{len(feat_cols)} features")

    print("\n── Extraction Summary ────────────────────────────────────────")
    print(f"  Patients extracted  : {len(features_df)}")
    print(f"  Features per patient: {len(feat_cols)}")
    print(f"  NaN features        : {len(high_nan)}")
    print(f"  Output              : {OUTPUT_CSV.name}")
    print("──────────────────────────────────────────────────────────────\n")

    print("Next step: Feature_Integration_Example.ipynb")
    print("  Inputs: nsclc_radiomics_features.csv + nsclc_deep_features.csv")
    print("  Runs: FeatureIntegrator → ComBat (batch_column='manufacturer') → LASSO")

    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NSCLC radiomics extraction")
    parser.add_argument("--workers",  type=int, default=6,
                        help="Number of parallel worker processes (default: 6)")
    parser.add_argument("--dry-run",  type=int, default=0,
                        help="Process only first N patients (0 = full cohort)")
    args = parser.parse_args()

    run(n_workers=args.workers, dry_run=args.dry_run)
