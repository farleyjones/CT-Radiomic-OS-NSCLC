"""
PyRadiomics-based feature extraction with standardized settings.

Extracts handcrafted radiomic features including shape, first-order,
and texture features (GLCM, GLRLM, GLSZM, GLDM).
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor

from ..config import PipelineConfig, RadiomicsConfig

logger = logging.getLogger(__name__)


def _extract_worker(args: Tuple) -> Dict:
    """
    Module-level worker for parallel extraction.

    Each worker process instantiates its own RadiomicsFeatureExtractor
    (not picklable, so cannot be shared across processes) and loads its
    own images from disk. Returns a feature dict or an error dict.
    """
    image_path, mask_path, patient_id, extractor_settings, feature_classes = args

    # Silence PyRadiomics per-patient logging in worker processes
    import logging as _logging
    _logging.getLogger("radiomics").setLevel(_logging.ERROR)

    try:
        import numpy as np

        image = sitk.ReadImage(str(image_path))
        mask  = sitk.ReadImage(str(mask_path))

        # The DICOM SEG NIfTI files are 4D (x, y, z, n_segments).
        # Collapse to 3D via max-projection across the segment axis,
        # then restore 3D spatial metadata from the CT image.
        if mask.GetDimension() == 4:
            arr = sitk.GetArrayFromImage(mask)   # numpy: (n, z, y, x)
            arr_3d = arr.max(axis=0)             # collapse → (z, y, x)
            mask_3d = sitk.GetImageFromArray(arr_3d.astype(np.int32))
            mask_3d.SetSpacing(image.GetSpacing())
            mask_3d.SetOrigin(image.GetOrigin())
            mask_3d.SetDirection(image.GetDirection())
            mask = mask_3d
        elif mask.GetPixelID() != sitk.sitkInt32:
            arr = sitk.GetArrayFromImage(mask)
            mask_new = sitk.GetImageFromArray(arr.astype(np.int32))
            mask_new.SetSpacing(mask.GetSpacing())
            mask_new.SetOrigin(mask.GetOrigin())
            mask_new.SetDirection(mask.GetDirection())
            mask = mask_new

        ext = featureextractor.RadiomicsFeatureExtractor(**extractor_settings)
        ext.disableAllFeatures()
        for fc in feature_classes:
            ext.enableFeatureClassByName(fc)

        # Enable additional image types if specified
        image_types = extractor_settings.get("_image_types", ["Original"])
        for img_type in image_types:
            if img_type != "Original":
                ext.enableImageTypeByName(img_type)

        result = ext.execute(image, mask)

        feature_dict = {"patient_id": patient_id}
        for key, value in result.items():
            if not key.startswith("diagnostics"):
                feature_dict[key] = float(value)

        return feature_dict

    except Exception as e:
        return {"patient_id": patient_id, "extraction_error": str(e)}


class RadiomicsExtractor:
    """
    PyRadiomics-based feature extraction with standardized settings.
    
    Provides consistent radiomic feature extraction across all cohorts
    with configurable feature classes and extraction parameters.
    """
    
    def __init__(
        self, 
        config: Optional[RadiomicsConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None
    ):
        self.config = config or RadiomicsConfig()
        self.pipeline_config = pipeline_config
        self.extractor = self._configure_extractor()
        self.feature_names: Optional[List[str]] = None
    
    def _configure_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        """Configure PyRadiomics extractor with standardized settings."""
        settings = {
            'binWidth': self.config.bin_width,
            'resampledPixelSpacing': None,  # Already resampled in preprocessing
            'interpolator': 'sitkBSpline',
            'normalize': self.config.normalize,
            'normalizeScale': self.config.normalize_scale,
            'force2D': self.config.force_2d,
        }
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        
        # Enable configured feature classes
        for feature_class in self.config.feature_classes:
            extractor.enableFeatureClassByName(feature_class)
        
        logger.info(
            f"Radiomics extractor configured with classes: "
            f"{self.config.feature_classes}"
        )
        
        return extractor
    
    def extract(
        self, 
        image: sitk.Image, 
        mask: sitk.Image,
        patient_id: str
    ) -> Dict:
        """
        Extract radiomic features from a single image/mask pair.
        
        Args:
            image: Preprocessed CT image
            mask: Tumor segmentation mask
            patient_id: Patient identifier
            
        Returns:
            Dictionary of feature_name: value pairs
        """
        # Ensure mask is integer type
        if mask.GetPixelID() != sitk.sitkInt32:
            mask = sitk.Cast(mask, sitk.sitkInt32)
        
        try:
            features = self.extractor.execute(image, mask)
            
            # Filter out diagnostic info, keep only features
            feature_dict = {'patient_id': patient_id}
            for key, value in features.items():
                if not key.startswith('diagnostics'):
                    feature_dict[key] = float(value)
            
            # Store feature names on first extraction
            if self.feature_names is None:
                self.feature_names = [
                    k for k in feature_dict.keys() if k != 'patient_id'
                ]
                logger.info(f"Extracted {len(self.feature_names)} radiomic features")
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {patient_id}: {e}")
            return {'patient_id': patient_id, 'extraction_error': str(e)}
    
    def extract_batch(
        self, 
        samples: List[Tuple[sitk.Image, sitk.Image, str]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from multiple samples.
        
        Args:
            samples: List of (image, mask, patient_id) tuples
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with features for all samples
        """
        all_features = []
        
        iterator = samples
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc="Extracting radiomics")
            except ImportError:
                pass
        
        for image, mask, patient_id in iterator:
            features = self.extract(image, mask, patient_id)
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        
        # Move patient_id to first column
        if 'patient_id' in df.columns:
            cols = ['patient_id'] + [c for c in df.columns if c != 'patient_id']
            df = df[cols]
        
        return df
    
    def extract_batch_from_paths(
        self,
        samples: List[Tuple[Path, Path, str]],
        n_workers: int = 6,
        checkpoint_path: Optional[Path] = None,
        image_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Parallel extraction from file paths — preferred for large cohorts.

        Each worker process loads its own images and instantiates its own
        extractor, so n_workers processes run truly in parallel.

        Checkpointing: if checkpoint_path is provided, completed patient IDs
        are read on startup and skipped, and results are appended to the file
        after every patient. Safe to interrupt and resume.

        Args:
            samples:          List of (image_path, mask_path, patient_id)
            n_workers:        Number of parallel processes (default 6)
            checkpoint_path:  Optional CSV path for incremental save/resume

        Returns:
            DataFrame with features for all samples (including any previously
            checkpointed results if checkpoint_path was supplied).
        """
        # Resume from checkpoint if available
        completed_ids: set = set()
        prior_rows: List[Dict] = []

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            prior_df = pd.read_csv(checkpoint_path)
            completed_ids = set(prior_df["patient_id"].tolist())
            prior_rows = prior_df.to_dict("records")
            logger.info(f"Resuming: {len(completed_ids)} patients already done")

        _image_types = image_types or ["Original"]
        pending = [
            (str(img), str(msk), pid,
             {
                 "binWidth":              self.config.bin_width,
                 "resampledPixelSpacing": [1, 1, 1],
                 "interpolator":          "sitkBSpline",
                 "normalize":             self.config.normalize,
                 "normalizeScale":        self.config.normalize_scale,
                 "force2D":               self.config.force_2d,
                 "_image_types":          _image_types,
             },
             self.config.feature_classes)
            for img, msk, pid in samples
            if pid not in completed_ids
        ]

        logger.info(
            f"Extracting {len(pending)} patients "
            f"using {n_workers} workers..."
        )

        all_features = list(prior_rows)

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        with mp.Pool(processes=n_workers) as pool:
            iterator = pool.imap_unordered(_extract_worker, pending)

            if use_tqdm:
                iterator = tqdm(iterator, total=len(pending),
                                desc="Extracting radiomics")

            for result in iterator:
                all_features.append(result)

                if "extraction_error" in result:
                    logger.warning(
                        f"  {result['patient_id']}: {result['extraction_error']}"
                    )

                # Incremental checkpoint after each completed patient
                if checkpoint_path is not None:
                    pd.DataFrame(all_features).to_csv(checkpoint_path, index=False)

        df = pd.DataFrame(all_features)

        # Store feature names from the first successful extraction
        if self.feature_names is None:
            self.feature_names = [
                c for c in df.columns
                if c not in ("patient_id", "extraction_error")
            ]

        # patient_id first
        if "patient_id" in df.columns:
            cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
            df = df[cols]

        n_err = int(df["extraction_error"].notna().sum()) if "extraction_error" in df.columns else 0
        logger.info(f"Extraction complete: {len(df) - n_err} ok, {n_err} errors")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of extracted feature names."""
        if self.feature_names is None:
            raise ValueError(
                "No features extracted yet. Call extract() first."
            )
        return self.feature_names
    
    def get_feature_count(self) -> int:
        """Get number of features being extracted."""
        return len(self.feature_names) if self.feature_names else 0
