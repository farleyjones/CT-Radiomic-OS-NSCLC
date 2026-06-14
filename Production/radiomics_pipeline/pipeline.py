"""
Pipeline orchestrators for radiomics and hybrid feature extraction.

Provides two main pipelines:
1. RadiomicsPipeline: Traditional PyRadiomics-only extraction
2. HybridFeaturePipeline: Combined radiomics + deep learning features
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from sklearn.model_selection import train_test_split

from .config import (
    PipelineConfig, CohortConfig, PretrainingConfig,
    DeepFeatureConfig, RadiomicsConfig, HarmonizationConfig,
    FeatureMode
)
from .preprocessing import ImagePreprocessor, AutomatedQA
from .features.radiomics import RadiomicsExtractor
from .harmonization import EndpointHarmonizer, ComBatHarmonizer

logger = logging.getLogger(__name__)


class RadiomicsPipeline:
    """
    Traditional radiomics pipeline using PyRadiomics.
    
    Orchestrates preprocessing, QA, feature extraction, and harmonization
    for multi-cohort radiomic analysis.
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        radiomics_config: Optional[RadiomicsConfig] = None,
        harmonization_config: Optional[HarmonizationConfig] = None
    ):
        self.config = config
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config)
        self.qa_checker = AutomatedQA(config)
        self.radiomics_extractor = RadiomicsExtractor(
            config=radiomics_config,
            pipeline_config=config
        )
        self.endpoint_harmonizer = EndpointHarmonizer()
        self.combat_harmonizer = ComBatHarmonizer(harmonization_config)
        
        # Results storage
        self.processed_patients: List[str] = []
        self.failed_patients: List[Tuple[str, str]] = []
    
    def process_cohort(
        self,
        cohort_config: CohortConfig,
        clinical_df: pd.DataFrame,
        skip_qa: bool = False
    ) -> pd.DataFrame:
        """
        Process a single cohort through the full pipeline.
        
        Args:
            cohort_config: Configuration for this cohort
            clinical_df: Clinical data with endpoints
            skip_qa: Whether to skip QA checks
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Processing cohort: {cohort_config.name}")
        
        # Harmonize endpoints
        clinical_harmonized = self.endpoint_harmonizer.harmonize(
            clinical_df, cohort_config
        )
        
        all_features = []
        data_path = cohort_config.data_path
        
        for _, row in clinical_harmonized.iterrows():
            patient_id = row[cohort_config.patient_id_column]
            
            try:
                # Construct file paths
                image_path = data_path / cohort_config.image_pattern.format(
                    patient_id=patient_id
                )
                mask_path = data_path / cohort_config.mask_pattern.format(
                    patient_id=patient_id
                )
                
                if not image_path.exists() or not mask_path.exists():
                    logger.warning(f"Missing files for {patient_id}")
                    self.failed_patients.append((patient_id, "Missing files"))
                    continue
                
                # Load images
                image = sitk.ReadImage(str(image_path))
                mask = sitk.ReadImage(str(mask_path))
                
                # QA check
                if not skip_qa:
                    qa_result = self.qa_checker.run_qa(image, mask, patient_id)
                    if not qa_result.passed:
                        logger.warning(
                            f"{patient_id} QA issues: {qa_result.issues}"
                        )
                
                # Preprocess
                image_proc, mask_proc = self.preprocessor.preprocess(image, mask)
                
                # Extract features
                features = self.radiomics_extractor.extract(
                    image_proc, mask_proc, patient_id
                )
                
                # Add clinical data
                features['time_to_event'] = row['time_to_event']
                features['event_occurred'] = row['event_occurred']
                features['cohort'] = cohort_config.name
                
                if cohort_config.site_column and cohort_config.site_column in row:
                    features['site'] = row[cohort_config.site_column]
                if cohort_config.vendor_column and cohort_config.vendor_column in row:
                    features['vendor'] = row[cohort_config.vendor_column]
                
                all_features.append(features)
                self.processed_patients.append(patient_id)
                
            except Exception as e:
                logger.error(f"Failed to process {patient_id}: {e}")
                self.failed_patients.append((patient_id, str(e)))
        
        return pd.DataFrame(all_features)
    
    def run(
        self,
        cohort_configs: List[CohortConfig],
        clinical_dfs: List[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run complete pipeline across all cohorts.
        
        Returns:
            Dictionary with train/test splits and metadata
        """
        # Process each cohort
        all_cohort_features = []
        for cohort_config, clinical_df in zip(cohort_configs, clinical_dfs):
            cohort_features = self.process_cohort(cohort_config, clinical_df)
            all_cohort_features.append(cohort_features)
        
        # Combine all cohorts
        combined_df = pd.concat(all_cohort_features, ignore_index=True)
        logger.info(
            f"Combined {len(combined_df)} samples from "
            f"{len(cohort_configs)} cohorts"
        )
        
        # Create train/test split
        train_df, test_df = self._create_train_test_split(combined_df)
        
        # ComBat harmonization (fit on train only)
        if 'site' in train_df.columns:
            train_harmonized = self.combat_harmonizer.fit_transform(
                train_df, batch_column='site'
            )
            test_harmonized = self.combat_harmonizer.transform(
                test_df, batch_column='site'
            )
        else:
            logger.warning("No site column found, skipping ComBat")
            train_harmonized = train_df
            test_harmonized = test_df
        
        return {
            'train': train_harmonized,
            'test': test_harmonized,
            'qa_summary': self.qa_checker.get_summary(),
            'failed_patients': self.failed_patients,
            'endpoint_log': self.endpoint_harmonizer.get_summary(),
        }
    
    def _create_train_test_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/test split."""
        # Stratify by event and cohort
        df['_strat_key'] = (
            df['event_occurred'].astype(str) + '_' + 
            df['cohort'].astype(str)
        )
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df['_strat_key']
        )
        
        train_df = train_df.drop('_strat_key', axis=1)
        test_df = test_df.drop('_strat_key', axis=1)
        
        logger.info(
            f"Train/test split: {len(train_df)} train, {len(test_df)} test"
        )
        
        return train_df, test_df
    
    def reset(self):
        """Reset pipeline state."""
        self.processed_patients = []
        self.failed_patients = []
        self.qa_checker.reset()
        self.endpoint_harmonizer.reset()
        self.combat_harmonizer.reset()


class HybridFeaturePipeline:
    """
    Combined radiomics + deep learning feature pipeline.
    
    Workflow:
    1. Pretrain encoder on unlabeled data (optional, if checkpoint not provided)
    2. Extract radiomics features
    3. Extract deep latent features from pretrained encoder
    4. Concatenate feature vectors
    5. Harmonize and split for downstream modeling
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        pretraining_config: Optional[PretrainingConfig] = None,
        deep_feature_config: Optional[DeepFeatureConfig] = None,
        radiomics_config: Optional[RadiomicsConfig] = None,
        harmonization_config: Optional[HarmonizationConfig] = None
    ):
        self.config = config
        self.pretraining_config = pretraining_config
        self.deep_feature_config = deep_feature_config
        self.radiomics_config = radiomics_config
        self.harmonization_config = harmonization_config
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config)
        self.qa_checker = AutomatedQA(config)
        self.endpoint_harmonizer = EndpointHarmonizer()
        self.combat_harmonizer = ComBatHarmonizer(harmonization_config)
        
        # Feature extractors (initialized lazily)
        self._radiomics_extractor = None
        self._deep_extractor = None
        self._pretrainer = None
        
        # Results
        self.processed_patients: List[str] = []
        self.failed_patients: List[Tuple[str, str]] = []
    
    @property
    def radiomics_extractor(self):
        """Lazy initialization of radiomics extractor."""
        if self._radiomics_extractor is None:
            self._radiomics_extractor = RadiomicsExtractor(
                config=self.radiomics_config,
                pipeline_config=self.config
            )
        return self._radiomics_extractor
    
    @property
    def deep_extractor(self):
        """Lazy initialization of deep feature extractor."""
        if self._deep_extractor is None:
            if self.deep_feature_config is None:
                raise ValueError(
                    "DeepFeatureConfig required for deep feature extraction"
                )
            from .features.deep_features import DeepFeatureExtractor
            self._deep_extractor = DeepFeatureExtractor(
                config=self.deep_feature_config,
                pipeline_config=self.config
            )
        return self._deep_extractor
    
    def pretrain(
        self,
        data_path: Path,
        val_path: Optional[Path] = None
    ) -> Path:
        """
        Pretrain encoder on unlabeled data.
        
        Args:
            data_path: Path to pretraining data (images + masks, no endpoints)
            val_path: Optional validation data path
            
        Returns:
            Path to saved encoder checkpoint
        """
        if self.pretraining_config is None:
            raise ValueError("PretrainingConfig required for pretraining")
        
        from .models.pretraining import SegmentationPretrainer
        
        self._pretrainer = SegmentationPretrainer(
            config=self.pretraining_config,
            pipeline_config=self.config
        )
        
        result = self._pretrainer.train(data_path, val_path)
        
        logger.info(
            f"Pretraining complete. Best loss: {result['best_loss']:.4f}"
        )
        
        return result['checkpoint_path']
    
    def process_cohort(
        self,
        cohort_config: CohortConfig,
        clinical_df: pd.DataFrame,
        feature_mode: Optional[FeatureMode] = None
    ) -> pd.DataFrame:
        """
        Process a single cohort extracting specified feature types.
        
        Args:
            cohort_config: Cohort configuration
            clinical_df: Clinical data with endpoints
            feature_mode: Which features to extract (default from config)
            
        Returns:
            DataFrame with features
        """
        feature_mode = feature_mode or self.config.feature_mode
        
        logger.info(
            f"Processing cohort {cohort_config.name} "
            f"with mode: {feature_mode.value}"
        )
        
        # Harmonize endpoints
        clinical_harmonized = self.endpoint_harmonizer.harmonize(
            clinical_df, cohort_config
        )
        
        # Collect samples for batch processing
        samples = []
        sample_metadata = []
        
        for _, row in clinical_harmonized.iterrows():
            patient_id = row[cohort_config.patient_id_column]
            
            try:
                image_path = cohort_config.data_path / cohort_config.image_pattern.format(
                    patient_id=patient_id
                )
                mask_path = cohort_config.data_path / cohort_config.mask_pattern.format(
                    patient_id=patient_id
                )
                
                if not image_path.exists() or not mask_path.exists():
                    self.failed_patients.append((patient_id, "Missing files"))
                    continue
                
                image = sitk.ReadImage(str(image_path))
                mask = sitk.ReadImage(str(mask_path))
                
                # QA
                qa_result = self.qa_checker.run_qa(image, mask, patient_id)
                
                # Preprocess
                image_proc, mask_proc = self.preprocessor.preprocess(image, mask)
                
                samples.append((image_proc, mask_proc, patient_id))
                sample_metadata.append({
                    'patient_id': patient_id,
                    'time_to_event': row['time_to_event'],
                    'event_occurred': row['event_occurred'],
                    'cohort': cohort_config.name,
                    'site': row.get(cohort_config.site_column),
                    'vendor': row.get(cohort_config.vendor_column),
                    'qa_passed': qa_result.passed,
                })
                
                self.processed_patients.append(patient_id)
                
            except Exception as e:
                logger.error(f"Failed: {patient_id}: {e}")
                self.failed_patients.append((patient_id, str(e)))
        
        # Extract features based on mode
        if feature_mode == FeatureMode.RADIOMICS_ONLY:
            features_df = self.radiomics_extractor.extract_batch(samples)
            
        elif feature_mode == FeatureMode.DEEP_ONLY:
            features_df = self.deep_extractor.extract_batch(samples)
            
        elif feature_mode == FeatureMode.HYBRID:
            # Extract both and merge
            radiomics_df = self.radiomics_extractor.extract_batch(samples)
            deep_df = self.deep_extractor.extract_batch(samples, show_progress=False)
            
            # Merge on patient_id
            deep_cols = [c for c in deep_df.columns if c != 'patient_id']
            features_df = pd.merge(
                radiomics_df,
                deep_df[['patient_id'] + deep_cols],
                on='patient_id',
                how='outer'
            )
        
        # Add metadata
        metadata_df = pd.DataFrame(sample_metadata)
        features_df = pd.merge(
            features_df,
            metadata_df,
            on='patient_id',
            how='left'
        )
        
        return features_df
    
    def run(
        self,
        cohort_configs: List[CohortConfig],
        clinical_dfs: List[pd.DataFrame],
        pretrain_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run complete hybrid pipeline.
        
        Args:
            cohort_configs: List of cohort configurations
            clinical_dfs: List of clinical DataFrames
            pretrain_path: Optional path to pretraining data (runs pretraining if provided)
            
        Returns:
            Dictionary with train/test splits and metadata
        """
        # Run pretraining if data provided
        if pretrain_path is not None:
            encoder_checkpoint = self.pretrain(pretrain_path)
            
            # Update deep feature config to use new checkpoint
            if self.deep_feature_config is not None:
                self.deep_feature_config.encoder_checkpoint = encoder_checkpoint
        
        # Process all cohorts
        all_features = []
        for cohort_config, clinical_df in zip(cohort_configs, clinical_dfs):
            cohort_features = self.process_cohort(cohort_config, clinical_df)
            all_features.append(cohort_features)
        
        # Combine
        combined_df = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} samples")
        
        # Train/test split
        train_df, test_df = self._create_train_test_split(combined_df)
        
        # Harmonization
        if 'site' in train_df.columns and train_df['site'].notna().any():
            train_harmonized = self.combat_harmonizer.fit_transform(
                train_df, batch_column='site'
            )
            test_harmonized = self.combat_harmonizer.transform(
                test_df, batch_column='site'
            )
        else:
            train_harmonized = train_df
            test_harmonized = test_df
        
        return {
            'train': train_harmonized,
            'test': test_harmonized,
            'qa_summary': self.qa_checker.get_summary(),
            'failed_patients': self.failed_patients,
            'endpoint_log': self.endpoint_harmonizer.get_summary(),
            'feature_mode': self.config.feature_mode.value,
        }
    
    def _create_train_test_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/test split."""
        df['_strat_key'] = (
            df['event_occurred'].astype(str) + '_' + 
            df['cohort'].astype(str)
        )
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df['_strat_key']
        )
        
        train_df = train_df.drop('_strat_key', axis=1)
        test_df = test_df.drop('_strat_key', axis=1)
        
        return train_df, test_df
    
    def reset(self):
        """Reset pipeline state."""
        self.processed_patients = []
        self.failed_patients = []
        self.qa_checker.reset()
        self.endpoint_harmonizer.reset()
        self.combat_harmonizer.reset()
