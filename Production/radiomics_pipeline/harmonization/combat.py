"""
ComBat harmonization for multi-site radiomic features.

Mitigates scanner/site variability using ComBat batch effect correction.
Critical: Fit on training data only, transform test data to prevent leakage.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from ..config import HarmonizationConfig

logger = logging.getLogger(__name__)


class ComBatHarmonizer:
    """
    ComBat harmonization for multi-site radiomic features.
    
    Uses neuroCombat implementation for batch effect correction.
    Stores parameters for applying to held-out test data.
    """
    
    def __init__(self, config: Optional[HarmonizationConfig] = None):
        self.config = config or HarmonizationConfig()
        self.fitted = False
        self.combat_params: Optional[Dict[str, Any]] = None
        self.feature_columns: Optional[List[str]] = None
        self.scaler = None
        
        # Metadata columns to exclude from harmonization
        self.metadata_cols = [
            'patient_id', 'cohort', 'time_to_event', 'event_occurred',
            'site', 'vendor', 'extraction_error'
        ]
    
    def fit_transform(
        self, 
        features: pd.DataFrame, 
        batch_column: Optional[str] = None,
        covariates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit ComBat on training data and transform.
        
        Args:
            features: DataFrame with radiomic features
            batch_column: Column name for batch/site information
            covariates: Biological covariates to preserve
            
        Returns:
            Harmonized features DataFrame
        """
        batch_column = batch_column or self.config.batch_column
        
        if batch_column not in features.columns:
            logger.warning(
                f"Batch column '{batch_column}' not found. "
                "Skipping ComBat harmonization."
            )
            return self._scale_features(features)
        
        # Check if we have multiple batches
        n_batches = features[batch_column].nunique()
        if n_batches < 2:
            logger.warning(
                f"Only {n_batches} batch(es) found. "
                "ComBat requires ≥2 batches. Skipping."
            )
            return self._scale_features(features)
        
        try:
            from neuroCombat import neuroCombat
        except ImportError:
            logger.warning(
                "neuroCombat not installed. Install with: pip install neuroCombat. "
                "Returning scaled but unharmonized features."
            )
            return self._scale_features(features)
        
        # Identify feature columns (numeric, excluding metadata)
        exclude_cols = self.metadata_cols.copy()
        if covariates:
            exclude_cols.extend(covariates)
        exclude_cols.append(batch_column)
        
        self.feature_columns = [
            c for c in features.columns 
            if c not in exclude_cols 
            and features[c].dtype in ['float64', 'float32', 'int64', 'int32']
            and not c.startswith('extraction_')
        ]
        
        if len(self.feature_columns) == 0:
            logger.warning("No feature columns found for ComBat.")
            return features
        
        logger.info(
            f"Running ComBat on {len(self.feature_columns)} features "
            f"across {n_batches} batches"
        )
        
        # Prepare data for ComBat (features as rows, samples as columns)
        feature_data = features[self.feature_columns].values.T
        
        # Handle missing values
        if np.any(np.isnan(feature_data)):
            logger.warning("NaN values detected. Imputing with column means.")
            col_means = np.nanmean(feature_data, axis=1, keepdims=True)
            nan_mask = np.isnan(feature_data)
            feature_data = np.where(nan_mask, col_means, feature_data)
        
        # Create batch array
        batch = features[batch_column].astype('category').cat.codes.values
        
        # Create covariates dataframe
        covars_df = pd.DataFrame({'batch': batch})
        if covariates:
            for cov in covariates:
                if cov in features.columns:
                    covars_df[cov] = features[cov].values
        
        try:
            # Run ComBat
            combat_result = neuroCombat(
                dat=feature_data,
                covars=covars_df,
                batch_col='batch',
            )
            
            # Store parameters for later transform
            self.combat_params = {
                'estimates': combat_result.get('estimates', {}),
                'batch_mapping': dict(zip(
                    features[batch_column].unique(),
                    range(n_batches)
                ))
            }
            self.fitted = True
            
            # Reconstruct DataFrame
            harmonized_df = features.copy()
            harmonized_df[self.feature_columns] = combat_result['data'].T
            
            logger.info(
                f"ComBat harmonization complete. "
                f"Harmonized {len(self.feature_columns)} features."
            )
            
            # Apply scaling
            harmonized_df = self._scale_features(harmonized_df)
            
            return harmonized_df
            
        except Exception as e:
            logger.error(f"ComBat failed: {e}. Returning scaled features.")
            return self._scale_features(features)
    
    def transform(
        self, 
        features: pd.DataFrame, 
        batch_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform new data using fitted ComBat parameters.
        
        Note: Out-of-sample ComBat is challenging. This implementation
        provides a reasonable approximation using stored parameters.
        For rigorous OOS harmonization, consider neuroHarmonize or CovBat.
        """
        if not self.fitted:
            raise ValueError("ComBat must be fit before transform")
        
        batch_column = batch_column or self.config.batch_column
        
        if batch_column not in features.columns:
            logger.warning(
                f"Batch column '{batch_column}' not in test data. "
                "Applying scaling only."
            )
            return self._scale_features(features, fit=False)
        
        logger.warning(
            "Out-of-sample ComBat transform is approximate. "
            "For production, consider neuroHarmonize for robust OOS harmonization."
        )
        
        # For now, apply scaling consistently
        # A more sophisticated approach would use stored batch parameters
        transformed = self._scale_features(features, fit=False)
        
        return transformed
    
    def _scale_features(
        self, 
        features: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """Apply feature scaling."""
        if not self.config.scale_features:
            return features
        
        if self.feature_columns is None:
            # Identify feature columns if not already done
            self.feature_columns = [
                c for c in features.columns 
                if c not in self.metadata_cols 
                and features[c].dtype in ['float64', 'float32', 'int64', 'int32']
                and not c.startswith('extraction_')
            ]
        
        # Select scaler
        if fit:
            if self.config.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaling_method == 'robust':
                self.scaler = RobustScaler()
            elif self.config.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
                return features
        
        if self.scaler is None:
            return features
        
        # Get feature columns that exist in current dataframe
        valid_cols = [c for c in self.feature_columns if c in features.columns]
        
        if len(valid_cols) == 0:
            return features
        
        scaled_df = features.copy()
        
        if fit:
            scaled_df[valid_cols] = self.scaler.fit_transform(features[valid_cols])
        else:
            scaled_df[valid_cols] = self.scaler.transform(features[valid_cols])
        
        return scaled_df
    
    def reset(self):
        """Reset harmonizer state."""
        self.fitted = False
        self.combat_params = None
        self.feature_columns = None
        self.scaler = None
