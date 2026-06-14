"""
Feature integration layer for multi-stream concatenation.

Combines multiple feature sources (radiomics, deep learning, clinical)
into a unified feature matrix ready for downstream modeling.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureSource(Enum):
    """Identifies the source of features."""
    RADIOMICS = "radiomics"
    DEEP_CNN = "deep_cnn"
    DEEP_TRANSFORMER = "deep_transformer"
    CLINICAL = "clinical"
    GENOMIC = "genomic"
    HANDCRAFTED = "handcrafted"
    CUSTOM = "custom"


class MissingStrategy(Enum):
    """Strategy for handling missing features."""
    DROP = "drop"              # Drop patients with any missing
    IMPUTE_MEAN = "mean"       # Impute with feature mean
    IMPUTE_MEDIAN = "median"   # Impute with feature median
    IMPUTE_ZERO = "zero"       # Impute with zeros
    IMPUTE_KNN = "knn"         # K-nearest neighbors imputation


@dataclass
class FeatureStream:
    """
    Represents a single stream of features.
    
    Attributes:
        name: Identifier for this stream
        source: Type of feature source
        features: DataFrame with patient_id and feature columns
        prefix: Optional prefix to add to feature names
        weight: Optional weight for this stream in ensemble
    """
    name: str
    source: FeatureSource
    features: pd.DataFrame
    prefix: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and setup stream."""
        if 'patient_id' not in self.features.columns:
            raise ValueError(f"Stream '{self.name}' missing 'patient_id' column")
        
        # Auto-generate prefix if not provided
        if self.prefix is None:
            self.prefix = f"{self.source.value}_"
    
    @property
    def feature_columns(self) -> List[str]:
        """Get list of feature column names (excluding metadata)."""
        exclude = {'patient_id', 'cohort', 'site', 'vendor', 
                   'time_to_event', 'event_occurred', 'extraction_error'}
        return [c for c in self.features.columns if c not in exclude]
    
    @property
    def n_features(self) -> int:
        """Number of features in this stream."""
        return len(self.feature_columns)
    
    @property
    def n_samples(self) -> int:
        """Number of samples in this stream."""
        return len(self.features)


@dataclass
class IntegratedFeatureSet:
    """
    Output of feature integration - ready for ML modeling.
    
    Provides clean separation between:
    - Feature matrix (X)
    - Target/outcome data (y, event indicators)
    - Patient/sample metadata
    """
    # Feature matrix
    X: np.ndarray
    feature_names: List[str]
    feature_sources: Dict[str, FeatureSource]
    
    # Target data
    y: Optional[np.ndarray] = None  # time_to_event
    event: Optional[np.ndarray] = None  # event indicator
    
    # Metadata
    patient_ids: List[str] = field(default_factory=list)
    cohorts: Optional[List[str]] = None
    metadata_df: Optional[pd.DataFrame] = None
    
    # Integration info
    stream_indices: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert back to DataFrame."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df.insert(0, 'patient_id', self.patient_ids)
        if self.y is not None:
            df['time_to_event'] = self.y
        if self.event is not None:
            df['event_occurred'] = self.event
        if self.cohorts is not None:
            df['cohort'] = self.cohorts
        return df
    
    def to_survival_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, time, event) tuple for survival models."""
        if self.y is None or self.event is None:
            raise ValueError("Survival data not available")
        return self.X, self.y, self.event
    
    def to_torch(self) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Convert to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        X_tensor = torch.from_numpy(self.X).float()
        y_tensor = torch.from_numpy(self.y).float() if self.y is not None else None
        event_tensor = torch.from_numpy(self.event).float() if self.event is not None else None
        
        return X_tensor, y_tensor, event_tensor
    
    def get_stream_features(self, stream_name: str) -> np.ndarray:
        """Extract features from a specific stream."""
        if stream_name not in self.stream_indices:
            raise KeyError(f"Stream '{stream_name}' not found")
        start, end = self.stream_indices[stream_name]
        return self.X[:, start:end]
    
    def get_stream_names(self, stream_name: str) -> List[str]:
        """Get feature names for a specific stream."""
        if stream_name not in self.stream_indices:
            raise KeyError(f"Stream '{stream_name}' not found")
        start, end = self.stream_indices[stream_name]
        return self.feature_names[start:end]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of feature matrix."""
        return self.X.shape
    
    def summary(self) -> str:
        """Return summary string."""
        lines = [
            f"IntegratedFeatureSet:",
            f"  Samples: {self.X.shape[0]}",
            f"  Features: {self.X.shape[1]}",
            f"  Streams: {list(self.stream_indices.keys())}",
        ]
        for name, (start, end) in self.stream_indices.items():
            lines.append(f"    - {name}: {end - start} features")
        if self.y is not None:
            lines.append(f"  Has survival data: Yes")
            lines.append(f"    Event rate: {np.mean(self.event):.1%}")
        return "\n".join(lines)


class FeatureIntegrator:
    """
    Integrates multiple feature streams into a unified feature matrix.
    
    Handles:
    - Multi-stream alignment by patient_id
    - Missing value strategies
    - Feature prefixing for source identification
    - Metadata preservation
    - Export to various formats
    """
    
    # Columns to preserve as metadata, not features
    METADATA_COLUMNS = {
        'patient_id', 'cohort', 'site', 'vendor', 
        'time_to_event', 'event_occurred',
        'qa_passed', 'extraction_error'
    }
    
    def __init__(
        self,
        missing_strategy: MissingStrategy = MissingStrategy.IMPUTE_MEDIAN,
        require_all_streams: bool = False,
        verbose: bool = True
    ):
        """
        Initialize integrator.
        
        Args:
            missing_strategy: How to handle missing features
            require_all_streams: If True, drop patients missing any stream
            verbose: Whether to log integration details
        """
        self.missing_strategy = missing_strategy
        self.require_all_streams = require_all_streams
        self.verbose = verbose
        
        self.streams: List[FeatureStream] = []
        self.integration_log: List[Dict] = []
    
    def add_stream(
        self,
        features: pd.DataFrame,
        name: str,
        source: FeatureSource,
        prefix: Optional[str] = None,
        weight: float = 1.0
    ) -> "FeatureIntegrator":
        """
        Add a feature stream.
        
        Args:
            features: DataFrame with patient_id and features
            name: Identifier for this stream
            source: Type of feature source
            prefix: Prefix to add to feature names
            weight: Weight for ensemble (optional)
            
        Returns:
            Self for chaining
        """
        stream = FeatureStream(
            name=name,
            source=source,
            features=features.copy(),
            prefix=prefix,
            weight=weight
        )
        
        self.streams.append(stream)
        
        if self.verbose:
            logger.info(
                f"Added stream '{name}': {stream.n_features} features, "
                f"{stream.n_samples} samples"
            )
        
        return self
    
    def add_radiomics(
        self,
        features: pd.DataFrame,
        name: str = "radiomics"
    ) -> "FeatureIntegrator":
        """Convenience method to add radiomics stream."""
        return self.add_stream(
            features, name, FeatureSource.RADIOMICS, prefix="rad_"
        )
    
    def add_deep_features(
        self,
        features: pd.DataFrame,
        name: str = "deep_cnn",
        source: FeatureSource = FeatureSource.DEEP_CNN
    ) -> "FeatureIntegrator":
        """Convenience method to add deep learning features."""
        return self.add_stream(
            features, name, source, prefix="deep_"
        )
    
    def add_clinical(
        self,
        features: pd.DataFrame,
        name: str = "clinical"
    ) -> "FeatureIntegrator":
        """Convenience method to add clinical features."""
        return self.add_stream(
            features, name, FeatureSource.CLINICAL, prefix="clin_"
        )
    
    def integrate(
        self,
        target_columns: Optional[List[str]] = None
    ) -> IntegratedFeatureSet:
        """
        Integrate all streams into unified feature set.
        
        Args:
            target_columns: Columns to use as targets (default: time_to_event, event_occurred)
            
        Returns:
            IntegratedFeatureSet ready for modeling
        """
        if len(self.streams) == 0:
            raise ValueError("No streams added. Use add_stream() first.")
        
        target_columns = target_columns or ['time_to_event', 'event_occurred']
        
        # Step 1: Find common patients
        patient_sets = [set(s.features['patient_id']) for s in self.streams]
        
        if self.require_all_streams:
            common_patients = set.intersection(*patient_sets)
        else:
            common_patients = set.union(*patient_sets)
        
        if len(common_patients) == 0:
            raise ValueError("No common patients across streams")
        
        patient_list = sorted(list(common_patients))
        
        if self.verbose:
            logger.info(f"Integrating {len(self.streams)} streams")
            logger.info(f"  Total patients: {len(patient_list)}")
            for s in self.streams:
                overlap = len(set(s.features['patient_id']) & common_patients)
                logger.info(f"  {s.name}: {overlap}/{s.n_samples} patients")
        
        # Step 2: Align and concatenate features
        all_features = []
        feature_names = []
        feature_sources = {}
        stream_indices = {}
        
        current_idx = 0
        
        for stream in self.streams:
            # Reindex to common patients
            stream_df = stream.features.set_index('patient_id')
            aligned = stream_df.reindex(patient_list)
            
            # Get feature columns only
            feat_cols = [c for c in aligned.columns if c not in self.METADATA_COLUMNS]
            
            # Apply prefix
            prefixed_names = [f"{stream.prefix}{c}" for c in feat_cols]
            
            # Extract values
            stream_features = aligned[feat_cols].values
            
            all_features.append(stream_features)
            feature_names.extend(prefixed_names)
            
            # Track source for each feature
            for name in prefixed_names:
                feature_sources[name] = stream.source
            
            # Track stream indices
            stream_indices[stream.name] = (current_idx, current_idx + len(feat_cols))
            current_idx += len(feat_cols)
        
        # Concatenate all features
        X = np.hstack(all_features)
        
        # Step 3: Handle missing values
        X = self._handle_missing(X, feature_names)
        
        # Step 4: Extract targets and metadata
        y, event, metadata_df = self._extract_targets(
            patient_list, target_columns
        )
        
        # Get cohort info if available
        cohorts = None
        if metadata_df is not None and 'cohort' in metadata_df.columns:
            cohorts = metadata_df['cohort'].tolist()
        
        # Log integration
        self.integration_log.append({
            'n_streams': len(self.streams),
            'n_samples': len(patient_list),
            'n_features': X.shape[1],
            'missing_strategy': self.missing_strategy.value,
            'stream_features': {s.name: s.n_features for s in self.streams}
        })
        
        return IntegratedFeatureSet(
            X=X,
            feature_names=feature_names,
            feature_sources=feature_sources,
            y=y,
            event=event,
            patient_ids=patient_list,
            cohorts=cohorts,
            metadata_df=metadata_df,
            stream_indices=stream_indices
        )
    
    def _handle_missing(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Apply missing value strategy."""
        n_missing_before = np.isnan(X).sum()
        
        if n_missing_before == 0:
            return X
        
        if self.missing_strategy == MissingStrategy.DROP:
            # This should be handled by require_all_streams=True
            logger.warning(
                f"Dropping {np.isnan(X).any(axis=1).sum()} samples with missing values"
            )
            # Can't drop here without affecting patient alignment
            # Fall back to median imputation
            return self._impute_median(X)
        
        elif self.missing_strategy == MissingStrategy.IMPUTE_MEAN:
            return self._impute_mean(X)
        
        elif self.missing_strategy == MissingStrategy.IMPUTE_MEDIAN:
            return self._impute_median(X)
        
        elif self.missing_strategy == MissingStrategy.IMPUTE_ZERO:
            X = np.nan_to_num(X, nan=0.0)
            return X
        
        elif self.missing_strategy == MissingStrategy.IMPUTE_KNN:
            return self._impute_knn(X)
        
        return X
    
    def _impute_mean(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values with column means."""
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        return X
    
    def _impute_median(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values with column medians."""
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        return X
    
    def _impute_knn(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Impute using KNN."""
        try:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=n_neighbors)
            return imputer.fit_transform(X)
        except ImportError:
            logger.warning("sklearn KNNImputer not available, using median")
            return self._impute_median(X)
    
    def _extract_targets(
        self,
        patient_list: List[str],
        target_columns: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Extract target variables and metadata."""
        # Find stream with target information
        target_stream = None
        for stream in self.streams:
            if all(col in stream.features.columns for col in target_columns):
                target_stream = stream
                break
        
        if target_stream is None:
            logger.warning("No stream contains target columns")
            return None, None, None
        
        # Align to patient list
        df = target_stream.features.set_index('patient_id').reindex(patient_list)
        
        # Extract targets
        y = None
        event = None
        
        if 'time_to_event' in df.columns:
            y = df['time_to_event'].values.astype(np.float32)
        
        if 'event_occurred' in df.columns:
            event = df['event_occurred'].values.astype(np.float32)
        
        # Extract metadata
        meta_cols = [c for c in df.columns if c in self.METADATA_COLUMNS]
        metadata_df = df[meta_cols].reset_index() if meta_cols else None
        
        return y, event, metadata_df
    
    def reset(self):
        """Clear all streams."""
        self.streams = []
        self.integration_log = []


class FeatureSelector:
    """
    Feature selection interface for integrated feature sets.
    
    Provides unified interface for various selection methods.
    """
    
    def __init__(self, method: str = "variance", n_features: Optional[int] = None):
        """
        Initialize selector.
        
        Args:
            method: Selection method (variance, correlation, lasso, mrmr, shap)
            n_features: Number of features to select (None = auto)
        """
        self.method = method
        self.n_features = n_features
        self.selected_features: Optional[List[str]] = None
        self.selector = None
    
    def fit(
        self,
        feature_set: IntegratedFeatureSet,
        y: Optional[np.ndarray] = None
    ) -> "FeatureSelector":
        """
        Fit feature selector.
        
        Args:
            feature_set: Integrated features
            y: Target variable (uses feature_set.y if not provided)
        """
        X = feature_set.X
        y = y if y is not None else feature_set.y
        
        if self.method == "variance":
            self._fit_variance(X, feature_set.feature_names)
        elif self.method == "correlation":
            self._fit_correlation(X, y, feature_set.feature_names)
        elif self.method == "lasso":
            self._fit_lasso(X, y, feature_set.feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, feature_set: IntegratedFeatureSet) -> IntegratedFeatureSet:
        """Apply feature selection."""
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        # Find indices of selected features
        indices = [
            feature_set.feature_names.index(f) 
            for f in self.selected_features
        ]
        
        # Create new feature set with selected features
        return IntegratedFeatureSet(
            X=feature_set.X[:, indices],
            feature_names=self.selected_features,
            feature_sources={
                f: feature_set.feature_sources[f] 
                for f in self.selected_features
            },
            y=feature_set.y,
            event=feature_set.event,
            patient_ids=feature_set.patient_ids,
            cohorts=feature_set.cohorts,
            metadata_df=feature_set.metadata_df,
            stream_indices={}  # Indices no longer valid after selection
        )
    
    def fit_transform(
        self,
        feature_set: IntegratedFeatureSet,
        y: Optional[np.ndarray] = None
    ) -> IntegratedFeatureSet:
        """Fit and transform in one step."""
        return self.fit(feature_set, y).transform(feature_set)
    
    def _fit_variance(self, X: np.ndarray, feature_names: List[str]):
        """Select features by variance threshold."""
        from sklearn.feature_selection import VarianceThreshold
        
        # Remove near-zero variance
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        
        mask = selector.get_support()
        self.selected_features = [f for f, m in zip(feature_names, mask) if m]
        
        if self.n_features and len(self.selected_features) > self.n_features:
            # Further reduce by variance ranking
            variances = np.var(X[:, mask], axis=0)
            top_indices = np.argsort(variances)[-self.n_features:]
            self.selected_features = [
                self.selected_features[i] for i in sorted(top_indices)
            ]
        
        logger.info(f"Variance selection: {len(self.selected_features)} features")
    
    def _fit_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ):
        """Select features by correlation with target."""
        if y is None:
            raise ValueError("Target required for correlation selection")
        
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            # Handle constant features
            if np.std(X[:, i]) == 0:
                correlations.append(0)
            else:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        correlations = np.array(correlations)
        n_select = self.n_features or max(10, X.shape[1] // 10)
        
        top_indices = np.argsort(correlations)[-n_select:]
        self.selected_features = [feature_names[i] for i in sorted(top_indices)]
        
        logger.info(f"Correlation selection: {len(self.selected_features)} features")
    
    def _fit_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ):
        """Select features using LASSO."""
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        
        if y is None:
            raise ValueError("Target required for LASSO selection")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LASSO
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Get non-zero coefficients
        nonzero_mask = lasso.coef_ != 0
        self.selected_features = [
            f for f, m in zip(feature_names, nonzero_mask) if m
        ]
        
        # Limit to n_features if specified
        if self.n_features and len(self.selected_features) > self.n_features:
            coef_abs = np.abs(lasso.coef_[nonzero_mask])
            top_indices = np.argsort(coef_abs)[-self.n_features:]
            self.selected_features = [
                self.selected_features[i] for i in sorted(top_indices)
            ]
        
        logger.info(f"LASSO selection: {len(self.selected_features)} features")
