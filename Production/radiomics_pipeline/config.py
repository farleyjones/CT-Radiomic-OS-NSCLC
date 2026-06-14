"""
Configuration dataclasses for the radiomics pipeline.

Defines all configurable parameters for preprocessing, feature extraction,
pretraining, and harmonization.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EndpointType(Enum):
    """Standardized endpoint types across cohorts."""
    SURVIVAL_DAYS = "survival_days"
    SURVIVAL_MONTHS = "survival_months"
    RECURRENCE_DATE = "recurrence_date"
    RECURRENCE_DAYS = "recurrence_days"
    PFS_DAYS = "pfs_days"
    OS_STATUS = "os_status"
    LOCAL_CONTROL = "local_control"


class FeatureMode(Enum):
    """Feature extraction mode."""
    RADIOMICS_ONLY = "radiomics"
    DEEP_ONLY = "deep"
    HYBRID = "hybrid"  # Concatenate radiomics + deep features


@dataclass
class PipelineConfig:
    """Global pipeline configuration."""
    
    # Resampling
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Intensity clamping (Hounsfield Units)
    hu_min: float = -1000.0
    hu_max: float = 400.0
    
    # QA thresholds
    min_mask_volume_mm3: float = 100.0
    max_mask_volume_mm3: float = 500000.0
    hu_tissue_range: Tuple[float, float] = (-1000, 400)
    
    # Train/test split
    test_size: float = 0.10
    random_state: int = 42
    
    # Cross-validation
    n_folds: int = 5
    
    # Feature extraction mode
    feature_mode: FeatureMode = FeatureMode.HYBRID
    
    # Output paths
    output_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None


@dataclass
class CohortConfig:
    """Configuration for a training cohort with endpoints."""
    name: str
    data_path: Path
    endpoint_type: EndpointType
    endpoint_column: str
    event_column: str
    patient_id_column: str = "patient_id"
    image_pattern: str = "{patient_id}.nii.gz"
    mask_pattern: str = "{patient_id}_mask.nii.gz"
    diagnosis_date_column: Optional[str] = None
    site_column: Optional[str] = None
    vendor_column: Optional[str] = None
    additional_metadata: Dict = field(default_factory=dict)


@dataclass
class PretrainingConfig:
    """Configuration for foundation model pretraining."""
    
    # Data
    data_path: Path = None
    image_pattern: str = "*.nii.gz"
    mask_pattern: str = "*_mask.nii.gz"
    
    # Model architecture
    encoder_type: str = "resnet3d"  # resnet3d, unet_encoder, vit3d
    encoder_depth: int = 50  # For ResNet: 18, 34, 50
    latent_dim: int = 512
    
    # Pretraining task
    task: str = "segmentation"  # segmentation, contrastive, mae
    num_classes: int = 3  # e.g., background, nodule, vessel
    
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Patch-based training (for memory efficiency)
    use_patches: bool = True
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    patches_per_volume: int = 4
    
    # Checkpointing
    checkpoint_path: Optional[Path] = None
    resume_from: Optional[Path] = None


@dataclass
class DeepFeatureConfig:
    """Configuration for deep feature extraction from pretrained encoder."""
    
    # Pretrained model
    encoder_checkpoint: Path = None
    encoder_type: str = "resnet3d"
    latent_dim: int = 512
    
    # Feature extraction
    extraction_layer: str = "avgpool"  # Layer to extract features from
    use_attention_pooling: bool = False  # Learnable attention over spatial dims
    
    # Batch processing
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class RadiomicsConfig:
    """Configuration for PyRadiomics feature extraction."""
    
    # Binning
    bin_width: int = 25
    
    # Feature classes to extract
    feature_classes: List[str] = field(default_factory=lambda: [
        "shape",
        "firstorder", 
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
    ])
    
    # Normalization
    normalize: bool = True
    normalize_scale: int = 100
    
    # Settings
    force_2d: bool = False
    resample: bool = False  # Already resampled in preprocessing


@dataclass 
class HarmonizationConfig:
    """Configuration for multi-site harmonization."""
    
    # ComBat settings
    use_combat: bool = True
    batch_column: str = "site"
    preserve_covariates: Optional[List[str]] = None
    
    # Feature scaling
    scale_features: bool = True
    scaling_method: str = "standard"  # standard, robust, minmax
