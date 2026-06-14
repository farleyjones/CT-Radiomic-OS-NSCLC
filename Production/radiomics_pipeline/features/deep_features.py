"""
Deep learning feature extraction from pretrained encoders.

Extracts latent representations from CNN encoders pretrained on
segmentation or self-supervised tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config import DeepFeatureConfig, PipelineConfig
from ..preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class DeepFeatureExtractor:
    """
    Extract deep learning features from a pretrained encoder.
    
    Uses a CNN encoder (e.g., ResNet3D, UNet encoder) pretrained on
    the segmentation task to extract transferable latent representations.
    """
    
    def __init__(
        self,
        config: DeepFeatureConfig,
        pipeline_config: Optional[PipelineConfig] = None,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep feature extraction")
        
        self.config = config
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(self.pipeline_config)
        
        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # Load encoder
        self.encoder = self._load_encoder()
        self.encoder.eval()
        self.encoder.to(self.device)
        
        logger.info(
            f"Deep feature extractor initialized on {self.device}, "
            f"latent_dim={config.latent_dim}"
        )
    
    def _load_encoder(self) -> nn.Module:
        """Load pretrained encoder from checkpoint."""
        from ..models.encoder import create_encoder
        
        encoder = create_encoder(
            encoder_type=self.config.encoder_type,
            latent_dim=self.config.latent_dim,
            pretrained=False  # We'll load weights from checkpoint
        )
        
        if self.config.encoder_checkpoint is not None:
            checkpoint_path = Path(self.config.encoder_checkpoint)
            if checkpoint_path.exists():
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location=self.device
                )
                
                # Handle different checkpoint formats
                if 'encoder_state_dict' in checkpoint:
                    encoder.load_state_dict(checkpoint['encoder_state_dict'])
                elif 'state_dict' in checkpoint:
                    # Filter encoder weights from full model
                    state_dict = {
                        k.replace('encoder.', ''): v 
                        for k, v in checkpoint['state_dict'].items()
                        if k.startswith('encoder.')
                    }
                    encoder.load_state_dict(state_dict)
                else:
                    encoder.load_state_dict(checkpoint)
                
                logger.info(f"Loaded encoder weights from {checkpoint_path}")
            else:
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}. "
                    "Using random initialization."
                )
        
        return encoder
    
    def extract(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        patient_id: str
    ) -> Dict[str, float]:
        """
        Extract deep features from a single image.
        
        Args:
            image: CT image (SimpleITK)
            mask: Segmentation mask (used for ROI cropping)
            patient_id: Patient identifier
            
        Returns:
            Dictionary with patient_id and latent features
        """
        # Preprocess to numpy array
        image_array, _ = self.preprocessor.preprocess_for_deep_learning(
            image, mask
        )
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add batch dim
        image_tensor = image_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.encoder(image_tensor)
            
            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]  # Take first output if tuple
            
            features = features.cpu().numpy().flatten()
        
        # Create feature dictionary
        feature_dict = {'patient_id': patient_id}
        for i, val in enumerate(features):
            feature_dict[f'deep_feat_{i:04d}'] = float(val)
        
        return feature_dict
    
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
            DataFrame with deep features for all samples
        """
        all_features = []
        
        iterator = samples
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc="Extracting deep features")
            except ImportError:
                pass
        
        for image, mask, patient_id in iterator:
            try:
                features = self.extract(image, mask, patient_id)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Deep feature extraction failed for {patient_id}: {e}")
                all_features.append({
                    'patient_id': patient_id, 
                    'extraction_error': str(e)
                })
        
        df = pd.DataFrame(all_features)
        
        # Move patient_id to first column
        if 'patient_id' in df.columns:
            cols = ['patient_id'] + [c for c in df.columns if c != 'patient_id']
            df = df[cols]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of deep feature names."""
        return [f'deep_feat_{i:04d}' for i in range(self.config.latent_dim)]
    
    def get_feature_count(self) -> int:
        """Get number of features."""
        return self.config.latent_dim


class HybridFeatureExtractor:
    """
    Combines traditional radiomics and deep learning features.
    
    Extracts both feature types and concatenates them into a
    unified feature vector.
    """
    
    def __init__(
        self,
        radiomics_extractor,  # RadiomicsExtractor
        deep_extractor: DeepFeatureExtractor
    ):
        self.radiomics_extractor = radiomics_extractor
        self.deep_extractor = deep_extractor
    
    def extract(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        patient_id: str
    ) -> Dict[str, float]:
        """Extract both radiomics and deep features."""
        # Extract both feature types
        radiomics_features = self.radiomics_extractor.extract(image, mask, patient_id)
        deep_features = self.deep_extractor.extract(image, mask, patient_id)
        
        # Merge dictionaries (patient_id appears in both)
        combined = {**radiomics_features}
        for k, v in deep_features.items():
            if k != 'patient_id':
                combined[k] = v
        
        return combined
    
    def extract_batch(
        self,
        samples: List[Tuple[sitk.Image, sitk.Image, str]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Extract combined features from multiple samples."""
        # Extract separately for better error handling
        radiomics_df = self.radiomics_extractor.extract_batch(samples, show_progress)
        deep_df = self.deep_extractor.extract_batch(samples, show_progress=False)
        
        # Merge on patient_id
        combined_df = pd.merge(
            radiomics_df, 
            deep_df.drop(columns=['extraction_error'], errors='ignore'),
            on='patient_id',
            how='outer'
        )
        
        return combined_df
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return (
            self.radiomics_extractor.get_feature_names() + 
            self.deep_extractor.get_feature_names()
        )
