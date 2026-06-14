"""
Image preprocessing for standardized radiomics and deep learning input.

Operations:
1. Resample to isotropic resolution
2. Clamp intensity to valid HU range
3. Optional z-score normalization
"""

import logging
from typing import Tuple, Optional
import numpy as np
import SimpleITK as sitk

from ..config import PipelineConfig

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Standardized image preprocessing for radiomics and deep learning.
    
    Ensures consistent spatial resolution and intensity range across
    all input images regardless of acquisition parameters.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessing_log = []
    
    def preprocess(
        self, 
        image: sitk.Image, 
        mask: sitk.Image,
        normalize: bool = False,
        return_metadata: bool = False
    ) -> Tuple[sitk.Image, sitk.Image]:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Input CT image
            mask: Tumor segmentation mask
            normalize: Whether to apply z-score normalization
            return_metadata: Whether to return preprocessing metadata
            
        Returns:
            Tuple of (preprocessed_image, resampled_mask) or
            Tuple of (preprocessed_image, resampled_mask, metadata)
        """
        metadata = {
            'original_spacing': image.GetSpacing(),
            'original_size': image.GetSize(),
            'original_origin': image.GetOrigin(),
        }
        
        # Step 1: Resample to isotropic
        image_resampled = self._resample_image(image, is_mask=False)
        mask_resampled = self._resample_image(mask, is_mask=True)
        
        # Step 2: Intensity clamping
        image_clamped = self._clamp_intensity(image_resampled)
        
        # Step 3: Optional normalization
        if normalize:
            image_clamped = self._zscore_normalize(image_clamped, mask_resampled)
        
        metadata.update({
            'new_spacing': image_resampled.GetSpacing(),
            'new_size': image_resampled.GetSize(),
            'normalized': normalize,
        })
        
        self.preprocessing_log.append(metadata)
        
        if return_metadata:
            return image_clamped, mask_resampled, metadata
        return image_clamped, mask_resampled
    
    def preprocess_for_deep_learning(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        target_size: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess and convert to numpy arrays for deep learning.
        
        Args:
            image: Input CT image
            mask: Segmentation mask
            target_size: Optional fixed output size (D, H, W)
            
        Returns:
            Tuple of (image_array, mask_array) as numpy arrays
        """
        # Standard preprocessing
        image_proc, mask_proc = self.preprocess(image, mask, normalize=True)
        
        # Convert to arrays
        image_array = sitk.GetArrayFromImage(image_proc).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask_proc).astype(np.int64)
        
        # Optional resize to fixed size
        if target_size is not None:
            image_array = self._resize_array(image_array, target_size)
            mask_array = self._resize_array(mask_array, target_size, is_mask=True)
        
        # Add channel dimension (C, D, H, W)
        image_array = image_array[np.newaxis, ...]
        
        return image_array, mask_array
    
    def _resample_image(
        self, 
        image: sitk.Image, 
        is_mask: bool = False
    ) -> sitk.Image:
        """Resample image to target isotropic spacing."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        new_spacing = self.config.target_spacing
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        
        if is_mask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetDefaultPixelValue(-1000)
        
        return resampler.Execute(image)
    
    def _clamp_intensity(self, image: sitk.Image) -> sitk.Image:
        """Clamp image intensity to valid HU range."""
        return sitk.Clamp(
            image, 
            sitk.sitkFloat32, 
            self.config.hu_min, 
            self.config.hu_max
        )
    
    def _zscore_normalize(
        self, 
        image: sitk.Image, 
        mask: sitk.Image
    ) -> sitk.Image:
        """Z-score normalization within tumor region."""
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Calculate stats within tumor
        tumor_values = image_array[mask_array > 0]
        if len(tumor_values) > 0:
            mean_val = np.mean(tumor_values)
            std_val = np.std(tumor_values)
            if std_val > 0:
                image_array = (image_array - mean_val) / std_val
        
        normalized = sitk.GetImageFromArray(image_array)
        normalized.CopyInformation(image)
        return normalized
    
    def _resize_array(
        self,
        array: np.ndarray,
        target_size: Tuple[int, int, int],
        is_mask: bool = False
    ) -> np.ndarray:
        """Resize 3D array to target size using scipy."""
        from scipy.ndimage import zoom
        
        current_size = array.shape
        zoom_factors = [t / c for t, c in zip(target_size, current_size)]
        
        order = 0 if is_mask else 3  # Nearest for masks, cubic for images
        return zoom(array, zoom_factors, order=order)
