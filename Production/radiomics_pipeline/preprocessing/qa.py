"""
Automated Quality Assurance for CT images and segmentation masks.

QA checks:
- Mask volume: Plausible tumor size range
- Centroid-in-lung: Tumor centroid falls within lung region
- HU plausibility: Intensity values within expected range
- Registration alignment: Mask-image spatial alignment
"""

import logging
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Quality assurance result for a single image/mask pair."""
    patient_id: str
    passed: bool
    mask_volume_mm3: float
    centroid_in_lung: bool
    hu_plausible: bool
    alignment_valid: bool
    issues: List[str] = field(default_factory=list)
    requires_manual_review: bool = False
    requires_nnunet: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'patient_id': self.patient_id,
            'passed': self.passed,
            'mask_volume_mm3': self.mask_volume_mm3,
            'centroid_in_lung': self.centroid_in_lung,
            'hu_plausible': self.hu_plausible,
            'alignment_valid': self.alignment_valid,
            'n_issues': len(self.issues),
            'issues': '; '.join(self.issues),
            'requires_manual': self.requires_manual_review,
            'requires_nnunet': self.requires_nnunet,
        }


class AutomatedQA:
    """
    Automated quality assurance for CT images and segmentation masks.
    
    Performs multiple validation checks and flags cases requiring
    manual review or automated re-segmentation.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.qa_results: List[QAResult] = []
    
    def run_qa(
        self, 
        image: sitk.Image, 
        mask: sitk.Image, 
        patient_id: str
    ) -> QAResult:
        """
        Run full QA pipeline on image/mask pair.
        
        Args:
            image: CT image
            mask: Segmentation mask
            patient_id: Patient identifier
            
        Returns:
            QAResult with pass/fail status and details
        """
        issues = []
        
        # Check 1: Mask volume
        mask_volume = self._calculate_mask_volume(mask)
        volume_ok = (
            self.config.min_mask_volume_mm3 <= mask_volume <= 
            self.config.max_mask_volume_mm3
        )
        if not volume_ok:
            issues.append(
                f"Mask volume {mask_volume:.1f}mm³ outside range "
                f"[{self.config.min_mask_volume_mm3}, {self.config.max_mask_volume_mm3}]"
            )
        
        # Check 2: Centroid in lung
        centroid_ok = self._check_centroid_in_lung(image, mask)
        if not centroid_ok:
            issues.append("Tumor centroid not in lung region")
        
        # Check 3: HU plausibility
        hu_ok = self._check_hu_plausibility(image, mask)
        if not hu_ok:
            issues.append("HU values outside plausible range")
        
        # Check 4: Registration alignment
        alignment_ok = self._check_alignment(image, mask)
        if not alignment_ok:
            issues.append("Image-mask spatial alignment mismatch")
        
        # Determine actions needed
        passed = len(issues) == 0
        requires_manual = not centroid_ok or not alignment_ok
        requires_nnunet = mask_volume < self.config.min_mask_volume_mm3 * 0.5
        
        result = QAResult(
            patient_id=patient_id,
            passed=passed,
            mask_volume_mm3=mask_volume,
            centroid_in_lung=centroid_ok,
            hu_plausible=hu_ok,
            alignment_valid=alignment_ok,
            issues=issues,
            requires_manual_review=requires_manual,
            requires_nnunet=requires_nnunet,
        )
        
        self.qa_results.append(result)
        
        if not passed:
            logger.warning(f"QA failed for {patient_id}: {issues}")
        
        return result
    
    def _calculate_mask_volume(self, mask: sitk.Image) -> float:
        """Calculate mask volume in mm³."""
        spacing = mask.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        
        mask_array = sitk.GetArrayFromImage(mask)
        n_voxels = np.sum(mask_array > 0)
        
        return n_voxels * voxel_volume
    
    def _check_centroid_in_lung(
        self, 
        image: sitk.Image, 
        mask: sitk.Image
    ) -> bool:
        """Check if tumor centroid falls within lung region."""
        mask_array = sitk.GetArrayFromImage(mask)
        image_array = sitk.GetArrayFromImage(image)
        
        tumor_coords = np.where(mask_array > 0)
        if len(tumor_coords[0]) == 0:
            return False
        
        centroid = [
            int(np.mean(tumor_coords[0])),
            int(np.mean(tumor_coords[1])),
            int(np.mean(tumor_coords[2])),
        ]
        
        # Check surrounding region
        z, y, x = centroid
        region = image_array[
            max(0, z-5):z+5, 
            max(0, y-10):y+10, 
            max(0, x-10):x+10
        ]
        
        has_air = np.any(region < -500)
        not_all_soft_tissue = np.mean(region) < 100
        
        return has_air or not_all_soft_tissue
    
    def _check_hu_plausibility(
        self, 
        image: sitk.Image, 
        mask: sitk.Image
    ) -> bool:
        """Check if HU values within tumor are plausible."""
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        tumor_values = image_array[mask_array > 0]
        if len(tumor_values) == 0:
            return False
        
        mean_hu = np.mean(tumor_values)
        return -300 <= mean_hu <= 300
    
    def _check_alignment(
        self, 
        image: sitk.Image, 
        mask: sitk.Image
    ) -> bool:
        """Verify image and mask are spatially aligned."""
        if image.GetSize() != mask.GetSize():
            return False
        
        img_spacing = np.array(image.GetSpacing())
        mask_spacing = np.array(mask.GetSpacing())
        if not np.allclose(img_spacing, mask_spacing, rtol=0.01):
            return False
        
        if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1.0):
            return False
        
        return True
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all QA results."""
        records = [r.to_dict() for r in self.qa_results]
        return pd.DataFrame(records)
    
    def get_flagged_patients(self) -> dict:
        """Get patients flagged for manual review or nnU-Net."""
        return {
            'manual_review': [
                r.patient_id for r in self.qa_results 
                if r.requires_manual_review
            ],
            'nnunet': [
                r.patient_id for r in self.qa_results 
                if r.requires_nnunet
            ],
            'passed': [
                r.patient_id for r in self.qa_results 
                if r.passed
            ],
        }
    
    def reset(self):
        """Clear all stored QA results."""
        self.qa_results = []
