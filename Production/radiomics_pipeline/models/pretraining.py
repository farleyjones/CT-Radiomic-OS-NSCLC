"""
Pretraining module for foundation model training on unlabeled data.

Supports segmentation-based pretraining using the ~1000 studies
without survival endpoints to learn transferable representations.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from ..config import PretrainingConfig, PipelineConfig
from ..preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    """
    Dataset for pretraining on images with segmentation masks.
    
    Loads CT volumes and segmentation masks, applies preprocessing,
    and optionally extracts random patches for memory efficiency.
    """
    
    def __init__(
        self,
        data_path: Path,
        config: PretrainingConfig,
        pipeline_config: Optional[PipelineConfig] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PretrainingDataset")
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK required for PretrainingDataset")
        
        self.data_path = Path(data_path)
        self.config = config
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.transform = transform
        self.return_metadata = return_metadata
        
        self.preprocessor = ImagePreprocessor(self.pipeline_config)
        
        # Find all image/mask pairs
        self.samples = self._find_samples()
        logger.info(f"Found {len(self.samples)} samples for pretraining")
    
    def _find_samples(self) -> List[Tuple[Path, Path, str]]:
        """Find all image/mask pairs in data directory."""
        samples = []
        
        # Standard Medical Decathlon structure
        images_dir = self.data_path / "imagesTr"
        labels_dir = self.data_path / "labelsTr"
        
        if images_dir.exists() and labels_dir.exists():
            for img_path in images_dir.glob("*.nii.gz"):
                # Expected: lung_001.nii.gz -> lung_001.nii.gz in labels
                mask_path = labels_dir / img_path.name
                if mask_path.exists():
                    patient_id = img_path.stem.replace('.nii', '')
                    samples.append((img_path, mask_path, patient_id))
        else:
            # Generic structure: image.nii.gz and image_mask.nii.gz
            for img_path in self.data_path.glob("*.nii.gz"):
                if "_mask" in img_path.name or "_seg" in img_path.name:
                    continue
                
                # Try common mask naming patterns
                stem = img_path.stem.replace('.nii', '')
                mask_candidates = [
                    img_path.parent / f"{stem}_mask.nii.gz",
                    img_path.parent / f"{stem}_seg.nii.gz",
                    img_path.parent / f"{stem}_label.nii.gz",
                ]
                
                for mask_path in mask_candidates:
                    if mask_path.exists():
                        samples.append((img_path, mask_path, stem))
                        break
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, patient_id = self.samples[idx]
        
        # Load images
        image = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))
        
        # Preprocess
        image_array, mask_array = self.preprocessor.preprocess_for_deep_learning(
            image, mask
        )
        
        # Extract patches if configured
        if self.config.use_patches:
            image_array, mask_array = self._extract_patch(image_array, mask_array)
        
        # Apply transforms (augmentation)
        if self.transform is not None:
            image_array, mask_array = self.transform(image_array, mask_array)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_array).float()
        mask_tensor = torch.from_numpy(mask_array).long()
        
        sample = {
            'image': image_tensor,
            'mask': mask_tensor,
            'patient_id': patient_id,
        }
        
        if self.return_metadata:
            sample['image_path'] = str(img_path)
            sample['mask_path'] = str(mask_path)
        
        return sample
    
    def _extract_patch(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch, preferring regions with foreground."""
        patch_size = self.config.patch_size
        
        # Image shape: (C, D, H, W)
        _, d, h, w = image.shape
        pd, ph, pw = patch_size
        
        # Find foreground voxels to sample around
        foreground = np.where(mask > 0)
        
        if len(foreground[0]) > 0 and np.random.random() > 0.3:
            # 70% chance to center patch on foreground
            idx = np.random.randint(len(foreground[0]))
            center_d = foreground[0][idx]
            center_h = foreground[1][idx]
            center_w = foreground[2][idx]
            
            # Calculate start positions
            start_d = max(0, min(center_d - pd // 2, d - pd))
            start_h = max(0, min(center_h - ph // 2, h - ph))
            start_w = max(0, min(center_w - pw // 2, w - pw))
        else:
            # Random patch
            start_d = np.random.randint(0, max(1, d - pd + 1))
            start_h = np.random.randint(0, max(1, h - ph + 1))
            start_w = np.random.randint(0, max(1, w - pw + 1))
        
        image_patch = image[
            :, 
            start_d:start_d+pd, 
            start_h:start_h+ph, 
            start_w:start_w+pw
        ]
        mask_patch = mask[
            start_d:start_d+pd, 
            start_h:start_h+ph, 
            start_w:start_w+pw
        ]
        
        return image_patch, mask_patch


class SegmentationDecoder(nn.Module):
    """
    Decoder for segmentation pretraining task.
    
    Takes encoded features and produces segmentation logits.
    """
    
    def __init__(
        self,
        encoder_channels: int,
        num_classes: int = 3,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Upsample and decode
            nn.ConvTranspose3d(encoder_channels, base_channels * 4, 4, 2, 1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(base_channels, base_channels // 2, 4, 2, 1),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(base_channels // 2, num_classes, 4, 2, 1),
        )
    
    def forward(self, x):
        return self.decoder(x)


class SegmentationPretrainer:
    """
    Pretrains encoder on segmentation task.
    
    Uses images with segmentation masks (no survival endpoints needed)
    to learn representations that can transfer to survival prediction.
    """
    
    def __init__(
        self,
        config: PretrainingConfig,
        pipeline_config: Optional[PipelineConfig] = None,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for pretraining")
        
        self.config = config
        self.pipeline_config = pipeline_config or PipelineConfig()
        
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def setup_model(self):
        """Initialize encoder and decoder models."""
        from .encoder import create_encoder
        
        self.encoder = create_encoder(
            encoder_type=self.config.encoder_type,
            latent_dim=self.config.latent_dim,
            in_channels=1
        )
        
        # Get encoder output channels for decoder
        # For ResNet, the final feature map has encoder_channels channels
        if 'resnet' in self.config.encoder_type.lower():
            if self.config.encoder_depth >= 50:
                encoder_channels = 512 * 4  # Bottleneck expansion
            else:
                encoder_channels = 512
        else:
            encoder_channels = 512
        
        self.decoder = SegmentationDecoder(
            encoder_channels=encoder_channels,
            num_classes=self.config.num_classes
        )
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Setup optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info(
            f"Model initialized: {self.config.encoder_type}, "
            f"latent_dim={self.config.latent_dim}"
        )
    
    def create_dataloader(self, data_path: Path) -> DataLoader:
        """Create dataloader for pretraining."""
        dataset = PretrainingDataset(
            data_path=data_path,
            config=self.config,
            pipeline_config=self.pipeline_config
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass through encoder (get feature maps, not latent)
            features = self.encoder.forward_features(images)
            
            # Decode to segmentation
            logits = self.decoder(features)
            
            # Resize logits to match mask size if needed
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(
                    logits, 
                    size=masks.shape[1:], 
                    mode='trilinear', 
                    align_corners=False
                )
            
            # Calculate loss
            loss = F.cross_entropy(logits, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(
        self, 
        data_path: Path,
        val_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run full pretraining.
        
        Args:
            data_path: Path to pretraining data
            val_path: Optional validation data path
            
        Returns:
            Training history and final checkpoint path
        """
        self.setup_model()
        dataloader = self.create_dataloader(data_path)
        
        val_dataloader = None
        if val_path is not None:
            val_dataloader = self.create_dataloader(val_path)
        
        logger.info(f"Starting pretraining for {self.config.num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(dataloader)
            
            # Validate
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
            
            # Log progress
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            self.training_history.append(epoch_info)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}" +
                (f", Val Loss: {val_loss:.4f}" if val_loss else "")
            )
            
            # Save best model
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint('best')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final checkpoint
        checkpoint_path = self.save_checkpoint('final')
        
        return {
            'history': self.training_history,
            'best_loss': self.best_loss,
            'checkpoint_path': checkpoint_path
        }
    
    def validate(self, dataloader: DataLoader) -> float:
        """Run validation."""
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                features = self.encoder.forward_features(images)
                logits = self.decoder(features)
                
                if logits.shape[2:] != masks.shape[1:]:
                    logits = F.interpolate(
                        logits, 
                        size=masks.shape[1:], 
                        mode='trilinear', 
                        align_corners=False
                    )
                
                loss = F.cross_entropy(logits, masks)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint."""
        checkpoint_dir = self.config.checkpoint_path or Path('./checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"encoder_{name}.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': {
                'encoder_type': self.config.encoder_type,
                'latent_dim': self.config.latent_dim,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.setup_model()
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
