"""
3D CNN Encoder architectures for medical image feature extraction.

Provides ResNet3D and other encoder backbones that can be pretrained
on segmentation tasks and used for latent feature extraction.
"""

import logging
from typing import Optional, Tuple, List
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_encoder(
    encoder_type: str = "resnet3d",
    latent_dim: int = 512,
    in_channels: int = 1,
    pretrained: bool = False,
    **kwargs
) -> "nn.Module":
    """
    Factory function to create encoder by type.
    
    Args:
        encoder_type: Type of encoder ('resnet3d', 'resnet3d_18', 'resnet3d_50')
        latent_dim: Dimension of latent feature vector
        in_channels: Number of input channels (1 for CT)
        pretrained: Whether to load pretrained weights
        
    Returns:
        Encoder module
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for encoder models")
    
    encoder_type = encoder_type.lower()
    
    if encoder_type in ['resnet3d', 'resnet3d_50']:
        return ResNet3D(
            block=Bottleneck,
            layers=[3, 4, 6, 3],  # ResNet-50
            latent_dim=latent_dim,
            in_channels=in_channels,
            **kwargs
        )
    elif encoder_type == 'resnet3d_18':
        return ResNet3D(
            block=BasicBlock,
            layers=[2, 2, 2, 2],  # ResNet-18
            latent_dim=latent_dim,
            in_channels=in_channels,
            **kwargs
        )
    elif encoder_type == 'resnet3d_34':
        return ResNet3D(
            block=BasicBlock,
            layers=[3, 4, 6, 3],  # ResNet-34
            latent_dim=latent_dim,
            in_channels=in_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1
    
    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152."""
    expansion = 4
    
    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet encoder for volumetric medical images.
    
    Extracts hierarchical features and produces a fixed-size
    latent representation suitable for downstream tasks.
    """
    
    def __init__(
        self,
        block,
        layers: List[int],
        latent_dim: int = 512,
        in_channels: int = 1,
        base_width: int = 64,
        zero_init_residual: bool = True
    ):
        super().__init__()
        
        self.in_planes = base_width
        self.latent_dim = latent_dim
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels, base_width, kernel_size=7, 
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)
        
        # Global pooling and projection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        final_planes = base_width * 8 * block.expansion
        self.fc = nn.Linear(final_planes, latent_dim)
        
        # Weight initialization
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(
        self, 
        block, 
        planes: int, 
        blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward_features(self, x):
        """Extract features before global pooling."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
    def forward(self, x):
        """Forward pass returning latent features."""
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_feature_maps(self, x) -> dict:
        """Get intermediate feature maps for visualization/debugging."""
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['stem'] = x
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        return features


class UNetEncoder(nn.Module):
    """
    U-Net style encoder for segmentation pretraining.
    
    Returns both the latent representation and skip connections
    for use with a decoder during pretraining.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 512,
        depth: int = 4
    ):
        super().__init__()
        
        self.depth = depth
        self.latent_dim = latent_dim
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = in_channels
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.encoders.append(
                self._encoder_block(channels, out_channels)
            )
            self.pools.append(nn.MaxPool3d(2, 2))
            channels = out_channels
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** depth)
        self.bottleneck = self._encoder_block(channels, bottleneck_channels)
        
        # Global pooling and projection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(bottleneck_channels, latent_dim)
    
    def _encoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create double convolution encoder block."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x) -> torch.Tensor:
        """Forward pass returning latent features only."""
        # Encoder path
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global pooling and projection
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward_with_skips(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning latent features and skip connections."""
        skips = []
        
        # Encoder path with skip connections
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global pooling and projection for latent
        latent = self.avgpool(x)
        latent = torch.flatten(latent, 1)
        latent = self.fc(latent)
        
        return latent, skips, x  # latent, skips, bottleneck_features
