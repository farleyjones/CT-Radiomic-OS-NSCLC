"""
Radiomics Pipeline Package

A modular pipeline for radiomic and deep learning feature extraction
from medical imaging data with multi-cohort harmonization.

Architecture:
- Pretraining branch: Learn representations from unlabeled data (segmentation task)
- Traditional radiomics: PyRadiomics feature extraction
- Deep features: CNN encoder latent space extraction
- Harmonization: Endpoint unification + ComBat batch correction
- Integration: Multi-stream feature concatenation for unified modeling
"""

# Core config (no heavy dependencies)
from .config import PipelineConfig, CohortConfig, PretrainingConfig, EndpointType

# Integration layer (minimal dependencies)
from .integration import (
    FeatureIntegrator,
    FeatureStream,
    FeatureSource,
    IntegratedFeatureSet,
    FeatureSelector,
    MissingStrategy,
)

__version__ = "0.1.0"

# Lazy imports for pipeline and classifier classes with heavy dependencies
def __getattr__(name):
    """Lazy import for classes requiring sklearn/SimpleITK."""
    if name == "RadiomicsPipeline":
        from .pipeline import RadiomicsPipeline
        return RadiomicsPipeline
    elif name == "HybridFeaturePipeline":
        from .pipeline import HybridFeaturePipeline
        return HybridFeaturePipeline
    elif name == "ResponsePredictor":
        from .classifiers import ResponsePredictor
        return ResponsePredictor
    elif name == "ModelComparison":
        from .classifiers import ModelComparison
        return ModelComparison
    elif name == "ClassifierConfig":
        from .classifiers import ClassifierConfig
        return ClassifierConfig
    elif name == "ModelType":
        from .classifiers import ModelType
        return ModelType
    elif name == "CensoringStrategy":
        from .classifiers import CensoringStrategy
        return CensoringStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Config
    "PipelineConfig",
    "CohortConfig", 
    "PretrainingConfig",
    "EndpointType",
    # Pipelines (lazy loaded)
    "RadiomicsPipeline",
    "HybridFeaturePipeline",
    # Integration
    "FeatureIntegrator",
    "FeatureStream",
    "FeatureSource",
    "IntegratedFeatureSet",
    "FeatureSelector",
    "MissingStrategy",
    # Classifiers (lazy loaded)
    "ResponsePredictor",
    "ModelComparison",
    "ClassifierConfig",
    "ModelType",
    "CensoringStrategy",
]
