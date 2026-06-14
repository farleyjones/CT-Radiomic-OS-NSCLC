"""Feature extraction modules for radiomics and deep learning."""

from .radiomics import RadiomicsExtractor

def __getattr__(name):
    if name == "DeepFeatureExtractor":
        from .deep_features import DeepFeatureExtractor
        return DeepFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RadiomicsExtractor", "DeepFeatureExtractor"]
