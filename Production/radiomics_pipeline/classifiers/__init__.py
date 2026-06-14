"""
Classifiers module for response prediction.

Provides:
- Binary response classifiers (Logistic, XGBoost, Random Forest)
- Survival-aware classifiers (Cox, RSF)
- Ensemble methods combining multiple approaches
- ResponsePredictor for end-to-end workflow
- ModelComparison for benchmarking
"""

from .base import (
    # Enums
    CensoringStrategy,
    ModelType,
    
    # Config
    ClassifierConfig,
    ClassificationResult,
    
    # Base class
    BaseResponseClassifier,
    
    # Concrete classifiers
    LogisticResponseClassifier,
    XGBoostResponseClassifier,
    RandomForestResponseClassifier,
    CoxResponseClassifier,
    RSFResponseClassifier,
    EnsembleResponseClassifier,
    
    # Factory
    create_classifier,
)

from .response_predictor import (
    ResponsePredictor,
    ModelComparison,
)

__all__ = [
    # Enums
    'CensoringStrategy',
    'ModelType',
    
    # Configuration
    'ClassifierConfig',
    'ClassificationResult',
    
    # Base
    'BaseResponseClassifier',
    
    # Classifiers
    'LogisticResponseClassifier',
    'XGBoostResponseClassifier', 
    'RandomForestResponseClassifier',
    'CoxResponseClassifier',
    'RSFResponseClassifier',
    'EnsembleResponseClassifier',
    
    # Factory
    'create_classifier',
    
    # High-level API
    'ResponsePredictor',
    'ModelComparison',
]
