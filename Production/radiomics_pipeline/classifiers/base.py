"""
Response classifier for PFS at landmark time (e.g., 180 days).

Classifies patients as Responders (PFS ≥ threshold) vs Non-Responders.

Approaches:
1. Binary classifiers (Logistic, XGBoost, RF) with censoring handling
2. Survival models (Cox, RSF) with landmark probability extraction
3. Ensemble methods combining multiple approaches

Critical: Must handle censored observations properly.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CensoringStrategy(Enum):
    """Strategy for handling censored observations in binary classification."""
    EXCLUDE = "exclude"           # Drop patients censored before landmark
    TREAT_AS_RESPONDER = "responder"  # Assume censored before landmark are responders
    IPCW = "ipcw"                 # Inverse probability of censoring weighting
    PSEUDO_VALUES = "pseudo"      # Pseudo-observation approach


class ModelType(Enum):
    """Available model types."""
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    COX = "cox"
    RSF = "rsf"  # Random Survival Forest
    DEEPSURV = "deepsurv"
    ENSEMBLE = "ensemble"


@dataclass
class ClassifierConfig:
    """Configuration for response classifier."""
    
    # Landmark time for response definition
    landmark_days: int = 180
    
    # Model selection
    model_type: ModelType = ModelType.XGBOOST
    
    # Censoring handling
    censoring_strategy: CensoringStrategy = CensoringStrategy.EXCLUDE
    
    # Class imbalance handling
    class_weight: str = "balanced"  # balanced, None, or dict
    
    # Cross-validation
    n_folds: int = 5
    
    # Hyperparameters (defaults, can be tuned)
    logistic_params: Dict = field(default_factory=lambda: {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
    })
    
    xgboost_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    })
    
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
    })
    
    cox_params: Dict = field(default_factory=lambda: {
        'alpha': 0.1,  # L2 penalty
        'ties': 'breslow',
    })
    
    rsf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
    })
    
    # Threshold for probability-based classification
    probability_threshold: float = 0.5
    
    # Random state
    random_state: int = 42


@dataclass
class ClassificationResult:
    """Results from response classification."""
    
    # Predictions
    y_pred: np.ndarray              # Binary predictions
    y_prob: np.ndarray              # Probability of being responder
    y_true: np.ndarray              # True labels
    
    # Metrics
    metrics: Dict[str, float]
    
    # Model info
    model_type: str
    feature_importance: Optional[pd.DataFrame] = None
    
    # Cross-validation results
    cv_scores: Optional[Dict[str, List[float]]] = None
    
    # Metadata
    n_train: int = 0
    n_test: int = 0
    n_censored_excluded: int = 0
    landmark_days: int = 180
    
    def summary(self) -> str:
        """Return summary string."""
        lines = [
            f"Classification Results ({self.model_type}):",
            f"  Landmark: {self.landmark_days} days",
            f"  Train/Test: {self.n_train}/{self.n_test}",
            f"  Censored excluded: {self.n_censored_excluded}",
            "",
            "  Metrics:",
        ]
        for name, value in self.metrics.items():
            lines.append(f"    {name}: {value:.3f}")
        return "\n".join(lines)


class BaseResponseClassifier(ABC):
    """Abstract base class for response classifiers."""
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseResponseClassifier":
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary response."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of being responder."""
        pass
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Fit and predict."""
        self.fit(X, y, **kwargs)
        return self.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return None


class LogisticResponseClassifier(BaseResponseClassifier):
    """Logistic Regression classifier."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LogisticResponseClassifier":
        from sklearn.linear_model import LogisticRegression
        
        self.model = LogisticRegression(
            **self.config.logistic_params,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = kwargs.get('feature_names')
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = np.abs(self.model.coef_[0])
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'coefficient': self.model.coef_[0]
        }).sort_values('importance', ascending=False)


class XGBoostResponseClassifier(BaseResponseClassifier):
    """XGBoost classifier."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostResponseClassifier":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        # Handle class imbalance via scale_pos_weight
        if self.config.class_weight == "balanced":
            scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        else:
            scale_pos_weight = 1.0
        
        params = {
            **self.config.xgboost_params,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.config.random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = kwargs.get('feature_names')
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class RandomForestResponseClassifier(BaseResponseClassifier):
    """Random Forest classifier."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RandomForestResponseClassifier":
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            **self.config.rf_params,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = kwargs.get('feature_names')
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class CoxResponseClassifier(BaseResponseClassifier):
    """
    Cox Proportional Hazards with landmark probability extraction.
    
    Uses survival probability at landmark time as classification score.
    """
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        time: Optional[np.ndarray] = None,
        event: Optional[np.ndarray] = None,
        **kwargs
    ) -> "CoxResponseClassifier":
        """
        Fit Cox model.
        
        Note: For Cox, y is not used directly. Instead, use time and event.
        y is kept for API consistency.
        """
        try:
            from sksurv.linear_model import CoxPHSurvivalAnalysis
        except ImportError:
            raise ImportError(
                "scikit-survival not installed. "
                "Install with: pip install scikit-survival"
            )
        
        if time is None or event is None:
            raise ValueError("Cox model requires time and event arrays")
        
        # Create structured array
        self.y_surv = np.array(
            [(bool(e), t) for e, t in zip(event, time)],
            dtype=[('event', bool), ('time', float)]
        )
        
        self.model = CoxPHSurvivalAnalysis(
            alpha=self.config.cox_params.get('alpha', 0.1)
        )
        self.model.fit(X, self.y_surv)
        self.is_fitted = True
        self.feature_names = kwargs.get('feature_names')
        
        # Store baseline survival function for prediction
        self._compute_baseline_survival(X)
        
        return self
    
    def _compute_baseline_survival(self, X: np.ndarray):
        """Compute baseline survival for probability extraction."""
        self.baseline_surv = self.model.predict_survival_function(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary response based on landmark survival probability."""
        probs = self.predict_proba(X)
        return (probs >= self.config.probability_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of being responder at landmark time.
        
        Returns P(T > landmark_days) = survival probability at landmark.
        """
        surv_funcs = self.model.predict_survival_function(X)
        
        probs = []
        for sf in surv_funcs:
            # Get survival probability at landmark time
            # sf.x contains time points, sf.y contains survival probabilities
            landmark = self.config.landmark_days
            if landmark <= sf.x[-1]:
                prob = sf(landmark)
            else:
                # Extrapolate if landmark beyond observed times
                prob = sf.y[-1]
            probs.append(prob)
        
        return np.array(probs)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = np.abs(self.model.coef_)
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'coefficient': self.model.coef_
        }).sort_values('importance', ascending=False)


class RSFResponseClassifier(BaseResponseClassifier):
    """
    Random Survival Forest with landmark probability extraction.
    """
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        time: Optional[np.ndarray] = None,
        event: Optional[np.ndarray] = None,
        **kwargs
    ) -> "RSFResponseClassifier":
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError:
            raise ImportError(
                "scikit-survival not installed. "
                "Install with: pip install scikit-survival"
            )
        
        if time is None or event is None:
            raise ValueError("RSF model requires time and event arrays")
        
        # Create structured array
        y_surv = np.array(
            [(bool(e), t) for e, t in zip(event, time)],
            dtype=[('event', bool), ('time', float)]
        )
        
        self.model = RandomSurvivalForest(
            **self.config.rsf_params,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y_surv)
        self.is_fitted = True
        self.feature_names = kwargs.get('feature_names')
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.config.probability_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get survival probability at landmark time."""
        surv_funcs = self.model.predict_survival_function(X)
        
        probs = []
        for sf in surv_funcs:
            landmark = self.config.landmark_days
            if landmark <= sf.x[-1]:
                idx = np.searchsorted(sf.x, landmark)
                prob = sf.y[min(idx, len(sf.y) - 1)]
            else:
                prob = sf.y[-1]
            probs.append(prob)
        
        return np.array(probs)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class EnsembleResponseClassifier(BaseResponseClassifier):
    """
    Ensemble classifier combining multiple models.
    
    Uses soft voting (probability averaging) by default.
    """
    
    def __init__(
        self, 
        config: ClassifierConfig,
        models: Optional[List[Tuple[str, BaseResponseClassifier]]] = None
    ):
        super().__init__(config)
        
        if models is None:
            # Default ensemble: Logistic + XGBoost + RF
            self.models = [
                ('logistic', LogisticResponseClassifier(config)),
                ('xgboost', XGBoostResponseClassifier(config)),
                ('rf', RandomForestResponseClassifier(config)),
            ]
        else:
            self.models = models
        
        self.weights: Optional[np.ndarray] = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        time: Optional[np.ndarray] = None,
        event: Optional[np.ndarray] = None,
        **kwargs
    ) -> "EnsembleResponseClassifier":
        """Fit all models in ensemble."""
        for name, model in self.models:
            logger.info(f"Fitting {name}...")
            
            if isinstance(model, (CoxResponseClassifier, RSFResponseClassifier)):
                model.fit(X, y, time=time, event=event, **kwargs)
            else:
                model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        self.feature_names = kwargs.get('feature_names')
        
        # Equal weights by default
        self.weights = np.ones(len(self.models)) / len(self.models)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.config.probability_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of model probabilities."""
        all_probs = []
        
        for name, model in self.models:
            probs = model.predict_proba(X)
            all_probs.append(probs)
        
        all_probs = np.array(all_probs)
        weighted_avg = np.average(all_probs, axis=0, weights=self.weights)
        
        return weighted_avg
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each model separately."""
        return {name: model.predict_proba(X) for name, model in self.models}
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Aggregate feature importance across models."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance_dfs = []
        for name, model in self.models:
            imp_df = model.get_feature_importance()
            if imp_df is not None:
                imp_df = imp_df.rename(columns={'importance': f'importance_{name}'})
                importance_dfs.append(imp_df[['feature', f'importance_{name}']])
        
        if not importance_dfs:
            return None
        
        # Merge all importance scores
        merged = importance_dfs[0]
        for df in importance_dfs[1:]:
            merged = pd.merge(merged, df, on='feature', how='outer')
        
        # Calculate mean importance
        imp_cols = [c for c in merged.columns if c.startswith('importance_')]
        merged['importance_mean'] = merged[imp_cols].mean(axis=1)
        merged = merged.sort_values('importance_mean', ascending=False)
        
        return merged


def create_classifier(config: ClassifierConfig) -> BaseResponseClassifier:
    """Factory function to create classifier by type."""
    model_map = {
        ModelType.LOGISTIC: LogisticResponseClassifier,
        ModelType.XGBOOST: XGBoostResponseClassifier,
        ModelType.RANDOM_FOREST: RandomForestResponseClassifier,
        ModelType.COX: CoxResponseClassifier,
        ModelType.RSF: RSFResponseClassifier,
        ModelType.ENSEMBLE: EnsembleResponseClassifier,
    }
    
    if config.model_type not in model_map:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model_map[config.model_type](config)
