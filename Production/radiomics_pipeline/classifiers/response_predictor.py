"""
Response prediction pipeline for PFS at landmark time.

Handles the full workflow:
1. Label generation from survival data
2. Censoring handling strategies
3. Model training with cross-validation
4. Proper evaluation metrics
5. Treatment-specific modeling
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

from .base import (
    ClassifierConfig,
    ClassificationResult,
    CensoringStrategy,
    ModelType,
    BaseResponseClassifier,
    create_classifier,
)

logger = logging.getLogger(__name__)


class ResponsePredictor:
    """
    Main class for response prediction at landmark time.
    
    Usage:
        predictor = ResponsePredictor(config)
        result = predictor.fit_evaluate(feature_set)
        
        # Or with treatment-specific models
        result = predictor.fit_evaluate_by_treatment(
            feature_set, 
            treatment_column='treatment_arm'
        )
    """
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.classifier: Optional[BaseResponseClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.treatment_classifiers: Dict[str, BaseResponseClassifier] = {}
        
        # Store metadata
        self.feature_names: Optional[List[str]] = None
        self.n_censored_excluded: int = 0
    
    def prepare_labels(
        self,
        time: np.ndarray,
        event: np.ndarray,
        landmark_days: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binary response labels from survival data.
        
        Responder: PFS >= landmark_days
        Non-Responder: PFS < landmark_days (and had event)
        
        Args:
            time: Time to event/censoring
            event: Event indicator (1=event, 0=censored)
            landmark_days: Classification threshold (default from config)
            
        Returns:
            Tuple of (labels, mask) where mask indicates valid samples
        """
        landmark = landmark_days or self.config.landmark_days
        n_samples = len(time)
        
        labels = np.zeros(n_samples, dtype=int)
        valid_mask = np.ones(n_samples, dtype=bool)
        
        for i in range(n_samples):
            t, e = time[i], event[i]
            
            if t >= landmark:
                # Survived beyond landmark = Responder
                labels[i] = 1
            elif e == 1:
                # Event before landmark = Non-Responder
                labels[i] = 0
            else:
                # Censored before landmark - handle based on strategy
                if self.config.censoring_strategy == CensoringStrategy.EXCLUDE:
                    valid_mask[i] = False
                elif self.config.censoring_strategy == CensoringStrategy.TREAT_AS_RESPONDER:
                    labels[i] = 1
                elif self.config.censoring_strategy == CensoringStrategy.IPCW:
                    # IPCW handled separately in fit
                    valid_mask[i] = False  # Exclude for now, weight in training
                else:
                    valid_mask[i] = False
        
        self.n_censored_excluded = (~valid_mask).sum()
        
        if self.n_censored_excluded > 0:
            logger.info(
                f"Censoring: {self.n_censored_excluded}/{n_samples} patients "
                f"excluded (censored before {landmark} days)"
            )
        
        return labels, valid_mask
    
    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True
    ) -> "ResponsePredictor":
        """
        Fit the response predictor.
        
        Args:
            X: Feature matrix
            time: Time to event/censoring
            event: Event indicator
            feature_names: Optional feature names
            scale_features: Whether to standardize features
        """
        self.feature_names = feature_names
        
        # Generate labels
        y, valid_mask = self.prepare_labels(time, event)
        
        # Filter to valid samples
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        time_valid = time[valid_mask]
        event_valid = event[valid_mask]
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X_valid = self.scaler.fit_transform(X_valid)
        
        # Create and fit classifier
        self.classifier = create_classifier(self.config)
        
        if self.config.model_type in [ModelType.COX, ModelType.RSF]:
            self.classifier.fit(
                X_valid, y_valid, 
                time=time_valid, 
                event=event_valid,
                feature_names=feature_names
            )
        else:
            self.classifier.fit(X_valid, y_valid, feature_names=feature_names)
        
        logger.info(
            f"Fitted {self.config.model_type.value} classifier "
            f"on {len(y_valid)} samples"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary response."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of being responder."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.classifier.predict_proba(X)
    
    def fit_evaluate(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        time_test: Optional[np.ndarray] = None,
        event_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> ClassificationResult:
        """
        Fit model and evaluate performance.
        
        If test data not provided, performs train/test split.
        """
        # Generate labels
        y, valid_mask = self.prepare_labels(time, event)
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        time_valid = time[valid_mask]
        event_valid = event[valid_mask]
        
        # Train/test split if test data not provided
        if X_test is None:
            (X_train, X_test, y_train, y_test, 
             time_train, time_test, event_train, event_test) = train_test_split(
                X_valid, y_valid, time_valid, event_valid,
                test_size=test_size,
                random_state=self.config.random_state,
                stratify=y_valid
            )
        else:
            X_train, y_train = X_valid, y_valid
            time_train, event_train = time_valid, event_valid
            
            # Process test labels
            y_test, test_valid_mask = self.prepare_labels(time_test, event_test)
            X_test = X_test[test_valid_mask]
            y_test = y_test[test_valid_mask]
            time_test = time_test[test_valid_mask]
            event_test = event_test[test_valid_mask]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit classifier
        self.classifier = create_classifier(self.config)
        self.feature_names = feature_names
        
        if self.config.model_type in [ModelType.COX, ModelType.RSF]:
            self.classifier.fit(
                X_train_scaled, y_train,
                time=time_train, event=event_train,
                feature_names=feature_names
            )
        else:
            self.classifier.fit(X_train_scaled, y_train, feature_names=feature_names)
        
        # Predict
        y_pred = self.classifier.predict(X_test_scaled)
        y_prob = self.classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        # Cross-validation scores
        cv_scores = self._cross_validate(
            X_train_scaled, y_train, time_train, event_train, feature_names
        )
        
        return ClassificationResult(
            y_pred=y_pred,
            y_prob=y_prob,
            y_true=y_test,
            metrics=metrics,
            model_type=self.config.model_type.value,
            feature_importance=self.classifier.get_feature_importance(),
            cv_scores=cv_scores,
            n_train=len(y_train),
            n_test=len(y_test),
            n_censored_excluded=self.n_censored_excluded,
            landmark_days=self.config.landmark_days,
        )
    
    def fit_evaluate_by_treatment(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        treatment: np.ndarray,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Dict[str, ClassificationResult]:
        """
        Fit and evaluate separate models for each treatment arm.
        
        Useful for identifying treatment-specific predictive features.
        """
        results = {}
        treatment_values = np.unique(treatment)
        
        for treat in treatment_values:
            treat_mask = treatment == treat
            
            logger.info(f"Training model for treatment: {treat}")
            
            result = self.fit_evaluate(
                X[treat_mask],
                time[treat_mask],
                event[treat_mask],
                feature_names=feature_names,
                test_size=test_size
            )
            
            results[str(treat)] = result
            
            # Store classifier
            self.treatment_classifiers[str(treat)] = self.classifier
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['specificity'] = recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        )
        
        # Probabilistic metrics
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auroc'] = np.nan
        
        try:
            metrics['auprc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auprc'] = np.nan
        
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # Prevalence
        metrics['prevalence'] = y_true.mean()
        
        return metrics
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, List[float]]:
        """Perform stratified cross-validation."""
        cv_scores = {
            'accuracy': [],
            'auroc': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }
        
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            time_train, time_val = time[train_idx], time[val_idx]
            event_train, event_val = event[train_idx], event[val_idx]
            
            # Fit classifier
            classifier = create_classifier(self.config)
            
            if self.config.model_type in [ModelType.COX, ModelType.RSF]:
                classifier.fit(
                    X_train, y_train,
                    time=time_train, event=event_train,
                    feature_names=feature_names
                )
            else:
                classifier.fit(X_train, y_train, feature_names=feature_names)
            
            # Evaluate
            y_pred = classifier.predict(X_val)
            y_prob = classifier.predict_proba(X_val)
            
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            
            try:
                cv_scores['auroc'].append(roc_auc_score(y_val, y_prob))
            except ValueError:
                cv_scores['auroc'].append(np.nan)
        
        # Log CV results
        logger.info(
            f"CV Results ({self.config.n_folds}-fold): "
            f"AUROC={np.nanmean(cv_scores['auroc']):.3f}±{np.nanstd(cv_scores['auroc']):.3f}, "
            f"F1={np.mean(cv_scores['f1']):.3f}±{np.std(cv_scores['f1']):.3f}"
        )
        
        return cv_scores
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from fitted classifier."""
        if self.classifier is None:
            return None
        return self.classifier.get_feature_importance()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ax=None
    ):
        """Plot ROC curve."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, label=f'{self.config.model_type.value} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - PFS@{self.config.landmark_days}d Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        ax=None
    ):
        """Plot calibration curve."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(prob_pred, prob_true, marker='o', label=self.config.model_type.value)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


class ModelComparison:
    """
    Compare multiple model types on the same data.
    
    Usage:
        comparison = ModelComparison(landmark_days=180)
        results = comparison.compare(X, time, event)
        comparison.plot_comparison()
    """
    
    def __init__(
        self,
        landmark_days: int = 180,
        model_types: Optional[List[ModelType]] = None,
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.landmark_days = landmark_days
        self.model_types = model_types or [
            ModelType.LOGISTIC,
            ModelType.XGBOOST,
            ModelType.RANDOM_FOREST,
            ModelType.COX,
            ModelType.RSF,
            ModelType.ENSEMBLE,
        ]
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.results: Dict[str, ClassificationResult] = {}
    
    def compare(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Compare all model types.
        
        Returns DataFrame with metrics for each model.
        """
        for model_type in self.model_types:
            logger.info(f"Evaluating {model_type.value}...")
            
            config = ClassifierConfig(
                model_type=model_type,
                landmark_days=self.landmark_days,
                n_folds=self.n_folds,
                random_state=self.random_state
            )
            
            predictor = ResponsePredictor(config)
            
            try:
                result = predictor.fit_evaluate(
                    X, time, event,
                    feature_names=feature_names,
                    test_size=test_size
                )
                self.results[model_type.value] = result
            except Exception as e:
                logger.error(f"Failed for {model_type.value}: {e}")
        
        return self.get_comparison_table()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table of all models."""
        rows = []
        
        for model_name, result in self.results.items():
            row = {
                'model': model_name,
                **result.metrics
            }
            
            # Add CV metrics if available
            if result.cv_scores:
                for metric, scores in result.cv_scores.items():
                    row[f'cv_{metric}_mean'] = np.nanmean(scores)
                    row[f'cv_{metric}_std'] = np.nanstd(scores)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('auroc', ascending=False)
        
        return df
    
    def plot_comparison(self, metric: str = 'auroc'):
        """Plot comparison of models."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        df = self.get_comparison_table()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = df['model'].values
        values = df[metric].values
        
        bars = ax.barh(models, values)
        ax.set_xlabel(metric.upper())
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_comparison(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ):
        """Plot ROC curves for all models."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(result.y_true, result.y_prob)
            auc = result.metrics['auroc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - PFS@{self.landmark_days}d Response')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
