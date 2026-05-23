"""
Core15+ Tabular Feature Attribution
Provides Integrated Gradients for neural networks and SHAP for tree-based models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

@dataclass
class TabularAttributionConfig:
    """Configuration for tabular feature attribution"""
    use_integrated_gradients: bool = True
    use_shap: bool = True
    ig_steps: int = 50
    ig_baseline: str = "zero"  # "zero", "mean", "median"
    shap_explainer: str = "auto"  # "auto", "tree", "kernel", "deep", "linear"
    shap_background_samples: int = 100
    feature_importance_threshold: float = 0.01
    normalize_attributions: bool = True
    save_individual_attributions: bool = True

class IntegratedGradients:
    """Integrated Gradients implementation for neural networks"""

    def __init__(self, model: torch.nn.Module, config: TabularAttributionConfig):
        self.model = model
        self.config = config

    def compute_integrated_gradients(self, input_tensor: torch.Tensor,
                                   target_class: int = None,
                                   baseline: torch.Tensor = None) -> torch.Tensor:
        """Compute Integrated Gradients attribution"""
        if baseline is None:
            baseline = self._get_baseline(input_tensor)

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, self.config.ig_steps, device=input_tensor.device)
        interpolated_inputs = []

        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)

        # Stack interpolated inputs
        interpolated_batch = torch.stack(interpolated_inputs)

        # Compute gradients for each interpolated input
        gradients = self._compute_gradients(interpolated_batch, target_class)

        # Integrate gradients (trapezoidal rule)
        integrated_gradients = torch.mean(gradients, dim=0) * (input_tensor - baseline)

        return integrated_gradients

    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get baseline for integrated gradients"""
        if self.config.ig_baseline == "zero":
            return torch.zeros_like(input_tensor)
        elif self.config.ig_baseline == "mean":
            # This would require training data statistics
            # For now, return zeros as fallback
            return torch.zeros_like(input_tensor)
        elif self.config.ig_baseline == "median":
            # This would require training data statistics
            # For now, return zeros as fallback
            return torch.zeros_like(input_tensor)
        else:
            return torch.zeros_like(input_tensor)

    def _compute_gradients(self, interpolated_batch: torch.Tensor,
                          target_class: int = None) -> torch.Tensor:
        """Compute gradients for interpolated inputs"""
        interpolated_batch.requires_grad_(True)

        # Forward pass
        outputs = self.model(interpolated_batch)

        if target_class is None:
            target_class = outputs.argmax(dim=1)[0].item()

        # Backward pass
        gradients_list = []
        for i in range(interpolated_batch.shape[0]):
            self.model.zero_grad()
            outputs[i, target_class].backward(retain_graph=True)
            gradients_list.append(interpolated_batch.grad[i].clone())

        return torch.stack(gradients_list)

class SHAPExplainer:
    """SHAP-based explainer for tree models and others"""

    def __init__(self, model: BaseEstimator, X_background: np.ndarray,
                 config: TabularAttributionConfig):
        self.model = model
        self.X_background = X_background
        self.config = config
        self.explainer = None

        if SHAP_AVAILABLE:
            self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP explanations")

        # Auto-select explainer based on model type
        if self.config.shap_explainer == "auto":
            if isinstance(self.model, (RandomForestClassifier,)):
                self.explainer = shap.TreeExplainer(self.model)
            elif isinstance(self.model, LogisticRegression):
                self.explainer = shap.LinearExplainer(self.model, self.X_background)
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self.X_background[:self.config.shap_background_samples]
                )
        elif self.config.shap_explainer == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.config.shap_explainer == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.X_background)
        elif self.config.shap_explainer == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.X_background[:self.config.shap_background_samples]
            )

    def explain(self, X: np.ndarray) -> shap.Explanation:
        """Generate SHAP explanations"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP explanations")

        return self.explainer(X)

class Core15Explainer:
    """High-level explainer for Core15+ features"""

    def __init__(self, model_path: str, feature_names: List[str],
                 config: TabularAttributionConfig = None):
        self.model_path = model_path
        self.feature_names = feature_names
        self.config = config or TabularAttributionConfig()
        self.model = None
        self.model_type = None
        self.X_background = None

    def load_model(self):
        """Load Core15+ model"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)

            # Determine model type
            if isinstance(self.model, LogisticRegression):
                self.model_type = "logistic"
            elif isinstance(self.model, RandomForestClassifier):
                self.model_type = "random_forest"
            elif hasattr(self.model, 'predict'):
                self.model_type = "sklearn"
            else:
                # Assume PyTorch model
                self.model_type = "pytorch"
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                # Would need to reconstruct PyTorch model here

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def set_background_data(self, X_background: np.ndarray):
        """Set background data for SHAP explanations"""
        self.X_background = X_background

    def explain_prediction(self, features: np.ndarray, subject_id: str,
                         feature_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive feature attribution explanation"""
        if self.model is None:
            self.load_model()

        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction
        if self.model_type == "pytorch":
            prediction, probabilities = self._predict_pytorch(features)
        else:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

        predicted_class = prediction if isinstance(prediction, int) else int(prediction)
        confidence = probabilities.max()

        # Compute attributions
        attributions = {}

        # Integrated Gradients (for neural networks)
        if self.config.use_integrated_gradients and self.model_type == "pytorch":
            ig_explainer = IntegratedGradients(self.model, self.config)
            input_tensor = torch.from_numpy(features).float()
            ig_attributions = ig_explainer.compute_integrated_gradients(input_tensor, predicted_class)
            attributions['integrated_gradients'] = ig_attributions.cpu().numpy()

        # SHAP explanations
        if self.config.use_shap and SHAP_AVAILABLE and self.X_background is not None:
            try:
                shap_explainer = SHAPExplainer(self.model, self.X_background, self.config)
                shap_values = shap_explainer.explain(features)

                # Extract SHAP values for predicted class
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:  # Multi-class
                        attributions['shap'] = shap_values.values[0, :, predicted_class]
                    else:  # Binary classification
                        attributions['shap'] = shap_values.values[0]
                else:
                    attributions['shap'] = shap_values[0]

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # Analyze feature importance
        feature_importance = self._analyze_feature_importance(attributions, features[0])

        # Generate data hash
        data_hash = hashlib.sha256(features.tobytes()).hexdigest()[:16]

        explanation = {
            'subject_id': subject_id,
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist()
            },
            'attributions': {
                'feature_attributions': attributions,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'feature_values': features[0].tolist()
            },
            'metadata': {
                'model_path': self.model_path,
                'model_type': self.model_type,
                'data_hash': data_hash,
                'use_integrated_gradients': self.config.use_integrated_gradients,
                'use_shap': self.config.use_shap,
                'feature_metadata': feature_metadata or {}
            }
        }

        return explanation

    def _predict_pytorch(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Make prediction with PyTorch model"""
        input_tensor = torch.from_numpy(features).float()

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()

        return predicted_class, probabilities.cpu().numpy()[0]

    def _analyze_feature_importance(self, attributions: Dict[str, np.ndarray],
                                  feature_values: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance across different attribution methods"""
        importance_analysis = {}

        for method, attr_values in attributions.items():
            # Absolute importance
            abs_importance = np.abs(attr_values)

            # Top features
            top_indices = np.argsort(abs_importance)[::-1]
            top_k = min(10, len(top_indices))

            # Feature statistics
            stats = {
                'top_features': [
                    {
                        'feature_name': self.feature_names[i],
                        'feature_index': int(i),
                        'attribution_value': float(attr_values[i]),
                        'feature_value': float(feature_values[i]),
                        'abs_importance': float(abs_importance[i])
                    }
                    for i in top_indices[:top_k]
                ],
                'total_positive_attribution': float(attr_values[attr_values > 0].sum()),
                'total_negative_attribution': float(attr_values[attr_values < 0].sum()),
                'attribution_magnitude': float(abs_importance.sum()),
                'attribution_variance': float(attr_values.var()),
                'sparsity': float((abs_importance < self.config.feature_importance_threshold).mean())
            }

            importance_analysis[method] = stats

        # Cross-method comparison if multiple methods available
        if len(attributions) > 1:
            importance_analysis['cross_method'] = self._compare_attribution_methods(attributions)

        return importance_analysis

    def _compare_attribution_methods(self, attributions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compare different attribution methods"""
        methods = list(attributions.keys())
        comparisons = {}

        if len(methods) >= 2:
            # Rank correlation between methods
            from scipy.stats import spearmanr

            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    abs_attr1 = np.abs(attributions[method1])
                    abs_attr2 = np.abs(attributions[method2])

                    correlation, p_value = spearmanr(abs_attr1, abs_attr2)
                    comparisons[f"{method1}_vs_{method2}_correlation"] = float(correlation)
                    comparisons[f"{method1}_vs_{method2}_pvalue"] = float(p_value)

        return comparisons

    def visualize_attributions(self, explanation: Dict[str, Any],
                             save_path: Optional[Path] = None) -> plt.Figure:
        """Visualize feature attributions"""
        attributions = explanation['attributions']['feature_attributions']
        feature_names = explanation['attributions']['feature_names']

        n_methods = len(attributions)
        fig, axes = plt.subplots(n_methods, 2, figsize=(15, 6 * n_methods))

        if n_methods == 1:
            axes = axes.reshape(1, -1)

        for i, (method, attr_values) in enumerate(attributions.items()):
            # Attribution bar plot
            top_indices = np.argsort(np.abs(attr_values))[-10:]
            top_values = attr_values[top_indices]
            top_names = [feature_names[j] for j in top_indices]

            bars = axes[i, 0].barh(range(len(top_values)), top_values,
                                  color=['red' if v < 0 else 'blue' for v in top_values])
            axes[i, 0].set_yticks(range(len(top_values)))
            axes[i, 0].set_yticklabels(top_names)
            axes[i, 0].set_title(f'{method.title()} - Top Feature Attributions')
            axes[i, 0].set_xlabel('Attribution Value')

            # Attribution distribution
            axes[i, 1].hist(attr_values, bins=30, alpha=0.7, edgecolor='black')
            axes[i, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[i, 1].set_title(f'{method.title()} - Attribution Distribution')
            axes[i, 1].set_xlabel('Attribution Value')
            axes[i, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def batch_explain(self, features_batch: List[Tuple[np.ndarray, str]],
                     feature_metadata_batch: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate explanations for batch of feature sets"""
        explanations = []

        if feature_metadata_batch is None:
            feature_metadata_batch = [{}] * len(features_batch)

        for i, (features, subject_id) in enumerate(features_batch):
            try:
                explanation = self.explain_prediction(
                    features, subject_id, feature_metadata_batch[i]
                )
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain Core15+ prediction for {subject_id}: {e}")
                explanations.append({
                    'subject_id': subject_id,
                    'error': str(e)
                })

        return explanations