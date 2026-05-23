"""
External Validation Metrics and Statistical Analysis
Comprehensive evaluation suite for external validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedGroupKFold
import scipy.stats as stats

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive set of classification metrics.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        y_prob: Predicted probabilities for positive class

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Core classification metrics
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['sensitivity'] = recall_score(y_true, y_pred)  # TPR
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0)  # TNR
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)

    # ROC and PR metrics
    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    metrics['pr_auc'] = average_precision_score(y_true, y_prob)

    # Calibration metrics
    metrics['brier_score'] = brier_score_loss(y_true, y_prob)

    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    # Additional derived metrics
    metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Sample sizes
    metrics['n_total'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true))
    metrics['n_negative'] = int(len(y_true) - np.sum(y_true))

    return convert_numpy_types(metrics)

def permutation_test(y_true: np.ndarray, y_prob: np.ndarray,
                    n_perm: int = 1000, rng: np.random.Generator = None,
                    metric: str = 'balanced_accuracy') -> Dict[str, Any]:
    """
    Permutation test for statistical significance.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_perm: Number of permutations
        rng: Random number generator
        metric: Metric to test ('balanced_accuracy' or 'roc_auc')

    Returns:
        Dictionary with permutation test results
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Compute observed score
    y_pred = (y_prob >= 0.5).astype(int)
    if metric == 'balanced_accuracy':
        observed_score = balanced_accuracy_score(y_true, y_pred)
    elif metric == 'roc_auc':
        observed_score = roc_auc_score(y_true, y_prob)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Permutation scores
    perm_scores = []
    for i in range(n_perm):
        y_perm = rng.permutation(y_true)
        if metric == 'balanced_accuracy':
            perm_score = balanced_accuracy_score(y_perm, y_pred)
        else:  # roc_auc
            perm_score = roc_auc_score(y_perm, y_prob)
        perm_scores.append(perm_score)

    perm_scores = np.array(perm_scores)

    # Calculate p-value (two-tailed)
    p_value = (np.sum(np.abs(perm_scores - np.mean(perm_scores)) >=
                     np.abs(observed_score - np.mean(perm_scores))) + 1) / (n_perm + 1)

    results = {
        'observed_score': float(observed_score),
        'permutation_mean': float(np.mean(perm_scores)),
        'permutation_std': float(np.std(perm_scores)),
        'p_value': float(p_value),
        'n_permutations': n_perm,
        'metric': metric,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01
    }

    return convert_numpy_types(results)

def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                n_boot: int = 1000, rng: np.random.Generator = None,
                confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Bootstrap confidence intervals for performance metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_boot: Number of bootstrap samples
        rng: Random number generator
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Dictionary with bootstrap confidence intervals
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(y_true)
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    # Bootstrap samples
    ba_scores = []
    auc_scores = []

    for i in range(n_boot):
        # Sample with replacement
        boot_indices = rng.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[boot_indices]
        y_prob_boot = y_prob[boot_indices]
        y_pred_boot = (y_prob_boot >= 0.5).astype(int)

        # Compute metrics
        try:
            ba_boot = balanced_accuracy_score(y_true_boot, y_pred_boot)
            auc_boot = roc_auc_score(y_true_boot, y_prob_boot)
            ba_scores.append(ba_boot)
            auc_scores.append(auc_boot)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    ba_scores = np.array(ba_scores)
    auc_scores = np.array(auc_scores)

    results = {
        'balanced_accuracy': {
            'mean': float(np.mean(ba_scores)),
            'std': float(np.std(ba_scores)),
            'ci_lower': float(np.percentile(ba_scores, lower_percentile)),
            'ci_upper': float(np.percentile(ba_scores, upper_percentile)),
        },
        'roc_auc': {
            'mean': float(np.mean(auc_scores)),
            'std': float(np.std(auc_scores)),
            'ci_lower': float(np.percentile(auc_scores, lower_percentile)),
            'ci_upper': float(np.percentile(auc_scores, upper_percentile)),
        },
        'confidence_level': confidence_level,
        'n_bootstrap': len(ba_scores)
    }

    return convert_numpy_types(results)

def calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value)
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    return fraction_of_positives, mean_predicted_value

def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                     n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Expected calibration error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

def feature_shift_report(X_external: pd.DataFrame,
                        training_stats: Dict[str, Dict[str, float]],
                        top_k: int = 10) -> Dict[str, Any]:
    """
    Analyze feature distribution shift between training and external data.

    Args:
        X_external: External dataset features
        training_stats: Training feature statistics
        top_k: Number of top features to analyze

    Returns:
        Feature shift analysis report
    """
    shift_metrics = {}

    # Get top features by importance (could be from feature_list order)
    feature_names = list(training_stats.keys())[:top_k]

    for feature in feature_names:
        if feature not in X_external.columns:
            continue

        external_data = X_external[feature].values
        train_stats = training_stats[feature]

        # Compute external statistics
        ext_mean = float(np.mean(external_data))
        ext_std = float(np.std(external_data))
        ext_min = float(np.min(external_data))
        ext_max = float(np.max(external_data))

        # Standardized mean difference (Cohen's d equivalent)
        mean_diff = (ext_mean - train_stats['mean']) / train_stats['std']

        # Variance ratio
        var_ratio = (ext_std ** 2) / (train_stats['std'] ** 2)

        # KS test would require original training data - approximate with range check
        range_overlap = (
            max(train_stats['min'], ext_min) < min(train_stats['max'], ext_max)
        )

        shift_metrics[feature] = {
            'training_mean': train_stats['mean'],
            'training_std': train_stats['std'],
            'external_mean': ext_mean,
            'external_std': ext_std,
            'standardized_mean_diff': float(mean_diff),
            'variance_ratio': float(var_ratio),
            'range_overlap': bool(range_overlap),
            'shift_magnitude': float(abs(mean_diff))  # Simple shift indicator
        }

    # Summary metrics
    all_shifts = [metrics['shift_magnitude'] for metrics in shift_metrics.values()]
    summary = {
        'mean_shift': float(np.mean(all_shifts)),
        'max_shift': float(np.max(all_shifts)),
        'n_features_analyzed': len(shift_metrics),
        'features_with_large_shift': sum(1 for s in all_shifts if s > 0.5),
        'features_with_extreme_shift': sum(1 for s in all_shifts if s > 1.0)
    }

    return convert_numpy_types({
        'feature_shifts': shift_metrics,
        'summary': summary
    })