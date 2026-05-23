"""
Statistical validation utilities for Phase III
Permutation tests and bootstrap confidence intervals
"""

import numpy as np
import logging
from typing import Callable, List, Tuple, Dict, Optional
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def permutation_test(y_true: np.ndarray,
                    y_pred_fn: Callable,
                    X: np.ndarray,
                    groups: Optional[np.ndarray] = None,
                    n_permutations: int = 1000,
                    metric: Callable = balanced_accuracy_score,
                    random_state: int = 42) -> Dict:
    """
    Permutation test for statistical significance

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_fn : Callable
        Function that takes X and returns predictions
    X : np.ndarray
        Feature matrix
    groups : np.ndarray, optional
        Group labels for grouped permutation (e.g., subjects)
    n_permutations : int
        Number of permutation iterations
    metric : Callable
        Metric function (default: balanced_accuracy_score)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    results : dict
        Permutation test results including p-value
    """
    logger.info(f"Running permutation test with {n_permutations} iterations")

    # Compute true score
    y_pred = y_pred_fn(X)
    true_score = metric(y_true, y_pred)

    logger.info(f"True score: {true_score:.4f}")

    # Permutation testing
    np.random.seed(random_state)
    permutation_scores = []

    for i in range(n_permutations):
        if i % 100 == 0 and i > 0:
            logger.debug(f"Permutation {i}/{n_permutations}")

        # Shuffle labels
        if groups is not None:
            # Subject-wise permutation: shuffle labels within groups
            y_perm = _grouped_permutation(y_true, groups, random_state + i)
        else:
            # Simple permutation
            y_perm = shuffle(y_true, random_state=random_state + i)

        # Compute score with permuted labels
        try:
            y_pred_perm = y_pred_fn(X)  # Model doesn't change, just labels
            perm_score = metric(y_perm, y_pred_perm)
            permutation_scores.append(perm_score)
        except Exception as e:
            logger.warning(f"Permutation {i} failed: {e}")
            continue

    permutation_scores = np.array(permutation_scores)

    # Compute p-value
    p_value = float(np.sum(permutation_scores >= true_score) / len(permutation_scores))

    # Summary statistics
    perm_mean = float(np.mean(permutation_scores))
    perm_std = float(np.std(permutation_scores, ddof=1))

    results = {
        'true_score': float(true_score),
        'permutation_scores': permutation_scores.tolist(),
        'p_value': p_value,
        'null_mean': perm_mean,
        'null_std': perm_std,
        'n_permutations': len(permutation_scores),
        'metric_name': metric.__name__ if hasattr(metric, '__name__') else 'custom_metric'
    }

    logger.info(f"Permutation test completed: p = {p_value:.4f}, "
               f"null = {perm_mean:.3f} ± {perm_std:.3f}")

    return results

def bootstrap_confidence_interval(scores: np.ndarray,
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 1000,
                                random_state: int = 42) -> Dict:
    """
    Bootstrap confidence interval for performance metrics

    Parameters:
    -----------
    scores : np.ndarray
        Array of metric scores (e.g., from cross-validation)
    confidence_level : float
        Confidence level (default: 0.95 for 95% CI)
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed

    Returns:
    --------
    results : dict
        Bootstrap CI results
    """
    logger.info(f"Computing {confidence_level:.0%} bootstrap CI with {n_bootstrap} samples")

    np.random.seed(random_state)
    n_scores = len(scores)

    bootstrap_means = []
    for i in range(n_bootstrap):
        # Bootstrap sample with replacement
        bootstrap_sample = np.random.choice(scores, size=n_scores, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_means, upper_percentile))

    results = {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores, ddof=1)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'bootstrap_means': bootstrap_means.tolist()
    }

    logger.info(f"Bootstrap CI: {ci_lower:.3f} - {ci_upper:.3f} "
               f"(mean: {results['mean']:.3f})")

    return results

def grouped_cross_validation_scores(X: np.ndarray,
                                   y: np.ndarray,
                                   groups: np.ndarray,
                                   model,
                                   cv,
                                   scoring: str = 'balanced_accuracy') -> np.ndarray:
    """
    Compute cross-validation scores with group structure preserved

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    groups : np.ndarray
        Group labels (e.g., subject IDs)
    model : sklearn estimator
        Model to evaluate
    cv : sklearn cross-validator
        Cross-validation strategy
    scoring : str
        Scoring metric

    Returns:
    --------
    scores : np.ndarray
        Cross-validation scores
    """
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)
    scores = []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute score
        score = scorer._score_func(y_test, y_pred)
        scores.append(score)

    return np.array(scores)

def statistical_validation_report(scores: np.ndarray,
                                 permutation_results: Dict,
                                 confidence_level: float = 0.95,
                                 significance_threshold: float = 0.05) -> Dict:
    """
    Generate comprehensive statistical validation report

    Parameters:
    -----------
    scores : np.ndarray
        Performance scores
    permutation_results : dict
        Results from permutation test
    confidence_level : float
        Confidence level for intervals
    significance_threshold : float
        Significance threshold (alpha)

    Returns:
    --------
    report : dict
        Comprehensive validation report
    """
    # Bootstrap confidence interval
    bootstrap_results = bootstrap_confidence_interval(scores, confidence_level)

    # Statistical significance
    is_significant = permutation_results['p_value'] < significance_threshold

    # Effect size (Cohen's d against null hypothesis)
    null_mean = permutation_results['null_mean']
    null_std = permutation_results['null_std']
    observed_mean = float(np.mean(scores))

    if null_std > 0:
        cohens_d = (observed_mean - null_mean) / null_std
    else:
        cohens_d = float('inf') if observed_mean > null_mean else 0.0

    report = {
        'performance_summary': {
            'mean_score': observed_mean,
            'std_score': float(np.std(scores, ddof=1)),
            'n_folds': len(scores)
        },
        'confidence_interval': {
            'ci_lower': bootstrap_results['ci_lower'],
            'ci_upper': bootstrap_results['ci_upper'],
            'confidence_level': confidence_level,
            'width': bootstrap_results['ci_upper'] - bootstrap_results['ci_lower']
        },
        'statistical_significance': {
            'p_value': permutation_results['p_value'],
            'is_significant': is_significant,
            'alpha': significance_threshold,
            'n_permutations': permutation_results['n_permutations']
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'magnitude': _interpret_cohens_d(cohens_d),
            'null_mean': null_mean,
            'null_std': null_std
        },
        'clinical_relevance': {
            'exceeds_chance': observed_mean > 0.5,
            'clinically_meaningful': observed_mean >= 0.6,  # Clinical threshold
            'regulatory_grade': (is_significant and
                               bootstrap_results['ci_lower'] >= 0.55)
        }
    }

    return report

def _grouped_permutation(y: np.ndarray, groups: np.ndarray, random_state: int) -> np.ndarray:
    """Permute labels while preserving group structure"""
    np.random.seed(random_state)

    unique_groups = np.unique(groups)
    y_permuted = y.copy()

    # Shuffle group assignments
    shuffled_groups = np.random.permutation(unique_groups)

    for orig_group, new_group in zip(unique_groups, shuffled_groups):
        orig_mask = groups == orig_group
        new_mask = groups == new_group

        # Swap labels between groups
        y_permuted[orig_mask] = y[new_mask]

    return y_permuted

def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"