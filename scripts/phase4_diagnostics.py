#!/usr/bin/env python3
"""
Phase 4 Diagnostics and Performance Enhancement
Addresses poor model performance through feature analysis and optimization.

Based on CV results showing ~50% BA (chance level), this script:
1. Analyzes feature collinearity and removes redundant features
2. Tests alpha-only core feature sets vs full feature set
3. Optimizes class balance and decision thresholds
4. Provides regulatory-grade provenance tracking
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import ks_2samp

# ML imports
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, precision_recall_curve, classification_report,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.ml_pipeline import RegulatoryMLPipeline
from src.utils.logger import setup_logger

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "results" / "features"
RESULTS_DIR = BASE_DIR / "results" / "stats"
FIGURES_DIR = BASE_DIR / "results" / "figures" / "05_diagnostics"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Setup logger
logger = setup_logger(__name__, BASE_DIR / "logs" / "phase4_diagnostics.log")

def save_provenance(timestamp: str) -> None:
    """Save regulatory-grade provenance information."""
    logger.info("Saving provenance information...")

    import subprocess
    import platform

    provenance = {
        "timestamp": timestamp,
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine()
        },
        "random_seed": RANDOM_SEED,
        "data_source": "Phase 3 feature extraction results",
        "analysis_purpose": "Performance diagnostics and enhancement"
    }

    # Get pip freeze output
    try:
        pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
        with open(RESULTS_DIR / f"phase4_provenance_{timestamp}.txt", "w") as f:
            f.write(f"# Phase 4 Diagnostics Provenance\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Random seed: {RANDOM_SEED}\n\n")
            f.write(pip_freeze)

        logger.info(f"Provenance saved to phase4_provenance_{timestamp}.txt")

    except Exception as e:
        logger.warning(f"Could not save pip freeze: {e}")

def load_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Load and prepare data for diagnostics."""
    logger.info("Loading data for diagnostics...")

    # Load CSV data
    df = pd.read_csv(DATA_DIR / "feature_matrix.csv")

    # Extract components
    subjects = df['subject_id'].values
    labels = df['condition'].values

    # Get feature columns
    metadata_cols = ['subject_id', 'condition', 'session_number']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols].values
    feature_names = feature_cols

    logger.info(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(subjects))} subjects")

    return df, X, labels, subjects, feature_names

def analyze_feature_collinearity(X: np.ndarray, feature_names: List[str],
                                threshold: float = 0.95) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Remove highly collinear features."""
    logger.info(f"Analyzing feature collinearity (threshold: {threshold})...")

    # Calculate correlation matrix
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    for i, j in zip(*np.where(upper_triangle & (np.abs(corr_matrix) > threshold))):
        high_corr_pairs.append({
            'feature1': corr_matrix.columns[i],
            'feature2': corr_matrix.columns[j],
            'correlation': corr_matrix.iloc[i, j]
        })

    # Remove redundant features (keep first of each pair)
    features_to_remove = set()
    for pair in high_corr_pairs:
        features_to_remove.add(pair['feature2'])

    # Keep non-redundant features
    features_to_keep = [f for f in feature_names if f not in features_to_remove]
    feature_indices = [i for i, f in enumerate(feature_names) if f in features_to_keep]

    X_reduced = X[:, feature_indices]

    logger.info(f"Removed {len(features_to_remove)} redundant features")
    logger.info(f"Remaining: {len(features_to_keep)} features")

    return X_reduced, features_to_keep, pd.DataFrame(high_corr_pairs)

def create_alpha_core_features(feature_names: List[str]) -> List[int]:
    """Select core alpha-band features for focused analysis."""
    logger.info("Creating alpha-core feature set...")

    alpha_keywords = [
        'alpha_power', 'individual_alpha_freq', 'alpha_beta_ratio',
        'motor_alpha', 'posterior_alpha', 'frontal_alpha',
        'alpha_asymmetry', 'alpha_peak_freq', 'alpha_bandwidth'
    ]

    alpha_indices = []
    alpha_features = []

    for i, feature in enumerate(feature_names):
        if any(keyword in feature.lower() for keyword in alpha_keywords):
            alpha_indices.append(i)
            alpha_features.append(feature)

    logger.info(f"Selected {len(alpha_indices)} alpha-core features:")
    for feature in alpha_features[:10]:  # Log first 10
        logger.info(f"  - {feature}")

    return alpha_indices

def optimize_svm_parameters(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
    """Optimize SVM with wider parameter ranges."""
    logger.info("Optimizing SVM parameters with wider ranges...")

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    # Wider parameter grid as suggested
    param_grid = {
        'clf__C': [0.5, 1.0, 3.0, 10.0, 30.0, 100.0],
        'clf__gamma': [0.001, 0.01, 0.1, 'scale', 'auto']
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, class_weight='balanced', random_state=RANDOM_SEED))
    ])

    # Use GroupKFold for parameter optimization
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    grid_search.fit(X, y_encoded, groups=groups)

    logger.info(f"Best SVM parameters: {grid_search.best_params_}")
    logger.info(f"Best SVM CV score: {grid_search.best_score_:.3f}")

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }

def optimize_decision_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Find optimal decision threshold using Youden's J statistic."""
    logger.info("Optimizing decision threshold...")

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Youden's J statistic: Sensitivity + Specificity - 1
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]

    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]

    logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
    logger.info(f"Sensitivity: {optimal_sensitivity:.3f}")
    logger.info(f"Specificity: {optimal_specificity:.3f}")
    logger.info(f"Youden's J: {youdens_j[optimal_idx]:.3f}")

    return {
        'threshold': optimal_threshold,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'youdens_j': youdens_j[optimal_idx]
    }

def compare_feature_sets(X_full: np.ndarray, X_reduced: np.ndarray, X_alpha: np.ndarray,
                        y: np.ndarray, groups: np.ndarray,
                        full_names: List[str], reduced_names: List[str],
                        alpha_names: List[str]) -> Dict:
    """Compare performance across different feature sets."""
    logger.info("Comparing feature set performance...")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Define feature sets
    feature_sets = {
        'full': (X_full, full_names),
        'reduced': (X_reduced, reduced_names),
        'alpha_core': (X_alpha, alpha_names)
    }

    results = {}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for set_name, (X_set, names) in feature_sets.items():
        logger.info(f"Testing {set_name} feature set ({X_set.shape[1]} features)...")

        # Test with Random Forest (best performer from initial CV)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ])

        cv_scores = []
        for train_idx, test_idx in cv.split(X_set, y_encoded, groups):
            X_train, X_test = X_set[train_idx], X_set[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            score = balanced_accuracy_score(y_test, y_pred)
            cv_scores.append(score)

        results[set_name] = {
            'n_features': X_set.shape[1],
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'feature_names': names
        }

        logger.info(f"{set_name}: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    return results

def create_diagnostic_figures(results: Dict, corr_data: pd.DataFrame, timestamp: str) -> None:
    """Create comprehensive diagnostic visualizations."""
    logger.info("Creating diagnostic figures...")

    # Figure 1: Feature set comparison
    plt.figure(figsize=(12, 8))

    set_names = list(results.keys())
    means = [results[name]['mean_score'] for name in set_names]
    stds = [results[name]['std_score'] for name in set_names]
    n_features = [results[name]['n_features'] for name in set_names]

    plt.subplot(2, 2, 1)
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = plt.bar(set_names, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    plt.title('Feature Set Performance Comparison')
    plt.ylabel('Balanced Accuracy')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Target (70%)')
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Chance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add feature count labels
    for bar, n_feat in zip(bars, n_features):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{n_feat} features', ha='center', fontsize=9)

    # Subplot 2: CV score distributions
    plt.subplot(2, 2, 2)
    cv_data = []
    labels = []
    for name, result in results.items():
        cv_data.extend(result['cv_scores'])
        labels.extend([name] * len(result['cv_scores']))

    df_cv = pd.DataFrame({'score': cv_data, 'feature_set': labels})
    sns.boxplot(data=df_cv, x='feature_set', y='score', ax=plt.gca())
    plt.title('Cross-Validation Score Distributions')
    plt.ylabel('Balanced Accuracy')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    plt.grid(True, alpha=0.3)

    # Subplot 3: High correlation pairs (if any)
    plt.subplot(2, 2, 3)
    if len(corr_data) > 0:
        corr_data_plot = corr_data.head(20)  # Top 20 correlations
        y_pos = np.arange(len(corr_data_plot))
        plt.barh(y_pos, corr_data_plot['correlation'].abs(), alpha=0.7)
        plt.yticks(y_pos, [f"{row['feature1'][:15]}..." for _, row in corr_data_plot.iterrows()])
        plt.xlabel('|Correlation|')
        plt.title('Top Highly Correlated Feature Pairs')
        plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.7, label='Threshold')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No high correlations found', ha='center', va='center')
        plt.title('Feature Correlations')

    # Subplot 4: Performance vs Feature Count
    plt.subplot(2, 2, 4)
    plt.scatter(n_features, means, c=colors, s=100, alpha=0.7)
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Balanced Accuracy')
    plt.title('Performance vs Feature Count')

    for i, name in enumerate(set_names):
        plt.annotate(name, (n_features[i], means[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'diagnostic_analysis_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Diagnostic figures saved to {FIGURES_DIR}")

def main():
    """Main diagnostic analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    logger.info("="*80)
    logger.info("PHASE 4 DIAGNOSTICS: PERFORMANCE ENHANCEMENT")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().isoformat()}")

    try:
        # Save provenance first
        save_provenance(timestamp)

        # Load data
        df, X, y, subjects, feature_names = load_data()

        # 1. Analyze and remove collinear features
        X_reduced, reduced_names, corr_data = analyze_feature_collinearity(
            X, feature_names, threshold=0.95
        )

        # 2. Create alpha-core feature set
        alpha_indices = create_alpha_core_features(feature_names)
        X_alpha = X[:, alpha_indices]
        alpha_names = [feature_names[i] for i in alpha_indices]

        # 3. Compare feature sets
        comparison_results = compare_feature_sets(
            X, X_reduced, X_alpha, y, subjects,
            feature_names, reduced_names, alpha_names
        )

        # 4. Optimize SVM (if reduced set performs better)
        best_set_name = max(comparison_results.keys(),
                           key=lambda k: comparison_results[k]['mean_score'])
        logger.info(f"Best performing feature set: {best_set_name}")

        if best_set_name == 'reduced':
            X_best = X_reduced
        elif best_set_name == 'alpha_core':
            X_best = X_alpha
        else:
            X_best = X

        svm_results = optimize_svm_parameters(X_best, y, subjects)

        # 5. Create diagnostic visualizations
        create_diagnostic_figures(comparison_results, corr_data, timestamp)

        # 6. Save detailed results
        diagnostic_summary = {
            'timestamp': timestamp,
            'random_seed': RANDOM_SEED,
            'original_features': len(feature_names),
            'reduced_features': len(reduced_names),
            'alpha_features': len(alpha_names),
            'high_correlations_removed': len(corr_data),
            'feature_set_comparison': comparison_results,
            'optimized_svm': svm_results,
            'best_feature_set': best_set_name,
            'recommendations': []
        }

        # Add recommendations based on results
        best_score = comparison_results[best_set_name]['mean_score']
        if best_score < 0.6:
            diagnostic_summary['recommendations'].extend([
                "Performance below 60% suggests fundamental issues with feature discriminability",
                "Consider temporal features, connectivity measures, or event-related potentials",
                "Verify data quality and preprocessing pipeline",
                "Check for class label accuracy and potential mislabeling"
            ])
        elif best_score < 0.7:
            diagnostic_summary['recommendations'].extend([
                "Performance approaching target but needs improvement",
                f"Focus on {best_set_name} feature set optimization",
                "Consider ensemble methods or stacking",
                "Explore interaction features between alpha and motor regions"
            ])
        else:
            diagnostic_summary['recommendations'].append(
                f"Target achieved with {best_set_name} feature set!"
            )

        # Save results
        with open(RESULTS_DIR / f'diagnostic_summary_{timestamp}.json', 'w') as f:
            json.dump(diagnostic_summary, f, indent=2, default=str)

        # Final report
        logger.info("\n" + "="*60)
        logger.info("DIAGNOSTIC ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Original features: {len(feature_names)}")
        logger.info(f"After collinearity removal: {len(reduced_names)}")
        logger.info(f"Alpha-core features: {len(alpha_names)}")
        logger.info(f"Best feature set: {best_set_name}")
        logger.info(f"Best performance: {best_score:.3f} ± {comparison_results[best_set_name]['std_score']:.3f}")
        logger.info(f"Target (≥70%) {'ACHIEVED' if best_score >= 0.7 else 'NOT ACHIEVED'}")

        return diagnostic_summary

    except Exception as e:
        logger.error(f"Diagnostic analysis failed: {str(e)}")
        logger.error("", exc_info=True)
        raise

if __name__ == "__main__":
    results = main()