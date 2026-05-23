#!/usr/bin/env python3
"""
Phase 6: Beta Burst Model Training & Validation
Train classification models using temporal beta burst features

Target: ≥70% balanced accuracy with rigorous statistical validation
"""

import os
import sys
import json
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ML imports
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
import scipy.stats as stats
from sklearn.inspection import permutation_importance

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def setup_logging():
    """Configure logging for Phase 6."""
    import logging

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/phase6_burst_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_burst_features() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load beta burst features from Phase 5."""
    logger = logging.getLogger(__name__)

    features_path = Path("results/phase5_beta_bursts/beta_burst_features.csv")
    if not features_path.exists():
        raise FileNotFoundError(f"Beta burst features not found: {features_path}")

    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # Extract target and groups
    y_series = df['condition']
    subjects_series = df['subject_id']

    # Select feature columns (exclude metadata)
    metadata_columns = ['subject_id', 'condition', 'session_number']
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    X_df = df[feature_columns]

    logger.info(f"Feature matrix: {X_df.shape}")
    logger.info(f"Classes: {y_series.value_counts().to_dict()}")
    logger.info(f"Subjects: {len(subjects_series.unique())}")

    return X_df, y_series, subjects_series

def create_model_pipelines() -> Dict[str, Pipeline]:
    """Create ML model pipelines with standardization."""

    pipelines = {
        'logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),

        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(random_state=42, probability=True, kernel='rbf'))
        ]),

        'gradient_boost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
        ])
    }

    return pipelines

def run_cross_validation(X_df: pd.DataFrame, y_series: pd.Series, subjects_series: pd.Series,
                        pipelines: Dict[str, Pipeline]) -> Dict:
    """Run subject-wise cross-validation on all models."""
    logger = logging.getLogger(__name__)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    logger.info(f"Label encoding: {label_mapping}")

    # Setup cross-validation
    cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'label_mapping': label_mapping,
        'cv_results': {},
        'model_performances': {}
    }

    logger.info("Starting subject-wise cross-validation...")

    for model_name, pipeline in pipelines.items():
        logger.info(f"Training {model_name}...")

        fold_results = []
        y_true_all = []
        y_pred_all = []
        y_proba_all = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_df, y_encoded, subjects_series)):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Get unique subjects in each fold
            train_subjects = subjects_series.iloc[train_idx].unique()
            test_subjects = subjects_series.iloc[test_idx].unique()

            logger.info(f"  Fold {fold_idx + 1}/5:")
            logger.info(f"    Train: {len(X_train)} samples, {len(train_subjects)} subjects")
            logger.info(f"    Test:  {len(X_test)} samples, {len(test_subjects)} subjects")

            # Train model
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class

            # Metrics
            ba_score = balanced_accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)

            fold_result = {
                'fold': fold_idx + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_subjects': len(train_subjects),
                'test_subjects': len(test_subjects),
                'balanced_accuracy': ba_score,
                'auc': auc_score
            }

            fold_results.append(fold_result)

            # Collect predictions for overall metrics
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)

            logger.info(f"    BA: {ba_score:.3f}, AUC: {auc_score:.3f}")

        # Overall performance
        overall_ba = balanced_accuracy_score(y_true_all, y_pred_all)
        overall_auc = roc_auc_score(y_true_all, y_proba_all)

        results['cv_results'][model_name] = {
            'fold_results': fold_results,
            'overall_balanced_accuracy': overall_ba,
            'overall_auc': overall_auc,
            'mean_balanced_accuracy': np.mean([f['balanced_accuracy'] for f in fold_results]),
            'std_balanced_accuracy': np.std([f['balanced_accuracy'] for f in fold_results]),
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_proba': y_proba_all
        }

        results['model_performances'][model_name] = overall_ba
        logger.info(f"  {model_name} Overall BA: {overall_ba:.3f} (±{np.std([f['balanced_accuracy'] for f in fold_results]):.3f})")

    return results

def run_statistical_validation(X_df: pd.DataFrame, y_series: pd.Series, subjects_series: pd.Series,
                              best_model_name: str, pipelines: Dict[str, Pipeline]) -> Dict:
    """Run permutation test and bootstrap CI on best model."""
    logger = logging.getLogger(__name__)

    logger.info(f"Running statistical validation on {best_model_name}...")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)

    # Get best pipeline
    best_pipeline = pipelines[best_model_name]

    # Permutation test
    logger.info("Running permutation test (1000 iterations)...")
    cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_score(X, y, groups):
        scores = []
        for train_idx, test_idx in cv_splitter.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, y_pred))

        return np.mean(scores)

    # Observed score
    observed_score = cv_score(X_df, y_encoded, subjects_series)

    # Permutation scores
    n_permutations = 1000
    permutation_scores = []

    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            logger.info(f"  Permutation {i + 1}/{n_permutations}")

        # Permute labels within subjects to maintain group structure
        y_permuted = y_encoded.copy()
        unique_subjects = subjects_series.unique()

        for subject in unique_subjects:
            subject_mask = subjects_series == subject
            subject_labels = y_encoded[subject_mask]
            if len(subject_labels) > 1:
                # Shuffle labels within this subject
                np.random.shuffle(subject_labels)
                y_permuted[subject_mask] = subject_labels

        perm_score = cv_score(X_df, y_permuted, subjects_series)
        permutation_scores.append(perm_score)

    permutation_scores = np.array(permutation_scores)
    p_value = (permutation_scores >= observed_score).mean()

    # Bootstrap confidence intervals
    logger.info("Computing bootstrap confidence intervals...")
    n_bootstrap = 1000
    bootstrap_scores = []

    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            logger.info(f"  Bootstrap {i + 1}/{n_bootstrap}")

        # Sample with replacement at subject level
        unique_subjects = subjects_series.unique()
        bootstrap_subjects = np.random.choice(unique_subjects, size=len(unique_subjects), replace=True)

        # Create bootstrap sample
        bootstrap_indices = []
        for subject in bootstrap_subjects:
            subject_indices = subjects_series[subjects_series == subject].index
            bootstrap_indices.extend(subject_indices)

        X_bootstrap = X_df.loc[bootstrap_indices]
        y_bootstrap = y_encoded[bootstrap_indices]
        subjects_bootstrap = subjects_series.loc[bootstrap_indices]

        # Reset indices
        X_bootstrap = X_bootstrap.reset_index(drop=True)
        subjects_bootstrap = subjects_bootstrap.reset_index(drop=True)

        bootstrap_score = cv_score(X_bootstrap, y_bootstrap, subjects_bootstrap)
        bootstrap_scores.append(bootstrap_score)

    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)

    stats_results = {
        'observed_score': observed_score,
        'permutation_scores': permutation_scores.tolist(),
        'permutation_mean': permutation_scores.mean(),
        'permutation_std': permutation_scores.std(),
        'p_value': p_value,
        'bootstrap_scores': bootstrap_scores.tolist(),
        'bootstrap_mean': bootstrap_scores.mean(),
        'bootstrap_std': bootstrap_scores.std(),
        'confidence_interval_95': [ci_lower, ci_upper]
    }

    logger.info("Statistical validation results:")
    logger.info(f"  Observed BA: {observed_score:.3f}")
    logger.info(f"  Permutation null mean: {permutation_scores.mean():.3f} ± {permutation_scores.std():.3f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Bootstrap CI (95%): [{ci_lower:.3f}, {ci_upper:.3f}]")

    return stats_results

def create_visualizations(results: Dict, stats_results: Dict, output_dir: Path):
    """Create comprehensive visualization outputs."""
    logger = logging.getLogger(__name__)

    logger.info("Generating visualization outputs...")

    # Create directories
    (output_dir / "roc_curves").mkdir(parents=True, exist_ok=True)
    (output_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    (output_dir / "performance").mkdir(parents=True, exist_ok=True)

    # ROC curves for each model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('ROC Curves by Model', fontsize=16)

    for i, (model_name, model_results) in enumerate(results['cv_results'].items()):
        ax = axes[i]

        y_true = model_results['y_true']
        y_proba = model_results['y_proba']

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = model_results['overall_auc']

        ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves" / "all_models_roc.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Model performance comparison
    model_names = list(results['model_performances'].keys())
    performances = list(results['model_performances'].values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, performances, alpha=0.7, color=['blue', 'green', 'orange'])
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Target (70%)')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance Level')
    plt.xlabel('Model')
    plt.ylabel('Balanced Accuracy')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add values on bars
    for bar, perf in zip(bars, performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "performance" / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical validation visualization
    if stats_results:
        plt.figure(figsize=(12, 5))

        # Permutation test
        plt.subplot(1, 2, 1)
        plt.hist(stats_results['permutation_scores'], bins=50, alpha=0.7, density=True)
        plt.axvline(stats_results['observed_score'], color='red', linestyle='--',
                   label=f'Observed ({stats_results["observed_score"]:.3f})')
        plt.xlabel('Balanced Accuracy')
        plt.ylabel('Density')
        plt.title(f'Permutation Test\np-value = {stats_results["p_value"]:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Bootstrap CI
        plt.subplot(1, 2, 2)
        plt.hist(stats_results['bootstrap_scores'], bins=50, alpha=0.7, density=True)
        plt.axvline(stats_results['confidence_interval_95'][0], color='green', linestyle='--', alpha=0.7)
        plt.axvline(stats_results['confidence_interval_95'][1], color='green', linestyle='--', alpha=0.7)
        plt.axvline(stats_results['bootstrap_mean'], color='red', linestyle='-',
                   label=f'Mean ({stats_results["bootstrap_mean"]:.3f})')
        plt.xlabel('Balanced Accuracy')
        plt.ylabel('Density')
        plt.title(f'Bootstrap Distribution\n95% CI: [{stats_results["confidence_interval_95"][0]:.3f}, {stats_results["confidence_interval_95"][1]:.3f}]')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "performance" / "statistical_validation.png", dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")

def save_results(results: Dict, stats_results: Dict, output_dir: Path):
    """Save comprehensive results."""

    # Save cross-validation results
    cv_results_path = output_dir / "cv_results.json"
    # Convert numpy arrays to lists for JSON serialization
    cv_results_clean = {}
    for model_name, model_results in results['cv_results'].items():
        cv_results_clean[model_name] = {
            'fold_results': model_results['fold_results'],
            'overall_balanced_accuracy': model_results['overall_balanced_accuracy'],
            'overall_auc': model_results['overall_auc'],
            'mean_balanced_accuracy': model_results['mean_balanced_accuracy'],
            'std_balanced_accuracy': model_results['std_balanced_accuracy']
        }

    # Convert numpy types for JSON serialization
    results_to_save = convert_numpy_types({
        'label_mapping': results['label_mapping'],
        'cv_results': cv_results_clean,
        'model_performances': results['model_performances']
    })

    with open(cv_results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    # Save statistical results
    if stats_results:
        stats_path = output_dir / "statistical_validation.json"
        stats_to_save = convert_numpy_types(stats_results)
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)

def main():
    """Main execution function for Phase 6."""
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("PHASE 6: BETA BURST MODEL TRAINING & VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")

    try:
        # Load data
        X_df, y_series, subjects_series = load_burst_features()

        # Create models
        pipelines = create_model_pipelines()
        logger.info(f"Created {len(pipelines)} model pipelines")

        # Run cross-validation
        results = run_cross_validation(X_df, y_series, subjects_series, pipelines)

        # Find best model
        best_model_name = max(results['model_performances'], key=results['model_performances'].get)
        best_performance = results['model_performances'][best_model_name]

        logger.info(f"Best model: {best_model_name} (BA: {best_performance:.3f})")

        # Statistical validation on best model
        stats_results = None
        if best_performance > 0.6:  # Only run extensive validation if promising
            stats_results = run_statistical_validation(X_df, y_series, subjects_series, best_model_name, pipelines)

        # Create outputs
        output_dir = Path("results/phase6_burst_models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        save_results(results, stats_results, output_dir)

        # Create visualizations
        create_visualizations(results, stats_results, output_dir)

        # Summary
        logger.info("=" * 40)
        logger.info("PHASE 6 RESULTS SUMMARY")
        logger.info("=" * 40)

        for model_name, performance in results['model_performances'].items():
            logger.info(f"{model_name}: {performance:.3f}")

        if best_performance >= 0.7:
            logger.info("🎯 TARGET ACHIEVED! ≥70% balanced accuracy reached")
        elif best_performance >= 0.6:
            logger.info("✅ Strong performance achieved (≥60% BA)")
        else:
            logger.info("⚠️ Performance below target - investigate features")

        if stats_results and stats_results['p_value'] < 0.001:
            logger.info(f"📊 Highly significant performance (p = {stats_results['p_value']:.6f})")

        logger.info("=" * 80)
        logger.info("PHASE 6 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return {
            'status': 'success',
            'best_model': best_model_name,
            'best_performance': best_performance,
            'target_achieved': best_performance >= 0.7,
            'output_dir': str(output_dir)
        }

    except Exception as e:
        logger.error(f"Phase 6 failed: {str(e)}")
        logger.error("", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    results = main()
    if results['status'] == 'failed':
        sys.exit(1)