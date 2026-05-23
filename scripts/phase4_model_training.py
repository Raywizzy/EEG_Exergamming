#!/usr/bin/env python3
"""
Phase 4: Model Training, Cross-Validation & Statistical Validation
Regulatory-grade ML pipeline with subject-wise isolation and statistical rigor.

Target: ≥70% balanced accuracy with proper validation
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

# ML imports
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import scipy.stats as stats

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.ml_pipeline import RegulatoryMLPipeline
from src.utils.logger import setup_logger

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "features"
RESULTS_DIR = BASE_DIR / "results" / "models"
FIGURES_DIR = BASE_DIR / "results" / "figures" / "05_model_training"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Setup logger
logger = setup_logger(__name__, BASE_DIR / "logs" / "phase4_model_training.log")

def load_feature_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load feature matrix from Phase 3."""
    logger.info("Loading feature data from Phase 3...")

    feature_path = BASE_DIR / "results" / "features" / "feature_matrix.csv"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}")

    # Load feature matrix from CSV
    df = pd.read_csv(feature_path)

    # Extract metadata columns
    subjects = df['subject_id'].values
    labels = df['condition'].values

    # Get feature columns (exclude metadata columns)
    metadata_cols = ['subject_id', 'condition', 'session_number']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Extract feature matrix
    X = df[feature_cols].values
    feature_names = feature_cols
    groups = subjects

    # Convert labels to consistent format
    y = np.array(labels)

    logger.info(f"Loaded features: {X.shape}")
    logger.info(f"Feature names: {len(feature_names)}")
    logger.info(f"Class distribution: {np.unique(y, return_counts=True)}")
    logger.info(f"Number of subjects: {len(np.unique(groups))}")

    return X, y, groups, feature_names

def create_performance_figures(results: Dict, timestamp: str) -> None:
    """Create comprehensive performance visualization figures."""
    logger.info("Creating performance figures...")

    # Figure 1: Cross-validation results comparison
    plt.figure(figsize=(15, 10))

    # Extract CV results for all models
    model_names = []
    cv_scores = []

    for model_name, model_results in results['cross_validation_results'].items():
        model_names.append(model_name.replace('_', ' ').title())
        cv_scores.append(model_results['cv_scores'])

    # Subplot 1: Box plot of CV scores
    plt.subplot(2, 3, 1)
    plt.boxplot(cv_scores, labels=model_names)
    plt.title('Cross-Validation Balanced Accuracy')
    plt.ylabel('Balanced Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Mean CV scores with error bars
    plt.subplot(2, 3, 2)
    means = [np.mean(scores) for scores in cv_scores]
    stds = [np.std(scores) for scores in cv_scores]
    plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title('Mean CV Performance ± SD')
    plt.ylabel('Balanced Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Subplot 3: Best model confusion matrix
    best_model_name = results['best_model']['name']
    cm = results['best_model']['confusion_matrix']
    plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PD_REAL', 'PD_SHAM'],
                yticklabels=['PD_REAL', 'PD_SHAM'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Subplot 4: ROC curves
    plt.subplot(2, 3, 4)
    for model_name, model_results in results['cross_validation_results'].items():
        if 'roc_curve' in model_results:
            fpr, tpr, _ = model_results['roc_curve']
            auc = model_results['roc_auc']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 5: Precision-Recall curves
    plt.subplot(2, 3, 5)
    for model_name, model_results in results['cross_validation_results'].items():
        if 'pr_curve' in model_results:
            precision, recall, _ = model_results['pr_curve']
            plt.plot(recall, precision, label=model_name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 6: Feature importance (top 15)
    plt.subplot(2, 3, 6)
    if 'feature_importance' in results['best_model']:
        feature_importance = results['best_model']['feature_importance']
        top_features = feature_importance[:15]

        plt.barh(range(len(top_features)), [f[1] for f in top_features])
        plt.yticks(range(len(top_features)), [f[0] for f in top_features])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance')
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'model_performance_comprehensive_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Statistical validation results
    plt.figure(figsize=(12, 8))

    # Subplot 1: Permutation test results
    plt.subplot(2, 2, 1)
    perm_scores = results['statistical_validation']['permutation_scores']
    true_score = results['best_model']['balanced_accuracy']

    plt.hist(perm_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(true_score, color='red', linestyle='--', linewidth=2,
                label=f'True Score: {true_score:.3f}')
    plt.xlabel('Permuted Scores')
    plt.ylabel('Frequency')
    plt.title('Permutation Test Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Bootstrap confidence intervals
    plt.subplot(2, 2, 2)
    if 'bootstrap_ci' in results['statistical_validation']:
        bootstrap_scores = results['statistical_validation']['bootstrap_scores']
        ci_lower = results['statistical_validation']['bootstrap_ci'][0]
        ci_upper = results['statistical_validation']['bootstrap_ci'][1]

        plt.hist(bootstrap_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(ci_lower, color='red', linestyle='--', alpha=0.7, label=f'95% CI')
        plt.axvline(ci_upper, color='red', linestyle='--', alpha=0.7)
        plt.axvline(true_score, color='blue', linestyle='-', linewidth=2,
                    label=f'Mean Score: {true_score:.3f}')
        plt.xlabel('Bootstrap Scores')
        plt.ylabel('Frequency')
        plt.title('Bootstrap Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Subplot 3: Calibration curve
    plt.subplot(2, 2, 3)
    if 'calibration_curve' in results['best_model']:
        fraction_pos, mean_pred = results['best_model']['calibration_curve']
        plt.plot(mean_pred, fraction_pos, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Subplot 4: Subject-wise performance
    plt.subplot(2, 2, 4)
    if 'subject_performance' in results:
        subject_accs = list(results['subject_performance'].values())
        plt.hist(subject_accs, bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Subject-wise Accuracy')
        plt.ylabel('Number of Subjects')
        plt.title('Subject-wise Performance Distribution')
        plt.axvline(np.mean(subject_accs), color='red', linestyle='--',
                    label=f'Mean: {np.mean(subject_accs):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'statistical_validation_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Performance figures saved to {FIGURES_DIR}")

def save_results(results: Dict, timestamp: str) -> None:
    """Save all results to files."""
    logger.info("Saving results...")

    # Save full results as pickle
    with open(RESULTS_DIR / f'complete_results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save summary as JSON (convert numpy types)
    summary = {
        'timestamp': timestamp,
        'random_seed': RANDOM_SEED,
        'best_model': {
            'name': results['best_model']['name'],
            'balanced_accuracy': float(results['best_model']['balanced_accuracy']),
            'roc_auc': float(results['best_model']['roc_auc']),
            'precision': float(results['best_model']['precision']),
            'recall': float(results['best_model']['recall']),
            'f1_score': float(results['best_model']['f1_score'])
        },
        'statistical_validation': {
            'permutation_p_value': float(results['statistical_validation']['permutation_p_value']),
            'permutation_score': float(results['statistical_validation']['permutation_score'])
        },
        'target_achieved': results['best_model']['balanced_accuracy'] >= 0.70
    }

    if 'bootstrap_ci' in results['statistical_validation']:
        summary['statistical_validation']['bootstrap_ci'] = [
            float(results['statistical_validation']['bootstrap_ci'][0]),
            float(results['statistical_validation']['bootstrap_ci'][1])
        ]

    # Add model comparison
    summary['model_comparison'] = {}
    for model_name, model_results in results['cross_validation_results'].items():
        summary['model_comparison'][model_name] = {
            'mean_cv_score': float(np.mean(model_results['cv_scores'])),
            'std_cv_score': float(np.std(model_results['cv_scores'])),
            'best_params': model_results.get('best_params', {})
        }

    with open(RESULTS_DIR / f'results_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {RESULTS_DIR}")

def main():
    """Main execution function for Phase 4."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    logger.info("="*80)
    logger.info("PHASE 4: MODEL TRAINING & STATISTICAL VALIDATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Random seed: {RANDOM_SEED}")

    try:
        # Load data
        X, y, groups, feature_names = load_feature_data()

        # Initialize pipeline
        pipeline = RegulatoryMLPipeline(random_state=RANDOM_SEED)

        # Mark first todo as completed and move to second
        logger.info("Subject-wise data loading completed successfully")

        # Convert data to DataFrame for ML pipeline
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        groups_series = pd.Series(groups)

        # Execute cross-validation with all models
        logger.info("Starting grouped cross-validation...")
        cv_results = pipeline.grouped_cross_validation(X_df, y_series, groups_series)

        # Continue with pipeline execution
        logger.info("Grouped cross-validation completed successfully")

        # Find best model from cross-validation scores
        model_scores = {}
        for model_name, scores_list in cv_results['cv_scores'].items():
            # Extract balanced accuracy scores from the scores list
            if isinstance(scores_list[0], dict):
                ba_scores = [score['balanced_accuracy'] for score in scores_list]
            else:
                ba_scores = scores_list
            model_scores[model_name] = np.mean(ba_scores)

        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_score = model_scores[best_model_name]

        logger.info(f"Best model: {best_model_name} (Balanced Accuracy: {best_score:.3f})")

        # Statistical validation
        logger.info("Running statistical validation...")
        stat_results = pipeline.statistical_validation(X_df, y_series, groups_series, best_model_name)

        # Feature importance
        logger.info("Calculating feature importance...")
        importance_results = pipeline.model_interpretation(X_df, y_series, best_model_name)

        # Get final predictions using cross-validation
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        y_pred_all = np.zeros_like(y)
        y_prob_all = np.zeros(len(y))

        for train_idx, test_idx in cv_splitter.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            temp_pipeline = pipeline.pipelines[best_model_name]
            temp_pipeline.fit(X_train, y_train)
            y_pred_all[test_idx] = temp_pipeline.predict(X_test)
            y_prob_all[test_idx] = temp_pipeline.predict_proba(X_test)[:, 1]

        # Calculate final metrics
        final_balanced_acc = balanced_accuracy_score(y, y_pred_all)
        final_roc_auc = roc_auc_score(y == 'PD_SHAM', y_prob_all)

        # Compile comprehensive results
        results = {
            'cross_validation_results': cv_results,
            'best_model': {
                'name': best_model_name,
                'balanced_accuracy': final_balanced_acc,
                'roc_auc': final_roc_auc,
                'confusion_matrix': confusion_matrix(y, y_pred_all),
                'classification_report': classification_report(y, y_pred_all, output_dict=True),
                'feature_importance': importance_results[:20]  # Top 20 features
            },
            'statistical_validation': stat_results,
            'feature_names': feature_names,
            'metadata': {
                'timestamp': timestamp,
                'random_seed': RANDOM_SEED,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_subjects': len(np.unique(groups)),
                'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
            }
        }

        # Add precision, recall, f1
        report = classification_report(y, y_pred_all, output_dict=True)
        results['best_model']['precision'] = report['weighted avg']['precision']
        results['best_model']['recall'] = report['weighted avg']['recall']
        results['best_model']['f1_score'] = report['weighted avg']['f1-score']

        # Create figures
        create_performance_figures(results, timestamp)

        # Save results
        save_results(results, timestamp)

        # Final report
        logger.info("\n" + "="*60)
        logger.info("PHASE 4 COMPLETE - FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Balanced Accuracy: {final_balanced_acc:.3f}")
        logger.info(f"ROC AUC: {final_roc_auc:.3f}")
        logger.info(f"Target ≥70% Achieved: {'✓ YES' if final_balanced_acc >= 0.70 else '✗ NO'}")
        logger.info(f"Statistical Significance (p-value): {stat_results['permutation_p_value']:.6f}")

        if 'bootstrap_ci' in stat_results:
            ci_lower, ci_upper = stat_results['bootstrap_ci']
            logger.info(f"Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        logger.info(f"Results saved to: {RESULTS_DIR}")
        logger.info(f"Figures saved to: {FIGURES_DIR}")

        return results

    except Exception as e:
        logger.error(f"Phase 4 failed: {str(e)}")
        logger.error("", exc_info=True)
        raise

if __name__ == "__main__":
    results = main()