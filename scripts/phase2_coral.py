#!/usr/bin/env python3
"""
Phase II: CORAL Domain Adaptation for Cross-Dataset Validation
============================================================

CORAL (CORrelation ALignment) is a domain adaptation technique that aligns
the covariance structures of source and target domains without using target
labels. This addresses the critical cross-dataset performance collapse we
observed (57.7% → 50.1%).

Implementation based on:
"Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (Sun & Saenko, 2016)

Author: EEG Exergaming Development Team
Date: 2025-09-14
Target: Improve cross-dataset performance from 50.1% to ≥60%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, permutation_test_score
import logging
import sys
import json
from pathlib import Path
import warnings
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phase2_coral_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def coral_transform(X_source, X_target):
    """
    Apply CORAL transformation to align source domain to target domain

    Args:
        X_source: Source domain features (n_samples, n_features)
        X_target: Target domain features (n_samples, n_features)

    Returns:
        X_source_transformed: CORAL-aligned source features
        X_target: Unchanged target features (for consistency)
    """

    logger.info(f"Applying CORAL transformation...")
    logger.info(f"  Source shape: {X_source.shape}")
    logger.info(f"  Target shape: {X_target.shape}")

    # Center both domains
    X_source_centered = X_source - X_source.mean(0, keepdims=True)
    X_target_centered = X_target - X_target.mean(0, keepdims=True)

    # Compute covariance matrices with regularization
    n_features = X_source.shape[1]
    regularization = 1e-6

    C_source = np.cov(X_source_centered, rowvar=False) + regularization * np.eye(n_features)
    C_target = np.cov(X_target_centered, rowvar=False) + regularization * np.eye(n_features)

    logger.info(f"  Source covariance condition number: {np.linalg.cond(C_source):.2e}")
    logger.info(f"  Target covariance condition number: {np.linalg.cond(C_target):.2e}")

    # Eigendecomposition for whitening and coloring
    try:
        # Source domain whitening matrix
        eigenvals_s, eigenvecs_s = np.linalg.eigh(C_source)
        eigenvals_s = np.maximum(eigenvals_s, regularization)  # Ensure positive
        whitening_s = eigenvecs_s @ np.diag(1.0 / np.sqrt(eigenvals_s)) @ eigenvecs_s.T

        # Target domain coloring matrix
        eigenvals_t, eigenvecs_t = np.linalg.eigh(C_target)
        eigenvals_t = np.maximum(eigenvals_t, regularization)  # Ensure positive
        coloring_t = eigenvecs_t @ np.diag(np.sqrt(eigenvals_t)) @ eigenvecs_t.T

        # Apply CORAL transformation: center → whiten → color
        X_source_transformed = (X_source_centered @ whitening_s) @ coloring_t

        # Add back the target mean
        X_source_transformed += X_target.mean(0, keepdims=True)

        logger.info("✅ CORAL transformation completed successfully")

        return X_source_transformed, X_target

    except np.linalg.LinAlgError as e:
        logger.error(f"❌ CORAL transformation failed: {e}")
        logger.warning("  Falling back to standardized source data")

        # Fallback: just standardize to target statistics
        source_std = StandardScaler()
        target_std = StandardScaler()

        X_source_std = source_std.fit_transform(X_source)
        target_std.fit(X_target)

        # Transform source to have target statistics
        X_source_target_stats = (X_source_std * target_std.scale_) + target_std.mean_

        return X_source_target_stats, X_target

def load_training_data():
    """Load MRC BNDU training data (source domain)"""
    train_path = "results/phase5_beta_bursts/beta_burst_features.csv"

    if not Path(train_path).exists():
        logger.error(f"MRC BNDU training data not found: {train_path}")
        return None, None, None

    logger.info(f"Loading MRC BNDU training data from: {train_path}")
    df = pd.read_csv(train_path)

    logger.info(f"Loaded MRC BNDU: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Classes: {df['condition'].value_counts().to_dict()}")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'condition', 'session_number']]
    X_train = df[feature_cols].values
    y_train = df['condition'].values
    subjects_train = df['subject_id'].values

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Unique training subjects: {len(np.unique(subjects_train))}")

    return X_train, y_train, subjects_train

def load_test_data():
    """Load Leicester test data (target domain)"""
    test_path = "results/phase8_fixed_bursts/real_leicester_burst_features_FIXED.csv"

    if not Path(test_path).exists():
        logger.error(f"Leicester test data not found: {test_path}")
        return None, None, None

    logger.info(f"Loading Leicester test data from: {test_path}")
    df = pd.read_csv(test_path)

    logger.info(f"Loaded Leicester: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Classes: {df['condition'].value_counts().to_dict()}")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'condition', 'session_file', 'session_duration_s']]
    X_test = df[feature_cols].values
    y_test = df['condition'].values
    subjects_test = df['subject_id'].values

    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Unique test subjects: {len(np.unique(subjects_test))}")

    return X_test, y_test, subjects_test

def align_features(X_source, y_source, subjects_source, X_target, y_target, subjects_target):
    """Align features between source and target domains"""

    logger.info("Aligning features between domains...")

    # Load feature names from both datasets
    mrc_df = pd.read_csv("results/phase5_beta_bursts/beta_burst_features.csv")
    leicester_df = pd.read_csv("results/phase8_fixed_bursts/real_leicester_burst_features_FIXED.csv")

    mrc_features = [col for col in mrc_df.columns if col not in ['subject_id', 'condition', 'session_number']]
    leicester_features = [col for col in leicester_df.columns if col not in ['subject_id', 'condition', 'session_file', 'session_duration_s']]

    # Find common features
    common_features = list(set(mrc_features) & set(leicester_features))
    common_features = sorted(common_features)  # For reproducibility

    logger.info(f"Common features found: {len(common_features)}")
    logger.info(f"Common features: {common_features[:5]}... (showing first 5)")

    if len(common_features) < 5:
        logger.error(f"Too few common features ({len(common_features)}) for meaningful validation")
        return None, None, None, None, None, None

    # Select only common features
    mrc_indices = [mrc_features.index(f) for f in common_features]
    leicester_indices = [leicester_features.index(f) for f in common_features]

    X_source_aligned = X_source[:, mrc_indices]
    X_target_aligned = X_target[:, leicester_indices]

    logger.info(f"Aligned feature shapes:")
    logger.info(f"  Source (MRC BNDU): {X_source_aligned.shape}")
    logger.info(f"  Target (Leicester): {X_target_aligned.shape}")

    return X_source_aligned, y_source, subjects_source, X_target_aligned, y_target, subjects_target

def create_model_pipelines():
    """Create model pipelines for CORAL evaluation"""

    pipelines = {
        'logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]),

        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(random_state=42, probability=True, class_weight='balanced'))
        ]),

        'gradient_boost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
        ])
    }

    logger.info(f"Created {len(pipelines)} model pipelines")
    return pipelines

def evaluate_coral_performance(X_source, y_source, X_target, y_target):
    """Evaluate models with and without CORAL alignment"""

    logger.info("="*60)
    logger.info("CORAL PERFORMANCE EVALUATION")
    logger.info("="*60)

    # Create pipelines
    pipelines = create_model_pipelines()

    results = {
        'baseline': {},  # Without CORAL
        'coral': {}      # With CORAL
    }

    for model_name, pipeline in pipelines.items():
        logger.info(f"\nEvaluating {model_name}...")

        # Baseline: Train on source, test on target (no alignment)
        logger.info("  Baseline (no CORAL):")
        pipeline.fit(X_source, y_source)

        y_pred_baseline = pipeline.predict(X_target)
        y_pred_proba_baseline = pipeline.predict_proba(X_target)

        ba_baseline = balanced_accuracy_score(y_target, y_pred_baseline)
        auc_baseline = roc_auc_score(y_target, y_pred_proba_baseline[:, 1])

        logger.info(f"    BA: {ba_baseline:.3f}, AUC: {auc_baseline:.3f}")

        results['baseline'][model_name] = {
            'balanced_accuracy': float(ba_baseline),
            'auc': float(auc_baseline)
        }

        # CORAL: Train on CORAL-aligned source, test on target
        logger.info("  With CORAL alignment:")

        X_source_coral, X_target_coral = coral_transform(X_source, X_target)

        pipeline.fit(X_source_coral, y_source)

        y_pred_coral = pipeline.predict(X_target_coral)
        y_pred_proba_coral = pipeline.predict_proba(X_target_coral)

        ba_coral = balanced_accuracy_score(y_target, y_pred_coral)
        auc_coral = roc_auc_score(y_target, y_pred_proba_coral[:, 1])

        logger.info(f"    BA: {ba_coral:.3f}, AUC: {auc_coral:.3f}")
        logger.info(f"    Improvement: {ba_coral - ba_baseline:+.3f} BA, {auc_coral - auc_baseline:+.3f} AUC")

        results['coral'][model_name] = {
            'balanced_accuracy': float(ba_coral),
            'auc': float(auc_coral),
            'improvement_ba': float(ba_coral - ba_baseline),
            'improvement_auc': float(auc_coral - auc_baseline)
        }

    return results

def statistical_validation_coral(X_source, y_source, X_target, y_target, best_method, best_model):
    """Run statistical validation on best CORAL model"""

    logger.info(f"\nRunning statistical validation on {best_method}_{best_model}...")

    # Create pipeline
    pipelines = create_model_pipelines()
    pipeline = pipelines[best_model]

    if best_method == 'coral':
        # Apply CORAL transformation
        X_source_transformed, X_target_transformed = coral_transform(X_source, X_target)
        logger.info("Applied CORAL transformation for statistical validation")
    else:
        # Use original features
        X_source_transformed, X_target_transformed = X_source, X_target
        logger.info("Using baseline (no CORAL) for statistical validation")

    # Train and get observed score
    pipeline.fit(X_source_transformed, y_source)
    y_pred = pipeline.predict(X_target_transformed)
    observed_score = balanced_accuracy_score(y_target, y_pred)

    logger.info(f"Observed score: {observed_score:.6f}")

    # Permutation test (faster version with fewer iterations for demo)
    logger.info("Running permutation test (100 iterations)...")

    n_permutations = 100
    permutation_scores = []

    for i in range(n_permutations):
        if (i + 1) % 25 == 0:
            logger.info(f"  Permutation {i + 1}/{n_permutations}")

        # Shuffle target labels
        y_target_perm = np.random.permutation(y_target)

        # Re-fit and evaluate
        if best_method == 'coral':
            X_source_perm, X_target_perm = coral_transform(X_source, X_target)
        else:
            X_source_perm, X_target_perm = X_source, X_target

        pipeline.fit(X_source_perm, y_source)
        y_pred_perm = pipeline.predict(X_target_perm)
        score_perm = balanced_accuracy_score(y_target_perm, y_pred_perm)
        permutation_scores.append(score_perm)

    # Calculate statistics
    perm_mean = np.mean(permutation_scores)
    perm_std = np.std(permutation_scores)

    # P-value calculation
    n_extreme = np.sum(np.array(permutation_scores) >= observed_score)
    p_value = n_extreme / n_permutations

    logger.info(f"Permutation results:")
    logger.info(f"  Observed score: {observed_score:.6f}")
    logger.info(f"  Permutation mean: {perm_mean:.6f}")
    logger.info(f"  Permutation std: {perm_std:.6f}")
    logger.info(f"  P-value: {p_value:.6f}")

    return {
        'observed_score': float(observed_score),
        'permutation_mean': float(perm_mean),
        'permutation_std': float(perm_std),
        'p_value': float(p_value),
        'n_permutations': n_permutations
    }

def check_coral_acceptance_criteria(results, statistical_validation):
    """Check if CORAL results meet acceptance criteria"""

    logger.info("\n" + "="*60)
    logger.info("CORAL ACCEPTANCE CRITERIA CHECK")
    logger.info("="*60)

    # Find best performing method and model
    best_ba = 0
    best_method = None
    best_model = None

    for method in ['baseline', 'coral']:
        for model in results[method]:
            ba = results[method][model]['balanced_accuracy']
            if ba > best_ba:
                best_ba = ba
                best_method = method
                best_model = model

    logger.info(f"Best performance: {best_method}_{best_model} (BA: {best_ba:.3f})")

    # Acceptance criteria
    criteria = {
        'ba_threshold': 0.60,  # Target: ≥60% balanced accuracy
        'improvement_threshold': 0.05,  # CORAL should improve by ≥5%
        'p_value_threshold': 0.05
    }

    # Check criteria
    ba_pass = best_ba >= criteria['ba_threshold']

    # Check improvement (only relevant if CORAL is best)
    if best_method == 'coral':
        baseline_ba = results['baseline'][best_model]['balanced_accuracy']
        improvement = best_ba - baseline_ba
        improvement_pass = improvement >= criteria['improvement_threshold']
    else:
        improvement = 0
        improvement_pass = False

    p_value_pass = statistical_validation['p_value'] < criteria['p_value_threshold']

    logger.info(f"Acceptance criteria:")
    logger.info(f"  ✅ BA ≥ {criteria['ba_threshold']}: {best_ba:.3f} {'PASS' if ba_pass else 'FAIL'}")
    logger.info(f"  ✅ CORAL improvement ≥ {criteria['improvement_threshold']}: {improvement:+.3f} {'PASS' if improvement_pass else 'FAIL'}")
    logger.info(f"  ✅ p-value < {criteria['p_value_threshold']}: {statistical_validation['p_value']:.3f} {'PASS' if p_value_pass else 'FAIL'}")

    all_pass = ba_pass and improvement_pass and p_value_pass

    logger.info(f"\nOverall result: {'✅ ALL CRITERIA PASSED' if all_pass else '❌ SOME CRITERIA FAILED'}")

    return {
        'best_method': best_method,
        'best_model': best_model,
        'best_balanced_accuracy': float(best_ba),
        'improvement': float(improvement),
        'criteria_results': {
            'ba_pass': bool(ba_pass),
            'improvement_pass': bool(improvement_pass),
            'p_value_pass': bool(p_value_pass),
            'all_pass': bool(all_pass)
        },
        'criteria_thresholds': criteria
    }

def main():
    """Main CORAL domain adaptation execution"""

    logger.info("================================================================================")
    logger.info("PHASE II: CORAL DOMAIN ADAPTATION")
    logger.info("================================================================================")
    logger.info(f"Start time: {pd.Timestamp.now().isoformat()}")
    logger.info("Target: Improve cross-dataset performance from 50.1% to ≥60%")

    # Create output directory
    output_dir = Path("results/phase2_coral")
    output_dir.mkdir(exist_ok=True)

    # Load source domain (MRC BNDU)
    X_source, y_source, subjects_source = load_training_data()
    if X_source is None:
        logger.error("Failed to load source domain data")
        return False

    # Load target domain (Leicester)
    X_target, y_target, subjects_target = load_test_data()
    if X_target is None:
        logger.error("Failed to load target domain data")
        return False

    # Align features between domains
    X_source, y_source, subjects_source, X_target, y_target, subjects_target = align_features(
        X_source, y_source, subjects_source, X_target, y_target, subjects_target
    )

    if X_source is None:
        logger.error("Failed to align features between domains")
        return False

    # Evaluate CORAL performance
    performance_results = evaluate_coral_performance(X_source, y_source, X_target, y_target)

    # Find best model for statistical validation
    best_ba = 0
    best_method = None
    best_model = None

    for method in ['baseline', 'coral']:
        for model in performance_results[method]:
            ba = performance_results[method][model]['balanced_accuracy']
            if ba > best_ba:
                best_ba = ba
                best_method = method
                best_model = model

    # Statistical validation
    statistical_validation = statistical_validation_coral(
        X_source, y_source, X_target, y_target, best_method, best_model
    )

    # Check acceptance criteria
    acceptance_results = check_coral_acceptance_criteria(performance_results, statistical_validation)

    # Save results
    final_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'phase': 'Phase II - CORAL Domain Adaptation',
        'source_domain': 'MRC BNDU',
        'target_domain': 'Leicester',
        'source_samples': int(X_source.shape[0]),
        'target_samples': int(X_target.shape[0]),
        'n_features': int(X_source.shape[1]),
        'performance_results': performance_results,
        'statistical_validation': statistical_validation,
        'acceptance_criteria': acceptance_results
    }

    output_file = output_dir / f"coral_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    logger.info("================================================================================")
    logger.info("CORAL DOMAIN ADAPTATION COMPLETED")
    logger.info("================================================================================")

    return acceptance_results['criteria_results']['all_pass']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)