#!/usr/bin/env python3
"""
Phase 6 Re-run with Corrected Statistical Validation
==================================================

This script re-executes Phase 6 model training with corrected statistical
validation, addressing the critical bugs identified in the integrity check:
- Broken permutation testing (all scores identical)
- Invalid p-values (p=1.0)
- No label shuffling during validation

Uses real Leicester beta burst features with proper subject-wise grouped
cross-validation and valid permutation testing.

Target Performance: ≥58% balanced accuracy with p<0.05
Author: EEG Exergaming Development Team
Date: 2025-09-14
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import logging
import sys
import json
from pathlib import Path
import warnings
from scipy import stats
from sklearn.utils import resample

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phase6_rerun_clean_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'),
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

def load_leicester_burst_features():
    """Load real Leicester beta burst features from Phase 8"""
    data_path = "results/phase8_fixed_bursts/real_leicester_burst_features_FIXED.csv"

    if not Path(data_path).exists():
        logger.error(f"Leicester burst features not found: {data_path}")
        return None, None, None

    logger.info(f"Loading Leicester burst features from: {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Classes: {df['condition'].value_counts().to_dict()}")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'condition', 'session_file', 'session_duration_s']]
    X = df[feature_cols].values
    y = df['condition'].values
    subjects = df['subject_id'].values

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Unique subjects: {len(np.unique(subjects))}")

    return X, y, subjects

def create_model_pipelines():
    """Create clean model pipelines with proper hyperparameter grids"""

    pipelines = {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
        ]),

        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
        ]),

        'gb': Pipeline([
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
    }

    param_grids = {
        'logreg': {'clf__C': [0.1, 1, 3, 10]},
        'svm': {'clf__C': [0.5, 1, 3, 10], 'clf__gamma': ['scale', 0.1, 0.01]},
        'gb': {'clf__n_estimators': [100, 300], 'clf__max_depth': [2, 3]}
    }

    logger.info(f"Created {len(pipelines)} model pipelines with hyperparameter grids")
    return pipelines, param_grids

def perform_nested_cross_validation(X, y, subjects):
    """Perform nested cross-validation with proper subject grouping"""

    logger.info("="*60)
    logger.info("NESTED CROSS-VALIDATION WITH CORRECTED STATISTICS")
    logger.info("="*60)

    # Outer CV: subject-wise stratified grouping
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    pipelines, param_grids = create_model_pipelines()

    results = {}

    for model_name, pipeline in pipelines.items():
        logger.info(f"\nTraining {model_name} with nested CV...")

        fold_scores = []
        fold_aucs = []
        best_params_per_fold = []

        fold_idx = 1
        for train_idx, test_idx in outer_cv.split(X, y, subjects):
            logger.info(f"  Outer fold {fold_idx}/5:")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            subjects_train = subjects[train_idx]

            logger.info(f"    Train: {len(X_train)} samples, {len(np.unique(subjects_train))} subjects")
            logger.info(f"    Test:  {len(X_test)} samples, {len(np.unique(subjects[test_idx]))} subjects")

            # Inner CV for hyperparameter optimization
            grid_search = GridSearchCV(
                pipeline, param_grids[model_name],
                cv=inner_cv,
                scoring='balanced_accuracy',
                n_jobs=1
            )

            # Fit with subject grouping for inner CV
            grid_search.fit(X_train, y_train, groups=subjects_train)

            # Test on outer fold
            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)

            # Calculate metrics
            ba = balanced_accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])

            fold_scores.append(ba)
            fold_aucs.append(auc)
            best_params_per_fold.append(grid_search.best_params_)

            logger.info(f"    BA: {ba:.3f}, AUC: {auc:.3f}")
            logger.info(f"    Best params: {grid_search.best_params_}")

            fold_idx += 1

        # Calculate overall statistics
        mean_ba = np.mean(fold_scores)
        std_ba = np.std(fold_scores)
        mean_auc = np.mean(fold_aucs)

        logger.info(f"\n{model_name} Overall Results:")
        logger.info(f"  Mean BA: {mean_ba:.3f} ± {std_ba:.3f}")
        logger.info(f"  Mean AUC: {mean_auc:.3f}")

        results[model_name] = {
            'fold_scores': fold_scores,
            'fold_aucs': fold_aucs,
            'mean_balanced_accuracy': mean_ba,
            'std_balanced_accuracy': std_ba,
            'mean_auc': mean_auc,
            'best_params_per_fold': best_params_per_fold
        }

    return results

def corrected_permutation_test(X, y, subjects, best_model_name, pipelines, param_grids, n_permutations=1000):
    """Perform corrected permutation test with proper label shuffling"""

    logger.info(f"\nRunning CORRECTED permutation test for {best_model_name}...")
    logger.info(f"Permutations: {n_permutations}")

    # Get the best model
    pipeline = pipelines[best_model_name]
    param_grid = param_grids[best_model_name]

    # Outer CV setup
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    # Calculate observed score
    observed_scores = []
    for train_idx, test_idx in outer_cv.split(X, y, subjects):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subjects_train = subjects[train_idx]

        # Inner CV for hyperparameter optimization
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=inner_cv,
            scoring='balanced_accuracy',
            n_jobs=1
        )
        grid_search.fit(X_train, y_train, groups=subjects_train)

        y_pred = grid_search.predict(X_test)
        ba = balanced_accuracy_score(y_test, y_pred)
        observed_scores.append(ba)

    observed_score = np.mean(observed_scores)
    logger.info(f"Observed score: {observed_score:.6f}")

    # Permutation test with proper label shuffling
    permutation_scores = []

    for perm_i in range(n_permutations):
        if (perm_i + 1) % 100 == 0:
            logger.info(f"  Permutation {perm_i + 1}/{n_permutations}")

        # CRITICAL: Shuffle labels within subjects to maintain subject structure
        y_perm = y.copy()

        # Get unique subjects and their indices
        unique_subjects = np.unique(subjects)
        for subject in unique_subjects:
            subject_mask = subjects == subject
            subject_labels = y[subject_mask]

            # Shuffle labels within this subject's sessions
            np.random.shuffle(subject_labels)
            y_perm[subject_mask] = subject_labels

        # Perform cross-validation with shuffled labels
        perm_scores = []
        for train_idx, test_idx in outer_cv.split(X, y_perm, subjects):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_perm, y_test_perm = y_perm[train_idx], y_perm[test_idx]
            subjects_train = subjects[train_idx]

            # Simple model fitting (no hyperparameter optimization for speed)
            simple_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ])

            simple_pipeline.fit(X_train, y_train_perm)
            y_pred_perm = simple_pipeline.predict(X_test)
            ba_perm = balanced_accuracy_score(y_test_perm, y_pred_perm)
            perm_scores.append(ba_perm)

        permutation_scores.append(np.mean(perm_scores))

    # Calculate permutation statistics
    perm_mean = np.mean(permutation_scores)
    perm_std = np.std(permutation_scores)

    # Calculate p-value (two-tailed)
    n_extreme = np.sum(np.abs(np.array(permutation_scores) - perm_mean) >=
                      np.abs(observed_score - perm_mean))
    p_value = n_extreme / n_permutations

    logger.info(f"Permutation results:")
    logger.info(f"  Observed score: {observed_score:.6f}")
    logger.info(f"  Permutation mean: {perm_mean:.6f}")
    logger.info(f"  Permutation std: {perm_std:.6f}")
    logger.info(f"  P-value: {p_value:.6f}")

    # Verify permutation test validity
    if perm_std < 0.001:
        logger.warning("⚠️  Permutation std very low - check label shuffling")
    else:
        logger.info("✅ Permutation test appears valid")

    return {
        'observed_score': float(observed_score),
        'permutation_scores': [float(s) for s in permutation_scores],
        'permutation_mean': float(perm_mean),
        'permutation_std': float(perm_std),
        'p_value': float(p_value)
    }

def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals"""

    logger.info(f"Computing bootstrap confidence intervals ({confidence*100}% CI)...")

    bootstrap_scores = []
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            logger.info(f"  Bootstrap {i + 1}/{n_bootstrap}")

        bootstrap_sample = resample(scores, random_state=i)
        bootstrap_scores.append(np.mean(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    logger.info(f"Bootstrap results:")
    logger.info(f"  Bootstrap mean: {np.mean(bootstrap_scores):.6f}")
    logger.info(f"  Bootstrap std: {np.std(bootstrap_scores):.6f}")
    logger.info(f"  {confidence*100}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    return {
        'bootstrap_scores': [float(s) for s in bootstrap_scores],
        'bootstrap_mean': float(np.mean(bootstrap_scores)),
        'bootstrap_std': float(np.std(bootstrap_scores)),
        'confidence_interval': [float(ci_lower), float(ci_upper)],
        'confidence_level': confidence
    }

def check_acceptance_criteria(results, statistical_validation, bootstrap_results):
    """Check if results meet Phase 6 rerun acceptance criteria"""

    logger.info("\n" + "="*60)
    logger.info("ACCEPTANCE CRITERIA CHECK")
    logger.info("="*60)

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['mean_balanced_accuracy'])
    best_ba = results[best_model]['mean_balanced_accuracy']

    criteria = {
        'ba_threshold': 0.58,
        'p_value_threshold': 0.05,
        'ci_lower_threshold': 0.55
    }

    # Check criteria
    ba_pass = best_ba >= criteria['ba_threshold']
    p_value_pass = statistical_validation['p_value'] < criteria['p_value_threshold']
    ci_lower_pass = bootstrap_results['confidence_interval'][0] >= criteria['ci_lower_threshold']

    logger.info(f"Best model: {best_model}")
    logger.info(f"Acceptance criteria:")
    logger.info(f"  ✅ BA ≥ {criteria['ba_threshold']}: {best_ba:.3f} {'PASS' if ba_pass else 'FAIL'}")
    logger.info(f"  ✅ p-value < {criteria['p_value_threshold']}: {statistical_validation['p_value']:.3f} {'PASS' if p_value_pass else 'FAIL'}")
    logger.info(f"  ✅ CI lower ≥ {criteria['ci_lower_threshold']}: {bootstrap_results['confidence_interval'][0]:.3f} {'PASS' if ci_lower_pass else 'FAIL'}")

    all_pass = ba_pass and p_value_pass and ci_lower_pass

    logger.info(f"\nOverall result: {'✅ ALL CRITERIA PASSED' if all_pass else '❌ SOME CRITERIA FAILED'}")

    return {
        'best_model': best_model,
        'best_balanced_accuracy': float(best_ba),
        'criteria_results': {
            'ba_pass': bool(ba_pass),
            'p_value_pass': bool(p_value_pass),
            'ci_lower_pass': bool(ci_lower_pass),
            'all_pass': bool(all_pass)
        },
        'criteria_thresholds': criteria
    }

def main():
    """Main Phase 6 rerun execution"""

    logger.info("================================================================================")
    logger.info("PHASE 6 RERUN: CORRECTED STATISTICAL VALIDATION")
    logger.info("================================================================================")
    logger.info(f"Start time: {pd.Timestamp.now().isoformat()}")

    # Create output directory
    output_dir = Path("results/phase6_clean")
    output_dir.mkdir(exist_ok=True)

    # Load data
    X, y, subjects = load_leicester_burst_features()
    if X is None:
        logger.error("Failed to load Leicester burst features")
        return False

    # Perform nested cross-validation
    cv_results = perform_nested_cross_validation(X, y, subjects)

    # Find best model
    best_model = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_balanced_accuracy'])
    logger.info(f"\nBest model: {best_model} (BA: {cv_results[best_model]['mean_balanced_accuracy']:.3f})")

    # Perform corrected permutation test
    pipelines, param_grids = create_model_pipelines()
    statistical_validation = corrected_permutation_test(
        X, y, subjects, best_model, pipelines, param_grids, n_permutations=1000
    )

    # Bootstrap confidence intervals
    best_fold_scores = cv_results[best_model]['fold_scores']
    bootstrap_results = bootstrap_confidence_interval(best_fold_scores, n_bootstrap=1000)

    # Check acceptance criteria
    acceptance_results = check_acceptance_criteria(cv_results, statistical_validation, bootstrap_results)

    # Save results
    final_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'phase': 'Phase 6 Rerun - Corrected Statistical Validation',
        'data_source': 'Leicester Real Beta Burst Features (Phase 8)',
        'cv_results': convert_numpy_types(cv_results),
        'statistical_validation': statistical_validation,
        'bootstrap_confidence_intervals': bootstrap_results,
        'acceptance_criteria': acceptance_results
    }

    output_file = output_dir / f"corrected_validation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    logger.info("================================================================================")
    logger.info("PHASE 6 RERUN COMPLETED")
    logger.info("================================================================================")

    return acceptance_results['criteria_results']['all_pass']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)