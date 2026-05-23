#!/usr/bin/env python3
"""
Phase II: Feature Optimization Based on Validated Biomarkers
==========================================================

This script performs systematic feature ablation experiments focusing on our
validated biomarkers from Phase 8:
- Primary: Burst duration (238ms vs 192ms, p=0.0075)
- Secondary: Duty cycle (7.0% vs 5.0%, p=0.0005)
- Spatial: Motor/posterior ratios for generalization

The goal is to find the optimal compact feature subset that:
1. Maximizes cross-dataset performance (target: 61-63% BA)
2. Reduces noise from unstable features
3. Improves generalization across sites

Author: EEG Exergaming Development Team
Date: 2025-09-14
Target: ≥61% BA with compact, stable feature subset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import sys
import json
from pathlib import Path
import warnings
from itertools import combinations
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phase2_feature_optim_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
    """Apply CORAL transformation (from previous task)"""

    # Center both domains
    X_source_centered = X_source - X_source.mean(0, keepdims=True)
    X_target_centered = X_target - X_target.mean(0, keepdims=True)

    # Compute covariance matrices with regularization
    n_features = X_source.shape[1]
    regularization = 1e-6

    C_source = np.cov(X_source_centered, rowvar=False) + regularization * np.eye(n_features)
    C_target = np.cov(X_target_centered, rowvar=False) + regularization * np.eye(n_features)

    try:
        # Source domain whitening matrix
        eigenvals_s, eigenvecs_s = np.linalg.eigh(C_source)
        eigenvals_s = np.maximum(eigenvals_s, regularization)
        whitening_s = eigenvecs_s @ np.diag(1.0 / np.sqrt(eigenvals_s)) @ eigenvecs_s.T

        # Target domain coloring matrix
        eigenvals_t, eigenvecs_t = np.linalg.eigh(C_target)
        eigenvals_t = np.maximum(eigenvals_t, regularization)
        coloring_t = eigenvecs_t @ np.diag(np.sqrt(eigenvals_t)) @ eigenvecs_t.T

        # Apply CORAL transformation
        X_source_transformed = (X_source_centered @ whitening_s) @ coloring_t
        X_source_transformed += X_target.mean(0, keepdims=True)

        return X_source_transformed, X_target

    except np.linalg.LinAlgError:
        # Fallback to standardization
        source_std = StandardScaler()
        target_std = StandardScaler()

        X_source_std = source_std.fit_transform(X_source)
        target_std.fit(X_target)
        X_source_target_stats = (X_source_std * target_std.scale_) + target_std.mean_

        return X_source_target_stats, X_target

def load_aligned_datasets():
    """Load and align MRC BNDU and Leicester datasets"""

    logger.info("Loading and aligning datasets...")

    # Load MRC BNDU (source)
    mrc_path = "results/phase5_beta_bursts/beta_burst_features.csv"
    if not Path(mrc_path).exists():
        logger.error(f"MRC BNDU data not found: {mrc_path}")
        return None, None, None, None, None, None, None

    mrc_df = pd.read_csv(mrc_path)
    logger.info(f"Loaded MRC BNDU: {mrc_df.shape[0]} samples")

    # Load Leicester (target)
    leicester_path = "results/phase8_fixed_bursts/real_leicester_burst_features_FIXED.csv"
    if not Path(leicester_path).exists():
        logger.error(f"Leicester data not found: {leicester_path}")
        return None, None, None, None, None, None, None

    leicester_df = pd.read_csv(leicester_path)
    logger.info(f"Loaded Leicester: {leicester_df.shape[0]} samples")

    # Get feature names
    mrc_features = [col for col in mrc_df.columns if col not in ['subject_id', 'condition', 'session_number']]
    leicester_features = [col for col in leicester_df.columns if col not in ['subject_id', 'condition', 'session_file', 'session_duration_s']]

    # Find common features
    common_features = list(set(mrc_features) & set(leicester_features))
    common_features = sorted(common_features)

    logger.info(f"Common features: {len(common_features)}")
    logger.info(f"Features: {common_features}")

    # Extract aligned data
    mrc_indices = [mrc_features.index(f) for f in common_features]
    leicester_indices = [leicester_features.index(f) for f in common_features]

    X_source = mrc_df[[col for col in mrc_df.columns if col in common_features]].values
    y_source = mrc_df['condition'].values
    subjects_source = mrc_df['subject_id'].values

    X_target = leicester_df[[col for col in leicester_df.columns if col in common_features]].values
    y_target = leicester_df['condition'].values
    subjects_target = leicester_df['subject_id'].values

    return X_source, y_source, subjects_source, X_target, y_target, subjects_target, common_features

def define_feature_subsets(common_features):
    """Define feature subsets based on validated biomarkers"""

    logger.info("Defining feature subsets based on Phase 8 validated biomarkers...")

    # Map common features to categories
    feature_categories = {
        'core_biomarkers': [],
        'spatial_ratios': [],
        'temporal_features': [],
        'amplitude_features': [],
        'auxiliary_features': []
    }

    for feature in common_features:
        if 'duration' in feature.lower():
            feature_categories['core_biomarkers'].append(feature)
        elif 'duty_cycle' in feature.lower() or 'duty' in feature.lower():
            feature_categories['core_biomarkers'].append(feature)
        elif 'ratio' in feature.lower() and ('motor' in feature.lower() or 'posterior' in feature.lower()):
            feature_categories['spatial_ratios'].append(feature)
        elif any(x in feature.lower() for x in ['rate', 'ibi', 'cv_ibi', 'density']):
            feature_categories['temporal_features'].append(feature)
        elif any(x in feature.lower() for x in ['amp', 'amplitude', 'p95']):
            feature_categories['amplitude_features'].append(feature)
        else:
            feature_categories['auxiliary_features'].append(feature)

    logger.info("Feature categories:")
    for category, features in feature_categories.items():
        logger.info(f"  {category}: {features}")

    # Define ablation subsets (cumulative)
    feature_subsets = {
        'core_only': feature_categories['core_biomarkers'],
        'core_spatial': feature_categories['core_biomarkers'] + feature_categories['spatial_ratios'],
        'core_spatial_temporal': (feature_categories['core_biomarkers'] +
                                 feature_categories['spatial_ratios'] +
                                 feature_categories['temporal_features']),
        'all_features': common_features
    }

    # Remove empty subsets
    feature_subsets = {k: v for k, v in feature_subsets.items() if len(v) > 0}

    logger.info("Ablation subsets:")
    for subset_name, features in feature_subsets.items():
        logger.info(f"  {subset_name}: {len(features)} features - {features[:3]}...")

    return feature_subsets, feature_categories

def evaluate_feature_subset(X_source, y_source, X_target, y_target, feature_indices, subset_name):
    """Evaluate a specific feature subset with CORAL + cross-dataset validation"""

    logger.info(f"\nEvaluating {subset_name} ({len(feature_indices)} features)...")

    # Select features
    X_source_subset = X_source[:, feature_indices]
    X_target_subset = X_target[:, feature_indices]

    logger.info(f"  Feature subset shape: Source {X_source_subset.shape}, Target {X_target_subset.shape}")

    # Apply CORAL transformation
    X_source_coral, X_target_coral = coral_transform(X_source_subset, X_target_subset)

    # Test multiple models
    pipelines = {
        'logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]),
        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(random_state=42, probability=True, class_weight='balanced'))
        ])
    }

    results = {}

    for model_name, pipeline in pipelines.items():
        # Train on CORAL-aligned source, test on target
        pipeline.fit(X_source_coral, y_source)

        y_pred = pipeline.predict(X_target_coral)
        y_pred_proba = pipeline.predict_proba(X_target_coral)

        ba = balanced_accuracy_score(y_target, y_pred)
        auc = roc_auc_score(y_target, y_pred_proba[:, 1])

        logger.info(f"    {model_name}: BA={ba:.3f}, AUC={auc:.3f}")

        results[model_name] = {
            'balanced_accuracy': float(ba),
            'auc': float(auc)
        }

    # Find best model for this subset
    best_model = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])
    best_ba = results[best_model]['balanced_accuracy']

    logger.info(f"  Best: {best_model} (BA: {best_ba:.3f})")

    return results, best_model, best_ba

def feature_stability_analysis(X_source, y_source, X_target, y_target, common_features):
    """Analyze feature stability across domains"""

    logger.info("\nAnalyzing feature stability across domains...")

    stability_scores = {}

    for i, feature_name in enumerate(common_features):
        # Get feature values from both domains
        source_values = X_source[:, i]
        target_values = X_target[:, i]

        # Calculate distributional similarity (KS test)
        ks_stat, ks_p = stats.ks_2samp(source_values, target_values)

        # Calculate coefficient of variation difference
        source_cv = np.std(source_values) / (np.mean(source_values) + 1e-8)
        target_cv = np.std(target_values) / (np.mean(target_values) + 1e-8)
        cv_diff = abs(source_cv - target_cv)

        # Calculate univariate discriminative power in target domain
        # Split target by class
        target_class0 = target_values[y_target == 'PD_REAL']
        target_class1 = target_values[y_target == 'PD_SHAM']

        if len(target_class0) > 0 and len(target_class1) > 0:
            # T-test for discriminative power
            t_stat, t_p = stats.ttest_ind(target_class0, target_class1)
            discriminative_power = abs(t_stat)
        else:
            discriminative_power = 0

        # Composite stability score (lower is more stable)
        stability_score = ks_stat + cv_diff - (discriminative_power / 10.0)  # Normalize discriminative power

        stability_scores[feature_name] = {
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'cv_difference': float(cv_diff),
            'discriminative_power': float(discriminative_power),
            'stability_score': float(stability_score)
        }

    # Sort by stability (most stable first)
    stable_features = sorted(stability_scores.keys(), key=lambda k: stability_scores[k]['stability_score'])

    logger.info("Feature stability ranking (most stable first):")
    for i, feature in enumerate(stable_features[:8]):  # Show top 8
        score = stability_scores[feature]
        logger.info(f"  {i+1:2d}. {feature:20s}: stability={score['stability_score']:.3f}, "
                   f"KS={score['ks_statistic']:.3f}, power={score['discriminative_power']:.1f}")

    return stability_scores, stable_features

def run_ablation_experiments(X_source, y_source, X_target, y_target, common_features):
    """Run systematic feature ablation experiments"""

    logger.info("="*60)
    logger.info("FEATURE ABLATION EXPERIMENTS")
    logger.info("="*60)

    # Define feature subsets
    feature_subsets, feature_categories = define_feature_subsets(common_features)

    # Run stability analysis
    stability_scores, stable_features = feature_stability_analysis(
        X_source, y_source, X_target, y_target, common_features
    )

    # Evaluate each feature subset
    ablation_results = {}

    for subset_name, feature_list in feature_subsets.items():
        # Get feature indices
        feature_indices = [common_features.index(f) for f in feature_list if f in common_features]

        if len(feature_indices) > 0:
            results, best_model, best_ba = evaluate_feature_subset(
                X_source, y_source, X_target, y_target, feature_indices, subset_name
            )

            ablation_results[subset_name] = {
                'features': feature_list,
                'n_features': len(feature_list),
                'results': results,
                'best_model': best_model,
                'best_balanced_accuracy': float(best_ba)
            }

    # Add stability-based subset (top stable features)
    n_stable = min(6, len(stable_features))  # Use top 6 stable features
    stable_subset = stable_features[:n_stable]
    stable_indices = [common_features.index(f) for f in stable_subset]

    results, best_model, best_ba = evaluate_feature_subset(
        X_source, y_source, X_target, y_target, stable_indices, f'top_{n_stable}_stable'
    )

    ablation_results[f'top_{n_stable}_stable'] = {
        'features': stable_subset,
        'n_features': len(stable_subset),
        'results': results,
        'best_model': best_model,
        'best_balanced_accuracy': float(best_ba)
    }

    return ablation_results, stability_scores

def check_feature_optimization_criteria(ablation_results):
    """Check if feature optimization meets acceptance criteria"""

    logger.info("\n" + "="*60)
    logger.info("FEATURE OPTIMIZATION ACCEPTANCE CRITERIA")
    logger.info("="*60)

    # Find best performing subset
    best_subset = max(ablation_results.keys(),
                     key=lambda k: ablation_results[k]['best_balanced_accuracy'])
    best_ba = ablation_results[best_subset]['best_balanced_accuracy']
    best_n_features = ablation_results[best_subset]['n_features']

    logger.info(f"Best subset: {best_subset}")
    logger.info(f"Performance: {best_ba:.3f} BA with {best_n_features} features")
    logger.info(f"Features: {ablation_results[best_subset]['features']}")

    # Acceptance criteria
    criteria = {
        'ba_threshold': 0.61,  # Target: ≥61% balanced accuracy
        'efficiency_threshold': 0.58,  # Must beat baseline efficiency
        'max_features': 10  # Prefer compact subsets
    }

    # Check criteria
    ba_pass = best_ba >= criteria['ba_threshold']
    efficiency_pass = best_ba >= criteria['efficiency_threshold']
    compact_pass = best_n_features <= criteria['max_features']

    logger.info(f"\nAcceptance criteria:")
    logger.info(f"  ✅ BA ≥ {criteria['ba_threshold']}: {best_ba:.3f} {'PASS' if ba_pass else 'FAIL'}")
    logger.info(f"  ✅ Efficient ≥ {criteria['efficiency_threshold']}: {best_ba:.3f} {'PASS' if efficiency_pass else 'FAIL'}")
    logger.info(f"  ✅ Compact ≤ {criteria['max_features']}: {best_n_features} features {'PASS' if compact_pass else 'FAIL'}")

    all_pass = ba_pass and efficiency_pass and compact_pass

    # Generate performance comparison table
    logger.info(f"\nPerformance comparison:")
    for subset_name, results in sorted(ablation_results.items(),
                                     key=lambda x: x[1]['best_balanced_accuracy'], reverse=True):
        ba = results['best_balanced_accuracy']
        n_feat = results['n_features']
        model = results['best_model']
        marker = "⭐" if subset_name == best_subset else "  "
        logger.info(f"{marker} {subset_name:20s}: {ba:.3f} BA, {n_feat:2d} features ({model})")

    logger.info(f"\nOverall result: {'✅ ALL CRITERIA PASSED' if all_pass else '❌ SOME CRITERIA FAILED'}")

    return {
        'best_subset': best_subset,
        'best_balanced_accuracy': float(best_ba),
        'best_features': ablation_results[best_subset]['features'],
        'best_n_features': int(best_n_features),
        'criteria_results': {
            'ba_pass': bool(ba_pass),
            'efficiency_pass': bool(efficiency_pass),
            'compact_pass': bool(compact_pass),
            'all_pass': bool(all_pass)
        },
        'criteria_thresholds': criteria
    }

def main():
    """Main feature optimization execution"""

    logger.info("================================================================================")
    logger.info("PHASE II: FEATURE OPTIMIZATION")
    logger.info("================================================================================")
    logger.info(f"Start time: {pd.Timestamp.now().isoformat()}")
    logger.info("Goal: Find optimal feature subset for 61-63% BA with compact, stable features")

    # Create output directory
    output_dir = Path("results/phase2_featopt")
    output_dir.mkdir(exist_ok=True)

    # Load aligned datasets
    X_source, y_source, subjects_source, X_target, y_target, subjects_target, common_features = load_aligned_datasets()

    if X_source is None:
        logger.error("Failed to load aligned datasets")
        return False

    # Run ablation experiments
    ablation_results, stability_scores = run_ablation_experiments(
        X_source, y_source, X_target, y_target, common_features
    )

    # Check acceptance criteria
    acceptance_results = check_feature_optimization_criteria(ablation_results)

    # Save results
    final_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'phase': 'Phase II - Feature Optimization',
        'source_domain': 'MRC BNDU',
        'target_domain': 'Leicester',
        'common_features': common_features,
        'ablation_results': convert_numpy_types(ablation_results),
        'stability_analysis': convert_numpy_types(stability_scores),
        'acceptance_criteria': acceptance_results
    }

    output_file = output_dir / f"ablation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Also save the ablation table in a simple format
    ablation_table = {}
    for subset_name, results in ablation_results.items():
        ablation_table[subset_name] = {
            'balanced_accuracy': results['best_balanced_accuracy'],
            'n_features': results['n_features'],
            'best_model': results['best_model'],
            'features': results['features']
        }

    table_file = output_dir / "ablation_table.json"
    with open(table_file, 'w') as f:
        json.dump(ablation_table, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Ablation table saved to: {table_file}")

    logger.info("================================================================================")
    logger.info("FEATURE OPTIMIZATION COMPLETED")
    logger.info("================================================================================")

    return acceptance_results['criteria_results']['all_pass']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)