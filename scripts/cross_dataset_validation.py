#!/usr/bin/env python3
"""
Cross-Dataset Validation: MRC BNDU → Leicester
==============================================

This script performs cross-dataset validation by training models on MRC BNDU
beta burst features and testing on real Leicester EEG burst features extracted
in Phase 8.

This addresses the critical simulation-to-reality gap identified in Phase 7-8
by combining the training power of MRC BNDU with the real-world validation
of Leicester data.

Author: EEG Exergaming Development Team
Date: 2025-09-14
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import permutation_test_score
import logging
import sys
import json
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/cross_dataset_validation_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'),
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

def load_mrc_training_data():
    """Load MRC BNDU training data"""
    train_path = "results/phase5_beta_bursts/beta_burst_features.csv"

    if not Path(train_path).exists():
        logger.error(f"MRC BNDU training data not found: {train_path}")
        return None, None

    logger.info(f"Loading MRC BNDU training data from: {train_path}")
    df = pd.read_csv(train_path)

    logger.info(f"Loaded MRC data: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Classes: {df['condition'].value_counts().to_dict()}")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'condition', 'session_number']]
    X_train = df[feature_cols].values
    y_train = df['condition'].values

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Feature columns: {feature_cols[:5]}... ({len(feature_cols)} total)")

    return X_train, y_train

def load_leicester_test_data():
    """Load real Leicester EEG test data from Phase 8"""
    test_path = "results/phase8_fixed_bursts/real_leicester_burst_features_FIXED.csv"

    if not Path(test_path).exists():
        logger.error(f"Leicester test data not found: {test_path}")
        return None, None

    logger.info(f"Loading Leicester test data from: {test_path}")
    df = pd.read_csv(test_path)

    logger.info(f"Loaded Leicester data: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Classes: {df['condition'].value_counts().to_dict()}")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'condition', 'session_file', 'session_duration_s']]
    X_test = df[feature_cols].values
    y_test = df['condition'].values

    logger.info(f"Test features shape: {X_test.shape}")

    return X_test, y_test

def create_model_pipelines():
    """Create model pipelines for cross-dataset validation"""

    pipelines = {
        'logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),

        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(random_state=42, probability=True))
        ]),

        'gradient_boost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
        ])
    }

    logger.info(f"Created {len(pipelines)} model pipelines")
    return pipelines

def evaluate_cross_dataset_performance(X_train, y_train, X_test, y_test):
    """Evaluate models trained on MRC BNDU, tested on Leicester"""

    logger.info("="*60)
    logger.info("CROSS-DATASET VALIDATION: MRC BNDU → LEICESTER")
    logger.info("="*60)

    # Create pipelines
    pipelines = create_model_pipelines()

    results = {}

    for model_name, pipeline in pipelines.items():
        logger.info(f"\nTraining {model_name}...")

        # Train on MRC BNDU
        logger.info(f"  Training on MRC BNDU: {X_train.shape[0]} samples")
        pipeline.fit(X_train, y_train)

        # Test on Leicester
        logger.info(f"  Testing on Leicester: {X_test.shape[0]} samples")
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)

        # Calculate metrics
        ba = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        logger.info(f"  Balanced Accuracy: {ba:.3f}")
        logger.info(f"  AUC: {auc:.3f}")

        # Store results
        results[model_name] = {
            'balanced_accuracy': ba,
            'auc': auc,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }

    return results

def run_statistical_validation(X_train, y_train, X_test, y_test, best_model_name, results):
    """Run statistical validation on best performing model"""

    logger.info(f"\nRunning statistical validation on {best_model_name}...")

    # Recreate best model pipeline
    pipelines = create_model_pipelines()
    best_pipeline = pipelines[best_model_name]

    # Get observed score
    observed_score = results[best_model_name]['balanced_accuracy']
    logger.info(f"Observed BA: {observed_score:.6f}")

    # Run permutation test on test set only (training on full MRC BNDU)
    logger.info("Running permutation test (100 iterations for speed)...")

    # Custom scoring function that trains on MRC BNDU and tests on Leicester
    def cross_dataset_scorer(estimator, X, y):
        # X and y here are the Leicester test data being permuted
        # We still train on the original MRC BNDU data
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X)
        return balanced_accuracy_score(y, y_pred)

    # Run permutation test
    score, perm_scores, pvalue = permutation_test_score(
        best_pipeline, X_test, y_test,
        scoring=cross_dataset_scorer,
        n_permutations=100,
        random_state=42,
        n_jobs=1
    )

    logger.info(f"Permutation test results:")
    logger.info(f"  Observed score: {score:.6f}")
    logger.info(f"  Permutation mean: {np.mean(perm_scores):.6f}")
    logger.info(f"  Permutation std: {np.std(perm_scores):.6f}")
    logger.info(f"  P-value: {pvalue:.6f}")

    return {
        'observed_score': float(score),
        'permutation_scores': [float(s) for s in perm_scores],
        'permutation_mean': float(np.mean(perm_scores)),
        'permutation_std': float(np.std(perm_scores)),
        'p_value': float(pvalue)
    }

def generate_cross_dataset_report(results, statistical_validation):
    """Generate comprehensive cross-dataset validation report"""

    logger.info("\n" + "="*60)
    logger.info("CROSS-DATASET VALIDATION RESULTS SUMMARY")
    logger.info("="*60)

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])
    best_ba = results[best_model]['balanced_accuracy']

    logger.info(f"\nModel Performance (MRC BNDU → Leicester):")
    for model_name in sorted(results.keys()):
        ba = results[model_name]['balanced_accuracy']
        auc = results[model_name]['auc']
        marker = "⭐" if model_name == best_model else "  "
        logger.info(f"{marker} {model_name:15}: BA={ba:.3f}, AUC={auc:.3f}")

    logger.info(f"\nBest model: {best_model} (BA: {best_ba:.3f})")

    # Compare with Phase 8 within-dataset results
    phase8_ba = 0.577  # From Phase 8 final results
    improvement = best_ba - phase8_ba

    logger.info(f"\nComparison with Phase 8 (within-dataset):")
    logger.info(f"  Phase 8 (Leicester train/test): {phase8_ba:.3f}")
    logger.info(f"  Cross-dataset (MRC→Leicester): {best_ba:.3f}")
    logger.info(f"  Difference: {improvement:+.3f}")

    if improvement > 0:
        logger.info("  ✅ Cross-dataset training improves performance!")
    else:
        logger.info("  ⚠️  Cross-dataset training shows performance drop")

    # Statistical significance
    if statistical_validation and statistical_validation['p_value'] < 0.05:
        logger.info(f"  ✅ Statistically significant (p={statistical_validation['p_value']:.3f})")
    else:
        pval = statistical_validation['p_value'] if statistical_validation else "N/A"
        logger.info(f"  ❌ Not statistically significant (p={pval})")

    return {
        'best_model': best_model,
        'best_balanced_accuracy': float(best_ba),
        'phase8_comparison': {
            'phase8_ba': phase8_ba,
            'cross_dataset_ba': float(best_ba),
            'improvement': float(improvement)
        },
        'all_results': convert_numpy_types(results),
        'statistical_validation': statistical_validation
    }

def main():
    """Main cross-dataset validation execution"""

    logger.info("================================================================================")
    logger.info("CROSS-DATASET VALIDATION: MRC BNDU TRAIN → LEICESTER TEST")
    logger.info("================================================================================")
    logger.info(f"Start time: {pd.Timestamp.now().isoformat()}")

    # Load training data (MRC BNDU)
    X_train, y_train = load_mrc_training_data()
    if X_train is None:
        logger.error("Failed to load MRC BNDU training data")
        return False

    # Load test data (Leicester)
    X_test, y_test = load_leicester_test_data()
    if X_test is None:
        logger.error("Failed to load Leicester test data")
        return False

    # Handle feature alignment by selecting common features
    logger.info(f"Feature alignment: MRC BNDU has {X_train.shape[1]} features, Leicester has {X_test.shape[1]}")

    # Get feature names from both datasets
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
        return False

    # Select only common features
    mrc_common_indices = [mrc_features.index(f) for f in common_features]
    leicester_common_indices = [leicester_features.index(f) for f in common_features]

    X_train = X_train[:, mrc_common_indices]
    X_test = X_test[:, leicester_common_indices]

    logger.info(f"✅ Aligned to {len(common_features)} common features")
    logger.info(f"Final shapes: MRC BNDU {X_train.shape}, Leicester {X_test.shape}")

    # Run cross-dataset evaluation
    results = evaluate_cross_dataset_performance(X_train, y_train, X_test, y_test)

    # Find best model for statistical validation
    best_model = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])

    # Run statistical validation
    statistical_validation = run_statistical_validation(
        X_train, y_train, X_test, y_test, best_model, results
    )

    # Generate final report
    report = generate_cross_dataset_report(results, statistical_validation)

    # Save results
    output_dir = Path("results/cross_dataset_validation")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"mrc_to_leicester_validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

    final_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'validation_type': 'Cross-Dataset',
        'training_dataset': 'MRC BNDU',
        'test_dataset': 'Leicester',
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        **report
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    logger.info("================================================================================")
    logger.info("CROSS-DATASET VALIDATION COMPLETED")
    logger.info("================================================================================")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)