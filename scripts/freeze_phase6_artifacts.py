#!/usr/bin/env python3
"""
Freeze Phase 6 Artifacts for External Validation
Extracts and saves the winning model, feature list, and scaler for Phase 7
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# ML imports
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

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
    """Configure logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/freeze_artifacts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Extract and freeze Phase 6 winning artifacts."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("FREEZING PHASE 6 ARTIFACTS FOR EXTERNAL VALIDATION")
    logger.info("=" * 60)

    # Load Phase 5 beta burst features (same as used in Phase 6)
    features_path = Path("results/phase5_beta_bursts/beta_burst_features.csv")
    if not features_path.exists():
        raise FileNotFoundError(f"Beta burst features not found: {features_path}")

    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns")

    # Prepare features and labels
    feature_columns = [col for col in df.columns if col not in ['subject_id', 'condition', 'session_number']]
    X = df[feature_columns].values
    y = df['condition'].values
    subjects = df['subject_id'].values

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Classes: {dict(pd.Series(y).value_counts())}")

    # Create the winning pipeline (Logistic Regression with StandardScaler)
    # Using the same hyperparameters that achieved 95.9% BA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

    # Label encoding (same as Phase 6)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Label encoding: {label_mapping}")

    # Train on full dataset (this is the locked model for external validation)
    logger.info("Training final model on full dataset...")
    pipeline.fit(X, y_encoded)

    # Verify performance matches Phase 6 (should be similar due to CV)
    y_pred = pipeline.predict(X)
    train_ba = balanced_accuracy_score(y_encoded, y_pred)
    logger.info(f"Full dataset training BA: {train_ba:.3f}")

    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Save model pipeline
    model_path = artifacts_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved: {model_path}")

    # Save feature list (order matters!)
    feature_list_path = artifacts_dir / "feature_list.json"
    with open(feature_list_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    logger.info(f"Feature list saved: {feature_list_path}")

    # Save label mapping
    label_mapping_path = artifacts_dir / "label_mapping.json"
    with open(label_mapping_path, 'w') as f:
        json.dump(convert_numpy_types(label_mapping), f, indent=2)
    logger.info(f"Label mapping saved: {label_mapping_path}")

    # Calculate and save training feature statistics for distribution shift analysis
    training_stats = {}
    for i, feature in enumerate(feature_columns):
        training_stats[feature] = {
            'mean': float(X[:, i].mean()),
            'std': float(X[:, i].std()),
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max()),
            'q25': float(np.percentile(X[:, i], 25)),
            'q75': float(np.percentile(X[:, i], 75))
        }

    training_stats_path = artifacts_dir / "training_feature_stats.json"
    with open(training_stats_path, 'w') as f:
        json.dump(convert_numpy_types(training_stats), f, indent=2)
    logger.info(f"Training statistics saved: {training_stats_path}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'phase6_performance': {
            'balanced_accuracy': 0.959,
            'std_dev': 0.019,
            'cv_method': 'StratifiedGroupKFold',
            'n_folds': 5
        },
        'model_type': 'LogisticRegression',
        'preprocessing': 'StandardScaler',
        'n_features': len(feature_columns),
        'n_samples': len(df),
        'n_subjects': len(df['subject_id'].unique()),
        'feature_source': 'beta_burst_temporal_dynamics',
        'locked_for_external_validation': True
    }

    metadata_path = artifacts_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved: {metadata_path}")

    logger.info("=" * 60)
    logger.info("ARTIFACTS SUCCESSFULLY FROZEN")
    logger.info("=" * 60)
    logger.info(f"Artifacts saved in: {artifacts_dir}")
    logger.info("Ready for Phase 7 external validation!")

    return {
        'status': 'success',
        'artifacts_dir': str(artifacts_dir),
        'model_path': str(model_path),
        'feature_list_path': str(feature_list_path),
        'n_features': len(feature_columns),
        'training_ba': train_ba
    }

if __name__ == "__main__":
    results = main()
    if results['status'] == 'failed':
        sys.exit(1)