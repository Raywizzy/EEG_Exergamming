#!/usr/bin/env python3
"""
Phase 3: Feature Extraction for EEG Exergaming Project
Extracts comprehensive spectral features according to CLAUDE.md:

- Alpha (8-12 Hz) and Beta (13-30 Hz) absolute & relative power
- Individual Alpha Frequency (IAF)
- Band ratios (alpha/beta, motor/posterior, anterior/posterior)
- Advanced features (entropy, bursts, bandwidth)
- Spatial features (asymmetry, regional differences)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from features.feature_extractor import EEGFeatureProcessor
from preprocessing.data_loader import EEGDataLoader
from utils.figure_manager import save_figure, create_figure, close_all_figures

def print_timestamp(message):
    """Print message with ISO timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    iso_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"[{timestamp}] {message}")
    print(f"ISO Date: {iso_date}")

def load_preprocessed_data():
    """Load preprocessed data from Phase 2."""
    print_timestamp("Loading preprocessed data from Phase 2")

    # Check if preprocessing results exist
    preproc_summary_path = Path("data/interim/preprocessing_summary.json")
    if not preproc_summary_path.exists():
        print("ERROR: Preprocessing results not found. Run Phase 2 first.")
        return None, None

    # Load metadata and data
    metadata_path = "data/interim/leicester_subject_metadata.json"
    data_loader = EEGDataLoader(
        data_path="data/external/leicester_dataset",
        metadata_path=metadata_path
    )

    # Load subjects for feature extraction
    subjects_data = data_loader.load_all_subjects(conditions=['PD_REAL', 'PD_SHAM'])

    print(f"✓ Loaded {len(subjects_data)} subjects for feature extraction")
    return data_loader, subjects_data

def simulate_preprocessed_data(subjects_data: Dict) -> Dict:
    """Simulate preprocessed data structure for feature extraction."""
    print_timestamp("Simulating preprocessed data structure")

    # Add scripts to path for import
    scripts_path = Path(__file__).parent
    sys.path.append(str(scripts_path))

    from phase2_preprocessing import PSDPreprocessor

    preprocessor = PSDPreprocessor()
    simulated_preprocessed = {}

    for subject_id, subject_data in subjects_data.items():
        print(f"Processing {subject_id}...")
        preprocessed_subject = preprocessor.preprocess_subject(subject_data)
        simulated_preprocessed[subject_id] = preprocessed_subject

    return simulated_preprocessed

def extract_features_all_subjects(preprocessed_data: Dict) -> Dict:
    """Extract features from all subjects."""
    print_timestamp("Extracting features from all subjects")

    processor = EEGFeatureProcessor()
    subjects_features = {}

    for subject_id, subject_data in preprocessed_data.items():
        print(f"Extracting features for {subject_id}...")

        try:
            subject_features = processor.process_subject_data(subject_data)
            subjects_features[subject_id] = subject_features

            # Print summary
            n_sessions = subject_features['total_sessions_with_features']
            condition = subject_features['subject_info']['condition']
            print(f"  ✓ {n_sessions} sessions processed ({condition})")

        except Exception as e:
            print(f"  ✗ Error processing {subject_id}: {e}")

    return subjects_features

def create_feature_matrix(subjects_features: Dict) -> pd.DataFrame:
    """Create feature matrix for machine learning."""
    print_timestamp("Creating feature matrix")

    processor = EEGFeatureProcessor()
    feature_df = processor.create_feature_matrix(subjects_features)

    if feature_df.empty:
        print("ERROR: No features extracted")
        return None

    print(f"✓ Feature matrix created:")
    print(f"  - Shape: {feature_df.shape}")
    print(f"  - Subjects: {feature_df['subject_id'].nunique()}")
    print(f"  - Sessions: {len(feature_df)}")

    # Class distribution
    class_counts = feature_df['condition'].value_counts()
    print(f"  - Class distribution:")
    for condition, count in class_counts.items():
        print(f"    {condition}: {count} sessions")

    return feature_df

def analyze_feature_distributions(feature_df: pd.DataFrame):
    """Analyze and visualize feature distributions."""
    print_timestamp("Analyzing feature distributions")

    # Get feature columns (exclude metadata)
    metadata_cols = ['subject_id', 'condition', 'session_number',
                    'original_epochs', 'rejected_epochs']
    feature_cols = [col for col in feature_df.columns if col not in metadata_cols]

    print(f"Analyzing {len(feature_cols)} features...")

    # 1. Alpha vs Beta features comparison
    alpha_features = [f for f in feature_cols if 'alpha' in f.lower()]
    beta_features = [f for f in feature_cols if 'beta' in f.lower()]

    fig, axes = create_figure(2, 2, figsize=(14, 10))
    fig.suptitle("Alpha vs Beta Features Analysis", fontsize=16)

    # Alpha power comparison
    ax = axes[0, 0]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'alpha_power_abs_mean' in data.columns:
            ax.hist(data['alpha_power_abs_mean'], bins=20, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Alpha Power (Mean)')
    ax.set_ylabel('Density')
    ax.set_title('Alpha Power Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta power comparison
    ax = axes[0, 1]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'beta_power_abs_mean' in data.columns:
            ax.hist(data['beta_power_abs_mean'], bins=20, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Beta Power (Mean)')
    ax.set_ylabel('Density')
    ax.set_title('Beta Power Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Alpha/Beta ratio
    ax = axes[1, 0]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'alpha_beta_ratio_mean' in data.columns:
            ax.hist(data['alpha_beta_ratio_mean'], bins=20, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Alpha/Beta Ratio (Mean)')
    ax.set_ylabel('Density')
    ax.set_title('Alpha/Beta Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Individual Alpha Frequency
    ax = axes[1, 1]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'individual_alpha_freq_mean' in data.columns:
            ax.hist(data['individual_alpha_freq_mean'], bins=15, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Individual Alpha Frequency (Hz)')
    ax.set_ylabel('Density')
    ax.set_title('IAF Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "feature_distributions_alpha_beta", "features", "spectral_analysis")
    close_all_figures()

    # 2. Feature correlation matrix
    fig, ax = create_figure(1, 1, figsize=(12, 10))

    # Select subset of key features for correlation
    key_features = [f for f in feature_cols if any(keyword in f.lower()
                   for keyword in ['alpha_power', 'beta_power', 'ratio', 'iaf', 'entropy'])][:20]

    if len(key_features) > 1:
        corr_data = feature_df[key_features].corr()
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Matrix (Key Features)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        ax.text(0.5, 0.5, "Insufficient features for correlation analysis",
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    save_figure(fig, "feature_correlation_matrix", "features", "correlations")
    close_all_figures()

    # 3. Spatial and advanced features
    fig, axes = create_figure(2, 2, figsize=(14, 10))
    fig.suptitle("Spatial and Advanced Features", fontsize=16)

    # Anterior/Posterior ratio
    ax = axes[0, 0]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'anterior_posterior_ratio_mean' in data.columns:
            ax.hist(data['anterior_posterior_ratio_mean'], bins=15, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Anterior/Posterior Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Spatial Feature: A/P Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral entropy
    ax = axes[0, 1]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'spectral_entropy_mean' in data.columns:
            ax.hist(data['spectral_entropy_mean'], bins=15, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Spectral Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Advanced Feature: Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta burst rate
    ax = axes[1, 0]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'beta_burst_rate_mean' in data.columns:
            ax.hist(data['beta_burst_rate_mean'], bins=15, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Beta Burst Rate')
    ax.set_ylabel('Density')
    ax.set_title('Advanced Feature: Beta Bursts')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Alpha asymmetry
    ax = axes[1, 1]
    for condition in ['PD_REAL', 'PD_SHAM']:
        data = feature_df[feature_df['condition'] == condition]
        if 'alpha_asymmetry_mean' in data.columns:
            ax.hist(data['alpha_asymmetry_mean'], bins=15, alpha=0.6,
                   label=condition, density=True)
    ax.set_xlabel('Alpha Asymmetry')
    ax.set_ylabel('Density')
    ax.set_title('Spatial Feature: Asymmetry')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "spatial_advanced_features", "features", "comprehensive")
    close_all_figures()

def create_feature_space_visualization(feature_df: pd.DataFrame):
    """Create PCA and UMAP visualizations of feature space."""
    print_timestamp("Creating feature space visualizations")

    # Get feature columns
    metadata_cols = ['subject_id', 'condition', 'session_number',
                    'original_epochs', 'rejected_epochs']
    feature_cols = [col for col in feature_df.columns if col not in metadata_cols]

    if len(feature_cols) < 2:
        print("⚠ Insufficient features for dimensionality reduction")
        return

    # Prepare data
    X = feature_df[feature_cols].fillna(0)  # Fill NaN with 0
    y = feature_df['condition']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = create_figure(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Space Visualization", fontsize=16)

    # PCA
    ax = axes[0]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    colors = {'PD_REAL': 'red', 'PD_SHAM': 'blue'}
    for condition in ['PD_REAL', 'PD_SHAM']:
        mask = y == condition
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[condition], alpha=0.6, label=condition, s=30)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA: Feature Space')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # UMAP (if available and enough samples)
    ax = axes[1]
    if UMAP_AVAILABLE and len(X_scaled) >= 15:
        try:
            umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
            X_umap = umap_reducer.fit_transform(X_scaled)

            for condition in ['PD_REAL', 'PD_SHAM']:
                mask = y == condition
                ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                          c=colors[condition], alpha=0.6, label=condition, s=30)

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP: Feature Space')
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"UMAP failed: {str(e)[:50]}...",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('UMAP: Error')
    elif not UMAP_AVAILABLE:
        ax.text(0.5, 0.5, "UMAP not available\n(pip install umap-learn)",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('UMAP: Not Installed')
    else:
        ax.text(0.5, 0.5, f"UMAP requires more samples\n(have {len(X_scaled)}, need ≥15)",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('UMAP: Insufficient Samples')

    plt.tight_layout()
    save_figure(fig, "feature_space_visualization", "features", "dimensionality_reduction")
    close_all_figures()

    # Print PCA summary
    print(f"PCA Analysis:")
    print(f"  - Total variance explained by PC1-PC2: {sum(pca.explained_variance_ratio_[:2]):.1%}")
    print(f"  - PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  - PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")

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

def save_feature_results(feature_df: pd.DataFrame, subjects_features: Dict):
    """Save feature extraction results."""
    print_timestamp("Saving feature extraction results")

    # Save feature matrix
    features_path = Path("results/features/feature_matrix.csv")
    features_path.parent.mkdir(exist_ok=True)
    feature_df.to_csv(features_path, index=False)

    # Create feature summary
    metadata_cols = ['subject_id', 'condition', 'session_number',
                    'original_epochs', 'rejected_epochs']
    feature_cols = [col for col in feature_df.columns if col not in metadata_cols]

    # Group features by type
    processor = EEGFeatureProcessor()
    feature_groups = processor.get_feature_importance_groups()

    # Create summary
    summary = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_features": len(feature_cols),
        "total_sessions": len(feature_df),
        "total_subjects": feature_df['subject_id'].nunique(),
        "class_distribution": feature_df['condition'].value_counts().to_dict(),
        "feature_groups": {k: len(v) for k, v in feature_groups.items()},
        "feature_statistics": {
            "mean_sessions_per_subject": len(feature_df) / feature_df['subject_id'].nunique(),
            "feature_matrix_shape": list(feature_df.shape),
            "missing_values": feature_df[feature_cols].isnull().sum().sum()
        },
        "feature_names_by_group": feature_groups
    }

    # Save summary
    summary_path = Path("results/features/feature_extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)

    # Save detailed subject features (structure only)
    subjects_structure = {}
    for subject_id, data in subjects_features.items():
        subjects_structure[subject_id] = {
            "condition": data['subject_info']['condition'],
            "total_sessions_with_features": data['total_sessions_with_features'],
            "session_numbers": list(data['session_features'].keys())
        }

    structure_path = Path("data/interim/subjects_features_structure.json")
    with open(structure_path, 'w') as f:
        json.dump(convert_numpy_types({
            "subjects_structure": subjects_structure,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), f, indent=2)

    print(f"✓ Feature matrix saved: {features_path}")
    print(f"✓ Feature summary saved: {summary_path}")
    print(f"✓ Subjects structure saved: {structure_path}")

    return summary

def main():
    """Execute Phase 3 feature extraction."""
    print("=" * 80)
    print("PHASE 3: FEATURE EXTRACTION - EEG EXERGAMING PROJECT")
    print("=" * 80)
    print_timestamp("Starting Phase 3 feature extraction")

    # Step 1: Load preprocessed data
    data_loader, subjects_data = load_preprocessed_data()
    if not subjects_data:
        print("❌ Failed to load preprocessed data")
        return False

    # Step 2: Simulate preprocessing (since we don't persist full processed data)
    preprocessed_data = simulate_preprocessed_data(subjects_data)

    # Step 3: Extract features
    subjects_features = extract_features_all_subjects(preprocessed_data)
    if not subjects_features:
        print("❌ Feature extraction failed")
        return False

    # Step 4: Create feature matrix
    feature_df = create_feature_matrix(subjects_features)
    if feature_df is None:
        print("❌ Feature matrix creation failed")
        return False

    # Step 5: Analyze features
    analyze_feature_distributions(feature_df)
    create_feature_space_visualization(feature_df)

    # Step 6: Save results
    summary = save_feature_results(feature_df, subjects_features)

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"✓ Extracted features from {summary['total_subjects']} subjects")
    print(f"✓ Total sessions: {summary['total_sessions']}")
    print(f"✓ Total features: {summary['total_features']}")
    print(f"✓ Feature matrix shape: {summary['feature_statistics']['feature_matrix_shape']}")

    for condition, count in summary['class_distribution'].items():
        print(f"✓ {condition}: {count} sessions")

    print("\nFeature Groups:")
    for group, count in summary['feature_groups'].items():
        print(f"  - {group}: {count} features")

    print("\nFiles Created:")
    files_created = [
        "results/features/feature_matrix.csv",
        "results/features/feature_extraction_summary.json",
        "data/interim/subjects_features_structure.json",
        "results/figures/04_psd_features/ (visualization plots)"
    ]
    for file in files_created:
        print(f"  - {file}")

    print("\nNext Steps:")
    print("1. Run Phase 4: Model Training & Validation")
    print("2. Implement subject-wise cross-validation")
    print("3. Train Logistic Regression, SVM, Random Forest")
    print("4. Target ≥70% balanced accuracy")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)