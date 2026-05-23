#!/usr/bin/env python3
"""
Phase 4 Diagnostic Plots Generation
Regulatory-compliant diagnostic visualization of EEG features

This script generates comprehensive diagnostic plots from existing Phase 3 features
without any synthetic data generation, following hard compliance rules.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Skipping UMAP plots.")

def setup_logging():
    """Configure logging for diagnostic plots generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/phase4_diagnostic_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_phase3_features() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load Phase 3 feature extraction results."""
    logger = logging.getLogger(__name__)

    # Load feature matrix CSV file
    features_path = "results/features/feature_matrix.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature matrix not found at: {features_path}")

    logger.info(f"Loading features from: {features_path}")

    # Load the CSV file
    df = pd.read_csv(features_path)

    # Extract required columns
    y_series = df['condition']
    subjects_series = df['subject_id']

    # Get feature columns (exclude metadata columns)
    metadata_columns = ['subject_id', 'condition', 'session_number', 'original_epochs', 'rejected_epochs']
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    X_df = df[feature_columns]

    logger.info(f"Loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    logger.info(f"Classes: {y_series.value_counts().to_dict()}")
    logger.info(f"Subjects: {len(subjects_series.unique())}")

    return X_df, y_series, subjects_series

def create_output_directories():
    """Create output directories for diagnostic plots."""
    output_base = "results/figures/05_diagnostics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dirs = [
        f"{output_base}/distributions_{timestamp}",
        f"{output_base}/dimensionality_{timestamp}",
        f"{output_base}/correlations_{timestamp}",
        f"{output_base}/subject_analysis_{timestamp}"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return dirs, timestamp

def plot_feature_distributions(X_df: pd.DataFrame, y_series: pd.Series, output_dir: str):
    """Generate feature distribution plots by class."""
    logger = logging.getLogger(__name__)
    logger.info("Generating feature distribution plots...")

    # Select key alpha and beta features for visualization
    alpha_features = [col for col in X_df.columns if col.startswith('alpha_')][:12]
    beta_features = [col for col in X_df.columns if col.startswith('beta_')][:12]

    # Alpha feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Alpha Band Feature Distributions by Class', fontsize=16)

    for i, feature in enumerate(alpha_features):
        ax = axes[i//4, i%4]

        for class_name in ['PD_REAL', 'PD_SHAM']:
            data = X_df.loc[y_series == class_name, feature]
            ax.hist(data, alpha=0.6, label=class_name, bins=20)

        ax.set_title(feature.replace('alpha_', ''), fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/alpha_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Beta feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Beta Band Feature Distributions by Class', fontsize=16)

    for i, feature in enumerate(beta_features):
        ax = axes[i//4, i%4]

        for class_name in ['PD_REAL', 'PD_SHAM']:
            data = X_df.loc[y_series == class_name, feature]
            ax.hist(data, alpha=0.6, label=class_name, bins=20)

        ax.set_title(feature.replace('beta_', ''), fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/beta_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dimensionality_reduction(X_df: pd.DataFrame, y_series: pd.Series, output_dir: str):
    """Generate PCA, t-SNE, and UMAP plots."""
    logger = logging.getLogger(__name__)
    logger.info("Generating dimensionality reduction plots...")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # PCA Analysis
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # Plot PCA
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Principal Component Analysis', fontsize=16)

    # PCA scatter plot
    ax = axes[0, 0]
    for class_name in ['PD_REAL', 'PD_SHAM']:
        mask = y_series == class_name
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, label=class_name, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Explained variance
    ax = axes[0, 1]
    ax.bar(range(1, 11), pca.explained_variance_ratio_[:10])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance by PC')
    ax.grid(True, alpha=0.3)

    # Cumulative variance
    ax = axes[1, 0]
    ax.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), 'bo-')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Variance Explained')
    ax.grid(True, alpha=0.3)

    # PC3 vs PC4
    ax = axes[1, 1]
    for class_name in ['PD_REAL', 'PD_SHAM']:
        mask = y_series == class_name
        ax.scatter(X_pca[mask, 2], X_pca[mask, 3], alpha=0.6, label=class_name, s=20)
    ax.set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)')
    ax.set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]:.1%} var)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # t-SNE (sample subset for speed)
    logger.info("Computing t-SNE projection...")
    n_samples = min(500, len(X_df))
    indices = np.random.choice(len(X_df), n_samples, replace=False)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
    X_tsne = tsne.fit_transform(X_scaled[indices])

    plt.figure(figsize=(10, 8))
    for class_name in ['PD_REAL', 'PD_SHAM']:
        mask = y_series.iloc[indices] == class_name
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.6, label=class_name, s=30)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Projection (Sample of Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/tsne_projection.png", dpi=300, bbox_inches='tight')
    plt.close()

    # UMAP if available
    if UMAP_AVAILABLE:
        logger.info("Computing UMAP projection...")
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X_scaled[indices])

        plt.figure(figsize=(10, 8))
        for class_name in ['PD_REAL', 'PD_SHAM']:
            mask = y_series.iloc[indices] == class_name
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], alpha=0.6, label=class_name, s=30)

        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Projection (Sample of Data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/umap_projection.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_correlation_analysis(X_df: pd.DataFrame, output_dir: str):
    """Generate feature correlation heatmaps."""
    logger = logging.getLogger(__name__)
    logger.info("Generating correlation analysis plots...")

    # Compute correlation matrix
    corr_matrix = X_df.corr()

    # Full correlation heatmap (clustered)
    plt.figure(figsize=(20, 16))
    sns.clustermap(corr_matrix,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   annot=False,
                   cbar_kws={'label': 'Correlation'})
    plt.savefig(f"{output_dir}/correlation_heatmap_clustered.png", dpi=300, bbox_inches='tight')
    plt.close()

    # High correlation pairs
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Plot top 20 high correlations
    if corr_pairs:
        top_pairs = corr_pairs[:min(20, len(corr_pairs))]

        features1 = [pair[0] for pair in top_pairs]
        features2 = [pair[1] for pair in top_pairs]
        correlations = [pair[2] for pair in top_pairs]

        plt.figure(figsize=(12, 8))
        colors = ['red' if corr < 0 else 'blue' for corr in correlations]
        bars = plt.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
        plt.yticks(range(len(correlations)),
                  [f"{f1[:20]}...{f2[:20]}" for f1, f2 in zip(features1, features2)])
        plt.xlabel('Correlation Coefficient')
        plt.title('Top 20 Feature Correlations (|r| > 0.8)')
        plt.grid(True, alpha=0.3)

        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(corr/2, i, f'{corr:.3f}',
                    ha='center', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/high_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_subject_analysis(X_df: pd.DataFrame, y_series: pd.Series, subjects_series: pd.Series, output_dir: str):
    """Generate subject-level analysis plots."""
    logger = logging.getLogger(__name__)
    logger.info("Generating subject analysis plots...")

    # Subject distribution by class
    subject_class_df = pd.DataFrame({
        'subject': subjects_series,
        'class': y_series
    }).drop_duplicates()

    class_counts = subject_class_df['class'].value_counts()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.index, class_counts.values, alpha=0.7)
    plt.title('Subject Distribution by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Subjects')
    plt.grid(True, alpha=0.3)

    # Add counts on bars
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/subject_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Sessions per subject
    sessions_per_subject = subjects_series.value_counts().sort_values(ascending=False)

    plt.figure(figsize=(15, 8))
    plt.bar(range(len(sessions_per_subject)), sessions_per_subject.values, alpha=0.7)
    plt.title('Sessions per Subject (Sorted)')
    plt.xlabel('Subject Rank')
    plt.ylabel('Number of Sessions')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Sessions Statistics:
Mean: {sessions_per_subject.mean():.1f}
Median: {sessions_per_subject.median():.1f}
Std: {sessions_per_subject.std():.1f}
Min: {sessions_per_subject.min()}
Max: {sessions_per_subject.max()}"""

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sessions_per_subject.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Subject-wise feature means (sample key features)
    key_features = ['alpha_power_mean', 'beta_power_mean', 'alpha_beta_ratio_mean', 'frontal_alpha_mean']
    available_features = [f for f in key_features if f in X_df.columns]

    if available_features:
        subject_means = X_df.groupby(subjects_series)[available_features].mean()
        subject_classes = subject_class_df.set_index('subject')['class']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Subject-wise Feature Means by Class', fontsize=16)

        for i, feature in enumerate(available_features[:4]):
            ax = axes[i//2, i%2]

            for class_name in ['PD_REAL', 'PD_SHAM']:
                subjects_in_class = subject_classes[subject_classes == class_name].index
                values = subject_means.loc[subjects_in_class, feature]
                ax.scatter([class_name]*len(values), values, alpha=0.6, s=50)

            ax.set_title(feature)
            ax.set_ylabel('Feature Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/subject_feature_means.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_summary_report(X_df: pd.DataFrame, y_series: pd.Series, subjects_series: pd.Series,
                          output_dirs: List[str], timestamp: str):
    """Generate diagnostic summary report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating diagnostic summary report...")

    # Basic statistics
    n_samples = len(X_df)
    n_features = len(X_df.columns)
    n_subjects = len(subjects_series.unique())
    class_counts = y_series.value_counts()

    # Feature statistics
    feature_stats = {
        'alpha_features': len([f for f in X_df.columns if f.startswith('alpha_')]),
        'beta_features': len([f for f in X_df.columns if f.startswith('beta_')]),
        'spatial_features': len([f for f in X_df.columns if 'ratio' in f.lower()]),
        'missing_values': X_df.isnull().sum().sum()
    }

    # Subject statistics
    sessions_per_subject = subjects_series.value_counts()
    subject_stats = {
        'mean_sessions': sessions_per_subject.mean(),
        'median_sessions': sessions_per_subject.median(),
        'min_sessions': sessions_per_subject.min(),
        'max_sessions': sessions_per_subject.max()
    }

    # Generate report
    report_content = f"""# Phase 4 Diagnostic Analysis Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Timestamp**: {timestamp}

## Data Summary
- **Total Samples**: {n_samples:,}
- **Total Features**: {n_features:,}
- **Total Subjects**: {n_subjects}
- **Class Distribution**:
  - PD_REAL: {class_counts.get('PD_REAL', 0)} samples
  - PD_SHAM: {class_counts.get('PD_SHAM', 0)} samples

## Feature Breakdown
- **Alpha Band Features**: {feature_stats['alpha_features']}
- **Beta Band Features**: {feature_stats['beta_features']}
- **Spatial/Ratio Features**: {feature_stats['spatial_features']}
- **Missing Values**: {feature_stats['missing_values']}

## Subject Statistics
- **Mean Sessions per Subject**: {subject_stats['mean_sessions']:.1f}
- **Median Sessions per Subject**: {subject_stats['median_sessions']:.1f}
- **Session Range**: {subject_stats['min_sessions']} - {subject_stats['max_sessions']}

## Generated Plots
### Distribution Analysis
- `alpha_distributions.png`: Alpha band feature distributions by class
- `beta_distributions.png`: Beta band feature distributions by class

### Dimensionality Analysis
- `pca_analysis.png`: Principal component analysis
- `tsne_projection.png`: t-SNE projection
- `umap_projection.png`: UMAP projection (if available)

### Correlation Analysis
- `correlation_heatmap_clustered.png`: Clustered correlation heatmap
- `high_correlations.png`: Top high-correlation feature pairs

### Subject Analysis
- `subject_distribution.png`: Subject counts by class
- `sessions_per_subject.png`: Session distribution across subjects
- `subject_feature_means.png`: Subject-wise feature means

## Compliance Notes
- All plots generated from Phase 3 compliant features only
- No synthetic data generation performed
- Subject-wise analysis maintains data isolation principles
- All artifacts timestamped for reproducibility

---
*Generated by Phase 4 diagnostic plotting pipeline*
*Regulatory compliance maintained throughout analysis*
"""

    # Save report
    report_path = f"results/diagnostic_analysis_report_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(report_content)

    logger.info(f"Diagnostic report saved to: {report_path}")
    return report_path

def main():
    """Main execution function."""
    # Setup
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PHASE 4 DIAGNOSTIC PLOTS GENERATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")

    try:
        # Create output directories
        output_dirs, timestamp = create_output_directories()
        logger.info(f"Output directories created with timestamp: {timestamp}")

        # Load data
        X_df, y_series, subjects_series = load_phase3_features()

        # Generate plots
        plot_feature_distributions(X_df, y_series, output_dirs[0])
        plot_dimensionality_reduction(X_df, y_series, output_dirs[1])
        plot_correlation_analysis(X_df, output_dirs[2])
        plot_subject_analysis(X_df, y_series, subjects_series, output_dirs[3])

        # Generate summary report
        report_path = generate_summary_report(X_df, y_series, subjects_series, output_dirs, timestamp)

        logger.info("=" * 80)
        logger.info("DIAGNOSTIC PLOTS GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"All plots saved to: results/figures/05_diagnostics/*_{timestamp}/")
        logger.info(f"Summary report: {report_path}")

        return {
            'status': 'success',
            'timestamp': timestamp,
            'output_dirs': output_dirs,
            'report_path': report_path,
            'n_samples': len(X_df),
            'n_features': len(X_df.columns),
            'n_subjects': len(subjects_series.unique())
        }

    except Exception as e:
        logger.error(f"Diagnostic analysis failed: {str(e)}")
        logger.error("", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    results = main()
    if results['status'] == 'failed':
        sys.exit(1)