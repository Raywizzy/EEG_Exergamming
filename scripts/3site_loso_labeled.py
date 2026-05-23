#!/usr/bin/env python3
"""
3-Site Leave-One-Site-Out Validation with Labeled Data
Using proper clinical labels: PD_REAL vs CONTROL
Following CLAUDE.md requirements for subject-wise validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

# CORAL domain adaptation
def coral_alignment(source_features, target_features):
    """Apply CORAL domain adaptation"""

    # Calculate covariance matrices
    cov_source = np.cov(source_features.T) + np.eye(source_features.shape[1]) * 1e-6
    cov_target = np.cov(target_features.T) + np.eye(target_features.shape[1]) * 1e-6

    # Eigendecomposition
    eigenvals_s, eigenvecs_s = np.linalg.eigh(cov_source)
    eigenvals_t, eigenvecs_t = np.linalg.eigh(cov_target)

    # Transformation matrix
    sqrt_cov_s = eigenvecs_s @ np.diag(np.sqrt(np.maximum(eigenvals_s, 1e-12))) @ eigenvecs_s.T
    inv_sqrt_cov_s = eigenvecs_s @ np.diag(1.0 / np.sqrt(np.maximum(eigenvals_s, 1e-12))) @ eigenvecs_s.T
    sqrt_cov_t = eigenvecs_t @ np.diag(np.sqrt(np.maximum(eigenvals_t, 1e-12))) @ eigenvecs_t.T

    A_coral = sqrt_cov_t @ inv_sqrt_cov_s

    # Apply transformation
    aligned_source = source_features @ A_coral.T

    return aligned_source

def load_labeled_datasets():
    """Load all labeled feature datasets"""

    datasets = {}

    # Dataset configurations (some use 'label', some use 'diagnosis')
    dataset_configs = {
        'ds002778': {'label_col': 'label'},
        'ds003490': {'label_col': 'label'},
        'ds004584': {'label_col': 'diagnosis'}
    }

    # Load each labeled dataset
    for dataset_id, config in dataset_configs.items():
        file_path = Path(f"results/features/{dataset_id}_core5_labeled.csv")
        label_col = config['label_col']

        if file_path.exists():
            df = pd.read_csv(file_path)

            # Remove rows without labels
            df = df[df[label_col].notna()]

            # Standardize label column name
            df = df.rename(columns={label_col: 'label'})

            # Subject-level aggregation (mean across sessions/windows)
            feature_cols = ['duration_cv', 'duty_cycle', 'mean_duration_ms',
                          'median_duration_ms', 'motor_posterior_duty_ratio']

            subject_df = df.groupby(['subject_id', 'label']).agg({
                **{col: 'mean' for col in feature_cols},
                'dataset': 'first'
            }).reset_index()

            datasets[dataset_id] = subject_df
            print(f"Loaded {dataset_id}: {len(subject_df)} subjects")

            # Print label distribution
            label_counts = subject_df['label'].value_counts()
            print(f"  {dict(label_counts)}")

        else:
            print(f"⚠️  Missing: {file_path}")

    return datasets

def prepare_loso_data(datasets):
    """Prepare data for LOSO validation"""

    # Combine all datasets
    combined_dfs = []
    for dataset_id, df in datasets.items():
        df['site'] = dataset_id
        combined_dfs.append(df)

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    print(f"\nCombined dataset summary:")
    print(f"Total subjects: {len(combined_df)}")
    print("By site:")
    print(combined_df['site'].value_counts())
    print("\nBy label:")
    print(combined_df['label'].value_counts())

    return combined_df

def run_loso_validation(combined_df):
    """Run Leave-One-Site-Out validation with CORAL"""

    feature_cols = ['duration_cv', 'duty_cycle', 'mean_duration_ms',
                   'median_duration_ms', 'motor_posterior_duty_ratio']

    sites = combined_df['site'].unique()
    results = []

    print("\n" + "="*80)
    print("3-SITE LOSO VALIDATION WITH LABELED DATA")
    print("="*80)
    print(f"Sites: {sites}")

    # Convert labels to binary
    combined_df['label_binary'] = (combined_df['label'] == 'PD_REAL').astype(int)

    for test_site in sites:
        print(f"\n--- FOLD: Test Site = {test_site} ---")

        # Split data
        train_data = combined_df[combined_df['site'] != test_site].copy()
        test_data = combined_df[combined_df['site'] == test_site].copy()

        train_sites = train_data['site'].unique()
        print(f"Train sites: {list(train_sites)}")
        print(f"Train: {len(train_data)} subjects, Test: {len(test_data)} subjects")

        # Extract features and labels
        X_train = train_data[feature_cols].values
        y_train = train_data['label_binary'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['label_binary'].values

        print(f"Train labels: {y_train.sum()}/{len(y_train)} PD")
        print(f"Test labels: {y_test.sum()}/{len(y_test)} PD")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply CORAL domain adaptation
        X_train_coral = coral_alignment(X_train_scaled, X_test_scaled)

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_coral, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        ba = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Sensitivity/Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"Balanced Accuracy: {ba:.3f} ({ba*100:.1f}%)")
        print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
        print(f"AUC: {auc:.3f}")

        # Store results
        results.append({
            'test_site': test_site,
            'train_sites': list(train_sites),
            'n_train': len(train_data),
            'n_test': len(test_data),
            'balanced_accuracy': ba,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })

    return results

def analyze_results(results):
    """Analyze LOSO results and compute statistics"""

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    # Extract balanced accuracies
    bas = [r['balanced_accuracy'] for r in results]
    sites = [r['test_site'] for r in results]

    print(f"\nBalanced Accuracies by Fold:")
    for site, ba in zip(sites, bas):
        print(f"  {site}: {ba:.3f} ({ba*100:.1f}%)")

    # Summary statistics
    mean_ba = np.mean(bas)
    std_ba = np.std(bas, ddof=1)
    n_folds = len(bas)

    print(f"\nSummary Statistics:")
    print(f"Mean BA: {mean_ba:.3f} ({mean_ba*100:.1f}%)")
    print(f"Std BA: {std_ba:.3f}")
    print(f"N folds: {n_folds}")

    # Statistical testing
    print(f"\nStatistical Testing:")

    # One-sample t-test vs 50%
    t_stat, p_value = stats.ttest_1samp(bas, 0.5)
    print(f"One-sample t-test vs 50%:")
    print(f"  t({n_folds-1}) = {t_stat:.3f}")
    print(f"  p = {p_value:.6f} ({'<0.05' if p_value < 0.05 else '≥0.05'})")

    # Effect size (Cohen's d)
    cohens_d = (mean_ba - 0.5) / std_ba
    print(f"  Cohen's d = {cohens_d:.3f}")

    # Confidence interval
    sem = std_ba / np.sqrt(n_folds)
    ci_95 = stats.t.interval(0.95, n_folds-1, mean_ba, sem)
    print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"  Effect size: {effect_size}")

    # Clinical threshold analysis
    clinical_threshold = 0.65  # 65% BA threshold for clinical utility
    print(f"\nClinical Threshold Analysis:")
    print(f"  Threshold: {clinical_threshold*100:.0f}%")
    print(f"  Mean BA: {'ABOVE' if mean_ba >= clinical_threshold else 'BELOW'} threshold")

    folds_above = sum(1 for ba in bas if ba >= clinical_threshold)
    print(f"  Folds above threshold: {folds_above}/{n_folds}")

    return {
        'balanced_accuracies': bas,
        'sites': sites,
        'mean_ba': mean_ba,
        'std_ba': std_ba,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_95': ci_95,
        'clinical_threshold': clinical_threshold,
        'folds_above_threshold': folds_above
    }

def save_results(results, stats_summary):
    """Save detailed results"""

    # Create output directory
    output_dir = Path("results/loso_validation")
    output_dir.mkdir(exist_ok=True)

    # Save summary
    summary_df = pd.DataFrame([{
        'test_site': r['test_site'],
        'balanced_accuracy': r['balanced_accuracy'],
        'sensitivity': r['sensitivity'],
        'specificity': r['specificity'],
        'auc': r['auc'],
        'n_train': r['n_train'],
        'n_test': r['n_test']
    } for r in results])

    summary_file = output_dir / "3site_loso_labeled_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved LOSO summary: {summary_file}")

    # Save detailed results
    detailed_file = output_dir / "3site_loso_labeled_detailed.pkl"
    with open(detailed_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'statistics': stats_summary
        }, f)
    print(f"Saved detailed results: {detailed_file}")

def main():
    """Main execution"""

    start_time = datetime.now()
    print(f"Started 3-site LOSO validation with labeled data at: {start_time}")
    print("="*80)

    # Load datasets
    print("Loading labeled datasets...")
    datasets = load_labeled_datasets()

    if len(datasets) < 2:
        print("❌ Need at least 2 datasets for LOSO validation")
        return 1

    # Prepare data
    print("\nPreparing LOSO datasets...")
    combined_df = prepare_loso_data(datasets)

    # Run validation
    results = run_loso_validation(combined_df)

    # Analyze results
    stats_summary = analyze_results(results)

    # Save results
    save_results(results, stats_summary)

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("3-SITE LOSO VALIDATION WITH LABELED DATA COMPLETE")
    print(f"Started: {start_time}")
    print(f"Completed: {end_time}")
    print(f"Duration: {duration}")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())