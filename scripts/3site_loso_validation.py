#!/usr/bin/env python3
"""
3-Site LOSO Validation with Statistical Significance Testing
Iowa (ds004584) + UCSD (ds002778) + UNM/Iowa (ds003490)

This unlocks df=2 for proper statistical testing vs 50% chance
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import yaml

# Add src to path
sys.path.append('/Users/user/Desktop/EEG_Exergamming/src')

def load_existing_features():
    """Load processed features from all three sites"""

    features_dir = Path('/Users/user/Desktop/EEG_Exergamming/results/features')

    datasets = {}

    # Try to load existing feature files
    for dataset_id in ['ds004584', 'ds002778', 'ds003490']:
        feature_file = features_dir / f"{dataset_id}_core5.csv"

        if feature_file.exists():
            df = pd.read_csv(feature_file)
            print(f"Loaded {dataset_id}: {len(df)} subjects")
            datasets[dataset_id] = df
        else:
            print(f"Warning: {feature_file} not found")

    return datasets

def prepare_loso_datasets(datasets):
    """Prepare datasets for LOSO validation"""

    # Check we have the minimum required datasets
    required_datasets = ['ds004584', 'ds002778']  # These we know exist
    for req_dataset in required_datasets:
        if req_dataset not in datasets:
            print(f"Warning: Required dataset {req_dataset} not found")

    print(f"Available datasets: {list(datasets.keys())}")

    if len(datasets) < 2:
        print("Warning: Need at least 2 sites for meaningful LOSO validation")

    combined_data = []

    for dataset_id, df in datasets.items():
        # Add site identifier
        df_copy = df.copy()
        df_copy['site'] = dataset_id
        combined_data.append(df_copy)

    # Combine all datasets
    all_data = pd.concat(combined_data, ignore_index=True)

    print(f"\\nCombined dataset summary:")
    print(f"Total subjects: {len(all_data)}")
    print("By site:")
    print(all_data['site'].value_counts())

    return all_data

def run_loso_validation(all_data):
    """Run Leave-One-Site-Out cross-validation"""

    print("\\n" + "=" * 80)
    print("3-SITE LOSO VALIDATION")
    print("=" * 80)

    # Define Core5 features
    core5_features = ['duration_cv', 'duty_cycle', 'mean_duration_ms',
                     'median_duration_ms', 'motor_posterior_duty_ratio']

    # Check features exist
    missing_features = [f for f in core5_features if f not in all_data.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Use only available features
        core5_features = [f for f in core5_features if f in all_data.columns]
        print(f"Using available features: {core5_features}")

    if len(core5_features) == 0:
        raise ValueError("No Core5 features found in data")

    # Get unique sites
    sites = all_data['site'].unique()
    print(f"Sites available for LOSO: {sites}")

    if len(sites) < 2:
        raise ValueError("Need at least 2 sites for LOSO validation")

    # Prepare features and labels
    X = all_data[core5_features].values

    # Try to get labels - check multiple possible column names
    label_columns = ['label', 'group', 'diagnosis', 'condition']
    y_column = None
    for col in label_columns:
        if col in all_data.columns:
            y_column = col
            break

    if y_column is None:
        print("Warning: No label column found. Creating mock labels for testing...")
        # Create balanced mock labels for testing
        np.random.seed(42)
        all_data['label'] = np.random.choice([0, 1], size=len(all_data))
        y_column = 'label'

    y = all_data[y_column].values
    site_labels = all_data['site'].values

    print(f"\\nFeature matrix: {X.shape}")
    print(f"Labels: {len(y)} ({np.sum(y)} positive)")
    print(f"Sites: {len(sites)} unique sites")

    # Run LOSO validation
    loso_results = []

    for test_site in sites:
        print(f"\\n--- FOLD: Test Site = {test_site} ---")

        # Split data
        test_mask = site_labels == test_site
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_sites = site_labels[train_mask]

        print(f"Train sites: {np.unique(train_sites)}")
        print(f"Train: {len(X_train)} subjects, Test: {len(X_test)} subjects")
        print(f"Train labels: {np.sum(y_train)}/{len(y_train)} positive")
        print(f"Test labels: {np.sum(y_test)}/{len(y_test)} positive")

        if len(X_test) == 0:
            print(f"Warning: No test subjects for site {test_site}")
            continue

        if len(np.unique(y_train)) < 2:
            print(f"Warning: Only one class in training data for site {test_site}")
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier
        clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        ba = balanced_accuracy_score(y_test, y_pred)

        # Individual class metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() if len(np.unique(y_test)) == 2 else (0, 0, 0, len(y_test))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_result = {
            'test_site': test_site,
            'train_sites': '+'.join(np.unique(train_sites)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_test_pos': np.sum(y_test),
            'n_test_neg': len(y_test) - np.sum(y_test),
            'balanced_accuracy': ba,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_test
        }

        loso_results.append(fold_result)

        print(f"Balanced Accuracy: {ba:.3f} ({ba*100:.1f}%)")
        print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")

    return loso_results

def calculate_statistics(loso_results):
    """Calculate statistical significance and effect sizes"""

    if len(loso_results) == 0:
        print("No LOSO results to analyze")
        return None

    print("\\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Extract balanced accuracies
    bas = [result['balanced_accuracy'] for result in loso_results]

    print(f"\\nBalanced Accuracies by Fold:")
    for i, result in enumerate(loso_results):
        print(f"  {result['test_site']}: {result['balanced_accuracy']:.3f} ({result['balanced_accuracy']*100:.1f}%)")

    # Summary statistics
    mean_ba = np.mean(bas)
    std_ba = np.std(bas, ddof=1) if len(bas) > 1 else 0

    print(f"\\nSummary Statistics:")
    print(f"Mean BA: {mean_ba:.3f} ({mean_ba*100:.1f}%)")
    print(f"Std BA: {std_ba:.3f}")
    print(f"N folds: {len(bas)}")

    # Statistical significance testing
    if len(bas) >= 2:
        # One-sample t-test vs 50% chance
        t_stat, p_value = stats.ttest_1samp(bas, 0.5)

        # Effect size (Cohen's d)
        cohens_d = (mean_ba - 0.5) / std_ba if std_ba > 0 else float('inf')

        # Confidence interval
        if len(bas) > 1:
            se = std_ba / np.sqrt(len(bas))
            t_critical = stats.t.ppf(0.975, len(bas) - 1)  # 95% CI
            ci_lower = mean_ba - t_critical * se
            ci_upper = mean_ba + t_critical * se
        else:
            ci_lower = ci_upper = mean_ba

        print(f"\\nStatistical Testing:")
        print(f"One-sample t-test vs 50%:")
        print(f"  t({len(bas)-1}) = {t_stat:.3f}")
        print(f"  p = {p_value:.6f} ({'<0.05' if p_value < 0.05 else '≥0.05'})")
        print(f"  Cohen's d = {cohens_d:.3f}")
        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        print(f"  Effect size: {effect_interp}")

        # Clinical threshold assessment
        threshold = 0.65
        above_threshold = mean_ba >= threshold
        folds_above = sum(1 for ba in bas if ba >= threshold)

        print(f"\\nClinical Threshold Analysis:")
        print(f"  Threshold: {threshold*100:.0f}%")
        print(f"  Mean BA: {'ABOVE' if above_threshold else 'BELOW'} threshold")
        print(f"  Folds above threshold: {folds_above}/{len(bas)}")

        stats_summary = {
            'mean_ba': mean_ba,
            'std_ba': std_ba,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_interp,
            'above_threshold': above_threshold,
            'folds_above_threshold': folds_above,
            'total_folds': len(bas),
            'df': len(bas) - 1
        }

        return stats_summary

    else:
        print("\\nInsufficient data for statistical testing (need ≥2 folds)")
        return {'mean_ba': mean_ba, 'n_folds': len(bas)}

def save_results(loso_results, stats_summary):
    """Save results to CSV and pickle files"""

    # Create results directory
    results_dir = Path('/Users/user/Desktop/EEG_Exergamming/results/loso_validation')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save LOSO summary
    loso_df = pd.DataFrame([
        {
            'test_site': r['test_site'],
            'train_sites': r['train_sites'],
            'n_test': r['n_test'],
            'n_test_pos': r['n_test_pos'],
            'n_test_neg': r['n_test_neg'],
            'balanced_accuracy': r['balanced_accuracy'],
            'sensitivity': r['sensitivity'],
            'specificity': r['specificity']
        } for r in loso_results
    ])

    loso_file = results_dir / '3site_loso_summary.csv'
    loso_df.to_csv(loso_file, index=False)
    print(f"\\nSaved LOSO summary: {loso_file}")

    # Save detailed results
    detailed_file = results_dir / '3site_loso_detailed.pkl'
    with open(detailed_file, 'wb') as f:
        pickle.dump({
            'loso_results': loso_results,
            'stats_summary': stats_summary,
            'timestamp': datetime.now()
        }, f)
    print(f"Saved detailed results: {detailed_file}")

    return loso_file, detailed_file

def main():
    """Main execution function"""

    start_time = datetime.now()
    print(f"Started 3-site LOSO validation at: {start_time}")
    print("=" * 80)

    # Load datasets
    print("Loading datasets...")
    datasets = load_existing_features()

    if len(datasets) == 0:
        print("No datasets found. Please ensure feature files exist.")
        return

    # Prepare for LOSO
    print("\\nPreparing LOSO datasets...")
    all_data = prepare_loso_datasets(datasets)

    # Run LOSO validation
    loso_results = run_loso_validation(all_data)

    # Calculate statistics
    stats_summary = calculate_statistics(loso_results)

    # Save results
    if loso_results:
        save_results(loso_results, stats_summary)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\\n" + "=" * 80)
    print(f"3-SITE LOSO VALIDATION COMPLETE")
    print(f"Started: {start_time}")
    print(f"Completed: {end_time}")
    print(f"Duration: {duration}")
    print("=" * 80)

if __name__ == "__main__":
    main()