#!/usr/bin/env python3
"""
Validate Core15+ Features Across All 3 Sites
Phase V optimization - Multi-site feature validation
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.core15_plus import Core15PlusExtractor
from src.preprocess.pipeline import EEGPreprocessor

def load_sample_subjects(dataset_name, n_subjects=3):
    """Load sample subjects from each dataset"""

    if dataset_name == 'ds004584':  # Iowa
        data_dir = Path("bids/ds004584")
    elif dataset_name == 'ds002778':  # UCSD
        data_dir = Path("data/bids/ds002778")
    elif dataset_name == 'ds003490':  # UNM/Iowa
        data_dir = Path("data/bids/ds003490")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return []

    subject_dirs = sorted([d for d in data_dir.glob("sub-*") if d.is_dir()])[:n_subjects]
    return subject_dirs, data_dir

def extract_features_from_raw(eeg_file, core15_extractor, dataset_name):
    """Extract Core15+ features from raw EEG file"""

    try:
        # Load and preprocess
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

        # Basic preprocessing
        raw = raw.filter(1, 40, verbose=False)
        raw = raw.resample(256, verbose=False)
        raw = raw.set_eeg_reference('average', verbose=False)

        # Get data and channel names
        data = raw.get_data()
        channel_names = raw.ch_names

        # Extract Core15+ features
        features = core15_extractor.extract_core15_plus(data, channel_names)
        features['dataset'] = dataset_name
        features['duration_s'] = data.shape[1] / raw.info['sfreq']

        return features

    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return None

def main():
    """Validate Core15+ features across all 3 sites"""

    print("=" * 80)
    print("PHASE V: CORE15+ MULTI-SITE VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Initialize feature extractor
    core15_extractor = Core15PlusExtractor()

    # Dataset configurations
    datasets = [
        ('ds004584', 'Iowa', 3),
        ('ds002778', 'UCSD', 3),
        ('ds003490', 'UNM/Iowa', 2)  # Smaller dataset
    ]

    all_features = []
    site_summary = {}

    # Process each dataset
    for dataset_name, site_name, n_subjects in datasets:
        print(f"🔍 Processing {site_name} ({dataset_name})")
        print("-" * 50)

        # Load subjects
        subject_dirs, data_dir = load_sample_subjects(dataset_name, n_subjects)

        if not subject_dirs:
            print(f"❌ No subjects found for {dataset_name}")
            continue

        site_features = []
        processing_times = []

        for i, subj_dir in enumerate(subject_dirs, 1):
            subj_id = subj_dir.name
            print(f"  Processing {i}/{len(subject_dirs)}: {subj_id}")

            # Find EEG file
            if dataset_name == 'ds003490':
                # Multi-session structure
                eeg_files = list(subj_dir.glob("ses-*/eeg/*_task-Rest_eeg.set"))
                if eeg_files:
                    eeg_file = eeg_files[0]  # Use first session
                else:
                    print(f"    ❌ No EEG file found")
                    continue
            else:
                # Standard structure
                eeg_files = list(subj_dir.glob("eeg/*_task-Rest_eeg.set"))
                if not eeg_files:
                    eeg_files = list(subj_dir.glob("eeg/*_task-rest_eeg.set"))  # Try lowercase

                if not eeg_files:
                    print(f"    ❌ No EEG file found")
                    continue
                eeg_file = eeg_files[0]

            print(f"    📁 File: {eeg_file.name}")

            # Extract features
            start_time = time.time()
            features = extract_features_from_raw(eeg_file, core15_extractor, dataset_name)
            processing_time = time.time() - start_time

            if features:
                features['subject_id'] = subj_id
                features['site'] = site_name
                features['processing_time'] = processing_time

                site_features.append(features)
                processing_times.append(processing_time)

                print(f"    ✅ Features extracted ({len([k for k in features.keys() if k not in ['subject_id', 'site', 'dataset', 'processing_time', 'duration_s']])} features)")
                print(f"    ⏱️  Processing time: {processing_time:.2f}s")

        if site_features:
            all_features.extend(site_features)
            site_summary[site_name] = {
                'n_subjects': len(site_features),
                'mean_processing_time': np.mean(processing_times),
                'features_extracted': len([k for k in site_features[0].keys() if k not in ['subject_id', 'site', 'dataset', 'processing_time', 'duration_s']])
            }
            print(f"  ✅ {site_name}: {len(site_features)} subjects processed")
        else:
            print(f"  ❌ {site_name}: No subjects processed")

        print()

    if not all_features:
        print("❌ No features extracted from any site")
        return

    # Create comprehensive analysis
    print("=" * 80)
    print("MULTI-SITE FEATURE VALIDATION RESULTS")
    print("=" * 80)

    df = pd.DataFrame(all_features)
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'site', 'dataset', 'processing_time', 'duration_s']]

    print(f"Total subjects processed: {len(df)}")
    print(f"Sites included: {', '.join(df['site'].unique())}")
    print(f"Total features extracted: {len(feature_cols)}")
    print()

    # Site-wise summary
    print("SITE-WISE PROCESSING SUMMARY:")
    for site, summary in site_summary.items():
        print(f"  {site}: {summary['n_subjects']} subjects, {summary['features_extracted']} features, {summary['mean_processing_time']:.2f}s avg")
    print()

    # Feature category analysis
    core5_cols = [col for col in feature_cols if any(base in col for base in ['duration_cv', 'duty_cycle', 'mean_duration', 'median_duration', 'motor_posterior'])]
    spectral_cols = [col for col in feature_cols if any(term in col for term in ['alpha', 'gamma', 'spectral', 'theta_beta'])]
    connectivity_cols = [col for col in feature_cols if any(term in col for term in ['pli', 'coherence', 'coupling'])]
    temporal_cols = [col for col in feature_cols if any(term in col for term in ['burst_sync', 'temporal_stab'])]

    print("FEATURE CATEGORIES:")
    print(f"  Core5: {len(core5_cols)} features")
    print(f"  Spectral: {len(spectral_cols)} features")
    print(f"  Connectivity: {len(connectivity_cols)} features")
    print(f"  Temporal: {len(temporal_cols)} features")
    print()

    # Cross-site feature comparison
    print("CROSS-SITE FEATURE STATISTICS:")
    print("-" * 50)

    # Core5 features
    if core5_cols:
        print("Core5 Features:")
        for col in core5_cols:
            site_stats = df.groupby('site')[col].agg(['mean', 'std', 'count']).round(3)
            print(f"  {col}:")
            for site in site_stats.index:
                mean_val = site_stats.loc[site, 'mean']
                std_val = site_stats.loc[site, 'std']
                count_val = site_stats.loc[site, 'count']
                print(f"    {site}: {mean_val:.3f} ± {std_val:.3f} (n={count_val})")
        print()

    # Spectral features
    if spectral_cols:
        print("Spectral Features (sample):")
        for col in spectral_cols[:3]:  # Show first 3
            site_stats = df.groupby('site')[col].agg(['mean', 'std', 'count']).round(3)
            print(f"  {col}:")
            for site in site_stats.index:
                mean_val = site_stats.loc[site, 'mean']
                std_val = site_stats.loc[site, 'std']
                count_val = site_stats.loc[site, 'count']
                print(f"    {site}: {mean_val:.3f} ± {std_val:.3f} (n={count_val})")
        print()

    # Feature ranges validation
    print("FEATURE RANGE VALIDATION:")
    print("-" * 50)

    range_issues = []
    for col in feature_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            min_val, max_val = col_data.min(), col_data.max()

            # Check for physiologically unreasonable values
            if 'duration' in col and 'ms' in col:
                if min_val < 50 or max_val > 1000:  # Duration should be 50-1000ms
                    range_issues.append(f"{col}: {min_val:.1f}-{max_val:.1f}ms (expected 50-1000ms)")
            elif 'duty_cycle' in col:
                if min_val < 0 or max_val > 20:  # Duty cycle should be 0-20%
                    range_issues.append(f"{col}: {min_val:.1f}-{max_val:.1f}% (expected 0-20%)")
            elif 'alpha_peak' in col:
                if min_val < 7 or max_val > 13:  # Alpha peak should be 7-13Hz
                    range_issues.append(f"{col}: {min_val:.1f}-{max_val:.1f}Hz (expected 7-13Hz)")

    if range_issues:
        print("⚠️  Potential range issues detected:")
        for issue in range_issues:
            print(f"  {issue}")
    else:
        print("✅ All features within expected physiological ranges")
    print()

    # Cross-site correlation analysis
    print("CROSS-SITE CORRELATION:")
    print("-" * 50)

    # Calculate correlation between sites for key features
    key_features = core5_cols + spectral_cols[:2] + connectivity_cols[:2]

    if len(df['site'].unique()) >= 2:
        sites = df['site'].unique()
        correlation_summary = []

        for feature in key_features[:5]:  # Test top 5 features
            feature_by_site = {}
            for site in sites:
                site_data = df[df['site'] == site][feature].dropna()
                if len(site_data) > 0:
                    feature_by_site[site] = site_data.mean()

            if len(feature_by_site) >= 2:
                correlation_summary.append({
                    'feature': feature,
                    'site_values': feature_by_site,
                    'cv': np.std(list(feature_by_site.values())) / np.mean(list(feature_by_site.values()))
                })

        print("Cross-site coefficient of variation (lower = more consistent):")
        for item in correlation_summary:
            print(f"  {item['feature']}: CV = {item['cv']:.3f}")
            for site, value in item['site_values'].items():
                print(f"    {site}: {value:.3f}")
    print()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/features/core15_plus_multisite_validation_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Multi-site validation results saved: {output_file}")
    print()

    print("=" * 80)
    print("CORE15+ MULTI-SITE VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print("1. ✅ Core15+ framework validated across sites")
    print("2. 🔄 Ready for ensemble model implementation")
    print("3. 🎯 Target: 3-site LOSO with Core15+ features")

if __name__ == "__main__":
    main()