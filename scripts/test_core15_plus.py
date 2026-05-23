#!/usr/bin/env python3
"""
Test Core15+ Feature Extraction on Iowa Dataset (ds004584)
Phase V optimization - Core5 → Core15+ expansion
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.core15_plus import Core15PlusExtractor
from src.preprocess.pipeline import EEGPreprocessor

def main():
    """Test Core15+ feature extraction on Iowa dataset sample"""

    print("=" * 80)
    print("PHASE V: CORE15+ FEATURE EXTRACTION TEST")
    print("=" * 80)
    print(f"Dataset: Iowa (ds004584)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Initialize preprocessor and feature extractor
    config_path = Path("config/preprocessing_config.yaml")
    preprocessor = EEGPreprocessor(config_path)
    core15_extractor = Core15PlusExtractor()

    # Test on first 5 subjects from Iowa dataset
    data_dir = Path("bids/ds004584")
    subject_dirs = sorted([d for d in data_dir.glob("sub-*") if d.is_dir()])[:5]

    print(f"Testing Core15+ extraction on {len(subject_dirs)} subjects:")
    for i, subj_dir in enumerate(subject_dirs, 1):
        print(f"  {i}. {subj_dir.name}")
    print()

    results = []
    processing_times = []

    for i, subj_dir in enumerate(subject_dirs, 1):
        subj_id = subj_dir.name
        print(f"Processing {i}/{len(subject_dirs)}: {subj_id}")

        # Find EEG file
        eeg_files = list(subj_dir.glob("eeg/*_task-Rest_eeg.set"))
        if not eeg_files:
            print(f"  ❌ No EEG file found for {subj_id}")
            continue

        eeg_file = eeg_files[0]
        print(f"  📁 File: {eeg_file.name}")

        try:
            start_time = time.time()

            # Load and preprocess
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

            # Basic preprocessing
            raw = raw.filter(1, 40, verbose=False)
            raw = raw.resample(256, verbose=False)
            raw = raw.set_eeg_reference('average', verbose=False)

            # Get data and channel names
            data = raw.get_data()
            channel_names = raw.ch_names

            print(f"  📊 Data shape: {data.shape}")
            print(f"  ⏱️  Duration: {data.shape[1]/raw.info['sfreq']:.1f}s")

            # Extract Core15+ features
            features = core15_extractor.extract_core15_plus(data, channel_names)

            # Add metadata
            features['subject_id'] = subj_id
            features['dataset'] = 'ds004584'
            features['processing_time'] = time.time() - start_time

            results.append(features)
            processing_times.append(features['processing_time'])

            print(f"  ✅ Core15+ features extracted ({len(features)-3} features)")
            print(f"  ⏱️  Processing time: {features['processing_time']:.2f}s")

            # Show sample features
            core5_features = [k for k in features.keys() if k.startswith(('duration_cv', 'duty_cycle', 'mean_duration', 'median_duration', 'motor_posterior'))]
            spectral_features = [k for k in features.keys() if 'alpha' in k or 'gamma' in k or 'spectral' in k or 'theta_beta' in k]
            connectivity_features = [k for k in features.keys() if 'pli' in k or 'coherence' in k or 'coupling' in k]
            temporal_features = [k for k in features.keys() if 'burst_sync' in k or 'temporal_stab' in k]

            print(f"    Core5: {len(core5_features)} features")
            print(f"    Spectral: {len(spectral_features)} features")
            print(f"    Connectivity: {len(connectivity_features)} features")
            print(f"    Temporal: {len(temporal_features)} features")
            print()

        except Exception as e:
            print(f"  ❌ Error processing {subj_id}: {str(e)}")
            continue

    if not results:
        print("❌ No subjects processed successfully")
        return

    # Create summary
    print("=" * 80)
    print("CORE15+ EXTRACTION SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(results)

    print(f"Subjects processed: {len(df)}")
    print(f"Total features: {len(df.columns) - 3}")  # Exclude metadata
    print(f"Mean processing time: {np.mean(processing_times):.2f}s ± {np.std(processing_times):.2f}s")
    print()

    # Feature categories
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'dataset', 'processing_time']]

    core5_cols = [col for col in feature_cols if any(base in col for base in ['duration_cv', 'duty_cycle', 'mean_duration', 'median_duration', 'motor_posterior'])]
    spectral_cols = [col for col in feature_cols if any(term in col for term in ['alpha', 'gamma', 'spectral', 'theta_beta'])]
    connectivity_cols = [col for col in feature_cols if any(term in col for term in ['pli', 'coherence', 'coupling'])]
    temporal_cols = [col for col in feature_cols if any(term in col for term in ['burst_sync', 'temporal_stab'])]

    print("FEATURE BREAKDOWN:")
    print(f"  Core5 (original): {len(core5_cols)} features")
    print(f"  Spectral: {len(spectral_cols)} features")
    print(f"  Connectivity: {len(connectivity_cols)} features")
    print(f"  Temporal: {len(temporal_cols)} features")
    print(f"  Total: {len(feature_cols)} features")
    print()

    # Sample feature values
    print("SAMPLE FEATURE VALUES:")
    if core5_cols:
        print("Core5:")
        for col in core5_cols[:3]:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"  {col}: {values.mean():.3f} ± {values.std():.3f}")

    if spectral_cols:
        print("Spectral:")
        for col in spectral_cols[:3]:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"  {col}: {values.mean():.3f} ± {values.std():.3f}")

    if connectivity_cols:
        print("Connectivity:")
        for col in connectivity_cols[:3]:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"  {col}: {values.mean():.3f} ± {values.std():.3f}")

    if temporal_cols:
        print("Temporal:")
        for col in temporal_cols[:3]:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"  {col}: {values.mean():.3f} ± {values.std():.3f}")
    print()

    # Save results
    output_file = f"results/features/ds004584_core15_plus_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Results saved: {output_file}")

    print("=" * 80)
    print("CORE15+ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()