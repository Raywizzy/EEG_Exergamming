#!/usr/bin/env python3
"""
Boundary-safe ds003490 processing with session aggregation
Following the technical plan: handle boundary events, per-session processing, subject-level aggregation
"""

import sys
import os
import gc
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path for Core5 imports
sys.path.append('/Users/user/Desktop/EEG_Exergamming/src')
from src.features.core5 import BetaBurstDetector, Core5FeatureExtractor

# Configure MNE to be less verbose
mne.set_log_level('WARNING')

def load_config():
    """Load preprocessing configuration"""
    config_path = Path('/Users/user/Desktop/EEG_Exergamming/config/preprocessing_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_boundary_spans(raw):
    """Extract boundary event time spans to avoid"""
    ann = raw.annotations
    bad_spans = []
    for a in ann:
        if a['description'].lower().startswith('boundary'):
            bad_spans.append((a['onset'], a['onset'] + a['duration']))
    return bad_spans

def get_safe_windows(raw, window_length=60, bad_spans=None):
    """Get continuous windows that don't cross boundary events"""
    if bad_spans is None:
        bad_spans = []

    duration = raw.times[-1]
    windows = []

    start = 0
    while start + window_length <= duration:
        end = start + window_length

        # Check if this window overlaps any boundary spans
        window_safe = True
        for bad_start, bad_end in bad_spans:
            if not (end <= bad_start or start >= bad_end):  # Window overlaps bad span
                window_safe = False
                break

        if window_safe:
            windows.append((start, end))

        start += 30  # 30s stride

    return windows

def process_session(eeg_file, config, extractor):
    """Process a single session file with boundary handling"""
    try:
        print(f"    Loading: {eeg_file.name}")

        # Load data with preload for filtering
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

        print(f"    Original: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # Filter and resample to reduce memory footprint
        raw.filter(l_freq=1, h_freq=40, fir_design='firwin', n_jobs=1, verbose=False)
        raw.resample(256, npad="auto", verbose=False)

        print(f"    Processed: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # Get boundary spans
        bad_spans = get_boundary_spans(raw)
        print(f"    Found {len(bad_spans)} boundary events")

        # Get safe windows
        safe_windows = get_safe_windows(raw, window_length=60, bad_spans=bad_spans)
        print(f"    Found {len(safe_windows)} safe 60s windows")

        if len(safe_windows) < 2:  # Need at least 2 minutes of clean data
            print(f"    Warning: Only {len(safe_windows)} safe windows, skipping")
            return None

        # Apply CAR reference
        if len(raw.ch_names) > 1:
            raw.set_eeg_reference('average', projection=False, verbose=False)

        # Process each safe window and collect features
        window_features = []

        for i, (start, end) in enumerate(safe_windows[:3]):  # Limit to first 3 windows for memory
            try:
                # Extract window
                window_raw = raw.copy().crop(tmin=start, tmax=end)

                # Get data for Core5 processing
                data = window_raw.get_data()
                channel_names = window_raw.ch_names
                sfreq = window_raw.info['sfreq']

                # Extract Core5 features
                features = extractor.extract_core5_features(
                    beta_data=data,
                    sfreq=sfreq,
                    channel_names=channel_names,
                    subject_metadata={'window': i}
                )

                window_features.append(features)

            except Exception as e:
                print(f"    Warning: Window {i} failed: {e}")
                continue

        # Clear memory
        del raw
        gc.collect()

        if len(window_features) == 0:
            print(f"    Error: No windows processed successfully")
            return None

        # Aggregate features across windows (take median)
        df_windows = pd.DataFrame(window_features)
        session_features = df_windows.median(numeric_only=True).to_dict()
        session_features['n_windows'] = len(window_features)
        session_features['n_boundary_events'] = len(bad_spans)

        print(f"    Success: {len(window_features)} windows, Core5 extracted")
        return session_features

    except Exception as e:
        print(f"    Error: {e}")
        return None

def process_ds003490_boundary_safe():
    """Main processing function with boundary-safe approach"""

    print("=" * 80)
    print("DS003490 BOUNDARY-SAFE PROCESSING")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Initialize Core5 extractor
    extractor = Core5FeatureExtractor(str(Path('/Users/user/Desktop/EEG_Exergamming/config/preprocessing_config.yaml')))

    # Find all REST files
    dataset_path = Path('/Users/user/Desktop/EEG_Exergamming/data/bids/ds003490')
    rest_files = list(dataset_path.glob('sub-*/ses-*/eeg/*task-Rest_eeg.set'))

    print(f"Found {len(rest_files)} REST files")

    # Group files by subject
    subject_sessions = {}
    for eeg_file in rest_files:
        parts = eeg_file.name.split('_')
        subject_id = parts[0]  # sub-XXX
        session_id = parts[1]  # ses-XX

        if subject_id not in subject_sessions:
            subject_sessions[subject_id] = []
        subject_sessions[subject_id].append((session_id, eeg_file))

    print(f"Found {len(subject_sessions)} unique subjects")

    # Process all subjects (full cohort)
    subject_features = []
    processed_subjects = 0
    max_subjects = len(subject_sessions)  # Process all available subjects

    for subject_id, sessions in list(subject_sessions.items())[:max_subjects]:
        try:
            print(f"\n[{processed_subjects+1}/{max_subjects}] Processing subject: {subject_id}")
            print(f"  Sessions: {[s[0] for s in sessions]}")

            session_feature_list = []

            # Process each session
            for session_id, eeg_file in sessions:
                print(f"  Processing session: {session_id}")
                session_features = process_session(eeg_file, config, extractor)

                if session_features is not None:
                    session_features['session_id'] = session_id
                    session_feature_list.append(session_features)

            if len(session_feature_list) == 0:
                print(f"  Warning: No sessions processed for {subject_id}")
                continue

            # Aggregate sessions to subject level
            df_sessions = pd.DataFrame(session_feature_list)

            # Take median across sessions for Core5 features
            core5_features = ['duration_cv', 'duty_cycle', 'mean_duration_ms',
                             'median_duration_ms', 'motor_posterior_duty_ratio']

            subject_agg = {}
            for feature in core5_features:
                if feature in df_sessions.columns:
                    subject_agg[feature] = df_sessions[feature].median()

            # Add metadata
            subject_agg['subject_id'] = subject_id
            subject_agg['dataset'] = 'ds003490'
            subject_agg['n_sessions'] = len(session_feature_list)
            subject_agg['total_windows'] = df_sessions['n_windows'].sum()
            subject_agg['total_boundary_events'] = df_sessions['n_boundary_events'].sum()

            subject_features.append(subject_agg)
            processed_subjects += 1

            print(f"  Success: {len(session_feature_list)} sessions aggregated")

        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            continue

    if subject_features:
        # Save results
        df_subjects = pd.DataFrame(subject_features)

        output_dir = Path('/Users/user/Desktop/EEG_Exergamming/results/features')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'ds003490_core5.csv'
        df_subjects.to_csv(output_file, index=False)

        print(f"\n" + "=" * 80)
        print(f"PROCESSING COMPLETE")
        print(f"Subjects processed: {processed_subjects}/{len(subject_sessions)}")
        print(f"Output: {output_file}")
        print(f"Features: {list(df_subjects.columns)}")

        # Show Core5 feature summary
        core5_features = ['duration_cv', 'duty_cycle', 'mean_duration_ms',
                         'median_duration_ms', 'motor_posterior_duty_ratio']
        available_core5 = [f for f in core5_features if f in df_subjects.columns]

        if available_core5:
            print(f"\nCore5 Feature Summary:")
            print(df_subjects[available_core5].describe())

        print("=" * 80)

        return df_subjects
    else:
        print("No subjects processed successfully")
        return None

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Started at: {start_time}")

    results = process_ds003490_boundary_safe()

    end_time = datetime.now()
    print(f"Completed at: {end_time}")
    print(f"Duration: {end_time - start_time}")