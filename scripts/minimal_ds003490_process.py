#!/usr/bin/env python3
"""
Minimal ds003490 processing script to handle boundary events and extract basic features
"""

import sys
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure MNE to be less verbose
mne.set_log_level('WARNING')

def process_minimal_ds003490():
    """Process ds003490 with minimal preprocessing to extract basic features"""

    print("=" * 60)
    print("MINIMAL DS003490 PROCESSING")
    print("=" * 60)

    # Find all REST files
    dataset_path = Path('/Users/user/Desktop/EEG_Exergamming/data/bids/ds003490')
    rest_files = list(dataset_path.glob('sub-*/ses-*/eeg/*task-Rest_eeg.set'))

    print(f"Found {len(rest_files)} REST files")

    results = []
    processed = 0

    for i, eeg_file in enumerate(rest_files[:10]):  # Test first 10 files
        try:
            print(f"\n[{i+1}/{min(10, len(rest_files))}] Processing: {eeg_file.name}")

            # Extract subject and session info
            parts = eeg_file.name.split('_')
            subject_id = parts[0]  # sub-XXX
            session_id = parts[1]  # ses-XX

            # Load raw data with minimal processing
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

            print(f"  Loaded: {len(raw.ch_names)} channels, {len(raw.times)} samples, {raw.info['sfreq']} Hz")

            # Basic filtering only
            raw.filter(l_freq=1, h_freq=45, verbose=False)
            raw.filter(l_freq=13, h_freq=30, verbose=False)  # Beta band

            # Get data
            data = raw.get_data()

            # Calculate very basic features
            beta_power = np.mean(data ** 2, axis=1)  # Power per channel
            total_beta_power = np.mean(beta_power)

            # Create basic feature dict
            features = {
                'subject_id': subject_id,
                'session_id': session_id,
                'dataset': 'ds003490',
                'n_channels': len(raw.ch_names),
                'duration_s': len(raw.times) / raw.info['sfreq'],
                'beta_power': total_beta_power,
                'file_path': str(eeg_file)
            }

            results.append(features)
            processed += 1

            print(f"  Success: Beta power = {total_beta_power:.2e}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if results:
        # Save results
        df = pd.DataFrame(results)
        output_dir = Path('/Users/user/Desktop/EEG_Exergamming/results/features')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'ds003490_minimal_features.csv'
        df.to_csv(output_file, index=False)

        print(f"\n" + "=" * 60)
        print(f"PROCESSING COMPLETE")
        print(f"Processed: {processed}/{len(rest_files[:10])} files")
        print(f"Output: {output_file}")
        print(f"Features extracted: {list(df.columns)}")
        print("=" * 60)

        return df
    else:
        print("No files processed successfully")
        return None

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Started at: {start_time}")

    results = process_minimal_ds003490()

    end_time = datetime.now()
    print(f"Completed at: {end_time}")
    print(f"Duration: {end_time - start_time}")