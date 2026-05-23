#!/usr/bin/env python3
"""
Targeted processing script for ds003490 with session-based structure
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('/Users/user/Desktop/EEG_Exergamming/src')

from src.preprocess.pipeline import EEGPreprocessor
from src.features.core5 import Core5FeatureExtractor, BetaBurstDetector

def process_ds003490():
    """Process ds003490 dataset for Core5 feature extraction"""

    print("="*80)
    print("DS003490 SPECIFIC PROCESSING PIPELINE")
    print("="*80)

    # Initialize components
    config_path = Path('/Users/user/Desktop/EEG_Exergamming/config/preprocessing_config.yaml')
    preprocessor = EEGPreprocessor(config_path)
    extractor = Core5FeatureExtractor(str(config_path))

    # Dataset path
    dataset_path = Path('/Users/user/Desktop/EEG_Exergamming/data/bids/ds003490')

    # Find all REST EEG files
    rest_files = list(dataset_path.glob('sub-*/ses-*/eeg/*task-Rest_eeg.set'))
    print(f"Found {len(rest_files)} REST EEG files")

    if len(rest_files) == 0:
        print("No REST files found - checking file structure...")
        return

    # Process files and collect features
    features_list = []
    processed_count = 0
    max_subjects = 60  # Limit for initial processing

    for eeg_file in rest_files[:max_subjects]:
        try:
            print(f"Processing: {eeg_file.name}")

            # Extract subject and session info
            parts = eeg_file.name.split('_')
            subject_id = parts[0]  # sub-XXX
            session_id = parts[1]  # ses-XX

            # Load and preprocess
            raw = preprocessor.load_eeg(eeg_file)
            raw_processed = preprocessor.preprocess(raw)

            # Extract Core5 features (detector is embedded in extractor)
            data = raw_processed.get_data()  # Shape: [channels, times]
            channel_names = raw_processed.ch_names
            sfreq = raw_processed.info['sfreq']

            # Extract features using the main API
            features = extractor.extract_core5_features(
                beta_data=data,
                sfreq=sfreq,
                channel_names=channel_names,
                subject_metadata={'subject_id': subject_id, 'session_id': session_id}
            )

            # Add metadata
            features['subject_id'] = subject_id
            features['session_id'] = session_id
            features['file_path'] = str(eeg_file)
            features['dataset'] = 'ds003490'

            features_list.append(features)
            processed_count += 1

            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(rest_files[:max_subjects])} files")

        except Exception as e:
            print(f"Error processing {eeg_file.name}: {e}")
            continue

    if features_list:
        # Combine all features
        df_features = pd.DataFrame(features_list)

        # Add labels (will need participants.tsv for proper labeling)
        df_features['label'] = 0  # Placeholder - will update with real labels

        # Save features
        output_dir = Path('/Users/user/Desktop/EEG_Exergamming/results/features/labelled')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'ds003490_core5.csv'
        df_features.to_csv(output_file, index=False)

        print(f"\nProcessing complete!")
        print(f"Processed: {len(df_features)} subjects")
        print(f"Features saved to: {output_file}")
        print(f"Feature columns: {list(df_features.columns)}")

        # Display summary statistics
        core5_cols = ['duration_cv', 'duty_cycle', 'mean_duration_ms', 'median_duration_ms', 'motor_posterior_duty_ratio']
        print("\nCore5 Feature Summary:")
        print(df_features[core5_cols].describe())

        return df_features
    else:
        print("No features extracted - check processing pipeline")
        return None

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Started processing at: {start_time}")

    features = process_ds003490()

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Completed at: {end_time}")
    print(f"Total duration: {duration}")