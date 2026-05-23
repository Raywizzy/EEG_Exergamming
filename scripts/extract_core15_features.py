#!/usr/bin/env python3
"""
Extract Core15+ features from all 3 labeled datasets
Builds on Core5 features to achieve clinical utility threshold
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
import logging

# Add src to path for imports
sys.path.append('/Users/user/Desktop/EEG_Exergamming/src')
from src.features.core15 import Core15FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure MNE to be less verbose
mne.set_log_level('WARNING')

def load_config():
    """Load preprocessing configuration"""
    config_path = Path('/Users/user/Desktop/EEG_Exergamming/config/preprocessing_config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config if file doesn't exist
        return {
            'sfreq': 512,
            'beta_band_hz': [13, 30],
            'filter_low': 1.0,
            'filter_high': 40.0,
            'notch_freq': 50.0
        }

def preprocess_raw(raw, config):
    """Apply preprocessing to raw EEG data"""

    # Filter
    raw.filter(l_freq=config['filter_low'], h_freq=config['filter_high'], verbose=False)

    # Notch filter for line noise
    raw.notch_filter(config['notch_freq'], verbose=False)

    # Resample if needed
    if raw.info['sfreq'] != config['sfreq']:
        raw.resample(config['sfreq'], verbose=False)

    return raw

def extract_features_ds003490():
    """Extract Core15+ features from ds003490 dataset"""

    logger.info("Processing ds003490...")

    config = load_config()
    extractor = Core15FeatureExtractor(config)

    # Load labeled data to get subject list
    labeled_file = Path("results/features/ds003490_core5_labeled.csv")
    if not labeled_file.exists():
        logger.error(f"Labeled file not found: {labeled_file}")
        return None

    labeled_df = pd.read_csv(labeled_file)
    subject_ids = labeled_df['subject_id'].unique()

    logger.info(f"Found {len(subject_ids)} subjects to process")

    feature_records = []

    for subject_id in subject_ids:
        logger.info(f"Processing subject {subject_id}")

        try:
            # Find EEG files for this subject
            bids_path = Path("bids")
            eeg_files = list(bids_path.glob(f"**/*{subject_id}*/*.set"))

            if not eeg_files:
                logger.warning(f"No EEG files found for subject {subject_id}")
                continue

            for eeg_file in eeg_files[:3]:  # Process up to 3 sessions
                try:
                    # Load EEG data
                    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

                    # Preprocess
                    raw = preprocess_raw(raw, config)

                    # Get data and channel names
                    data = raw.get_data()
                    ch_names = raw.ch_names

                    # Extract Core15+ features
                    features = extractor.extract_all_features(data, ch_names)

                    # Add metadata
                    features['subject_id'] = subject_id
                    features['dataset'] = 'ds003490'
                    features['session_file'] = eeg_file.name

                    feature_records.append(features)

                    logger.info(f"Extracted features for {subject_id} - {eeg_file.name}")

                except Exception as e:
                    logger.error(f"Error processing {eeg_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {e}")
            continue

        # Memory cleanup
        gc.collect()

    return pd.DataFrame(feature_records)

def extract_features_ds002778():
    """Extract Core15+ features from ds002778 dataset"""

    logger.info("Processing ds002778...")

    config = load_config()
    extractor = Core15FeatureExtractor(config)

    # Load labeled data to get subject list
    labeled_file = Path("results/features/ds002778_core5_labeled.csv")
    if not labeled_file.exists():
        logger.error(f"Labeled file not found: {labeled_file}")
        return None

    labeled_df = pd.read_csv(labeled_file)
    subject_ids = labeled_df['subject_id'].unique()

    logger.info(f"Found {len(subject_ids)} subjects to process")

    feature_records = []

    for subject_id in subject_ids:
        logger.info(f"Processing subject {subject_id}")

        try:
            # Find EEG files for this subject
            bids_path = Path("bids")
            eeg_files = list(bids_path.glob(f"**/*{subject_id}*/*.set"))

            if not eeg_files:
                logger.warning(f"No EEG files found for subject {subject_id}")
                continue

            for eeg_file in eeg_files[:3]:  # Process up to 3 sessions
                try:
                    # Load EEG data
                    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

                    # Preprocess
                    raw = preprocess_raw(raw, config)

                    # Get data and channel names
                    data = raw.get_data()
                    ch_names = raw.ch_names

                    # Extract Core15+ features
                    features = extractor.extract_all_features(data, ch_names)

                    # Add metadata
                    features['subject_id'] = subject_id
                    features['dataset'] = 'ds002778'
                    features['session_file'] = eeg_file.name

                    feature_records.append(features)

                    logger.info(f"Extracted features for {subject_id} - {eeg_file.name}")

                except Exception as e:
                    logger.error(f"Error processing {eeg_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {e}")
            continue

        # Memory cleanup
        gc.collect()

    return pd.DataFrame(feature_records)

def extract_features_ds004584():
    """Extract Core15+ features from ds004584 dataset"""

    logger.info("Processing ds004584...")

    config = load_config()
    extractor = Core15FeatureExtractor(config)

    # Load labeled data to get subject list
    labeled_file = Path("results/features/ds004584_core5_labeled.csv")
    if not labeled_file.exists():
        logger.error(f"Labeled file not found: {labeled_file}")
        return None

    labeled_df = pd.read_csv(labeled_file)
    subject_ids = labeled_df['subject_id'].unique()

    logger.info(f"Found {len(subject_ids)} subjects to process")

    feature_records = []

    for subject_id in subject_ids:
        logger.info(f"Processing subject {subject_id}")

        try:
            # Find EEG files for this subject
            bids_path = Path("bids")
            eeg_files = list(bids_path.glob(f"**/*{subject_id}*/*.set"))

            if not eeg_files:
                logger.warning(f"No EEG files found for subject {subject_id}")
                continue

            for eeg_file in eeg_files[:3]:  # Process up to 3 sessions
                try:
                    # Load EEG data
                    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

                    # Preprocess
                    raw = preprocess_raw(raw, config)

                    # Get data and channel names
                    data = raw.get_data()
                    ch_names = raw.ch_names

                    # Extract Core15+ features
                    features = extractor.extract_all_features(data, ch_names)

                    # Add metadata
                    features['subject_id'] = subject_id
                    features['dataset'] = 'ds004584'
                    features['session_file'] = eeg_file.name

                    feature_records.append(features)

                    logger.info(f"Extracted features for {subject_id} - {eeg_file.name}")

                except Exception as e:
                    logger.error(f"Error processing {eeg_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {e}")
            continue

        # Memory cleanup
        gc.collect()

    return pd.DataFrame(feature_records)

def add_labels_to_features(features_df, dataset_id):
    """Add clinical labels to Core15+ features"""

    if dataset_id == 'ds004584':
        # Load subject mapping
        mapping_file = Path("metadata/ds004584_subject_map.csv")
        if mapping_file.exists():
            mapping_df = pd.read_csv(mapping_file)
            features_df = features_df.merge(
                mapping_df[['raw_id', 'diagnosis']],
                left_on='subject_id',
                right_on='raw_id',
                how='left'
            )
            features_df = features_df.rename(columns={'diagnosis': 'label'})
        else:
            logger.error(f"Subject mapping not found: {mapping_file}")
            return None
    else:
        # Load existing labels from Core5 labeled files
        labeled_file = Path(f"results/features/{dataset_id}_core5_labeled.csv")
        if labeled_file.exists():
            labeled_df = pd.read_csv(labeled_file)
            subject_labels = labeled_df[['subject_id', 'label']].drop_duplicates()
            features_df = features_df.merge(subject_labels, on='subject_id', how='left')
        else:
            logger.error(f"Labeled file not found: {labeled_file}")
            return None

    return features_df

def main():
    """Main execution"""

    start_time = datetime.now()
    logger.info(f"Started Core15+ feature extraction at: {start_time}")
    logger.info("="*80)

    # Create output directory
    output_dir = Path("results/features")
    output_dir.mkdir(exist_ok=True)

    # Process each dataset
    datasets = ['ds003490', 'ds002778', 'ds004584']
    extraction_functions = [
        extract_features_ds003490,
        extract_features_ds002778,
        extract_features_ds004584
    ]

    for dataset_id, extract_func in zip(datasets, extraction_functions):
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING {dataset_id.upper()}")
        logger.info(f"{'='*50}")

        try:
            # Extract features
            features_df = extract_func()

            if features_df is not None and len(features_df) > 0:
                # Add labels
                labeled_features_df = add_labels_to_features(features_df, dataset_id)

                if labeled_features_df is not None:
                    # Save labeled features
                    output_file = output_dir / f"{dataset_id}_core15_labeled.csv"
                    labeled_features_df.to_csv(output_file, index=False)

                    logger.info(f"✅ Saved {dataset_id}: {output_file}")
                    logger.info(f"Records: {len(labeled_features_df)}")
                    logger.info(f"Subjects: {len(labeled_features_df['subject_id'].unique())}")
                    logger.info(f"Features: {len([col for col in labeled_features_df.columns if col not in ['subject_id', 'dataset', 'session_file', 'label']])}")

                    # Check labels
                    if 'label' in labeled_features_df.columns:
                        label_counts = labeled_features_df['label'].value_counts()
                        logger.info(f"Labels: {dict(label_counts)}")
                else:
                    logger.error(f"❌ Failed to add labels to {dataset_id}")
            else:
                logger.error(f"❌ No features extracted for {dataset_id}")

        except Exception as e:
            logger.error(f"❌ Error processing {dataset_id}: {e}")

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "="*80)
    logger.info("CORE15+ FEATURE EXTRACTION COMPLETE")
    logger.info("="*80)

    # Check output files
    for dataset_id in datasets:
        output_file = output_dir / f"{dataset_id}_core15_labeled.csv"
        if output_file.exists():
            df = pd.read_csv(output_file)
            logger.info(f"✅ {dataset_id}: {len(df)} records, {len(df['subject_id'].unique())} subjects")
        else:
            logger.info(f"❌ {dataset_id}: Missing output file")

    logger.info(f"\nStarted: {start_time}")
    logger.info(f"Completed: {end_time}")
    logger.info(f"Duration: {duration}")

    return 0

if __name__ == "__main__":
    sys.exit(main())