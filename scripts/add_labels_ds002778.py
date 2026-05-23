#!/usr/bin/env python3
"""
Add clinical labels to ds002778 Core5 features based on participants.tsv
"""

import pandas as pd
from pathlib import Path
import sys

def add_labels_to_ds002778():
    """Add clinical labels to ds002778 features based on participants.tsv."""

    # File paths
    base_dir = Path(".")
    participants_file = base_dir / "data/bids/ds002778/participants.tsv"
    features_file = base_dir / "data/features/core5/ds002778_core5_features.csv"

    print(f"Reading participants file: {participants_file}")
    print(f"Reading features file: {features_file}")

    # Read participants info
    participants_df = pd.read_csv(participants_file, sep='\t')
    print(f"Found {len(participants_df)} participants in participants.tsv")

    # Read features
    features_df = pd.read_csv(features_file)
    print(f"Found {len(features_df)} subjects in features file")

    # Create mapping from subject_id to label
    label_mapping = {}
    for _, row in participants_df.iterrows():
        subject_id = row['participant_id']
        if subject_id.startswith('sub-hc'):
            label_mapping[subject_id] = 'CONTROL'
        elif subject_id.startswith('sub-pd'):
            label_mapping[subject_id] = 'PD_REAL'  # Using PD_REAL as default for PD subjects
        else:
            print(f"Warning: Unknown subject type: {subject_id}")

    print(f"Created label mapping for {len(label_mapping)} subjects")
    print(f"HC subjects: {sum(1 for v in label_mapping.values() if v == 'CONTROL')}")
    print(f"PD subjects: {sum(1 for v in label_mapping.values() if v == 'PD_REAL')}")

    # Add labels to features (convert to numeric: CONTROL=0, PD_REAL=1)
    string_labels = features_df['subject_id'].map(label_mapping)
    features_df['label'] = string_labels.apply(lambda x: 0 if x == 'CONTROL' else 1 if x == 'PD_REAL' else -1)

    # Check for missing labels
    missing_labels = features_df['label'].isna().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} subjects missing labels")
        print("Subjects with missing labels:")
        missing_subjects = features_df[features_df['label'].isna()]['subject_id'].tolist()
        print(missing_subjects)

    # Display label distribution in features
    label_counts = features_df['label'].value_counts()
    print(f"\nLabel distribution in processed features:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # Save updated features
    features_df.to_csv(features_file, index=False)
    print(f"\n✅ Updated features saved to {features_file}")

    # Display first few rows to verify
    print(f"\nFirst few rows with labels:")
    print(features_df[['subject_id', 'dataset_id', 'session_id', 'label']].head())

    return features_df

if __name__ == "__main__":
    add_labels_to_ds002778()