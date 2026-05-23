#!/usr/bin/env python3
"""Add clinical labels to Core5 features using BIDS participants metadata."""

import pandas as pd
import sys
from pathlib import Path

def add_labels_to_ds004584():
    """Add labels to ds004584 features from participants.tsv."""

    # Load participants metadata
    participants_file = Path("bids/ds004584/participants.tsv")
    features_file = Path("data/features/core5/ds004584_core5_features.csv")

    if not participants_file.exists():
        print(f"❌ Participants file not found: {participants_file}")
        return False

    if not features_file.exists():
        print(f"❌ Features file not found: {features_file}")
        return False

    # Load data
    participants_df = pd.read_csv(participants_file, sep='\t')
    features_df = pd.read_csv(features_file)

    print(f"📊 Loaded participants: {len(participants_df)}")
    print(f"📊 Loaded features: {len(features_df)}")

    # Create label mapping
    label_mapping = {}
    for _, row in participants_df.iterrows():
        subject_id = row['participant_id']
        group = row['GROUP']
        # PD = 1, Control = 0
        label = 1 if group == 'PD' else 0
        label_mapping[subject_id] = label

    # Add labels to features
    features_df['label'] = features_df['subject_id'].map(label_mapping)

    # Check for missing labels
    missing_labels = features_df['label'].isna().sum()
    if missing_labels > 0:
        print(f"⚠️  Warning: {missing_labels} subjects missing labels")
        print("Subjects with missing labels:")
        missing_subjects = features_df[features_df['label'].isna()]['subject_id'].tolist()
        print(missing_subjects)

        # Remove subjects with missing labels
        features_df = features_df.dropna(subset=['label'])
        print(f"Removed {missing_labels} subjects with missing labels")

    # Convert labels to int
    features_df['label'] = features_df['label'].astype(int)

    # Print label distribution
    label_counts = features_df['label'].value_counts()
    print(f"📈 Label distribution:")
    print(f"   PD (label=1): {label_counts.get(1, 0)}")
    print(f"   Control (label=0): {label_counts.get(0, 0)}")

    # Save updated features
    features_df.to_csv(features_file, index=False)
    print(f"✅ Updated features saved to: {features_file}")

    return True

if __name__ == "__main__":
    print("="*60)
    print("ADDING CLINICAL LABELS TO CORE5 FEATURES")
    print("="*60)

    success = add_labels_to_ds004584()

    if success:
        print("✅ Labels successfully added!")
    else:
        print("❌ Failed to add labels")
        sys.exit(1)