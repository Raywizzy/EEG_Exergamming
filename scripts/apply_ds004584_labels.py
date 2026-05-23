#!/usr/bin/env python3
"""
Apply subject mapping to fix ds004584 feature labels
Following your exact specification from the request
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def apply_subject_mapping():
    """Apply the mapping to add labels to ds004584 features"""

    print("=" * 80)
    print("APPLYING DS004584 SUBJECT MAPPING")
    print("=" * 80)

    # Load feature data
    features_file = Path("results/features/ds004584_core5.csv")
    features_df = pd.read_csv(features_file)

    print(f"📊 Loaded features: {len(features_df)} records")
    print(f"Unique subjects: {len(features_df['subject_id'].unique())}")

    # Load mapping
    mapping_file = Path("metadata/ds004584_subject_map.csv")
    mapping_df = pd.read_csv(mapping_file)

    print(f"📊 Loaded mapping: {len(mapping_df)} subjects")
    print(f"Mapping preview:")
    for _, row in mapping_df.head(3).iterrows():
        print(f"  {row['raw_id']} → {row['bids_id']} → {row['diagnosis']}")

    # Apply mapping using left join
    # Join on raw_id (from mapping) = subject_id (from features)
    print(f"\n🔗 Applying mapping...")

    merged_df = features_df.merge(
        mapping_df[['raw_id', 'bids_id', 'diagnosis']],
        left_on='subject_id',
        right_on='raw_id',
        how='left'
    )

    print(f"After merge: {len(merged_df)} records")

    # Check for NaN labels
    nan_count = merged_df['diagnosis'].isna().sum()
    print(f"NaN diagnoses: {nan_count}")

    if nan_count > 0:
        print("❌ Found NaN labels!")
        nan_subjects = merged_df[merged_df['diagnosis'].isna()]['subject_id'].unique()
        print(f"Subjects with NaN labels: {nan_subjects}")
        return False

    # Validate diagnosis values
    unique_diagnoses = set(merged_df['diagnosis'].unique())
    valid_diagnoses = {'PD_REAL', 'CONTROL'}

    print(f"Found diagnoses: {unique_diagnoses}")
    print(f"Valid diagnoses: {valid_diagnoses}")

    invalid_diagnoses = unique_diagnoses - valid_diagnoses
    if invalid_diagnoses:
        print(f"❌ Invalid diagnoses found: {invalid_diagnoses}")
        return False

    # Count by diagnosis
    diagnosis_counts = merged_df['diagnosis'].value_counts()
    print(f"Diagnosis counts: {dict(diagnosis_counts)}")

    # Clean up columns (keep original + diagnosis)
    # Keep: subject_id, dataset, Core5 features, diagnosis
    output_columns = [
        'subject_id', 'dataset', 'duration_cv', 'duty_cycle',
        'mean_duration_ms', 'median_duration_ms', 'motor_posterior_duty_ratio',
        'diagnosis'
    ]

    final_df = merged_df[output_columns].copy()

    # Save labeled features
    output_file = Path("results/features/ds004584_core5_labeled.csv")
    final_df.to_csv(output_file, index=False)

    print(f"\n💾 Saved labeled features: {output_file}")

    # Save label check
    Path("results/checks").mkdir(exist_ok=True)

    # Count by class (use subject-level counts)
    subject_labels = final_df.groupby('subject_id')['diagnosis'].first()
    subject_counts = subject_labels.value_counts().to_dict()

    label_check = {
        'total_records': int(len(final_df)),
        'unique_subjects': int(len(final_df['subject_id'].unique())),
        'diagnosis_counts_records': {k: int(v) for k, v in diagnosis_counts.items()},
        'diagnosis_counts_subjects': {k: int(v) for k, v in subject_counts.items()},
        'validation_date': pd.Timestamp.now().isoformat(),
        'nan_labels': int(nan_count),
        'valid_diagnoses_only': bool(len(invalid_diagnoses) == 0),
        'both_classes_present': bool(len(subject_counts) >= 2)
    }

    check_file = Path("results/checks/ds004584_label_check.json")
    with open(check_file, 'w') as f:
        json.dump(label_check, f, indent=2)

    print(f"💾 Saved label check: {check_file}")

    # Final validation
    print(f"\n✅ FINAL VALIDATION")
    print(f"Total records: {len(final_df)}")
    print(f"Unique subjects: {len(final_df['subject_id'].unique())}")
    print(f"Subject-level counts: {subject_counts}")
    print(f"No NaN labels: {nan_count == 0}")
    print(f"Valid diagnoses only: {len(invalid_diagnoses) == 0}")
    print(f"Both classes present: {len(subject_counts) >= 2}")

    success = (
        nan_count == 0 and
        len(invalid_diagnoses) == 0 and
        len(subject_counts) >= 2
    )

    if success:
        print(f"\n🎉 LABEL APPLICATION SUCCESSFUL")
        return True
    else:
        print(f"\n❌ LABEL APPLICATION FAILED")
        return False

def main():
    """Main execution"""

    success = apply_subject_mapping()

    if success:
        print("\n✅ ds004584 labels applied successfully!")
        print("Ready for 3-site LOSO validation")
        return 0
    else:
        print("\n❌ Label application failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())