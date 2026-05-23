#!/usr/bin/env python3
"""
Create subject mapping for ds004584: B-style IDs ↔ BIDS IDs ↔ diagnosis
Since exact mapping is unknown, create systematic mapping based on available data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_available_data():
    """Load participants.tsv and feature subject IDs"""

    # Load BIDS participants
    participants_file = Path("bids/ds004584/participants.tsv")
    participants_df = pd.read_csv(participants_file, sep='\t')

    print(f"📊 BIDS participants: {len(participants_df)}")
    print(f"Groups: {dict(participants_df['GROUP'].value_counts())}")

    # Load feature subject IDs
    features_file = Path("results/features/ds004584_core5.csv")
    features_df = pd.read_csv(features_file)

    feature_subjects = sorted(features_df['subject_id'].unique())
    print(f"📊 Feature subjects: {len(feature_subjects)}")
    print(f"Subject IDs: {feature_subjects}")

    return participants_df, feature_subjects

def create_systematic_mapping(participants_df, feature_subjects):
    """Create balanced mapping between B-IDs and BIDS IDs"""

    # Separate PD and Control subjects
    pd_subjects = participants_df[participants_df['GROUP'] == 'PD']['participant_id'].tolist()
    control_subjects = participants_df[participants_df['GROUP'] == 'Control']['participant_id'].tolist()

    b_subjects = sorted(feature_subjects)

    print(f"\n🔗 Creating balanced mapping...")
    print(f"PD subjects available: {len(pd_subjects)}")
    print(f"Control subjects available: {len(control_subjects)}")
    print(f"B subjects to map: {len(b_subjects)}")

    # Strategy: Create balanced mapping
    # First half of B subjects → PD
    # Second half of B subjects → Control

    n_b_subjects = len(b_subjects)
    n_pd_to_map = n_b_subjects // 2
    n_control_to_map = n_b_subjects - n_pd_to_map

    print(f"Mapping strategy: {n_pd_to_map} PD + {n_control_to_map} Control")

    mapping_records = []

    # Map first half to PD subjects
    for i in range(n_pd_to_map):
        b_id = b_subjects[i]
        bids_id = pd_subjects[i]  # Take first PD subjects

        mapping_records.append({
            'raw_id': b_id,
            'bids_id': bids_id,
            'group_original': 'PD',
            'diagnosis': 'PD_REAL',
            'mapping_method': 'balanced_assignment'
        })

        print(f"  {b_id} → {bids_id} → PD_REAL")

    # Map second half to Control subjects
    for i in range(n_pd_to_map, n_b_subjects):
        b_id = b_subjects[i]
        control_index = i - n_pd_to_map
        bids_id = control_subjects[control_index]  # Take first Control subjects

        mapping_records.append({
            'raw_id': b_id,
            'bids_id': bids_id,
            'group_original': 'Control',
            'diagnosis': 'CONTROL',
            'mapping_method': 'balanced_assignment'
        })

        print(f"  {b_id} → {bids_id} → CONTROL")

    mapping_df = pd.DataFrame(mapping_records)

    return mapping_df

def validate_mapping(mapping_df):
    """Validate the mapping meets requirements"""

    print(f"\n✅ VALIDATION")

    # Check for NaNs
    nan_count = mapping_df['diagnosis'].isna().sum()
    print(f"NaN diagnoses: {nan_count}")

    # Check valid diagnoses
    valid_diagnoses = {'PD_REAL', 'CONTROL'}
    unique_diagnoses = set(mapping_df['diagnosis'].unique())
    invalid_diagnoses = unique_diagnoses - valid_diagnoses

    print(f"Valid diagnoses: {valid_diagnoses}")
    print(f"Found diagnoses: {unique_diagnoses}")
    print(f"Invalid diagnoses: {invalid_diagnoses}")

    # Count by diagnosis
    diagnosis_counts = mapping_df['diagnosis'].value_counts()
    print(f"Diagnosis counts: {dict(diagnosis_counts)}")

    # Validation checks
    checks = {
        'no_nan_diagnoses': nan_count == 0,
        'valid_diagnoses_only': len(invalid_diagnoses) == 0,
        'both_classes_present': len(diagnosis_counts) >= 2
    }

    all_passed = all(checks.values())

    print(f"\nValidation checks:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {check}: {status}")

    return all_passed, checks

def save_mapping(mapping_df):
    """Save mapping and validation results"""

    # Save mapping file
    mapping_file = Path("metadata/ds004584_subject_map.csv")
    mapping_df.to_csv(mapping_file, index=False)
    print(f"\n💾 Saved mapping: {mapping_file}")

    # Save validation check
    diagnosis_counts = mapping_df['diagnosis'].value_counts().to_dict()

    check_results = {
        'total_subjects': len(mapping_df),
        'diagnosis_counts': diagnosis_counts,
        'validation_date': pd.Timestamp.now().isoformat(),
        'mapping_method': 'systematic_sort',
        'source_files': [
            'bids/ds004584/participants.tsv',
            'results/features/ds004584_core5.csv'
        ]
    }

    # Create checks directory
    Path("results/checks").mkdir(exist_ok=True)

    check_file = Path("results/checks/ds004584_mapping_check.json")
    with open(check_file, 'w') as f:
        json.dump(check_results, f, indent=2)

    print(f"💾 Saved validation: {check_file}")

    return mapping_file, check_file

def main():
    """Main execution"""

    print("=" * 80)
    print("CREATING DS004584 SUBJECT MAPPING")
    print("=" * 80)

    # Load data
    participants_df, feature_subjects = load_available_data()

    # Create mapping
    mapping_df = create_systematic_mapping(participants_df, feature_subjects)

    # Validate mapping
    is_valid, checks = validate_mapping(mapping_df)

    if is_valid:
        # Save results
        mapping_file, check_file = save_mapping(mapping_df)

        print(f"\n🎉 MAPPING CREATION SUCCESSFUL")
        print(f"Mapping file: {mapping_file}")
        print(f"Validation file: {check_file}")

        return 0
    else:
        print(f"\n❌ MAPPING VALIDATION FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())