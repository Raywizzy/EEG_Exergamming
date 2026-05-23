#!/usr/bin/env python3
"""
Stage 1: Dataset Documentation and Specification
Creates comprehensive documentation of the University of Leicester EEG dataset.
Must complete successfully before proceeding to Stage 2.

HARD RULES:
- Only use approved datasets from DATA_SOURCES.md
- No synthetic or fabricated data allowed
- All paths must exist and be verified
- Subject-wise isolation enforced
- Classes must be CONTROL, PD_REAL, PD_SHAM only
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def print_timestamp(message):
    """Print message with ISO timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    iso_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"[{timestamp}] {message}")
    print(f"ISO Date: {iso_date}")

def verify_data_path(data_path):
    """Verify that data path exists and contains expected files."""
    if not os.path.exists(data_path):
        print(f"ERROR: Data path does not exist: {data_path}")
        print("REQUIRED ACTION: Provide the correct path to the University of Leicester dataset")
        return False

    print(f"✓ Data path verified: {data_path}")
    return True

def document_dataset_structure(data_path, output_path):
    """
    Document the complete dataset structure.
    Must identify all subjects, sessions, and conditions.
    """
    print_timestamp("Starting dataset structure documentation")

    # This is a template - actual implementation depends on dataset format
    dataset_info = {
        "dataset_name": "University of Leicester EEG Exergaming Dataset",
        "documentation_date": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "signal_type": "EEG",
        "channels": None,  # To be determined from actual data
        "sampling_rate": None,  # To be determined
        "subjects": {},
        "sessions": {},
        "conditions": [],
        "file_format": None,  # To be determined
        "total_files": 0,
        "classes": {
            "CONTROL": {"count": 0, "subjects": []},
            "PD_REAL": {"count": 0, "subjects": []},
            "PD_SHAM": {"count": 0, "subjects": []}
        }
    }

    # Save initial documentation structure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"✓ Dataset structure template saved to: {output_path}")
    print("NEXT STEP: Complete dataset documentation with actual file analysis")

    return dataset_info

def validate_class_labels(labels):
    """Validate that only approved class labels are present."""
    approved_classes = {"CONTROL", "PD_REAL", "PD_SHAM"}
    unique_labels = set(labels)

    # Check for invalid labels
    invalid_labels = unique_labels - approved_classes
    if invalid_labels:
        print(f"ERROR: Invalid class labels found: {invalid_labels}")
        print(f"Only these classes are allowed: {approved_classes}")
        return False

    # Check for NaN or missing values
    if pd.isna(labels).any():
        print("ERROR: NaN values found in class labels")
        return False

    print(f"✓ Class labels validated: {unique_labels}")
    return True

def create_subject_summary(dataset_info):
    """Create summary of subjects and ensure subject-wise isolation."""
    summary = {
        "total_subjects": len(dataset_info["subjects"]),
        "subjects_per_class": dataset_info["classes"],
        "validation_approach": "subject-wise cross-validation",
        "isolation_verified": True  # Will be set based on actual verification
    }

    print("SUBJECT SUMMARY:")
    print(f"Total subjects: {summary['total_subjects']}")
    for class_name, info in summary['subjects_per_class'].items():
        print(f"{class_name}: {info['count']} subjects")

    return summary

def main():
    """Main execution function for Stage 1."""
    print("=" * 80)
    print("STAGE 1: DATASET DOCUMENTATION AND SPECIFICATION")
    print("=" * 80)

    print_timestamp("Stage 1 started")

    # Configuration
    # NOTE: This path MUST be provided by the user before proceeding
    data_path = os.getenv("EEG_DATA_PATH")
    if not data_path:
        print("ERROR: EEG_DATA_PATH environment variable not set")
        print("REQUIRED ACTION:")
        print("1. Set environment variable: export EEG_DATA_PATH=/path/to/leicester/data")
        print("2. Or modify this script with the correct path")
        print("CANNOT PROCEED without valid data path")
        sys.exit(1)

    output_dir = Path(__file__).parent.parent / "data" / "interim"
    dataset_doc_path = output_dir / "dataset_specification.json"

    # Step 1: Verify data path exists
    if not verify_data_path(data_path):
        print("STAGE 1 FAILED: Data path verification failed")
        sys.exit(1)

    # Step 2: Document dataset structure
    dataset_info = document_dataset_structure(data_path, dataset_doc_path)

    # Step 3: Create acceptance criteria checklist
    acceptance_criteria = {
        "data_path_verified": os.path.exists(data_path),
        "dataset_documented": os.path.exists(dataset_doc_path),
        "classes_validated": False,  # Will be True after actual data analysis
        "subject_isolation_verified": False,  # Will be True after validation
        "file_inventory_complete": False,  # Will be True after file analysis
        "no_synthetic_data": True,  # Must remain True
        "stage1_complete": False
    }

    # Save acceptance criteria
    criteria_path = output_dir / "stage1_acceptance_criteria.json"
    with open(criteria_path, 'w') as f:
        json.dump(acceptance_criteria, f, indent=2)

    print_timestamp("Stage 1 template completed")
    print("\nSTAGE 1 STATUS: TEMPLATE CREATED")
    print("=" * 80)
    print("REQUIRED ACTIONS TO COMPLETE STAGE 1:")
    print("1. Provide actual dataset path via EEG_DATA_PATH environment variable")
    print("2. Run complete file analysis and inventory")
    print("3. Validate all class labels")
    print("4. Verify subject-wise data organization")
    print("5. Complete acceptance criteria checklist")
    print("=" * 80)

    print(f"✓ Files created:")
    print(f"  - {dataset_doc_path}")
    print(f"  - {criteria_path}")

    return acceptance_criteria

if __name__ == "__main__":
    main()