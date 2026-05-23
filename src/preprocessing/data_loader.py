#!/usr/bin/env python3
"""
Data Loader for EEG Exergaming Project
Handles loading and parsing of EEG data from various formats.

Supports:
- Leicester CSV format (PSD per epoch)
- MRC BNDU format (when available)
- Subject-wise organization
- Condition mapping (PD_REAL vs PD_SHAM)
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
import re
from typing import Dict, List, Tuple, Optional

class EEGDataLoader:
    """Load and organize EEG data with proper subject isolation."""

    def __init__(self, data_path: str, metadata_path: str = None):
        """
        Initialize data loader.

        Parameters:
        -----------
        data_path : str
            Path to dataset directory
        metadata_path : str, optional
            Path to subject metadata JSON file
        """
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.metadata = self._load_metadata()
        self.subjects_data = {}

    def _load_metadata(self) -> Dict:
        """Load subject metadata if available."""
        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def load_leicester_subject(self, subject_id: str) -> Dict:
        """
        Load data for a single Leicester subject.

        Parameters:
        -----------
        subject_id : str
            Subject identifier (e.g., 'A1', 'B2')

        Returns:
        --------
        Dict with subject data including sessions, conditions, etc.
        """
        subject_path = self.data_path / subject_id

        if not subject_path.exists():
            raise ValueError(f"Subject directory not found: {subject_path}")

        # Get all CSV files for this subject
        csv_files = list(subject_path.glob("*.csv"))
        pickle_files = list(subject_path.glob("*_pickle_new"))

        sessions_data = {}
        for csv_file in csv_files:
            session_match = re.search(r'session(\d+)_', csv_file.name)
            if session_match:
                session_num = int(session_match.group(1))

                # Load CSV data
                df = pd.read_csv(csv_file)

                # Extract metadata from filename
                date_time_match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}_\d{2}_\d{2})', csv_file.name)
                session_date = date_time_match.group(1) if date_time_match else None
                session_time = date_time_match.group(2) if date_time_match else None

                sessions_data[session_num] = {
                    "csv_data": df,
                    "csv_path": str(csv_file),
                    "session_date": session_date,
                    "session_time": session_time,
                    "epoch_count": len(df),
                    "psd_stats": {
                        "mean": df['PSD'].mean(),
                        "std": df['PSD'].std(),
                        "min": df['PSD'].min(),
                        "max": df['PSD'].max(),
                        "median": df['PSD'].median()
                    }
                }

                # Check for corresponding pickle file
                pickle_pattern = csv_file.name.replace('.csv', '_pickle_new')
                pickle_path = subject_path / pickle_pattern
                if pickle_path.exists():
                    sessions_data[session_num]["pickle_path"] = str(pickle_path)

        # Get condition from metadata
        condition = "UNKNOWN"
        if self.metadata and "subjects" in self.metadata:
            subject_meta = self.metadata["subjects"].get(subject_id, {})
            condition = subject_meta.get("condition", "UNKNOWN")

        return {
            "subject_id": subject_id,
            "condition": condition,
            "sessions": sessions_data,
            "total_sessions": len(sessions_data),
            "csv_files": len(csv_files),
            "pickle_files": len(pickle_files)
        }

    def load_all_subjects(self, conditions: List[str] = None) -> Dict[str, Dict]:
        """
        Load data for all subjects or specified conditions.

        Parameters:
        -----------
        conditions : List[str], optional
            List of conditions to include (e.g., ['PD_REAL', 'PD_SHAM'])

        Returns:
        --------
        Dict mapping subject_id to subject data
        """
        if not conditions:
            conditions = ['PD_REAL', 'PD_SHAM']

        all_subjects = {}

        # Get list of subject directories
        subject_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name

            try:
                subject_data = self.load_leicester_subject(subject_id)

                # Filter by condition
                if subject_data["condition"] in conditions:
                    all_subjects[subject_id] = subject_data
                    print(f"✓ Loaded {subject_id}: {subject_data['condition']} "
                          f"({subject_data['total_sessions']} sessions)")

            except Exception as e:
                print(f"⚠ Warning: Could not load {subject_id}: {e}")

        print(f"\nLoaded {len(all_subjects)} subjects:")
        condition_counts = {}
        for subject_data in all_subjects.values():
            condition = subject_data["condition"]
            condition_counts[condition] = condition_counts.get(condition, 0) + 1

        for condition, count in condition_counts.items():
            print(f"  {condition}: {count} subjects")

        return all_subjects

    def get_subject_sessions_data(self, subject_id: str, session_numbers: List[int] = None) -> List[pd.DataFrame]:
        """
        Get session data for a subject as list of DataFrames.

        Parameters:
        -----------
        subject_id : str
            Subject identifier
        session_numbers : List[int], optional
            Specific session numbers to retrieve

        Returns:
        --------
        List of pandas DataFrames with session data
        """
        if subject_id not in self.subjects_data:
            self.subjects_data[subject_id] = self.load_leicester_subject(subject_id)

        subject_data = self.subjects_data[subject_id]
        sessions = subject_data["sessions"]

        if session_numbers:
            available_sessions = [num for num in session_numbers if num in sessions]
        else:
            available_sessions = list(sessions.keys())

        session_dataframes = []
        for session_num in sorted(available_sessions):
            df = sessions[session_num]["csv_data"].copy()
            df["subject_id"] = subject_id
            df["session_number"] = session_num
            df["condition"] = subject_data["condition"]
            session_dataframes.append(df)

        return session_dataframes

    def create_combined_dataset(self, subjects: List[str] = None,
                              max_sessions_per_subject: int = None) -> pd.DataFrame:
        """
        Create combined dataset with all subjects and sessions.

        Parameters:
        -----------
        subjects : List[str], optional
            List of subject IDs to include
        max_sessions_per_subject : int, optional
            Limit number of sessions per subject

        Returns:
        --------
        Combined DataFrame with all data
        """
        if not subjects:
            subjects = list(self.subjects_data.keys())

        combined_data = []

        for subject_id in subjects:
            session_dataframes = self.get_subject_sessions_data(subject_id)

            # Limit sessions if requested
            if max_sessions_per_subject:
                session_dataframes = session_dataframes[:max_sessions_per_subject]

            combined_data.extend(session_dataframes)

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            print(f"✓ Combined dataset created: {len(combined_df)} epochs from {len(subjects)} subjects")
            return combined_df
        else:
            print("⚠ No data found to combine")
            return pd.DataFrame()

    def get_subject_condition_mapping(self) -> Dict[str, str]:
        """Get mapping of subject IDs to conditions."""
        mapping = {}
        for subject_id, data in self.subjects_data.items():
            mapping[subject_id] = data["condition"]
        return mapping

    def validate_subject_isolation(self, train_subjects: List[str],
                                 test_subjects: List[str]) -> bool:
        """
        Validate that there's no subject leakage between train/test sets.

        Parameters:
        -----------
        train_subjects : List[str]
            Training subject IDs
        test_subjects : List[str]
            Test subject IDs

        Returns:
        --------
        bool: True if no overlap, False otherwise
        """
        train_set = set(train_subjects)
        test_set = set(test_subjects)
        overlap = train_set.intersection(test_set)

        if overlap:
            print(f"❌ Subject isolation violation! Overlapping subjects: {overlap}")
            return False
        else:
            print(f"✓ Subject isolation validated: {len(train_set)} train, {len(test_set)} test")
            return True

def load_pickle_data(pickle_path: str) -> any:
    """
    Load pickle file data.

    Parameters:
    -----------
    pickle_path : str
        Path to pickle file

    Returns:
    --------
    Loaded pickle data
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file {pickle_path}: {e}")
        return None

def create_data_summary(data_loader: EEGDataLoader, subjects_data: Dict) -> Dict:
    """
    Create comprehensive summary of loaded data.

    Parameters:
    -----------
    data_loader : EEGDataLoader
        Data loader instance
    subjects_data : Dict
        Dictionary of subject data

    Returns:
    --------
    Dict with data summary statistics
    """
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_subjects": len(subjects_data),
        "conditions": {},
        "sessions_stats": {
            "total_sessions": 0,
            "sessions_per_subject": {}
        },
        "epoch_stats": {
            "total_epochs": 0,
            "epochs_per_subject": {},
            "psd_statistics": {
                "overall_mean": 0,
                "overall_std": 0,
                "overall_min": float('inf'),
                "overall_max": float('-inf')
            }
        }
    }

    all_psd_values = []

    for subject_id, subject_data in subjects_data.items():
        condition = subject_data["condition"]
        sessions = subject_data["sessions"]

        # Condition counts
        if condition not in summary["conditions"]:
            summary["conditions"][condition] = 0
        summary["conditions"][condition] += 1

        # Session counts
        session_count = len(sessions)
        summary["sessions_stats"]["total_sessions"] += session_count
        summary["sessions_stats"]["sessions_per_subject"][subject_id] = session_count

        # Epoch statistics
        subject_epochs = 0
        subject_psd_values = []

        for session_num, session_data in sessions.items():
            epoch_count = session_data["epoch_count"]
            subject_epochs += epoch_count

            # Collect PSD values
            df = session_data["csv_data"]
            psd_values = df["PSD"].values
            subject_psd_values.extend(psd_values)
            all_psd_values.extend(psd_values)

        summary["epoch_stats"]["total_epochs"] += subject_epochs
        summary["epoch_stats"]["epochs_per_subject"][subject_id] = subject_epochs

    # Overall PSD statistics
    if all_psd_values:
        psd_array = np.array(all_psd_values)
        summary["epoch_stats"]["psd_statistics"] = {
            "overall_mean": float(np.mean(psd_array)),
            "overall_std": float(np.std(psd_array)),
            "overall_min": float(np.min(psd_array)),
            "overall_max": float(np.max(psd_array)),
            "overall_median": float(np.median(psd_array))
        }

    return summary