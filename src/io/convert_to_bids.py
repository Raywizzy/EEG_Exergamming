"""
BIDS Conversion Module for EEG Biomarker Validation Pipeline
Phase IV Multi-Site Clinical Validation

Converts raw EEG datasets to BIDS format with standardized structure.
Supports multiple dataset formats with configurable channel mapping.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
from datetime import datetime

import mne
import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BIDSConverter:
    """Convert raw EEG datasets to BIDS format with standardized structure."""

    def __init__(self, config_path: str):
        """Initialize BIDS converter with configuration.

        Args:
            config_path: Path to preprocessing configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.seed = self.config.get('seed', 42)
        np.random.seed(self.seed)

        # Create output directory structure
        self.bids_root = Path("data/bids")
        self.bids_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized BIDS converter with seed={self.seed}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _create_dataset_description(self, dataset_info: Dict, bids_dir: Path) -> None:
        """Create dataset_description.json file for BIDS compliance."""
        dataset_description = {
            "Name": dataset_info.get("name", "EEG Dataset"),
            "BIDSVersion": "1.7.0",
            "DatasetType": "raw",
            "Authors": ["Phase IV Validation Team"],
            "Acknowledgements": f"Dataset: {dataset_info.get('name', 'Unknown')}",
            "ReferencesAndLinks": dataset_info.get("references", []),
            "DatasetDOI": dataset_info.get("doi", ""),
            "License": "See original dataset license",
            "EthicsApprovals": ["Original dataset IRB/ethics approval"],
            "GeneratedBy": [{
                "Name": "EEG Biomarker Validation Pipeline",
                "Version": "1.0.0",
                "Container": {"Type": "conda", "Tag": "eeg-phd"}
            }]
        }

        desc_file = bids_dir / "dataset_description.json"
        with open(desc_file, 'w') as f:
            json.dump(dataset_description, f, indent=2)

        logger.info(f"Created dataset_description.json for {dataset_info['name']}")

    def _create_participants_tsv(self, subjects_info: List[Dict], bids_dir: Path) -> None:
        """Create participants.tsv file with subject metadata."""
        participants_data = []

        for subj in subjects_info:
            participants_data.append({
                "participant_id": subj["participant_id"],
                "age": subj.get("age", "n/a"),
                "sex": subj.get("sex", "n/a"),
                "group": subj.get("group", "n/a"),  # CONTROL, PD_REAL, PD_SHAM
                "medication_state": subj.get("medication_state", "n/a"),
                "disease_duration": subj.get("disease_duration", "n/a"),
                "UPDRS_III": subj.get("UPDRS_III", "n/a")
            })

        df = pd.DataFrame(participants_data)
        participants_file = bids_dir / "participants.tsv"
        df.to_csv(participants_file, sep='\t', index=False, na_rep='n/a')

        # Create participants.json with column descriptions
        participants_json = {
            "participant_id": {"Description": "Unique participant identifier"},
            "age": {"Description": "Age in years", "Units": "years"},
            "sex": {"Description": "Biological sex", "Levels": {"M": "male", "F": "female"}},
            "group": {"Description": "Clinical group", "Levels": {
                "CONTROL": "Healthy control",
                "PD_REAL": "Parkinson's disease, off medication",
                "PD_SHAM": "Parkinson's disease, on medication/sham"
            }},
            "medication_state": {"Description": "Medication status during recording"},
            "disease_duration": {"Description": "Years since PD diagnosis", "Units": "years"},
            "UPDRS_III": {"Description": "UPDRS Part III motor score", "Units": "points"}
        }

        with open(bids_dir / "participants.json", 'w') as f:
            json.dump(participants_json, f, indent=2)

        logger.info(f"Created participants.tsv with {len(participants_data)} subjects")

    def _load_channel_mapping(self, dataset_id: str) -> Optional[Dict]:
        """Load channel mapping configuration for dataset."""
        dataset_config = self.config['datasets'].get(dataset_id, {})
        mapping_file = dataset_config.get('mapping_file')

        if mapping_file and Path(mapping_file).exists():
            with open(mapping_file, 'r') as f:
                return yaml.safe_load(f)
        return None

    def _standardize_channels(self, raw: mne.io.Raw, dataset_id: str) -> mne.io.Raw:
        """Standardize channel names and select intersection montage."""
        # Get intersection channels from config
        intersection_channels = self.config['spatial_regions']['intersection_channels']

        # Load channel mapping if available
        channel_mapping = self._load_channel_mapping(dataset_id)
        if channel_mapping:
            raw.rename_channels(channel_mapping)
            logger.info(f"Applied channel mapping for {dataset_id}")

        # Select intersection channels that are present
        available_channels = [ch for ch in intersection_channels if ch in raw.ch_names]

        if len(available_channels) < 10:  # Minimum channel requirement
            logger.warning(f"Only {len(available_channels)} intersection channels found")

        # Pick intersection channels
        raw.pick_channels(available_channels)
        logger.info(f"Selected {len(available_channels)} intersection channels")

        return raw

    def _convert_single_file(self, input_path: Path, output_path: Path,
                           subject_id: str, session: str, task: str,
                           dataset_config: Dict) -> Tuple[mne.io.Raw, Dict]:
        """Convert single EEG file to BIDS format."""

        # Load EEG data based on file extension
        file_ext = input_path.suffix.lower()

        try:
            if file_ext in ['.vhdr', '.eeg']:
                raw = mne.io.read_raw_brainvision(str(input_path), preload=False)
            elif file_ext in ['.edf', '.bdf']:
                raw = mne.io.read_raw_edf(str(input_path), preload=False)
            elif file_ext == '.set':
                raw = mne.io.read_raw_eeglab(str(input_path), preload=False)
            elif file_ext in ['.fif', '.fiff']:
                raw = mne.io.read_raw_fif(str(input_path), preload=False)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            logger.error(f"Failed to load {input_path}: {e}")
            return None, {}

        # Standardize channels
        raw = self._standardize_channels(raw, list(dataset_config.keys())[0])

        # Set line noise frequency
        line_noise = dataset_config.get('line_noise_hz', self.config['line_noise_hz'])

        # Create BIDS filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as BIDS-compatible format (.edf)
        bids_path = output_path.with_suffix('.edf')
        raw.export(str(bids_path), fmt='edf', overwrite=True)

        # Create sidecar JSON with metadata
        sidecar_json = {
            "TaskName": task,
            "SamplingFrequency": raw.info['sfreq'],
            "PowerLineFrequency": line_noise,
            "EEGReference": "Average",  # Will be applied during preprocessing
            "EEGGround": "Unknown",
            "EEGPlacementScheme": "10-20",
            "Manufacturer": "Unknown",
            "ManufacturersModelName": "Unknown",
            "RecordingDuration": raw.times[-1] if len(raw.times) > 0 else 0,
            "RecordingType": "continuous",
            "SoftwareFilters": "n/a"
        }

        # Add channel information
        sidecar_json["EEGChannelCount"] = len(raw.ch_names)
        sidecar_json["EOGChannelCount"] = 0
        sidecar_json["EMGChannelCount"] = 0
        sidecar_json["MiscChannelCount"] = 0

        # Save sidecar JSON
        json_path = bids_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(sidecar_json, f, indent=2)

        # Create channels.tsv
        channels_data = []
        for ch_name in raw.ch_names:
            channels_data.append({
                "name": ch_name,
                "type": "EEG",
                "units": "µV",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "notch": "n/a",
                "sampling_frequency": raw.info['sfreq'],
                "reference": "n/a",
                "group": "EEG",
                "description": f"EEG electrode {ch_name}"
            })

        channels_df = pd.DataFrame(channels_data)
        channels_path = bids_path.with_suffix('.tsv').name.replace('_eeg.tsv', '_channels.tsv')
        channels_file = bids_path.parent / channels_path
        channels_df.to_csv(channels_file, sep='\t', index=False, na_rep='n/a')

        logger.info(f"Converted {input_path.name} to BIDS format")

        return raw, sidecar_json

    def convert_dataset(self, dataset_id: str) -> bool:
        """Convert entire dataset to BIDS format.

        Args:
            dataset_id: Dataset identifier from config

        Returns:
            bool: Success status
        """
        logger.info(f"Starting BIDS conversion for dataset: {dataset_id}")

        dataset_config = self.config['datasets'].get(dataset_id)
        if not dataset_config:
            logger.error(f"Dataset {dataset_id} not found in config")
            return False

        # Skip if already in BIDS format
        if dataset_config.get('type') == 'BIDS':
            logger.info(f"Dataset {dataset_id} already in BIDS format")
            return True

        # Create BIDS directory for dataset
        bids_dir = self.bids_root / dataset_id
        bids_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset description
        self._create_dataset_description(dataset_config, bids_dir)

        # Find all EEG files in raw directory
        raw_root = Path(dataset_config['root'])
        if not raw_root.exists():
            logger.error(f"Raw data directory not found: {raw_root}")
            return False

        # Supported extensions
        eeg_extensions = ['.vhdr', '.eeg', '.edf', '.bdf', '.set', '.fif', '.fiff']
        eeg_files = []

        for ext in eeg_extensions:
            eeg_files.extend(raw_root.rglob(f"*{ext}"))

        if not eeg_files:
            logger.error(f"No EEG files found in {raw_root}")
            return False

        logger.info(f"Found {len(eeg_files)} EEG files to convert")

        # Convert each file
        subjects_info = []
        converted_count = 0

        for eeg_file in eeg_files:
            try:
                # Extract subject ID from filename or path
                # This is dataset-specific and may need customization
                subject_id = self._extract_subject_id(eeg_file, dataset_config)
                session = "01"  # Default session
                task = dataset_config.get('rest_task_codes', ['rest'])[0]

                # Create BIDS path structure
                subj_dir = bids_dir / f"sub-{subject_id}"
                ses_dir = subj_dir / f"ses-{session}"
                eeg_dir = ses_dir / "eeg"

                # BIDS filename
                bids_filename = f"sub-{subject_id}_ses-{session}_task-{task}_eeg"
                output_path = eeg_dir / bids_filename

                # Convert file
                raw, metadata = self._convert_single_file(
                    eeg_file, output_path, subject_id, session, task, dataset_config
                )

                if raw is not None:
                    # Extract subject info (this would need to be enhanced with actual metadata)
                    subject_info = {
                        "participant_id": f"sub-{subject_id}",
                        "age": "n/a",  # Would extract from metadata if available
                        "sex": "n/a",
                        "group": "UNKNOWN",  # Would determine from metadata
                        "medication_state": dataset_config.get('medication_state', 'n/a')
                    }
                    subjects_info.append(subject_info)
                    converted_count += 1

            except Exception as e:
                logger.error(f"Failed to convert {eeg_file}: {e}")
                continue

        if subjects_info:
            # Create participants.tsv
            self._create_participants_tsv(subjects_info, bids_dir)

        logger.info(f"Successfully converted {converted_count}/{len(eeg_files)} files for {dataset_id}")
        return converted_count > 0

    def _extract_subject_id(self, file_path: Path, dataset_config: Dict) -> str:
        """Extract subject ID from file path or name.

        This is dataset-specific and may need customization for each dataset.
        """
        # Default implementation - extract number from filename
        import re

        # Look for patterns like sub-001, subject_001, s001, etc.
        filename = file_path.stem
        patterns = [
            r'sub-(\d+)',
            r'subject_?(\d+)',
            r's(\d+)',
            r'(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return match.group(1).zfill(3)  # Zero-pad to 3 digits

        # Fallback: use hash of filename
        import hashlib
        hash_obj = hashlib.md5(filename.encode())
        return hash_obj.hexdigest()[:6]

    def validate_bids_structure(self, dataset_id: str) -> bool:
        """Validate BIDS structure for converted dataset."""
        bids_dir = self.bids_root / dataset_id

        # Check required files
        required_files = [
            "dataset_description.json",
            "participants.tsv",
            "participants.json"
        ]

        for req_file in required_files:
            if not (bids_dir / req_file).exists():
                logger.error(f"Missing required BIDS file: {req_file}")
                return False

        # Check for subject directories
        subject_dirs = list(bids_dir.glob("sub-*"))
        if not subject_dirs:
            logger.error("No subject directories found")
            return False

        # Validate at least one subject has EEG data
        eeg_found = False
        for subj_dir in subject_dirs[:5]:  # Check first 5 subjects
            eeg_files = list(subj_dir.rglob("*_eeg.edf"))
            if eeg_files:
                eeg_found = True
                break

        if not eeg_found:
            logger.error("No EEG files found in BIDS structure")
            return False

        logger.info(f"BIDS validation passed for {dataset_id}")
        return True


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Convert raw EEG datasets to BIDS format")
    parser.add_argument("--config", required=True, help="Path to preprocessing config YAML")
    parser.add_argument("--dataset", help="Specific dataset ID to convert (converts all if not specified)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing BIDS structure")

    args = parser.parse_args()

    # Initialize converter
    converter = BIDSConverter(args.config)

    # Get datasets to process
    if args.dataset:
        dataset_ids = [args.dataset]
    else:
        # Convert all RAW datasets
        dataset_ids = [
            dataset_id for dataset_id, config in converter.config['datasets'].items()
            if config.get('to_bids', False) and config.get('type') == 'RAW'
        ]

    if not dataset_ids:
        logger.warning("No datasets found for conversion")
        return

    # Process datasets
    success_count = 0
    for dataset_id in dataset_ids:
        logger.info(f"Processing dataset: {dataset_id}")

        if args.validate_only:
            if converter.validate_bids_structure(dataset_id):
                success_count += 1
        else:
            if converter.convert_dataset(dataset_id):
                if converter.validate_bids_structure(dataset_id):
                    success_count += 1
                else:
                    logger.error(f"BIDS validation failed for {dataset_id}")

    logger.info(f"Successfully processed {success_count}/{len(dataset_ids)} datasets")

    if success_count == len(dataset_ids):
        logger.info("BIDS conversion completed successfully!")
    else:
        logger.error("Some datasets failed conversion")


if __name__ == "__main__":
    main()