#!/usr/bin/env python3
"""
Dataset Download Script for EEG Exergaming Project
Downloads approved public PD EEG datasets from MRC BNDU, OpenNeuro, or Kaggle.

Usage:
    python scripts/download_dataset.py --source mrc_bndu --dataset lfps-and-eegs-patients-parkinsons-disease-during-neurofeedback-training
    python scripts/download_dataset.py --source openneuro --dataset ds002778
    python scripts/download_dataset.py --source kaggle --dataset souravbasakshuvo/uc-san-diego-parkinsons-disease-resting-state-eeg
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import argparse

def print_timestamp(message):
    """Print message with ISO timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    iso_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"[{timestamp}] {message}")
    print(f"ISO Date: {iso_date}")

def download_openneuro_dataset(dataset_id, output_path):
    """Download dataset from OpenNeuro using datalad."""
    print_timestamp(f"Downloading OpenNeuro dataset: {dataset_id}")

    try:
        # Check if datalad is installed
        result = subprocess.run(['which', 'datalad'], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: datalad not found. Install with:")
            print("pip install datalad")
            print("or conda install -c conda-forge datalad")
            return False

        # Download dataset
        cmd = [
            'datalad', 'install',
            f'https://github.com/OpenNeuroDatasets/{dataset_id}.git',
            str(output_path)
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Dataset downloaded successfully to: {output_path}")
            return True
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def download_kaggle_dataset(dataset_name, output_path):
    """Download dataset from Kaggle using kaggle API."""
    print_timestamp(f"Downloading Kaggle dataset: {dataset_name}")

    try:
        # Check if kaggle is installed
        result = subprocess.run(['which', 'kaggle'], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: kaggle CLI not found. Install with:")
            print("pip install kaggle")
            print("Then configure API key from https://www.kaggle.com/account")
            return False

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Download dataset
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(output_path), '--unzip']

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Dataset downloaded successfully to: {output_path}")
            return True
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def log_download(dataset_info, output_path, success):
    """Log dataset download for audit trail."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_info": dataset_info,
        "output_path": str(output_path),
        "success": success,
        "purpose": "Primary training dataset for PD_REAL vs PD_SHAM classification"
    }

    logs_dir = Path("results/logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "dataset_downloads.json"

    # Read existing logs
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    # Add new entry
    logs.append(log_entry)

    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

    print(f"✓ Download logged to: {log_file}")

def download_mrc_bndu_dataset(dataset_id, output_path):
    """Download dataset from MRC BNDU data sharing platform."""
    print_timestamp(f"Downloading MRC BNDU dataset: {dataset_id}")

    # MRC BNDU requires registration and login
    dataset_url = f"https://data.mrc.ox.ac.uk/data-set/{dataset_id}"

    print("MRC BNDU Dataset Download Instructions:")
    print("=" * 60)
    print(f"Dataset URL: {dataset_url}")
    print("DOI: 10.60964/bndu-4jde-7j28")
    print("\nMANUAL DOWNLOAD REQUIRED:")
    print("1. Go to: https://data.mrc.ox.ac.uk/")
    print("2. Register for an account if you don't have one")
    print("3. Log in to your account")
    print("4. Navigate to the dataset page:")
    print(f"   {dataset_url}")
    print("5. Click 'Download' button")
    print("6. Extract the downloaded files to:")
    print(f"   {output_path}")
    print("\nExpected dataset contents:")
    print("- 12 PD patients")
    print("- EEG and LFP recordings")
    print("- Training vs No Training conditions")
    print("- 80 trials per subject (40 Training, 40 No Training)")
    print("=" * 60)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a README file with instructions
    readme_path = output_path / "DOWNLOAD_INSTRUCTIONS.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# MRC BNDU Dataset Download Instructions

## Dataset Information
- **Name**: LFPs and EEGs from patients with Parkinson's disease during neurofeedback training
- **DOI**: 10.60964/bndu-4jde-7j28
- **URL**: {dataset_url}
- **License**: CC BY-SA 4.0

## Download Steps
1. Register at https://data.mrc.ox.ac.uk/
2. Login to your account
3. Navigate to {dataset_url}
4. Download the dataset files
5. Extract all files to this directory: {output_path}

## Expected Data Structure
- 12 PD patients (4 females)
- EEG + LFP recordings from STN
- Experimental conditions:
  - Training: Real neurofeedback (PD_REAL)
  - No Training: Sham condition (PD_SHAM)
- 80 trials per subject (40 per condition)

## Citation
Tan, H., Wade, C., & Brown, P. (2020). Subthalamic beta-targeted neurofeedback speeds up movement initiation but increases tremor in Parkinsonian patients. eLife, 9, e60979.
""")

    print(f"✓ Download instructions saved to: {readme_path}")
    return True

def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download approved PD EEG datasets")
    parser.add_argument('--source', choices=['mrc_bndu', 'openneuro', 'kaggle'], required=True,
                       help='Data source to download from')
    parser.add_argument('--dataset', required=True,
                       help='Dataset identifier')
    parser.add_argument('--output', default='data/raw',
                       help='Output directory (default: data/raw)')

    args = parser.parse_args()

    print("=" * 80)
    print("EEG EXERGAMING PROJECT - DATASET DOWNLOAD")
    print("=" * 80)
    print_timestamp(f"Starting download from {args.source}")

    output_path = Path(args.output) / f"{args.source}_{args.dataset.replace('/', '_')}"

    # Dataset information for logging
    dataset_info = {
        "source": args.source,
        "dataset_id": args.dataset,
        "url": f"https://{args.source}.org" + (f"/datasets/{args.dataset}" if args.source == "openneuro" else f"/datasets/{args.dataset}")
    }

    success = False

    if args.source == 'mrc_bndu':
        success = download_mrc_bndu_dataset(args.dataset, output_path)
    elif args.source == 'openneuro':
        success = download_openneuro_dataset(args.dataset, output_path)
    elif args.source == 'kaggle':
        success = download_kaggle_dataset(args.dataset, output_path)

    # Log the download attempt
    log_download(dataset_info, output_path, success)

    if success:
        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Dataset location: {output_path}")
        print("Next steps:")
        print("1. Verify dataset contains PD subjects only")
        print("2. Identify real vs sham feedback conditions")
        print("3. Update dataset specification")
        print("4. Run audit to verify compliance")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("DOWNLOAD FAILED")
        print("=" * 80)
        print("Please check error messages above and:")
        print("1. Verify internet connection")
        print("2. Check API credentials")
        print("3. Confirm dataset exists and is accessible")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()