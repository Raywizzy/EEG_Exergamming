#!/usr/bin/env python3
"""
Test script for Phase III UCSD integration
Creates mock UCSD data to verify the pipeline works before downloading real dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mne
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_ucsd_data():
    """Create mock UCSD EEG data for pipeline testing"""

    logger.info("Creating mock UCSD dataset for pipeline testing...")

    # Create directory structure
    mock_dir = Path("data/raw/ucsd_ds002778_mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    # Mock EEG parameters
    sfreq = 500
    duration = 300  # 5 minutes
    n_channels = 64

    # Standard 10-20 channel names (subset)
    ch_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
        'FT9', 'FT10', 'TP9', 'TP10'
    ]

    # Pad to 64 channels if needed
    while len(ch_names) < n_channels:
        ch_names.append(f'EEG{len(ch_names)+1:02d}')

    ch_names = ch_names[:n_channels]

    # Create mock subjects
    n_subjects = 20

    for subj_id in range(1, n_subjects + 1):
        # Create 2 sessions per subject (rest and task, mapping to PD_SHAM and PD_REAL)
        for ses_id, condition in enumerate(['rest', 'task'], 1):

            # Generate realistic EEG-like data
            n_samples = int(sfreq * duration)

            # Base EEG with 1/f noise + alpha peak
            freqs = np.fft.fftfreq(n_samples, 1/sfreq)[:n_samples//2]

            data = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                # 1/f background
                noise = np.random.randn(n_samples)

                # Add alpha oscillation (8-12 Hz) - stronger in posterior
                if any(region in ch_names[ch].upper() for region in ['P', 'O']):
                    alpha_power = 2.0
                else:
                    alpha_power = 0.5

                t = np.arange(n_samples) / sfreq
                alpha_freq = 10 + np.random.randn() * 0.5  # 10 Hz ± jitter
                alpha = alpha_power * np.sin(2 * np.pi * alpha_freq * t)

                # Add beta bursts (13-30 Hz) - stronger in motor for "task"
                if condition == 'task' and any(region in ch_names[ch].upper() for region in ['C', 'FC']):
                    beta_power = 1.5
                else:
                    beta_power = 0.3

                # Create realistic beta bursts
                beta_freq = 20 + np.random.randn() * 2
                beta_base = beta_power * np.sin(2 * np.pi * beta_freq * t)

                # Modulate with burst envelope
                burst_rate = 0.1  # bursts per second
                burst_times = np.random.poisson(burst_rate * duration)
                burst_envelope = np.zeros_like(t)

                for _ in range(burst_times):
                    burst_start = np.random.rand() * duration
                    burst_dur = 0.1 + np.random.exponential(0.05)  # 100ms + exponential tail
                    burst_mask = (t >= burst_start) & (t <= burst_start + burst_dur)
                    burst_envelope[burst_mask] = 1.0

                beta = beta_base * (0.1 + 0.9 * burst_envelope)  # baseline + bursts

                # Combine components
                data[ch] = noise + alpha + beta

                # Scale to realistic EEG amplitude (microvolts)
                data[ch] *= 50e-6

            # Create MNE Raw object
            info = mne.create_info(ch_names, sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)

            # Add some realistic artifacts occasionally
            if np.random.rand() < 0.2:  # 20% chance of artifacts
                artifact_start = np.random.randint(0, n_samples - int(sfreq))
                artifact_duration = int(sfreq * 0.5)  # 500ms artifact
                artifact_channels = np.random.choice(n_channels, size=3, replace=False)

                for ch in artifact_channels:
                    raw._data[ch, artifact_start:artifact_start+artifact_duration] *= 10

            # Save as EDF file (common format for public datasets)
            subj_dir = mock_dir / f"sub-{subj_id:02d}" / f"ses-{ses_id:02d}" / "eeg"
            subj_dir.mkdir(parents=True, exist_ok=True)

            filename = subj_dir / f"sub-{subj_id:02d}_ses-{ses_id:02d}_task-{condition}_eeg.edf"

            mne.export.export_raw(filename, raw, fmt='edf', overwrite=True, verbose=False)

            logger.debug(f"Created mock file: {filename}")

    logger.info(f"Created mock UCSD dataset: {n_subjects} subjects, 2 sessions each")
    logger.info(f"Location: {mock_dir}")
    logger.info("Files ready for testing phase3_ucsd_ingest.py pipeline")

    return mock_dir

def test_ucsd_pipeline():
    """Test the UCSD processing pipeline with mock data"""

    logger.info("Testing UCSD processing pipeline...")

    # Create mock data
    mock_dir = create_mock_ucsd_data()

    # Update config to point to mock data
    mock_config = {
        'dataset_name': 'ucsd_ds002778_mock',
        'bids_root': str(mock_dir),
        'harmonized_root': 'data/harmonized/ucsd_mock',
        'features_out': 'results/features/ucsd_Core5_mock.csv',
        'ref': 'CAR',
        'notch_hz': 50,
        'bandpass': [13, 30],
        'resample_hz': 250,
        'motor': ['C3', 'Cz', 'C4'],
        'posterior': ['P3', 'Pz', 'P4', 'O1', 'O2'],
        'burst': {
            'method': 'hilbert_envelope',
            'threshold': 'median_plus_k_mad',
            'k': 2.0,
            'min_duration_ms': 100,
            'merge_gap_ms': 60
        },
        'core5_features': [
            'duty_cycle',
            'mean_duration_ms',
            'median_duration_ms',
            'duration_cv',
            'motor_posterior_duty_ratio'
        ],
        'site_tag': 'UCSD_MOCK'
    }

    # Save mock config
    import yaml
    mock_config_path = Path("config/phase3_ucsd_mock.yaml")
    with open(mock_config_path, 'w') as f:
        yaml.dump(mock_config, f)

    logger.info(f"Mock config saved: {mock_config_path}")
    logger.info("To test the pipeline, run:")
    logger.info(f"python scripts/phase3_ucsd_ingest.py --config {mock_config_path} --all")

    return mock_config_path

if __name__ == "__main__":
    test_ucsd_pipeline()