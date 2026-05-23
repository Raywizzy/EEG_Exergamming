#!/usr/bin/env python3
"""Debug single subject processing"""

import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)  # Less verbose

sys.path.append('src')
from preprocess.pipeline import EEGPreprocessor

# Initialize processor
processor = EEGPreprocessor('config/preprocessing_config.yaml')

# Load single subject
dataset_config = processor.config['datasets']['ds004584']
bids_path = 'bids/ds004584'

# Get first subject file
import glob
eeg_files = glob.glob(f"{bids_path}/sub-001/eeg/*_eeg.set")
if eeg_files:
    print(f"Processing: {eeg_files[0]}")

    import mne
    raw = mne.io.read_raw_eeglab(eeg_files[0], preload=True)
    print(f"Raw info: {raw.info}")
    print(f"Duration: {raw.times[-1]:.1f}s")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sample rate: {raw.info['sfreq']} Hz")

    # Try preprocessing step by step
    metadata = {'subject_id': 'sub-001', 'dataset_id': 'ds004584'}

    try:
        epochs, beta_data, qc_metrics = processor.preprocess_subject(raw, metadata)
        print(f"Success! Epochs: {len(epochs) if epochs else 'None'}")
        print(f"QC metrics: {qc_metrics}")

        # Debug: check variance of one epoch
        if epochs:
            data = epochs.get_data()
            epoch_var = np.var(data[0], axis=1)  # Variance of first epoch across time
            print(f"First epoch variance per channel: min={epoch_var.min():.6f}, max={epoch_var.max():.6f}, mean={epoch_var.mean():.6f}")
            print(f"flat_uV threshold: {processor.artifact_reject['flat_uV']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No EEG files found")