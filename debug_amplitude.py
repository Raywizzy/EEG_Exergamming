#!/usr/bin/env python3
"""Debug amplitude ranges after preprocessing"""

import sys
import numpy as np
import mne
sys.path.append('src')
from preprocess.pipeline import EEGPreprocessor

# Initialize processor
processor = EEGPreprocessor('config/preprocessing_config.yaml')

# Load and process single subject step by step
eeg_file = "bids/ds004584/sub-001/eeg/sub-001_task-Rest_eeg.set"
raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

print("=== Original data ===")
print(f"Range: {raw.get_data().min():.6f} to {raw.get_data().max():.6f}")
print(f"Std: {raw.get_data().std():.6f}")

# Apply unit conversion
data_std = raw.get_data().std()
if data_std < 1e-3:
    print("Converting volts to microvolts...")
    raw._data *= 1e6

print("=== After unit conversion ===")
print(f"Range: {raw.get_data().min():.1f} to {raw.get_data().max():.1f} µV")
print(f"Std: {raw.get_data().std():.1f} µV")

# Resample and filter step by step
raw = raw.resample(256)
print("=== After resampling ===")
print(f"Range: {raw.get_data().min():.1f} to {raw.get_data().max():.1f} µV")

# Check range after processing steps
raw = processor._apply_intersection_montage(raw)
print("=== After channel selection ===")
print(f"Range: {raw.get_data().min():.1f} to {raw.get_data().max():.1f} µV")
print(f"Channels: {raw.ch_names}")

raw = processor._apply_bandpass_and_notch(raw)
print("=== After filtering ===")
print(f"Range: {raw.get_data().min():.1f} to {raw.get_data().max():.1f} µV")
print(f"Peak-to-peak per channel:")
data = raw.get_data()
for i, ch_name in enumerate(raw.ch_names):
    ptp = np.ptp(data[i])
    print(f"  {ch_name}: {ptp:.1f} µV")

print(f"\nArtifact threshold: {processor.artifact_reject['peak_to_peak_uV']} µV")