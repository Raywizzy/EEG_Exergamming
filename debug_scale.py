#!/usr/bin/env python3
"""Debug data scaling issues"""

import mne
import numpy as np

# Load raw data
eeg_file = "bids/ds004584/sub-001/eeg/sub-001_task-Rest_eeg.set"
raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

print(f"Raw data shape: {raw.get_data().shape}")
print(f"Raw data range: {raw.get_data().min():.6f} to {raw.get_data().max():.6f}")
print(f"Raw data std: {raw.get_data().std():.6f}")
print(f"Raw data units: {raw.info['chs'][0]['unit']}")

# Sample a small segment
data_sample = raw.get_data()[:5, :1000]  # First 5 channels, first 1000 samples
print(f"Sample data stats:")
print(f"  Min: {data_sample.min():.6f}")
print(f"  Max: {data_sample.max():.6f}")
print(f"  Mean: {data_sample.mean():.6f}")
print(f"  Std: {data_sample.std():.6f}")

# Check if data is in volts (should be microvolts for EEG)
if data_sample.std() < 1e-3:
    print("Data appears to be in volts - need to convert to microvolts")
    print("Multiplying by 1e6...")
    raw._data *= 1e6
    data_sample = raw.get_data()[:5, :1000]
    print(f"After scaling:")
    print(f"  Min: {data_sample.min():.6f}")
    print(f"  Max: {data_sample.max():.6f}")
    print(f"  Std: {data_sample.std():.6f}")