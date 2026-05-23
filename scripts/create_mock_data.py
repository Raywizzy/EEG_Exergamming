#!/usr/bin/env python3
"""
Create mock EEG data for testing the persistent platform
"""

import numpy as np
import mne
from pathlib import Path

def create_mock_eeg_data():
    """Create mock EEG data file for testing."""
    # Create mock data
    sfreq = 512
    duration = 120  # 2 minutes

    # Standard 10-20 channel names only
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
    n_channels = len(ch_names)

    # Create realistic EEG-like data
    n_samples = int(sfreq * duration)
    data = np.random.randn(n_channels, n_samples) * 1e-6  # Scale to microvolts

    # Add some alpha rhythm (8-12 Hz) to posterior channels
    alpha_freq = 10
    t = np.arange(n_samples) / sfreq
    alpha_signal = 5e-6 * np.sin(2 * np.pi * alpha_freq * t)

    # Add alpha to occipital channels
    occipital_indices = [26, 27, 28]  # O1, Oz, O2
    for idx in occipital_indices:
        data[idx] += alpha_signal

    # Create info structure
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')

    # Create raw object
    raw = mne.io.RawArray(data, info)

    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    return raw

def main():
    """Create mock data files."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Create mock EEG file
    raw = create_mock_eeg_data()
    output_path = output_dir / "demo_eeg.fif"
    raw.save(output_path, overwrite=True)

    print(f"✅ Created mock EEG data: {output_path}")
    print(f"   Duration: {raw.times[-1]:.1f} seconds")
    print(f"   Channels: {len(raw.ch_names)}")
    print(f"   Sampling rate: {raw.info['sfreq']} Hz")

if __name__ == "__main__":
    main()