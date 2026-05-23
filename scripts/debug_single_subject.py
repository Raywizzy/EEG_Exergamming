#!/usr/bin/env python3
"""
Debug single subject preprocessing for ds004584
"""

import sys
sys.path.append('src')

import mne
import numpy as np
from pathlib import Path
from preprocess.pipeline import EEGPreprocessor

def debug_subject():
    # Initialize preprocessor
    preprocessor = EEGPreprocessor('config/preprocessing_config.yaml')

    # Load single subject manually
    eeg_file = Path('bids/ds004584/sub-001/eeg/sub-001_task-Rest_eeg.set')

    print(f"Loading: {eeg_file}")
    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True)

    print(f"Original data:")
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} seconds")
    print(f"  Data shape: {raw.get_data().shape}")
    print()

    # Check intersection channels
    available_channels = [ch for ch in preprocessor.intersection_channels if ch in raw.ch_names]
    print(f"Intersection channels available: {len(available_channels)}")
    print(f"Channels: {available_channels}")
    print()

    # Apply intersection montage
    raw_intersect = raw.copy().pick_channels(available_channels)
    print(f"After intersection montage:")
    print(f"  Channels: {len(raw_intersect.ch_names)}")
    print(f"  Data shape: {raw_intersect.get_data().shape}")
    print()

    # Resample
    target_sr = preprocessor.sampling_rate
    if abs(raw_intersect.info['sfreq'] - target_sr) > 1e-6:
        print(f"Resampling from {raw_intersect.info['sfreq']} Hz to {target_sr} Hz")
        raw_intersect.resample(target_sr)
        print(f"After resampling: {raw_intersect.get_data().shape}")
        print()

    # Apply filters
    print("Applying bandpass filter...")
    raw_filtered = raw_intersect.copy()
    raw_filtered.filter(
        l_freq=preprocessor.bandpass_hz[0],
        h_freq=preprocessor.bandpass_hz[1],
        method='iir',
        iir_params={'order': 4, 'ftype': 'butter'},
        verbose=False
    )

    # Apply notch filter
    raw_filtered.notch_filter(
        freqs=60,  # US line noise for ds004584
        notch_widths=2,
        method='iir',
        verbose=False
    )

    # Apply CAR
    raw_filtered.set_eeg_reference('average', projection=False, verbose=False)
    print("Applied filtering and CAR")
    print()

    # Check artifact detection
    print("Running artifact detection...")
    data, times = raw_filtered.get_data(return_times=True)
    data_uv = data * 1e6  # Convert to microvolts

    # Peak-to-peak check
    window_samples = int(2.0 * raw_filtered.info['sfreq'])
    pp_threshold = preprocessor.artifact_reject['peak_to_peak_uV']

    print(f"Artifact detection parameters:")
    print(f"  Window size: {window_samples} samples ({window_samples/raw_filtered.info['sfreq']:.1f} s)")
    print(f"  Peak-to-peak threshold: {pp_threshold} µV")
    print(f"  Flat threshold: {preprocessor.artifact_reject['flat_uV']} µV")
    print()

    clean_mask = np.ones(data.shape[1], dtype=bool)
    artifact_windows = 0

    for i in range(0, data.shape[1] - window_samples, window_samples // 2):
        window_data = data_uv[:, i:i+window_samples]
        pp_amplitude = np.ptp(window_data, axis=1)

        if np.any(pp_amplitude > pp_threshold):
            clean_mask[i:i+window_samples] = False
            artifact_windows += 1

    artifact_pct = (1 - np.mean(clean_mask)) * 100
    print(f"Artifact analysis:")
    print(f"  Windows with excessive amplitude: {artifact_windows}")
    print(f"  Total artifact percentage: {artifact_pct:.1f}%")
    print(f"  Clean percentage: {100-artifact_pct:.1f}%")
    print()

    # Try epoching
    print("Attempting epoching...")
    epoch_length_samples = int(preprocessor.epoch_len_s * raw_filtered.info['sfreq'])
    overlap_samples = int(preprocessor.epoch_overlap * epoch_length_samples)
    step_samples = epoch_length_samples - overlap_samples

    events = []
    clean_epochs = 0
    total_epochs = 0

    for start_sample in range(0, len(clean_mask) - epoch_length_samples, step_samples):
        end_sample = start_sample + epoch_length_samples
        epoch_mask = clean_mask[start_sample:end_sample]
        clean_proportion = np.mean(epoch_mask)
        total_epochs += 1

        if clean_proportion >= 0.8:
            events.append([start_sample, 0, 1])
            clean_epochs += 1

    print(f"Epoching results:")
    print(f"  Epoch length: {preprocessor.epoch_len_s} s ({epoch_length_samples} samples)")
    print(f"  Overlap: {preprocessor.epoch_overlap} ({overlap_samples} samples)")
    print(f"  Step size: {step_samples} samples")
    print(f"  Total possible epochs: {total_epochs}")
    print(f"  Clean epochs (≥80% clean): {clean_epochs}")
    print(f"  Clean epoch percentage: {clean_epochs/total_epochs*100:.1f}%")
    print()

    # Check quality control criteria
    min_epochs = preprocessor.artifact_reject['min_valid_epochs_pct'] / 100 * 60
    print(f"Quality control assessment:")
    print(f"  Minimum required epochs: {min_epochs}")
    print(f"  Actual clean epochs: {clean_epochs}")
    print(f"  Passes epoch count: {'✅' if clean_epochs >= min_epochs else '❌'}")

    if clean_epochs > 0:
        # Create epochs to test
        events_array = np.array(events[:10]) if len(events) > 10 else np.array(events)  # Limit for testing
        if len(events_array) > 0:
            epochs = mne.Epochs(
                raw_filtered,
                events_array,
                {'rest': 1},
                tmin=0,
                tmax=preprocessor.epoch_len_s - 1/raw_filtered.info['sfreq'],
                baseline=None,
                preload=True,
                verbose=False
            )

            # Signal quality check
            epoch_data = epochs.get_data()
            mean_variance = np.mean(np.var(epoch_data, axis=2))
            print(f"  Signal variance: {mean_variance:.2e}")
            print(f"  Passes signal quality: {'✅' if mean_variance >= 1e-12 else '❌'}")

            # Check for flat epochs
            flat_epochs = 0
            for epoch_data_single in epoch_data:
                if np.any(np.var(epoch_data_single, axis=1) < preprocessor.artifact_reject['flat_uV']):
                    flat_epochs += 1

            flat_proportion = flat_epochs / len(epochs)
            print(f"  Flat epochs: {flat_epochs}/{len(epochs)} ({flat_proportion:.1%})")
            print(f"  Passes flat check: {'✅' if flat_proportion <= 0.3 else '❌'}")

    print()
    print("Debug complete!")

if __name__ == "__main__":
    debug_subject()