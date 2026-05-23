#!/usr/bin/env python3
"""
Core15+ Feature Extraction Pipeline
Expanding from Core5 to comprehensive EEG biomarker panel

Phase V Optimization: 49% → 65%+ balanced accuracy target
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import mne
import warnings
warnings.filterwarnings('ignore')

class Core15PlusExtractor:
    """Enhanced feature extraction expanding Core5 to Core15+"""

    def __init__(self, sfreq=256, bands=None):
        """
        Initialize Core15+ extractor

        Parameters:
        -----------
        sfreq : float
            Sampling frequency in Hz
        bands : dict
            Frequency bands for analysis
        """
        self.sfreq = sfreq

        if bands is None:
            self.bands = {
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
        else:
            self.bands = bands

        # Motor and posterior channel groups
        self.motor_channels = ['C3', 'Cz', 'C4']
        self.posterior_channels = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

    def extract_core5_features(self, data, channel_names):
        """Extract original Core5 beta-burst features"""

        # Beta band filtering
        beta_data = self._filter_band(data, self.bands['beta'])

        # Burst detection using median + 2*MAD threshold
        bursts = self._detect_beta_bursts(beta_data)

        # Core5 features
        features = {}

        if len(bursts) > 0:
            durations = [b['duration'] for b in bursts]
            features['duration_cv'] = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
            features['duty_cycle'] = sum(durations) / (data.shape[1] / self.sfreq)
            features['mean_duration_ms'] = np.mean(durations) * 1000
            features['median_duration_ms'] = np.median(durations) * 1000

            # Motor/posterior ratio
            motor_duty = self._calculate_region_duty_cycle(bursts, channel_names, self.motor_channels)
            posterior_duty = self._calculate_region_duty_cycle(bursts, channel_names, self.posterior_channels)
            features['motor_posterior_duty_ratio'] = motor_duty / posterior_duty if posterior_duty > 0 else 1.0
        else:
            # Default values for no bursts detected
            features = {
                'duration_cv': 0.0,
                'duty_cycle': 0.0,
                'mean_duration_ms': 0.0,
                'median_duration_ms': 0.0,
                'motor_posterior_duty_ratio': 1.0
            }

        return features

    def extract_spectral_features(self, data, channel_names):
        """Extract Core6-10: Spectral features"""

        features = {}

        # Core6: Alpha peak frequency and bandwidth
        alpha_data = self._filter_band(data, self.bands['alpha'])
        freqs, psd = signal.welch(alpha_data, fs=self.sfreq, nperseg=int(2*self.sfreq))
        alpha_mask = (freqs >= self.bands['alpha'][0]) & (freqs <= self.bands['alpha'][1])
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = psd[:, alpha_mask]

        # Peak frequency
        peak_indices = np.argmax(alpha_psd, axis=1)
        features['alpha_peak_freq'] = np.mean(alpha_freqs[peak_indices])
        features['alpha_bandwidth'] = np.std(alpha_freqs[peak_indices])

        # Core7: Cross-frequency coupling (beta-gamma)
        beta_data = self._filter_band(data, self.bands['beta'])
        gamma_data = self._filter_band(data, self.bands['gamma'])
        features['beta_gamma_coupling'] = self._calculate_cross_frequency_coupling(beta_data, gamma_data)

        # Core8: Spectral entropy
        features['spectral_entropy'] = self._calculate_spectral_entropy(data)

        # Core9: Power ratios (known PD markers)
        theta_power = self._calculate_band_power(data, self.bands['theta'])
        alpha_power = self._calculate_band_power(data, self.bands['alpha'])
        beta_power = self._calculate_band_power(data, self.bands['beta'])

        features['theta_beta_ratio'] = theta_power / beta_power if beta_power > 0 else 0
        features['alpha_beta_ratio'] = alpha_power / beta_power if beta_power > 0 else 0

        # Core10: Spectral stability (variance across time windows)
        features['spectral_stability'] = self._calculate_spectral_stability(data)

        return features

    def extract_connectivity_features(self, data, channel_names):
        """Extract Core11-13: Connectivity features"""

        features = {}

        # Core11: Phase Locking Index (PLI) between motor and posterior
        motor_indices = [i for i, ch in enumerate(channel_names) if ch in self.motor_channels]
        posterior_indices = [i for i, ch in enumerate(channel_names) if ch in self.posterior_channels]

        if len(motor_indices) > 0 and len(posterior_indices) > 0:
            features['pli_motor_posterior'] = self._calculate_pli(data, motor_indices, posterior_indices)
        else:
            features['pli_motor_posterior'] = 0.0

        # Core12: Coherence metrics (bilateral symmetry)
        left_channels = ['C3', 'P3', 'O1']
        right_channels = ['C4', 'P4', 'O2']
        features['coherence_bilateral'] = self._calculate_bilateral_coherence(data, channel_names, left_channels, right_channels)

        # Core13: Beta-theta coupling
        beta_data = self._filter_band(data, self.bands['beta'])
        theta_data = self._filter_band(data, self.bands['theta'])
        features['beta_theta_coupling'] = self._calculate_cross_frequency_coupling(beta_data, theta_data)

        return features

    def extract_temporal_features(self, data, channel_names):
        """Extract Core14-15: Temporal features"""

        features = {}

        # Detect bursts for temporal analysis
        beta_data = self._filter_band(data, self.bands['beta'])
        bursts = self._detect_beta_bursts(beta_data)

        if len(bursts) > 2:
            # Core14: Burst synchronization (cross-channel overlap)
            features['burst_synchrony'] = self._calculate_burst_synchrony(bursts, data.shape[0])

            # Core15: Temporal stability (burst-to-burst variability)
            features['temporal_stability'] = self._calculate_temporal_stability(bursts)
        else:
            features['burst_synchrony'] = 0.0
            features['temporal_stability'] = 0.0

        return features

    def extract_core15_plus(self, data, channel_names, subject_metadata=None):
        """Extract complete Core15+ feature set"""

        features = {}

        # Core5: Original beta-burst features
        core5 = self.extract_core5_features(data, channel_names)
        features.update(core5)

        # Core6-10: Spectral features
        spectral = self.extract_spectral_features(data, channel_names)
        features.update(spectral)

        # Core11-13: Connectivity features
        connectivity = self.extract_connectivity_features(data, channel_names)
        features.update(connectivity)

        # Core14-15: Temporal features
        temporal = self.extract_temporal_features(data, channel_names)
        features.update(temporal)

        # Add metadata
        if subject_metadata:
            features.update(subject_metadata)

        return features

    def _filter_band(self, data, band):
        """Filter data to specific frequency band"""
        nyquist = self.sfreq / 2
        low = band[0] / nyquist
        high = band[1] / nyquist

        if high >= 1.0:
            high = 0.99

        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=1)

    def _detect_beta_bursts(self, beta_data):
        """Detect beta bursts using median + 2*MAD threshold"""

        # Calculate envelope using Hilbert transform
        analytic_signal = signal.hilbert(beta_data, axis=1)
        envelope = np.abs(analytic_signal)

        # Threshold: median + 2*MAD across all channels
        all_envelope = envelope.flatten()
        median_env = np.median(all_envelope)
        mad_env = np.median(np.abs(all_envelope - median_env))
        threshold = median_env + 2 * mad_env

        # Detect bursts (minimum 100ms duration, 50ms gap)
        min_duration_samples = int(0.1 * self.sfreq)  # 100ms
        min_gap_samples = int(0.05 * self.sfreq)      # 50ms

        bursts = []
        for ch in range(envelope.shape[0]):
            above_threshold = envelope[ch] > threshold

            # Find burst onset/offset
            diff = np.diff(above_threshold.astype(int))
            onsets = np.where(diff == 1)[0] + 1
            offsets = np.where(diff == -1)[0] + 1

            # Handle edge cases
            if above_threshold[0]:
                onsets = np.concatenate([[0], onsets])
            if above_threshold[-1]:
                offsets = np.concatenate([offsets, [len(above_threshold)]])

            # Extract bursts meeting duration criteria
            for onset, offset in zip(onsets, offsets):
                duration_samples = offset - onset
                if duration_samples >= min_duration_samples:
                    bursts.append({
                        'channel': ch,
                        'onset': onset / self.sfreq,
                        'offset': offset / self.sfreq,
                        'duration': duration_samples / self.sfreq
                    })

        return bursts

    def _calculate_region_duty_cycle(self, bursts, channel_names, region_channels):
        """Calculate duty cycle for specific brain region"""

        region_indices = [i for i, ch in enumerate(channel_names) if ch in region_channels]
        if not region_indices:
            return 0.0

        region_bursts = [b for b in bursts if b['channel'] in region_indices]
        if not region_bursts:
            return 0.0

        total_duration = sum(b['duration'] for b in region_bursts)
        return total_duration / len(region_indices)  # Normalize by number of channels

    def _calculate_cross_frequency_coupling(self, data1, data2):
        """Calculate cross-frequency coupling between two signals"""

        # Phase-amplitude coupling
        phase1 = np.angle(signal.hilbert(data1, axis=1))
        amplitude2 = np.abs(signal.hilbert(data2, axis=1))

        # Mean vector length (phase-amplitude coupling strength)
        coupling_values = []
        for ch in range(data1.shape[0]):
            complex_coupling = np.mean(amplitude2[ch] * np.exp(1j * phase1[ch]))
            coupling_values.append(np.abs(complex_coupling))

        return np.mean(coupling_values)

    def _calculate_spectral_entropy(self, data):
        """Calculate spectral entropy across frequency bands"""

        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=int(2*self.sfreq))

        # Normalize PSD to probabilities
        psd_norm = psd / np.sum(psd, axis=1, keepdims=True)

        # Calculate entropy for each channel
        entropies = []
        for ch in range(psd_norm.shape[0]):
            ch_entropy = entropy(psd_norm[ch] + 1e-12)  # Add small value for numerical stability
            entropies.append(ch_entropy)

        return np.mean(entropies)

    def _calculate_band_power(self, data, band):
        """Calculate power in specific frequency band"""

        filtered_data = self._filter_band(data, band)
        return np.mean(np.var(filtered_data, axis=1))

    def _calculate_spectral_stability(self, data, window_length=2.0):
        """Calculate spectral stability across time windows"""

        window_samples = int(window_length * self.sfreq)
        n_windows = data.shape[1] // window_samples

        if n_windows < 2:
            return 0.0

        # Calculate PSD for each window
        window_psds = []
        for w in range(n_windows):
            start_idx = w * window_samples
            end_idx = (w + 1) * window_samples
            window_data = data[:, start_idx:end_idx]

            freqs, psd = signal.welch(window_data, fs=self.sfreq, nperseg=min(window_samples, int(self.sfreq)))
            window_psds.append(np.mean(psd, axis=0))

        # Calculate variance across windows (lower = more stable)
        window_psds = np.array(window_psds)
        stability = 1.0 / (1.0 + np.mean(np.var(window_psds, axis=0)))

        return stability

    def _calculate_pli(self, data, indices1, indices2):
        """Calculate Phase Locking Index between two channel groups"""

        # Extract signals from channel groups
        signals1 = data[indices1]
        signals2 = data[indices2]

        # Calculate instantaneous phases
        phases1 = np.angle(signal.hilbert(signals1, axis=1))
        phases2 = np.angle(signal.hilbert(signals2, axis=1))

        # PLI calculation
        pli_values = []
        for s1 in range(phases1.shape[0]):
            for s2 in range(phases2.shape[0]):
                phase_diff = phases1[s1] - phases2[s2]
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli_values.append(pli)

        return np.mean(pli_values)

    def _calculate_bilateral_coherence(self, data, channel_names, left_channels, right_channels):
        """Calculate bilateral coherence between left and right hemispheres"""

        left_indices = [i for i, ch in enumerate(channel_names) if ch in left_channels]
        right_indices = [i for i, ch in enumerate(channel_names) if ch in right_channels]

        if not left_indices or not right_indices:
            return 0.0

        # Calculate coherence between corresponding left-right pairs
        coherence_values = []
        min_pairs = min(len(left_indices), len(right_indices))

        for i in range(min_pairs):
            left_signal = data[left_indices[i]]
            right_signal = data[right_indices[i]]

            freqs, coherence = signal.coherence(left_signal, right_signal, fs=self.sfreq)
            # Average coherence in beta band
            beta_mask = (freqs >= 13) & (freqs <= 30)
            coherence_values.append(np.mean(coherence[beta_mask]))

        return np.mean(coherence_values)

    def _calculate_burst_synchrony(self, bursts, n_channels):
        """Calculate synchrony of bursts across channels"""

        if len(bursts) < 2:
            return 0.0

        # Create binary burst matrix (channels x time)
        max_time = max(b['offset'] for b in bursts)
        time_resolution = 0.01  # 10ms resolution
        n_time_bins = int(max_time / time_resolution)

        burst_matrix = np.zeros((n_channels, n_time_bins))

        for burst in bursts:
            ch = burst['channel']
            start_bin = int(burst['onset'] / time_resolution)
            end_bin = int(burst['offset'] / time_resolution)
            burst_matrix[ch, start_bin:end_bin] = 1

        # Calculate synchrony as correlation between channels
        correlations = []
        for ch1 in range(n_channels):
            for ch2 in range(ch1 + 1, n_channels):
                if np.sum(burst_matrix[ch1]) > 0 and np.sum(burst_matrix[ch2]) > 0:
                    corr = np.corrcoef(burst_matrix[ch1], burst_matrix[ch2])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    def _calculate_temporal_stability(self, bursts):
        """Calculate temporal stability of burst patterns"""

        if len(bursts) < 3:
            return 0.0

        # Inter-burst intervals
        sorted_bursts = sorted(bursts, key=lambda x: x['onset'])
        intervals = []

        for i in range(1, len(sorted_bursts)):
            interval = sorted_bursts[i]['onset'] - sorted_bursts[i-1]['offset']
            if interval > 0:  # Only positive intervals
                intervals.append(interval)

        if len(intervals) < 2:
            return 0.0

        # Stability = 1 / (1 + coefficient of variation)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        stability = 1.0 / (1.0 + cv)

        return stability


def process_subject_core15_plus(eeg_file, config_path, extractor=None):
    """Process single subject for Core15+ features"""

    if extractor is None:
        extractor = Core15PlusExtractor()

    try:
        # Load EEG data
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

        # Basic preprocessing
        raw.filter(l_freq=1, h_freq=45, fir_design='firwin', verbose=False)
        raw.resample(256, npad="auto", verbose=False)

        if len(raw.ch_names) > 1:
            raw.set_eeg_reference('average', projection=False, verbose=False)

        # Extract data
        data = raw.get_data()
        channel_names = raw.ch_names

        # Extract Core15+ features
        subject_metadata = {
            'subject_id': eeg_file.stem.split('_')[0],
            'file_path': str(eeg_file)
        }

        features = extractor.extract_core15_plus(data, channel_names, subject_metadata)

        return features

    except Exception as e:
        print(f"Error processing {eeg_file}: {e}")
        return None


if __name__ == "__main__":
    # Test Core15+ extraction
    extractor = Core15PlusExtractor()

    # Generate synthetic test data
    n_channels, n_samples = 64, 30720  # 2 minutes at 256 Hz
    test_data = np.random.randn(n_channels, n_samples) * 1e-6

    # Add some realistic EEG characteristics
    for ch in range(n_channels):
        # Add alpha rhythm
        alpha_freq = 10 + np.random.randn() * 1
        alpha_signal = 2e-6 * np.sin(2 * np.pi * alpha_freq * np.arange(n_samples) / 256)
        test_data[ch] += alpha_signal

        # Add beta bursts
        for burst in range(5):
            start = np.random.randint(0, n_samples - 1000)
            end = start + np.random.randint(256, 1000)  # 1-4 seconds
            beta_freq = 20 + np.random.randn() * 5
            burst_signal = 5e-6 * np.sin(2 * np.pi * beta_freq * np.arange(end - start) / 256)
            test_data[ch, start:end] += burst_signal

    # Test feature extraction
    channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    features = extractor.extract_core15_plus(test_data, channel_names)

    print("Core15+ Feature Extraction Test Results:")
    print("=" * 50)
    for feature, value in features.items():
        print(f"{feature}: {value:.6f}")

    print(f"\nTotal features extracted: {len(features)}")
    print("Core15+ extraction pipeline ready for Phase V optimization!")