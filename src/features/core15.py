"""
Core15+ Feature Extraction Module for Enhanced EEG Biomarkers
Phase 3 Feature Expansion: Spectral + Connectivity + Temporal + Stability

Builds on Core5 beta-burst features (50.5% BA) to achieve clinical utility (≥65% BA).
Features: Spectral power, connectivity, advanced burst metrics, stability measures.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import coherence, welch, hilbert
from typing import Dict, List, Tuple, Optional
import mne
import logging

logger = logging.getLogger(__name__)

class Core15FeatureExtractor:
    """Enhanced feature extractor combining Core5 + spectral + connectivity + stability."""

    def __init__(self, config: Dict):
        """Initialize Core15+ feature extractor.

        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        self.sfreq = config.get('sfreq', 512)
        self.freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 40)
        }

        # Channel groups for spatial features
        self.motor_channels = ['C3', 'Cz', 'C4']
        self.posterior_channels = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
        self.frontal_channels = ['F3', 'Fz', 'F4']

        logger.info("Initialized Core15+ feature extractor")
        logger.info(f"Frequency bands: {self.freq_bands}")
        logger.info(f"Motor channels: {self.motor_channels}")

    def extract_spectral_features(self, data: np.ndarray, ch_names: List[str]) -> Dict:
        """Extract spectral power features across frequency bands.

        Args:
            data: EEG data (channels x time)
            ch_names: Channel names

        Returns:
            Dict of spectral features
        """
        features = {}

        # Calculate PSD for each channel
        freqs, psd = welch(data, fs=self.sfreq, nperseg=self.sfreq*2,
                          noverlap=self.sfreq, axis=-1)

        # Extract power in each band
        total_power = np.trapz(psd, freqs, axis=-1)

        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

            # Calculate band power
            band_power = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=-1)

            # Relative power (normalized by total power)
            rel_power = band_power / (total_power + 1e-10)

            # Average across motor channels
            motor_indices = [i for i, ch in enumerate(ch_names) if ch in self.motor_channels]
            if motor_indices:
                features[f'{band_name}_power_rel'] = np.mean(rel_power[motor_indices])
            else:
                features[f'{band_name}_power_rel'] = np.mean(rel_power)

        # Alpha/beta ratio
        alpha_power = features.get('alpha_power_rel', 0.1)
        beta_power = features.get('beta_power_rel', 0.1)
        features['alpha_beta_ratio'] = alpha_power / (beta_power + 1e-10)

        # Individual peak frequencies
        features['alpha_peak_freq'] = self._find_peak_frequency(
            freqs, psd, self.freq_bands['alpha'], ch_names
        )
        features['beta_peak_freq'] = self._find_peak_frequency(
            freqs, psd, self.freq_bands['beta'], ch_names
        )

        return features

    def _find_peak_frequency(self, freqs: np.ndarray, psd: np.ndarray,
                           band: Tuple[float, float], ch_names: List[str]) -> float:
        """Find peak frequency within a band."""
        low_freq, high_freq = band
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

        # Average PSD across motor channels
        motor_indices = [i for i, ch in enumerate(ch_names) if ch in self.motor_channels]
        if motor_indices:
            avg_psd = np.mean(psd[motor_indices], axis=0)
        else:
            avg_psd = np.mean(psd, axis=0)

        # Find peak within band
        band_freqs = freqs[freq_mask]
        band_psd = avg_psd[freq_mask]

        if len(band_psd) > 0:
            peak_idx = np.argmax(band_psd)
            return band_freqs[peak_idx]
        else:
            return (low_freq + high_freq) / 2  # Default to band center

    def extract_temporal_burst_features(self, data: np.ndarray, ch_names: List[str]) -> Dict:
        """Extract advanced temporal burst metrics.

        Args:
            data: EEG data (channels x time)
            ch_names: Channel names

        Returns:
            Dict of temporal burst features
        """
        features = {}

        # Focus on beta band for burst analysis
        beta_low, beta_high = self.freq_bands['beta']

        # Filter data to beta band
        sos = signal.butter(4, [beta_low, beta_high], btype='bandpass',
                           fs=self.sfreq, output='sos')
        beta_data = signal.sosfiltfilt(sos, data, axis=-1)

        # Calculate instantaneous amplitude via Hilbert transform
        analytic_signal = hilbert(beta_data, axis=-1)
        amplitude = np.abs(analytic_signal)

        # Average across motor channels
        motor_indices = [i for i, ch in enumerate(ch_names) if ch in self.motor_channels]
        if motor_indices:
            avg_amplitude = np.mean(amplitude[motor_indices], axis=0)
        else:
            avg_amplitude = np.mean(amplitude, axis=0)

        # Detect bursts using median + 2*MAD threshold
        threshold = np.median(avg_amplitude) + 2 * np.median(np.abs(avg_amplitude - np.median(avg_amplitude)))
        burst_mask = avg_amplitude > threshold

        # Find burst events
        burst_starts, burst_ends = self._find_burst_events(burst_mask)

        if len(burst_starts) > 0:
            # Burst rate (per minute)
            recording_duration_min = len(avg_amplitude) / (self.sfreq * 60)
            features['burst_rate'] = len(burst_starts) / recording_duration_min

            # Burst amplitudes
            burst_amplitudes = []
            for start, end in zip(burst_starts, burst_ends):
                burst_amplitudes.append(np.mean(avg_amplitude[start:end]))

            features['mean_burst_amplitude'] = np.mean(burst_amplitudes)
            features['burst_amplitude_cv'] = np.std(burst_amplitudes) / (np.mean(burst_amplitudes) + 1e-10)

            # Inter-burst intervals
            if len(burst_starts) > 1:
                inter_burst_intervals = []
                for i in range(1, len(burst_starts)):
                    interval_samples = burst_starts[i] - burst_ends[i-1]
                    interval_seconds = interval_samples / self.sfreq
                    inter_burst_intervals.append(interval_seconds)

                features['inter_burst_interval_mean'] = np.mean(inter_burst_intervals)
                features['inter_burst_interval_cv'] = np.std(inter_burst_intervals) / (np.mean(inter_burst_intervals) + 1e-10)
            else:
                features['inter_burst_interval_mean'] = np.nan
                features['inter_burst_interval_cv'] = np.nan
        else:
            # No bursts detected
            features['burst_rate'] = 0.0
            features['mean_burst_amplitude'] = np.nan
            features['burst_amplitude_cv'] = np.nan
            features['inter_burst_interval_mean'] = np.nan
            features['inter_burst_interval_cv'] = np.nan

        return features

    def _find_burst_events(self, burst_mask: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find start and end indices of burst events."""
        # Find transitions
        diff_mask = np.diff(burst_mask.astype(int))
        starts = np.where(diff_mask == 1)[0] + 1  # Start of burst
        ends = np.where(diff_mask == -1)[0] + 1   # End of burst

        # Handle edge cases
        if burst_mask[0]:
            starts = np.concatenate([[0], starts])
        if burst_mask[-1]:
            ends = np.concatenate([ends, [len(burst_mask)]])

        # Ensure equal number of starts and ends
        min_len = min(len(starts), len(ends))
        return starts[:min_len].tolist(), ends[:min_len].tolist()

    def extract_connectivity_features(self, data: np.ndarray, ch_names: List[str]) -> Dict:
        """Extract spatial connectivity features.

        Args:
            data: EEG data (channels x time)
            ch_names: Channel names

        Returns:
            Dict of connectivity features
        """
        features = {}

        # Beta band coherence analysis
        beta_low, beta_high = self.freq_bands['beta']
        alpha_low, alpha_high = self.freq_bands['alpha']

        # Motor interhemispheric coherence (C3-C4)
        c3_idx = self._find_channel_index('C3', ch_names)
        c4_idx = self._find_channel_index('C4', ch_names)

        if c3_idx is not None and c4_idx is not None:
            freqs, coh = coherence(data[c3_idx], data[c4_idx], fs=self.sfreq, nperseg=self.sfreq*2)
            beta_mask = (freqs >= beta_low) & (freqs <= beta_high)
            features['motor_coherence_beta'] = np.mean(coh[beta_mask])
        else:
            features['motor_coherence_beta'] = np.nan

        # Fronto-motor coherence (Fz-Cz)
        fz_idx = self._find_channel_index('Fz', ch_names)
        cz_idx = self._find_channel_index('Cz', ch_names)

        if fz_idx is not None and cz_idx is not None:
            freqs, coh = coherence(data[fz_idx], data[cz_idx], fs=self.sfreq, nperseg=self.sfreq*2)
            beta_mask = (freqs >= beta_low) & (freqs <= beta_high)
            features['fronto_motor_beta_coh'] = np.mean(coh[beta_mask])
        else:
            features['fronto_motor_beta_coh'] = np.nan

        # Motor-posterior alpha coherence (C3-P3)
        p3_idx = self._find_channel_index('P3', ch_names)

        if c3_idx is not None and p3_idx is not None:
            freqs, coh = coherence(data[c3_idx], data[p3_idx], fs=self.sfreq, nperseg=self.sfreq*2)
            alpha_mask = (freqs >= alpha_low) & (freqs <= alpha_high)
            features['motor_posterior_alpha_coh'] = np.mean(coh[alpha_mask])
        else:
            features['motor_posterior_alpha_coh'] = np.nan

        # Hemispheric asymmetry in beta power
        if c3_idx is not None and c4_idx is not None:
            freqs, psd_c3 = welch(data[c3_idx], fs=self.sfreq, nperseg=self.sfreq*2)
            freqs, psd_c4 = welch(data[c4_idx], fs=self.sfreq, nperseg=self.sfreq*2)

            beta_mask = (freqs >= beta_low) & (freqs <= beta_high)
            c3_beta_power = np.trapz(psd_c3[beta_mask], freqs[beta_mask])
            c4_beta_power = np.trapz(psd_c4[beta_mask], freqs[beta_mask])

            features['hemispheric_asymmetry_beta'] = (c3_beta_power - c4_beta_power) / (c3_beta_power + c4_beta_power + 1e-10)
        else:
            features['hemispheric_asymmetry_beta'] = np.nan

        return features

    def _find_channel_index(self, ch_name: str, ch_names: List[str]) -> Optional[int]:
        """Find index of channel name in list."""
        try:
            return ch_names.index(ch_name)
        except ValueError:
            return None

    def extract_stability_features(self, data: np.ndarray, ch_names: List[str]) -> Dict:
        """Extract temporal stability features across recording.

        Args:
            data: EEG data (channels x time)
            ch_names: Channel names

        Returns:
            Dict of stability features
        """
        features = {}

        # Split data into windows for stability analysis
        window_samples = int(4 * self.sfreq)  # 4-second windows
        overlap_samples = int(2 * self.sfreq)  # 50% overlap

        n_samples = data.shape[-1]
        window_starts = range(0, n_samples - window_samples + 1, window_samples - overlap_samples)

        if len(window_starts) < 3:
            # Not enough windows for stability analysis
            features['alpha_power_stability'] = np.nan
            features['beta_power_stability'] = np.nan
            features['burst_rate_stability'] = np.nan
            return features

        # Extract power in each window
        alpha_powers = []
        beta_powers = []
        burst_rates = []

        for start in window_starts:
            end = start + window_samples
            window_data = data[:, start:end]

            # Calculate PSD for this window
            freqs, psd = welch(window_data, fs=self.sfreq, nperseg=self.sfreq, axis=-1)

            # Motor channels average
            motor_indices = [i for i, ch in enumerate(ch_names) if ch in self.motor_channels]
            if motor_indices:
                avg_psd = np.mean(psd[motor_indices], axis=0)
            else:
                avg_psd = np.mean(psd, axis=0)

            # Alpha power
            alpha_mask = (freqs >= self.freq_bands['alpha'][0]) & (freqs <= self.freq_bands['alpha'][1])
            alpha_power = np.trapz(avg_psd[alpha_mask], freqs[alpha_mask])
            alpha_powers.append(alpha_power)

            # Beta power
            beta_mask = (freqs >= self.freq_bands['beta'][0]) & (freqs <= self.freq_bands['beta'][1])
            beta_power = np.trapz(avg_psd[beta_mask], freqs[beta_mask])
            beta_powers.append(beta_power)

            # Burst rate for this window
            window_burst_features = self.extract_temporal_burst_features(window_data, ch_names)
            burst_rates.append(window_burst_features.get('burst_rate', 0.0))

        # Calculate stability (CV across windows)
        features['alpha_power_stability'] = np.std(alpha_powers) / (np.mean(alpha_powers) + 1e-10)
        features['beta_power_stability'] = np.std(beta_powers) / (np.mean(beta_powers) + 1e-10)
        features['burst_rate_stability'] = np.std(burst_rates) / (np.mean(burst_rates) + 1e-10)

        return features

    def extract_all_features(self, data: np.ndarray, ch_names: List[str]) -> Dict:
        """Extract all Core15+ features.

        Args:
            data: EEG data (channels x time)
            ch_names: Channel names

        Returns:
            Dict containing all Core15+ features
        """
        features = {}

        # 1. Spectral features
        spectral_features = self.extract_spectral_features(data, ch_names)
        features.update(spectral_features)

        # 2. Temporal burst features
        temporal_features = self.extract_temporal_burst_features(data, ch_names)
        features.update(temporal_features)

        # 3. Connectivity features
        connectivity_features = self.extract_connectivity_features(data, ch_names)
        features.update(connectivity_features)

        # 4. Stability features
        stability_features = self.extract_stability_features(data, ch_names)
        features.update(stability_features)

        logger.info(f"Extracted {len(features)} Core15+ features")

        return features