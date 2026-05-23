"""
Core5 Feature Extraction Module for EEG Biomarker Validation
Phase IV Multi-Site Clinical Validation

Implements the locked Core5 biomarker feature set from Phase III optimization.
Features: duration_cv, duty_cycle, mean_duration_ms, median_duration_ms, motor_posterior_duty_ratio
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from datetime import datetime

import mne
import numpy as np
import pandas as pd
import yaml
from scipy import signal, stats
from scipy.signal import hilbert
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BetaBurstDetector:
    """Beta-burst detection using locked Phase III parameters."""

    def __init__(self, config: Dict):
        """Initialize burst detector with locked parameters.

        Args:
            config: Configuration dictionary with burst detection parameters
        """
        self.config = config
        self.burst_params = config['burst_detection']
        self.beta_band_hz = config['beta_band_hz']
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)

        # Locked threshold method
        self.threshold_method = self.burst_params['method']  # "median_plus_2mad"
        self.min_duration_ms = self.burst_params['min_duration_ms']
        self.min_gap_ms = self.burst_params['min_gap_ms']

        logger.info(f"Initialized beta-burst detector: {self.threshold_method}")
        logger.info(f"Min duration: {self.min_duration_ms} ms, Min gap: {self.min_gap_ms} ms")

    def _compute_amplitude_envelope(self, beta_signal: np.ndarray, sfreq: float) -> np.ndarray:
        """Compute amplitude envelope using Hilbert transform.

        Args:
            beta_signal: Beta-filtered signal [channels, times]
            sfreq: Sampling frequency

        Returns:
            Amplitude envelope [channels, times]
        """
        envelope = np.abs(hilbert(beta_signal, axis=1))
        return envelope

    def _compute_threshold_median_plus_2mad(self, envelope: np.ndarray) -> np.ndarray:
        """Compute burst threshold using median + 2×MAD method.

        Args:
            envelope: Amplitude envelope [channels, times]

        Returns:
            Threshold values [channels]
        """
        thresholds = np.zeros(envelope.shape[0])

        for ch_idx in range(envelope.shape[0]):
            ch_envelope = envelope[ch_idx, :]
            median_val = np.median(ch_envelope)
            mad_val = stats.median_abs_deviation(ch_envelope, scale='normal')
            thresholds[ch_idx] = median_val + 2 * mad_val

        return thresholds

    def _detect_burst_events(self, envelope: np.ndarray, threshold: float, sfreq: float) -> List[Tuple[int, int]]:
        """Detect burst events in single channel.

        Args:
            envelope: Single channel amplitude envelope
            threshold: Burst threshold
            sfreq: Sampling frequency

        Returns:
            List of (start_sample, end_sample) tuples
        """
        # Binarize signal
        above_threshold = envelope > threshold

        # Find transitions
        diff = np.diff(above_threshold.astype(int))
        burst_starts = np.where(diff == 1)[0] + 1
        burst_ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            burst_starts = np.concatenate([[0], burst_starts])
        if above_threshold[-1]:
            burst_ends = np.concatenate([burst_ends, [len(above_threshold)]])

        # Ensure starts and ends are paired
        if len(burst_starts) != len(burst_ends):
            min_len = min(len(burst_starts), len(burst_ends))
            burst_starts = burst_starts[:min_len]
            burst_ends = burst_ends[:min_len]

        # Filter by minimum duration
        min_duration_samples = int(self.min_duration_ms * sfreq / 1000)
        valid_bursts = []

        for start, end in zip(burst_starts, burst_ends):
            duration_samples = end - start
            if duration_samples >= min_duration_samples:
                valid_bursts.append((start, end))

        # Merge bursts separated by less than minimum gap
        min_gap_samples = int(self.min_gap_ms * sfreq / 1000)
        merged_bursts = []

        if valid_bursts:
            current_start, current_end = valid_bursts[0]

            for start, end in valid_bursts[1:]:
                gap = start - current_end
                if gap < min_gap_samples:
                    # Merge bursts
                    current_end = end
                else:
                    # Save current burst and start new one
                    merged_bursts.append((current_start, current_end))
                    current_start, current_end = start, end

            # Add final burst
            merged_bursts.append((current_start, current_end))

        return merged_bursts

    def detect_bursts(self, beta_data: np.ndarray, sfreq: float, channel_names: List[str]) -> Dict:
        """Detect beta-bursts in multi-channel data.

        Args:
            beta_data: Beta-filtered signal [epochs, channels, times] or [channels, times]
            sfreq: Sampling frequency
            channel_names: List of channel names

        Returns:
            Dictionary with burst detection results
        """
        if beta_data.ndim == 3:
            # Concatenate epochs for burst detection
            n_epochs, n_channels, n_times = beta_data.shape
            concat_data = beta_data.reshape(n_channels, -1)
        else:
            concat_data = beta_data

        n_channels, total_samples = concat_data.shape

        # Compute amplitude envelope
        envelope = self._compute_amplitude_envelope(concat_data, sfreq)

        # Compute thresholds
        thresholds = self._compute_threshold_median_plus_2mad(envelope)

        # Detect bursts for each channel
        all_bursts = {}
        burst_masks = np.zeros_like(envelope, dtype=bool)

        for ch_idx, ch_name in enumerate(channel_names):
            bursts = self._detect_burst_events(envelope[ch_idx], thresholds[ch_idx], sfreq)

            # Create burst mask
            for start, end in bursts:
                burst_masks[ch_idx, start:end] = True

            all_bursts[ch_name] = {
                'bursts': bursts,
                'threshold': thresholds[ch_idx],
                'n_bursts': len(bursts),
                'burst_mask': burst_masks[ch_idx]
            }

        # Calculate overall burst statistics
        total_bursts = sum(len(all_bursts[ch]['bursts']) for ch in channel_names)
        total_burst_time = np.sum(burst_masks)
        total_time = burst_masks.size

        results = {
            'channel_bursts': all_bursts,
            'summary': {
                'total_bursts': total_bursts,
                'burst_rate_hz': total_bursts / (total_samples / sfreq),
                'duty_cycle_overall': total_burst_time / total_time,
                'thresholds': dict(zip(channel_names, thresholds))
            },
            'burst_masks': burst_masks,
            'envelope': envelope
        }

        logger.info(f"Detected {total_bursts} bursts across {n_channels} channels")
        return results


class Core5FeatureExtractor:
    """Extract Core5 biomarker features from beta-burst data."""

    def __init__(self, config_path: str):
        """Initialize feature extractor with configuration.

        Args:
            config_path: Path to preprocessing configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.seed = self.config.get('seed', 42)
        np.random.seed(self.seed)

        # Initialize burst detector
        self.burst_detector = BetaBurstDetector(self.config)

        # Define spatial regions
        self.spatial_regions = self.config['spatial_regions']
        self.motor_channels = self.spatial_regions['motor']
        self.posterior_channels = self.spatial_regions['posterior']

        # Core5 feature definitions
        self.core5_features = self.config['core5_features']

        logger.info(f"Initialized Core5 feature extractor")
        logger.info(f"Motor channels: {self.motor_channels}")
        logger.info(f"Posterior channels: {self.posterior_channels}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _extract_burst_durations(self, burst_results: Dict, sfreq: float) -> List[float]:
        """Extract all burst durations across channels.

        Args:
            burst_results: Results from burst detection
            sfreq: Sampling frequency

        Returns:
            List of burst durations in milliseconds
        """
        all_durations = []

        for ch_name, ch_data in burst_results['channel_bursts'].items():
            for start, end in ch_data['bursts']:
                duration_ms = (end - start) / sfreq * 1000
                all_durations.append(duration_ms)

        return all_durations

    def _extract_region_duty_cycles(self, burst_results: Dict, channel_names: List[str]) -> Dict[str, float]:
        """Extract duty cycles for spatial regions.

        Args:
            burst_results: Results from burst detection
            channel_names: List of channel names

        Returns:
            Dictionary with region duty cycles
        """
        region_duty_cycles = {}

        # Motor region
        motor_channels_present = [ch for ch in self.motor_channels if ch in channel_names]
        if motor_channels_present:
            motor_masks = []
            for ch in motor_channels_present:
                if ch in burst_results['channel_bursts']:
                    motor_masks.append(burst_results['channel_bursts'][ch]['burst_mask'])

            if motor_masks:
                motor_combined = np.any(motor_masks, axis=0)
                region_duty_cycles['motor'] = np.mean(motor_combined)
            else:
                region_duty_cycles['motor'] = 0.0
        else:
            region_duty_cycles['motor'] = 0.0

        # Posterior region
        posterior_channels_present = [ch for ch in self.posterior_channels if ch in channel_names]
        if posterior_channels_present:
            posterior_masks = []
            for ch in posterior_channels_present:
                if ch in burst_results['channel_bursts']:
                    posterior_masks.append(burst_results['channel_bursts'][ch]['burst_mask'])

            if posterior_masks:
                posterior_combined = np.any(posterior_masks, axis=0)
                region_duty_cycles['posterior'] = np.mean(posterior_combined)
            else:
                region_duty_cycles['posterior'] = 0.0
        else:
            region_duty_cycles['posterior'] = 0.0

        return region_duty_cycles

    def extract_core5_features(self, beta_data: np.ndarray, sfreq: float, channel_names: List[str],
                              subject_metadata: Dict) -> Dict:
        """Extract Core5 biomarker features.

        Args:
            beta_data: Beta-filtered signal [epochs, channels, times] or [channels, times]
            sfreq: Sampling frequency
            channel_names: List of channel names
            subject_metadata: Subject metadata

        Returns:
            Dictionary with Core5 features
        """
        logger.info(f"Extracting Core5 features for subject {subject_metadata.get('subject_id', 'unknown')}")

        # Detect beta-bursts
        burst_results = self.burst_detector.detect_bursts(beta_data, sfreq, channel_names)

        # Extract burst durations
        durations_ms = self._extract_burst_durations(burst_results, sfreq)

        # Initialize features with defaults
        core5_features = {
            'duration_cv': 0.0,
            'duty_cycle': 0.0,
            'mean_duration_ms': 0.0,
            'median_duration_ms': 0.0,
            'motor_posterior_duty_ratio': 0.0
        }

        if len(durations_ms) > 0:
            # Feature 1: Coefficient of variation of burst durations
            if np.std(durations_ms) > 0:
                core5_features['duration_cv'] = np.std(durations_ms) / np.mean(durations_ms)
            else:
                core5_features['duration_cv'] = 0.0

            # Feature 2: Overall duty cycle
            core5_features['duty_cycle'] = burst_results['summary']['duty_cycle_overall']

            # Feature 3: Mean burst duration
            core5_features['mean_duration_ms'] = np.mean(durations_ms)

            # Feature 4: Median burst duration
            core5_features['median_duration_ms'] = np.median(durations_ms)

        # Feature 5: Motor/Posterior duty cycle ratio
        region_duty_cycles = self._extract_region_duty_cycles(burst_results, channel_names)
        if region_duty_cycles['posterior'] > 0:
            core5_features['motor_posterior_duty_ratio'] = (
                region_duty_cycles['motor'] / region_duty_cycles['posterior']
            )
        else:
            core5_features['motor_posterior_duty_ratio'] = region_duty_cycles['motor']

        # Add metadata
        feature_vector = {
            'subject_id': subject_metadata.get('subject_id', 'unknown'),
            'dataset_id': subject_metadata.get('dataset_id', 'unknown'),
            'session_id': subject_metadata.get('session_id', 'unknown'),
            'task': subject_metadata.get('task', 'unknown'),
            'extraction_timestamp': datetime.now().isoformat(),
            **core5_features,
            # Additional burst statistics
            'n_bursts_total': burst_results['summary']['total_bursts'],
            'burst_rate_hz': burst_results['summary']['burst_rate_hz'],
            'n_channels': len(channel_names),
            'motor_duty_cycle': region_duty_cycles['motor'],
            'posterior_duty_cycle': region_duty_cycles['posterior']
        }

        logger.info(f"Extracted Core5 features: {core5_features}")
        return feature_vector

    def extract_dataset_features(self, dataset_id: str) -> pd.DataFrame:
        """Extract Core5 features for entire dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DataFrame with Core5 features for all subjects
        """
        logger.info(f"Extracting Core5 features for dataset: {dataset_id}")

        # Load preprocessed data
        interim_dir = Path("data/interim") / dataset_id
        if not interim_dir.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {interim_dir}")

        # This would need to be implemented to load the preprocessed epochs and beta data
        # For now, showing the structure
        features_list = []

        # Placeholder for loading preprocessed data
        # epochs_list, beta_data_list, metadata_list = load_preprocessed_data(interim_dir)

        # for epochs, beta_data, metadata in zip(epochs_list, beta_data_list, metadata_list):
        #     features = self.extract_core5_features(
        #         beta_data, epochs.info['sfreq'], epochs.ch_names, metadata
        #     )
        #     features_list.append(features)

        # Create DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)
        else:
            # Create empty DataFrame with expected columns
            expected_columns = [
                'subject_id', 'dataset_id', 'session_id', 'task', 'extraction_timestamp',
                'duration_cv', 'duty_cycle', 'mean_duration_ms', 'median_duration_ms',
                'motor_posterior_duty_ratio', 'n_bursts_total', 'burst_rate_hz',
                'n_channels', 'motor_duty_cycle', 'posterior_duty_cycle'
            ]
            features_df = pd.DataFrame(columns=expected_columns)

        # Save features
        output_dir = Path("data/features")
        output_dir.mkdir(parents=True, exist_ok=True)
        features_file = output_dir / f"{dataset_id}_core5_features.csv"
        features_df.to_csv(features_file, index=False)

        logger.info(f"Saved Core5 features to {features_file}")
        return features_df

    def validate_features(self, features_df: pd.DataFrame) -> Dict:
        """Validate extracted Core5 features.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Validation report
        """
        validation_report = {
            'n_subjects': len(features_df),
            'feature_completeness': {},
            'feature_ranges': {},
            'data_quality': {}
        }

        # Check feature completeness
        for feature in self.core5_features:
            non_null_count = features_df[feature].notna().sum()
            validation_report['feature_completeness'][feature] = non_null_count / len(features_df)

        # Check feature ranges
        for feature in self.core5_features:
            if len(features_df) > 0:
                values = features_df[feature].dropna()
                if len(values) > 0:
                    validation_report['feature_ranges'][feature] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }

        # Data quality checks
        validation_report['data_quality'] = {
            'has_negative_values': (features_df[self.core5_features] < 0).any().any(),
            'has_infinite_values': np.isinf(features_df[self.core5_features]).any().any(),
            'has_zero_variance': features_df[self.core5_features].var().min() < 1e-10
        }

        logger.info("Feature validation completed")
        return validation_report


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Extract Core5 biomarker features")
    parser.add_argument("--config", required=True, help="Path to preprocessing config YAML")
    parser.add_argument("--dataset", help="Specific dataset ID to process (processes all if not specified)")
    parser.add_argument("--output-dir", default="data/features", help="Output directory for features")

    args = parser.parse_args()

    # Initialize feature extractor
    extractor = Core5FeatureExtractor(args.config)

    # Get datasets to process
    if args.dataset:
        dataset_ids = [args.dataset]
    else:
        # Process all datasets
        dataset_ids = list(extractor.config['datasets'].keys())

    # Extract features for each dataset
    all_features = []
    for dataset_id in dataset_ids:
        try:
            features_df = extractor.extract_dataset_features(dataset_id)
            all_features.append(features_df)

            # Validate features
            validation_report = extractor.validate_features(features_df)
            logger.info(f"Dataset {dataset_id}: {validation_report['n_subjects']} subjects processed")

        except Exception as e:
            logger.error(f"Failed to extract features for dataset {dataset_id}: {e}")

    # Combine all features
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)

        # Save combined features
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_file = output_dir / "all_datasets_core5_features.csv"
        combined_features.to_csv(combined_file, index=False)

        logger.info(f"Saved combined features to {combined_file}")
        logger.info(f"Total subjects: {len(combined_features)}")

    else:
        logger.warning("No features extracted")


if __name__ == "__main__":
    main()