#!/usr/bin/env python3
"""
Phase 8: Real Leicester Beta Burst Extraction & Model Training
- Extract real beta bursts directly from Leicester EEG time series
- Replace simulation with actual temporal dynamics
- Bridge simulation-to-real gap identified in Phase 7

CRITICAL: This uses actual Leicester EEG data to extract real beta bursts
No simulation - only real temporal dynamics from PD patients
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from scipy import signal, stats
from scipy.signal import hilbert, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_logging():
    """Configure logging for Phase 8."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/phase8_real_bursts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_subject_metadata() -> Dict[str, Any]:
    """Load Leicester subject metadata."""
    metadata_path = Path("data/interim/leicester_subject_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_eeg_session(subject_id: str, session_file: str) -> Optional[np.ndarray]:
    """
    Load a single EEG session pickle file.

    Args:
        subject_id: Subject identifier (e.g., 'A1', 'B10')
        session_file: Session file name (already includes _pickle_new suffix)

    Returns:
        EEG data array (samples, 1) - continuous time series or None if failed
    """
    logger = logging.getLogger(__name__)

    # File name is already the pickle file name (e.g., "session2_2024-11-15_10_29_36_pickle_new")
    session_path = Path(f"data/external/leicester_dataset/{subject_id}/{session_file}")

    if not session_path.exists():
        logger.debug(f"File not found: {session_path}")
        return None

    try:
        # Load pickle data (list of epochs)
        with open(session_path, 'rb') as f:
            epoch_list = pickle.load(f)

        if not epoch_list:
            logger.debug(f"Empty epoch list in {session_path}")
            return None

        # Convert list of epochs to continuous time series
        # Each epoch: (512,) samples -> concatenate to (n_epochs * 512,)
        continuous_data = np.concatenate(epoch_list, axis=0)

        # Skip if too few samples (less than 10 seconds at 250 Hz)
        if len(continuous_data) < 2500:
            logger.debug(f"Session too short: {len(continuous_data)} samples in {session_path}")
            return None

        # Reshape to (samples, 1) for compatibility with multi-channel code
        eeg_data = continuous_data.reshape(-1, 1)

        logger.debug(f"Successfully loaded {session_path}: {eeg_data.shape}")
        return eeg_data

    except Exception as e:
        logger.debug(f"Error loading {session_path}: {e}")
        return None

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        data: Input signal (samples, channels) or (samples,)
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        fs: Sampling frequency
        order: Filter order

    Returns:
        Filtered signal with same shape as input
    """
    nyquist = fs * 0.5
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    # Handle both 1D and 2D arrays
    if data.ndim == 1:
        # 1D array - single channel
        return filtfilt(b, a, data)
    else:
        # 2D array - multiple channels
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
        return filtered_data

def extract_beta_bursts(eeg_data: np.ndarray, fs: float = 250.0,
                       beta_band: Tuple[float, float] = (13, 30),
                       min_duration_ms: float = 100,
                       merge_gap_ms: float = 60) -> Dict[str, List]:
    """
    Extract beta bursts from EEG data using He et al. 2020 methodology.

    Args:
        eeg_data: EEG data (samples, channels) or (samples,) for single channel
        fs: Sampling frequency
        beta_band: Beta frequency band (low, high)
        min_duration_ms: Minimum burst duration in ms
        merge_gap_ms: Maximum gap to merge nearby bursts in ms

    Returns:
        Dictionary with burst events per channel
    """
    # Ensure 2D array
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    # Filter to beta band
    filtered_data = bandpass_filter(eeg_data, beta_band[0], beta_band[1], fs)

    # For single channel, CAR has no effect (mean across 1 channel = channel itself)
    # But apply it for consistency
    if filtered_data.ndim == 1:
        car_data = filtered_data  # No change for 1D
    else:
        car_data = filtered_data - np.mean(filtered_data, axis=1, keepdims=True)

    # Extract amplitude envelope using Hilbert transform
    if car_data.ndim == 1:
        envelope = np.abs(hilbert(car_data)).reshape(-1, 1)
    else:
        # Apply Hilbert transform to each channel separately
        envelope = np.zeros_like(car_data)
        for ch in range(car_data.shape[1]):
            envelope[:, ch] = np.abs(hilbert(car_data[:, ch]))

    n_channels = envelope.shape[1]

    # For single channel data, treat it as motor region
    if n_channels == 1:
        motor_channels = [0]
        posterior_channels = []
    else:
        # Define channel groupings (simplified - assume first 32 channels are standard 10-20)
        n_channels = min(envelope.shape[1], 32)
        motor_channels = [ch for ch in [2, 8, 9] if ch < n_channels]  # Approximate C3, Cz, C4 positions
        posterior_channels = [ch for ch in [18, 19, 20, 28, 29] if ch < n_channels]  # Approximate P3, Pz, P4, O1, O2 positions

    burst_data = {
        'motor_bursts': [],
        'posterior_bursts': [],
        'all_channel_bursts': []
    }

    # Extract bursts for each region
    for region_name, channels in [('motor', motor_channels), ('posterior', posterior_channels)]:
        if not channels:
            continue

        # Average envelope across region channels
        region_envelope = np.mean(envelope[:, channels], axis=1)

        # Threshold: median + 2 * MAD
        median_amp = np.median(region_envelope)
        mad = stats.median_abs_deviation(region_envelope)
        threshold = median_amp + 2 * mad

        # Find periods above threshold
        above_threshold = region_envelope > threshold

        # Convert to burst events
        bursts = []
        in_burst = False
        burst_start = None

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_burst:
                # Start of burst
                in_burst = True
                burst_start = i
            elif not is_above and in_burst:
                # End of burst
                in_burst = False
                burst_duration_ms = (i - burst_start) / fs * 1000

                if burst_duration_ms >= min_duration_ms:
                    # Calculate burst properties
                    burst_envelope = region_envelope[burst_start:i]
                    burst_properties = {
                        'start_sample': burst_start,
                        'end_sample': i,
                        'duration_ms': burst_duration_ms,
                        'peak_amplitude': np.max(burst_envelope),
                        'mean_amplitude': np.mean(burst_envelope),
                        'peak_time': burst_start + np.argmax(burst_envelope)
                    }
                    bursts.append(burst_properties)

        # Handle case where burst extends to end of data
        if in_burst and burst_start is not None:
            burst_duration_ms = (len(above_threshold) - burst_start) / fs * 1000
            if burst_duration_ms >= min_duration_ms:
                burst_envelope = region_envelope[burst_start:]
                burst_properties = {
                    'start_sample': burst_start,
                    'end_sample': len(above_threshold),
                    'duration_ms': burst_duration_ms,
                    'peak_amplitude': np.max(burst_envelope),
                    'mean_amplitude': np.mean(burst_envelope),
                    'peak_time': burst_start + np.argmax(burst_envelope)
                }
                bursts.append(burst_properties)

        # Merge nearby bursts
        if bursts:
            merged_bursts = []
            current_burst = bursts[0].copy()

            for next_burst in bursts[1:]:
                gap_ms = (next_burst['start_sample'] - current_burst['end_sample']) / fs * 1000

                if gap_ms <= merge_gap_ms:
                    # Merge bursts
                    current_burst['end_sample'] = next_burst['end_sample']
                    current_burst['duration_ms'] = (current_burst['end_sample'] - current_burst['start_sample']) / fs * 1000
                    current_burst['peak_amplitude'] = max(current_burst['peak_amplitude'], next_burst['peak_amplitude'])
                else:
                    # Save current burst and start new one
                    merged_bursts.append(current_burst)
                    current_burst = next_burst.copy()

            # Add final burst
            merged_bursts.append(current_burst)
            bursts = merged_bursts

        burst_data[f'{region_name}_bursts'] = bursts

    # Store all bursts together
    burst_data['all_channel_bursts'] = burst_data['motor_bursts'] + burst_data['posterior_bursts']

    return burst_data

def compute_burst_features(burst_data: Dict[str, List], session_duration_s: float) -> Dict[str, float]:
    """
    Compute burst-based features for a session.

    Args:
        burst_data: Burst detection results
        session_duration_s: Total session duration in seconds

    Returns:
        Dictionary of burst features
    """
    features = {}

    # Basic burst counts
    motor_bursts = burst_data['motor_bursts']
    posterior_bursts = burst_data['posterior_bursts']
    all_bursts = burst_data['all_channel_bursts']

    # Rate features (bursts per minute)
    features['rate_per_min'] = len(all_bursts) / (session_duration_s / 60)
    features['motor_rate_per_min'] = len(motor_bursts) / (session_duration_s / 60)
    features['posterior_rate_per_min'] = len(posterior_bursts) / (session_duration_s / 60)

    # Duration features
    if all_bursts:
        durations = [burst['duration_ms'] for burst in all_bursts]
        features['mean_duration_ms'] = np.mean(durations)
        features['median_duration_ms'] = np.median(durations)
        features['duration_cv'] = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
    else:
        features['mean_duration_ms'] = 0
        features['median_duration_ms'] = 0
        features['duration_cv'] = 0

    # Amplitude features
    if all_bursts:
        amplitudes = [burst['peak_amplitude'] for burst in all_bursts]
        mean_amplitudes = [burst['mean_amplitude'] for burst in all_bursts]

        features['mean_peak_amp'] = np.mean(amplitudes)
        features['p95_peak_amp'] = np.percentile(amplitudes, 95)
        features['amplitude_cv'] = np.std(mean_amplitudes) / np.mean(mean_amplitudes) if np.mean(mean_amplitudes) > 0 else 0
    else:
        features['mean_peak_amp'] = 0
        features['p95_peak_amp'] = 0
        features['amplitude_cv'] = 0

    # Temporal features
    if all_bursts:
        # Calculate inter-burst intervals
        if len(all_bursts) > 1:
            # Sort by start time
            sorted_bursts = sorted(all_bursts, key=lambda x: x['start_sample'])
            ibis = []
            for i in range(1, len(sorted_bursts)):
                ibi_samples = sorted_bursts[i]['start_sample'] - sorted_bursts[i-1]['end_sample']
                ibi_ms = ibi_samples / 250 * 1000  # Assuming 250 Hz
                if ibi_ms > 0:
                    ibis.append(ibi_ms)

            if ibis:
                features['mean_ibi_ms'] = np.mean(ibis)
                features['cv_ibi'] = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
            else:
                features['mean_ibi_ms'] = 0
                features['cv_ibi'] = 0
        else:
            features['mean_ibi_ms'] = 0
            features['cv_ibi'] = 0

        # Duty cycle (proportion of time in bursts)
        total_burst_duration_ms = sum(burst['duration_ms'] for burst in all_bursts)
        session_duration_ms = session_duration_s * 1000
        features['duty_cycle'] = total_burst_duration_ms / session_duration_ms

        # Burst density (bursts per 10-second window)
        features['density_per_10s'] = len(all_bursts) / (session_duration_s / 10)
    else:
        features['mean_ibi_ms'] = 0
        features['cv_ibi'] = 0
        features['duty_cycle'] = 0
        features['density_per_10s'] = 0

    # Regional ratios (key discriminative features)
    if features['posterior_rate_per_min'] > 0:
        features['motor_posterior_rate_ratio'] = features['motor_rate_per_min'] / features['posterior_rate_per_min']
    else:
        features['motor_posterior_rate_ratio'] = features['motor_rate_per_min']  # Handle division by zero

    # Motor vs posterior duty cycle ratio
    if motor_bursts and posterior_bursts:
        motor_duty = sum(burst['duration_ms'] for burst in motor_bursts) / (session_duration_s * 1000)
        posterior_duty = sum(burst['duration_ms'] for burst in posterior_bursts) / (session_duration_s * 1000)

        if posterior_duty > 0:
            features['motor_posterior_duty_ratio'] = motor_duty / posterior_duty
        else:
            features['motor_posterior_duty_ratio'] = motor_duty
    else:
        features['motor_posterior_duty_ratio'] = 0

    return features

def extract_subject_features(subject_id: str, metadata: Dict, max_sessions: int = 10) -> Optional[pd.DataFrame]:
    """
    Extract burst features for all sessions from a subject.

    Args:
        subject_id: Subject identifier
        metadata: Subject metadata
        max_sessions: Maximum number of sessions to process per subject

    Returns:
        DataFrame with burst features per session
    """
    logger = logging.getLogger(__name__)

    if subject_id not in metadata['subjects']:
        return None

    subject_info = metadata['subjects'][subject_id]

    # Skip excluded subjects
    if not subject_info.get('include_in_analysis', False):
        return None

    condition = subject_info['condition']
    if condition not in ['PD_REAL', 'PD_SHAM']:
        return None

    # Get list of pickle files for this subject
    subject_dir = Path(f"data/external/leicester_dataset/{subject_id}")
    if not subject_dir.exists():
        return None

    pickle_files = list(subject_dir.glob("*_pickle_new"))

    # Limit number of sessions to avoid overprocessing
    if len(pickle_files) > max_sessions:
        pickle_files = pickle_files[:max_sessions]

    session_features = []
    processed_count = 0

    for pickle_file in pickle_files:
        try:
            # Load EEG data
            eeg_data = load_eeg_session(subject_id, pickle_file.name)
            if eeg_data is None:
                continue

            # Calculate session duration
            session_duration_s = len(eeg_data) / 250.0  # Assuming 250 Hz

            # Skip very short sessions
            if session_duration_s < 30:  # Less than 30 seconds
                continue

            # Extract beta bursts
            burst_data = extract_beta_bursts(eeg_data, fs=250.0, min_duration_ms=100)

            # Compute features
            features = compute_burst_features(burst_data, session_duration_s)

            # Add metadata
            features['subject_id'] = subject_id
            features['condition'] = condition
            features['session_file'] = pickle_file.name
            features['session_duration_s'] = session_duration_s

            session_features.append(features)
            processed_count += 1

            # Log progress
            if processed_count % 5 == 0:
                logger.info(f"  Processed {processed_count} sessions for {subject_id}")

        except Exception as e:
            logger.warning(f"Failed to process {pickle_file} for {subject_id}: {str(e)}")
            continue

    if session_features:
        logger.info(f"Successfully extracted features from {len(session_features)} sessions for {subject_id}")
        return pd.DataFrame(session_features)
    else:
        logger.warning(f"No valid sessions processed for {subject_id}")
        return None

def main():
    """Main execution for Phase 8."""
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("PHASE 8: REAL LEICESTER BETA BURST EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")

    try:
        # Load subject metadata
        logger.info("Loading Leicester subject metadata...")
        metadata = load_subject_metadata()

        # Get list of subjects to process
        valid_subjects = []
        for subject_id, info in metadata['subjects'].items():
            if info.get('include_in_analysis', False) and info['condition'] in ['PD_REAL', 'PD_SHAM']:
                valid_subjects.append(subject_id)

        logger.info(f"Processing {len(valid_subjects)} subjects")
        pd_real_count = sum(1 for s in valid_subjects if metadata['subjects'][s]['condition'] == 'PD_REAL')
        pd_sham_count = sum(1 for s in valid_subjects if metadata['subjects'][s]['condition'] == 'PD_SHAM')
        logger.info(f"PD_REAL: {pd_real_count}, PD_SHAM: {pd_sham_count}")

        # Extract features from each subject
        all_features = []

        for i, subject_id in enumerate(valid_subjects):
            logger.info(f"Processing subject {subject_id} ({i+1}/{len(valid_subjects)})...")

            subject_features = extract_subject_features(subject_id, metadata, max_sessions=8)

            if subject_features is not None and len(subject_features) > 0:
                all_features.append(subject_features)
                logger.info(f"  Added {len(subject_features)} sessions from {subject_id}")
            else:
                logger.warning(f"  No features extracted from {subject_id}")

        if not all_features:
            logger.error("No features extracted from any subject!")
            return {'status': 'failed', 'error': 'No valid data'}

        # Combine all features
        logger.info("Combining features from all subjects...")
        combined_features = pd.concat(all_features, ignore_index=True)

        logger.info(f"Total feature matrix: {combined_features.shape}")
        logger.info(f"Class distribution: {combined_features['condition'].value_counts().to_dict()}")

        # Create output directories
        output_dir = Path("results/phase8_real_bursts")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw features
        features_path = output_dir / "real_leicester_burst_features.csv"
        combined_features.to_csv(features_path, index=False)
        logger.info(f"Raw features saved: {features_path}")

        # Quick model training to validate features
        logger.info("Training quick validation model...")

        # Prepare features and labels
        feature_columns = [col for col in combined_features.columns
                          if col not in ['subject_id', 'condition', 'session_file', 'session_duration_s']]

        X = combined_features[feature_columns].values
        y = combined_features['condition'].values
        subjects = combined_features['subject_id'].values

        # Handle any NaN/inf values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Quick cross-validation
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        # Simple logistic regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Label encoding
        y_encoded = (y == 'PD_SHAM').astype(int)

        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded, subjects)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            ba_score = balanced_accuracy_score(y_test, y_pred)
            cv_scores.append(ba_score)

            logger.info(f"  Fold {fold+1}: BA = {ba_score:.3f}")

        mean_ba = np.mean(cv_scores)
        std_ba = np.std(cv_scores)

        logger.info("=" * 60)
        logger.info("REAL LEICESTER BURST FEATURES - VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Cross-validation BA: {mean_ba:.3f} ± {std_ba:.3f}")

        # Save results
        results = {
            'validation_results': {
                'mean_balanced_accuracy': float(mean_ba),
                'std_balanced_accuracy': float(std_ba),
                'cv_scores': [float(score) for score in cv_scores]
            },
            'dataset_info': {
                'n_samples': int(len(combined_features)),
                'n_features': len(feature_columns),
                'n_subjects': len(combined_features['subject_id'].unique()),
                'class_distribution': combined_features['condition'].value_counts().to_dict()
            },
            'timestamp': datetime.now().isoformat()
        }

        results_path = output_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 80)
        logger.info("PHASE 8 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Real Leicester burst features extracted from {len(combined_features)} sessions")
        logger.info(f"Validation performance: {mean_ba:.1%} ± {std_ba:.1%}")

        if mean_ba >= 0.70:
            logger.info("✅ TARGET ACHIEVED: Real burst features show >70% balanced accuracy!")
        elif mean_ba >= 0.60:
            logger.info("⚠️  PROMISING: Real burst features show good separation, optimization needed")
        else:
            logger.info("❌ CHALLENGE: Real burst features show limited separation, review needed")

        return {
            'status': 'success',
            'balanced_accuracy': mean_ba,
            'n_samples': len(combined_features),
            'n_subjects': len(combined_features['subject_id'].unique()),
            'features_path': str(features_path),
            'results_path': str(results_path)
        }

    except Exception as e:
        logger.error(f"Phase 8 failed: {str(e)}")
        logger.error("", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    results = main()
    if results['status'] == 'failed':
        sys.exit(1)