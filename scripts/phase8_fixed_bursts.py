#!/usr/bin/env python3
"""
Phase 8: Real Leicester Beta Burst Extraction - FIXED VERSION
Using battle-tested burst detection that avoids common traps
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from scipy.signal import butter, filtfilt, hilbert, iirnotch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
import pickle

warnings.filterwarnings('ignore')

def setup_logging():
    """Configure logging for Phase 8."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/phase8_fixed_bursts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# BATTLE-TESTED BURST EXTRACTION (FIXED VERSION)
# ============================================================================

def _butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='bandpass')
    return b, a

def _apply_notch(x, fs, f0=50.0, Q=30.0):
    # x: (samples, channels)
    b, a = iirnotch(w0=f0/(fs/2), Q=Q)
    return filtfilt(b, a, x, axis=0)

def _mad(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis, keepdims=True)

def _segments_above(bool_idx):
    """Return list of (start, end) sample indices for contiguous True regions."""
    if bool_idx.ndim != 1:
        raise ValueError("bool_idx must be 1D")
    diff = np.diff(bool_idx.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def _merge_segments(segments, max_gap):
    """Merge (start,end) if gap between consecutive segments <= max_gap."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def extract_beta_bursts_fixed(
    data,                 # np.ndarray shape (samples, channels) or (samples,)
    sfreq,                # sampling rate (float)
    motor_chs=None,       # list[str] or list[int]
    posterior_chs=None,   # list[str] or list[int]
    ch_names=None,        # list[str] if indexing by names
    band=(13, 30),
    car=True,
    notch=False,          # Skip notch for Leicester data
    thresh_k=2.0,         # envelope threshold = median + k * MAD (per channel)
    min_dur_s=0.100,      # >=100 ms
    merge_gap_s=0.060,    # merge segments with <=60 ms gap
    verbose=True
):
    """
    Fixed version of beta burst extraction that avoids all common traps.

    Returns:
      {
        'bursts': {ch_name: [(start_idx, end_idx, peak_amp), ...], ...},
        'features': {
            'motor': {...},
            'posterior': {...},
            'global': {...}
        }
      }
    """
    # Handle 1D input (single channel)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    assert data.ndim == 2, "data must be (samples, channels)"
    n_samp, n_ch = data.shape
    fs = float(sfreq)

    # --- Channel indexing
    if ch_names is None:
        ch_names = [f"ch{idx}" for idx in range(n_ch)]
    name_to_idx = {nm: i for i, nm in enumerate(ch_names)}

    def _idx_list(x):
        if x is None:
            return []
        if len(x) == 0:
            return []
        if isinstance(x[0], str):
            return [name_to_idx[nm] for nm in x if nm in name_to_idx]
        return list(x)

    # For single channel, use it as motor
    if n_ch == 1:
        motor_idx = [0]
        posterior_idx = []
    else:
        motor_idx     = _idx_list(motor_chs) or _idx_list(['C3','Cz','C4'])
        posterior_idx = _idx_list(posterior_chs) or _idx_list(['P3','Pz','P4','O1','O2'])

    # --- Copy to float64
    x = np.asarray(data, dtype=np.float64)

    # --- Notch (optional, skip for Leicester)
    if notch:
        x = _apply_notch(x, fs, f0=50.0, Q=30.0)

    # --- CAR (optional): subtract per-sample mean across channels
    # Skip CAR for single channel
    if car and n_ch > 1:
        x = x - np.mean(x, axis=1, keepdims=True)

    # --- Bandpass filter (13–30 Hz by default)
    b, a = _butter_bandpass(band[0], band[1], fs, order=4)
    xf = filtfilt(b, a, x, axis=0)

    # --- Hilbert envelope (axis=0 -> over time; result shape (samples, channels))
    env = np.abs(hilbert(xf, axis=0))

    if verbose:
        print(f"[DEBUG] env shape: {env.shape}, range: {np.min(env):.3f} to {np.max(env):.3f}")

    # --- Thresholds per channel (median + k*MAD) with keepdims to broadcast cleanly
    med = np.median(env, axis=0, keepdims=True)
    mad = _mad(env, axis=0)  # keepdims=True in helper
    thr = med + thresh_k * mad
    above = env > thr  # (samples, channels) boolean

    if verbose:
        print(f"[DEBUG] threshold range: {np.min(thr):.3f} to {np.max(thr):.3f}")
        print(f"[DEBUG] samples above threshold: {np.sum(above)} / {above.size} ({100*np.sum(above)/above.size:.1f}%)")

    # --- Convert durations to samples
    min_dur   = int(round(min_dur_s * fs))
    merge_gap = int(round(merge_gap_s * fs))

    if verbose:
        print(f"[DEBUG] min_dur: {min_dur} samples, merge_gap: {merge_gap} samples")

    bursts_dict = {}
    total_bursts = 0

    # --- Detect segments per channel
    for ci in range(n_ch):
        mask = above[:, ci].ravel()
        segs = _segments_above(mask)

        # Filter by min duration
        segs = [(s, e) for (s, e) in segs if (e - s) >= min_dur]
        segs = _merge_segments(segs, merge_gap)

        # Re-check min_dur after merge
        segs = [(s, e) for (s, e) in segs if (e - s) >= min_dur]

        # Collect with peak amplitude
        ch_bursts = []
        for (s, e) in segs:
            peak_amp = float(np.max(env[s:e, ci])) if e > s else 0.0
            ch_bursts.append((int(s), int(e), peak_amp))

        bursts_dict[ch_names[ci]] = ch_bursts
        total_bursts += len(ch_bursts)

        if verbose and len(ch_bursts) > 0:
            print(f"[DEBUG] Channel {ch_names[ci]}: {len(ch_bursts)} bursts")

    if verbose:
        print(f"[extract_beta_bursts_fixed] fs={fs} Hz | band={band} Hz | "
              f"min_dur={min_dur} samp | merge_gap={merge_gap} samp | "
              f"total_bursts={total_bursts}")

    # --- Region-wise features
    def _region_feats(idx_list, label):
        if not idx_list:
            return {
                'rate_per_min': 0.0, 'mean_dur_ms': 0.0,
                'mean_amp': 0.0, 'duty_cycle_pct': 0.0, 'n_bursts': 0
            }
        n_b = 0
        durs = []
        amps = []
        mask_any = np.zeros(n_samp, dtype=bool)
        for i in idx_list:
            for (s, e, amp) in bursts_dict[ch_names[i]]:
                n_b += 1
                durs.append((e - s) / fs)
                amps.append(amp)
                mask_any[s:e] = True
        dur_s = np.array(durs) if durs else np.array([])
        rate_per_min   = (n_b / (n_samp / fs)) * 60.0 if n_samp > 0 else 0.0
        mean_dur_ms    = (np.mean(dur_s) * 1000.0) if dur_s.size else 0.0
        mean_amp       = float(np.mean(amps)) if amps else 0.0
        duty_cycle_pct = (100.0 * mask_any.sum() / n_samp) if n_samp > 0 else 0.0
        return {
            'rate_per_min': rate_per_min,
            'mean_dur_ms': mean_dur_ms,
            'mean_amp': mean_amp,
            'duty_cycle_pct': duty_cycle_pct,
            'n_bursts': n_b
        }

    feats_motor     = _region_feats(motor_idx, 'motor')
    feats_post      = _region_feats(posterior_idx, 'posterior')
    motor_rate      = feats_motor['rate_per_min']
    posterior_rate  = feats_post['rate_per_min']
    mp_ratio = (motor_rate / posterior_rate) if (posterior_rate and posterior_rate > 0) else motor_rate

    features = {
        'motor': feats_motor,
        'posterior': feats_post,
        'global': {
            'motor_posterior_rate_ratio': mp_ratio
        }
    }

    return {'bursts': bursts_dict, 'features': features}

# ============================================================================
# LEICESTER DATA LOADING
# ============================================================================

def load_subject_metadata() -> Dict[str, Any]:
    """Load Leicester subject metadata."""
    metadata_path = Path("data/interim/leicester_subject_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_eeg_session(subject_id: str, session_file: str) -> Optional[np.ndarray]:
    """Load a single EEG session pickle file."""
    logger = logging.getLogger(__name__)

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
        continuous_data = np.concatenate(epoch_list, axis=0)

        # Skip if too few samples (less than 10 seconds at 250 Hz)
        if len(continuous_data) < 2500:
            logger.debug(f"Session too short: {len(continuous_data)} samples")
            return None

        logger.debug(f"Successfully loaded {session_path}: {continuous_data.shape}")
        return continuous_data  # Return 1D array

    except Exception as e:
        logger.debug(f"Error loading {session_path}: {e}")
        return None

def extract_subject_features(subject_id: str, metadata: Dict, max_sessions: int = 8) -> Optional[pd.DataFrame]:
    """Extract burst features for all sessions from a subject using FIXED extractor."""
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

    # Limit number of sessions
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
            session_duration_s = len(eeg_data) / 250.0

            # Skip very short sessions
            if session_duration_s < 30:  # Less than 30 seconds
                continue

            # Extract beta bursts using FIXED extractor
            result = extract_beta_bursts_fixed(
                eeg_data,
                sfreq=250.0,
                ch_names=['Cz'],  # Single channel
                motor_chs=['Cz'],
                posterior_chs=[],  # No posterior for single channel
                thresh_k=2.0,
                min_dur_s=0.100,
                verbose=False  # Set to True for debugging
            )

            # Extract features from result
            motor_feats = result['features']['motor']
            global_feats = result['features']['global']

            # Build feature vector (compatible with original format)
            features = {
                'rate_per_min': motor_feats['rate_per_min'],
                'motor_rate_per_min': motor_feats['rate_per_min'],
                'posterior_rate_per_min': 0.0,  # Single channel = no posterior
                'mean_duration_ms': motor_feats['mean_dur_ms'],
                'median_duration_ms': motor_feats['mean_dur_ms'],  # Approximate
                'duration_cv': 0.0,  # TODO: compute properly if needed
                'mean_peak_amp': motor_feats['mean_amp'],
                'p95_peak_amp': motor_feats['mean_amp'],  # Approximate
                'amplitude_cv': 0.0,  # TODO: compute properly if needed
                'mean_ibi_ms': 0.0,   # TODO: compute IBI if needed
                'cv_ibi': 0.0,
                'duty_cycle': motor_feats['duty_cycle_pct'] / 100.0,
                'density_per_10s': motor_feats['rate_per_min'] / 6.0,  # Convert from per-minute
                'motor_posterior_rate_ratio': global_feats['motor_posterior_rate_ratio'],
                'motor_posterior_duty_ratio': global_feats['motor_posterior_rate_ratio'],  # Approximate
                # Metadata
                'subject_id': subject_id,
                'condition': condition,
                'session_file': pickle_file.name,
                'session_duration_s': session_duration_s
            }

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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for Phase 8 FIXED."""
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("PHASE 8: REAL LEICESTER BETA BURST EXTRACTION - FIXED VERSION")
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

                # Show sample features for first few subjects
                if i < 3:
                    sample = subject_features.iloc[0]
                    logger.info(f"  Sample features: rate={sample['rate_per_min']:.1f}/min, dur={sample['mean_duration_ms']:.1f}ms, amp={sample['mean_peak_amp']:.2f}")
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

        # Check for non-zero features
        non_zero_rate = (combined_features['rate_per_min'] > 0).sum()
        logger.info(f"Non-zero burst rates: {non_zero_rate} / {len(combined_features)} ({100*non_zero_rate/len(combined_features):.1f}%)")

        # Create output directories
        output_dir = Path("results/phase8_fixed_bursts")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw features
        features_path = output_dir / "real_leicester_burst_features_FIXED.csv"
        combined_features.to_csv(features_path, index=False)
        logger.info(f"Raw features saved: {features_path}")

        # Quick model training if we have non-zero features
        if non_zero_rate > 0:
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
            logger.info("REAL LEICESTER BURST FEATURES - VALIDATION RESULTS (FIXED)")
            logger.info("=" * 60)
            logger.info(f"Cross-validation BA: {mean_ba:.3f} ± {std_ba:.3f}")

            # Save results
            results = {
                'validation_results': {
                    'mean_balanced_accuracy': float(mean_ba),
                    'std_balanced_accuracy': float(std_ba),
                    'cv_scores': [float(score) for score in cv_scores],
                    'non_zero_features': int(non_zero_rate)
                },
                'dataset_info': {
                    'n_samples': int(len(combined_features)),
                    'n_features': len(feature_columns),
                    'n_subjects': len(combined_features['subject_id'].unique()),
                    'class_distribution': combined_features['condition'].value_counts().to_dict()
                },
                'timestamp': datetime.now().isoformat()
            }

            results_path = output_dir / "validation_results_FIXED.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info("=" * 80)
            logger.info("PHASE 8 COMPLETED SUCCESSFULLY - FIXED VERSION")
            logger.info("=" * 80)
            logger.info(f"Real Leicester burst features extracted from {len(combined_features)} sessions")
            logger.info(f"Sessions with bursts: {non_zero_rate} / {len(combined_features)}")
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
                'sessions_with_bursts': non_zero_rate,
                'features_path': str(features_path),
                'results_path': str(results_path)
            }
        else:
            logger.error("No bursts detected in any session - algorithm still has issues")
            return {'status': 'failed', 'error': 'No bursts detected'}

    except Exception as e:
        logger.error(f"Phase 8 failed: {str(e)}")
        logger.error("", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    results = main()
    if results['status'] == 'failed':
        sys.exit(1)