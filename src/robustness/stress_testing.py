"""
Automated Stress Testing Framework
Comprehensive robustness validation across multiple perturbation types
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import json
import sqlite3
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class StressTestConfig:
    """Configuration for stress testing parameters"""
    # Noise stress tests
    noise_levels: List[float] = None  # SNR levels to test
    noise_types: List[str] = None     # 'gaussian', 'pink', 'powerline', 'muscle', 'eye'

    # Electrode dropout tests
    dropout_rates: List[float] = None # Percentage of electrodes to drop
    dropout_patterns: List[str] = None # 'random', 'contiguous', 'specific_regions'

    # Duration variation tests
    duration_factors: List[float] = None # Multipliers for recording length
    truncation_positions: List[str] = None # 'start', 'middle', 'end', 'random'

    # Amplitude variation tests
    amplitude_factors: List[float] = None # Amplitude scaling factors

    # Frequency domain tests
    filter_corruptions: List[Dict] = None # Various filter artifacts

    # Temporal tests
    sampling_rates: List[float] = None # Different sampling rates

    # Cross-validation parameters
    n_repetitions: int = 100
    confidence_level: float = 0.95
    parallel_workers: int = 4

    # Performance thresholds
    min_accuracy_threshold: float = 0.7
    max_performance_drop: float = 0.1

    def __post_init__(self):
        """Set default values if not provided"""
        if self.noise_levels is None:
            self.noise_levels = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # SNR values

        if self.noise_types is None:
            self.noise_types = ['gaussian', 'pink', 'powerline', 'muscle', 'eye']

        if self.dropout_rates is None:
            self.dropout_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        if self.dropout_patterns is None:
            self.dropout_patterns = ['random', 'contiguous', 'frontal', 'occipital']

        if self.duration_factors is None:
            self.duration_factors = [0.25, 0.5, 0.75, 1.5, 2.0, 3.0]

        if self.truncation_positions is None:
            self.truncation_positions = ['start', 'middle', 'end', 'random']

        if self.amplitude_factors is None:
            self.amplitude_factors = [0.1, 0.5, 0.8, 1.2, 2.0, 5.0]

        if self.filter_corruptions is None:
            self.filter_corruptions = [
                {'type': 'highpass', 'cutoff': 0.1},
                {'type': 'lowpass', 'cutoff': 30},
                {'type': 'notch', 'freq': 50, 'quality': 10},
                {'type': 'bandstop', 'low': 8, 'high': 12}
            ]

        if self.sampling_rates is None:
            self.sampling_rates = [125, 250, 500, 1000, 2000]

@dataclass
class StressTestResult:
    """Results from a single stress test"""
    test_id: str
    test_type: str
    perturbation_params: Dict[str, Any]
    subject_id: str
    baseline_metrics: Dict[str, float]
    stressed_metrics: Dict[str, float]
    performance_drop: Dict[str, float]
    passed_thresholds: bool
    execution_time: float
    timestamp: datetime
    model_id: str
    explanation_available: bool = False

class EEGPerturbationEngine:
    """Generate various EEG signal perturbations for stress testing"""

    def __init__(self, sampling_rate: float = 250.0):
        self.sampling_rate = sampling_rate

    def add_noise(self, eeg_data: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
        """Add various types of noise to EEG data"""
        signal_power = np.mean(eeg_data ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        if noise_type == 'gaussian':
            noise = np.random.normal(0, np.sqrt(noise_power), eeg_data.shape)

        elif noise_type == 'pink':
            # Generate pink noise (1/f)
            noise = self._generate_pink_noise(eeg_data.shape, noise_power)

        elif noise_type == 'powerline':
            # 50Hz powerline interference
            t = np.arange(eeg_data.shape[-1]) / self.sampling_rate
            powerline = np.sqrt(noise_power) * np.sin(2 * np.pi * 50 * t)
            noise = np.broadcast_to(powerline, eeg_data.shape)

        elif noise_type == 'muscle':
            # High-frequency muscle artifacts
            noise = self._generate_muscle_artifact(eeg_data.shape, noise_power)

        elif noise_type == 'eye':
            # Eye movement artifacts (low frequency, high amplitude)
            noise = self._generate_eye_artifact(eeg_data.shape, noise_power)

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return eeg_data + noise

    def dropout_electrodes(self, eeg_data: np.ndarray, dropout_rate: float,
                          pattern: str = 'random') -> np.ndarray:
        """Simulate electrode dropouts"""
        n_channels = eeg_data.shape[0]
        n_dropout = int(n_channels * dropout_rate)

        if pattern == 'random':
            dropout_indices = np.random.choice(n_channels, n_dropout, replace=False)

        elif pattern == 'contiguous':
            start_idx = np.random.randint(0, n_channels - n_dropout + 1)
            dropout_indices = np.arange(start_idx, start_idx + n_dropout)

        elif pattern == 'frontal':
            # Simulate frontal electrode dropout (first 25% of channels)
            frontal_channels = int(0.25 * n_channels)
            dropout_indices = np.random.choice(frontal_channels,
                                             min(n_dropout, frontal_channels), replace=False)

        elif pattern == 'occipital':
            # Simulate occipital electrode dropout (last 25% of channels)
            occipital_start = int(0.75 * n_channels)
            occipital_channels = np.arange(occipital_start, n_channels)
            dropout_indices = np.random.choice(occipital_channels,
                                             min(n_dropout, len(occipital_channels)), replace=False)

        # Create copy and zero out dropped electrodes
        corrupted_data = eeg_data.copy()
        corrupted_data[dropout_indices, :] = 0

        return corrupted_data

    def vary_duration(self, eeg_data: np.ndarray, factor: float,
                     position: str = 'random') -> np.ndarray:
        """Vary recording duration"""
        original_length = eeg_data.shape[-1]
        new_length = int(original_length * factor)

        if factor >= 1.0:
            # Extend by repetition
            repetitions = int(np.ceil(factor))
            extended = np.tile(eeg_data, (1, repetitions))
            return extended[:, :new_length]

        else:
            # Truncate
            if position == 'start':
                return eeg_data[:, :new_length]
            elif position == 'end':
                return eeg_data[:, -new_length:]
            elif position == 'middle':
                start_idx = (original_length - new_length) // 2
                return eeg_data[:, start_idx:start_idx + new_length]
            elif position == 'random':
                start_idx = np.random.randint(0, original_length - new_length + 1)
                return eeg_data[:, start_idx:start_idx + new_length]

    def scale_amplitude(self, eeg_data: np.ndarray, factor: float) -> np.ndarray:
        """Scale signal amplitude"""
        return eeg_data * factor

    def apply_filter_corruption(self, eeg_data: np.ndarray,
                               corruption: Dict[str, Any]) -> np.ndarray:
        """Apply various filter corruptions"""
        nyquist = self.sampling_rate / 2

        if corruption['type'] == 'highpass':
            sos = signal.butter(4, corruption['cutoff'] / nyquist, btype='high', output='sos')
            return signal.sosfilt(sos, eeg_data, axis=1)

        elif corruption['type'] == 'lowpass':
            sos = signal.butter(4, corruption['cutoff'] / nyquist, btype='low', output='sos')
            return signal.sosfilt(sos, eeg_data, axis=1)

        elif corruption['type'] == 'notch':
            freq = corruption['freq']
            quality = corruption['quality']
            sos = signal.iirnotch(freq / nyquist, quality, output='sos')
            return signal.sosfilt(sos, eeg_data, axis=1)

        elif corruption['type'] == 'bandstop':
            low = corruption['low'] / nyquist
            high = corruption['high'] / nyquist
            sos = signal.butter(4, [low, high], btype='bandstop', output='sos')
            return signal.sosfilt(sos, eeg_data, axis=1)

        return eeg_data

    def resample_data(self, eeg_data: np.ndarray, target_rate: float) -> np.ndarray:
        """Resample to different sampling rate"""
        if target_rate == self.sampling_rate:
            return eeg_data

        resample_factor = target_rate / self.sampling_rate
        new_length = int(eeg_data.shape[-1] * resample_factor)

        return signal.resample(eeg_data, new_length, axis=1)

    def _generate_pink_noise(self, shape: Tuple, power: float) -> np.ndarray:
        """Generate pink (1/f) noise"""
        n_samples = shape[-1]
        n_channels = shape[0] if len(shape) > 1 else 1

        # Generate white noise in frequency domain
        freqs = np.fft.fftfreq(n_samples)
        freqs[0] = 1e-10  # Avoid division by zero

        # Create 1/f spectrum
        spectrum = 1 / np.sqrt(np.abs(freqs))
        spectrum[0] = 0  # DC component

        noise = np.zeros(shape)
        for ch in range(n_channels):
            # Random phase
            phase = np.random.uniform(0, 2*np.pi, len(spectrum))
            complex_spectrum = spectrum * np.exp(1j * phase)

            # Convert to time domain
            time_series = np.fft.ifft(complex_spectrum).real
            noise[ch if len(shape) > 1 else 0] = time_series

        # Scale to desired power
        current_power = np.mean(noise ** 2)
        if current_power > 0:
            noise *= np.sqrt(power / current_power)

        return noise

    def _generate_muscle_artifact(self, shape: Tuple, power: float) -> np.ndarray:
        """Generate muscle artifact (high-frequency noise)"""
        # High-frequency components (50-200 Hz)
        noise = np.random.normal(0, 1, shape)

        # Apply high-pass filter
        nyquist = self.sampling_rate / 2
        sos = signal.butter(4, 50 / nyquist, btype='high', output='sos')
        noise = signal.sosfilt(sos, noise, axis=-1)

        # Scale to desired power
        current_power = np.mean(noise ** 2)
        if current_power > 0:
            noise *= np.sqrt(power / current_power)

        return noise

    def _generate_eye_artifact(self, shape: Tuple, power: float) -> np.ndarray:
        """Generate eye movement artifacts"""
        # Low-frequency, high-amplitude transients
        n_samples = shape[-1]
        t = np.arange(n_samples) / self.sampling_rate

        # Random eye movements (0.1-2 Hz)
        n_movements = np.random.randint(1, 5)
        artifact = np.zeros(shape)

        for ch in range(shape[0] if len(shape) > 1 else 1):
            ch_artifact = np.zeros(n_samples)

            for _ in range(n_movements):
                # Random timing and amplitude
                start_time = np.random.uniform(0, t[-1] * 0.8)
                duration = np.random.uniform(0.5, 2.0)
                amplitude = np.random.uniform(0.5, 2.0)

                # Create transient
                mask = (t >= start_time) & (t <= start_time + duration)
                ch_artifact[mask] += amplitude * np.exp(-(t[mask] - start_time))

            artifact[ch if len(shape) > 1 else 0] = ch_artifact

        # Scale to desired power
        current_power = np.mean(artifact ** 2)
        if current_power > 0:
            artifact *= np.sqrt(power / current_power)

        return artifact

class StressTestSuite:
    """Comprehensive stress testing suite for EEG models"""

    def __init__(self, config: StressTestConfig, db_path: str):
        self.config = config
        self.db_path = Path(db_path)
        self.perturbation_engine = EEGPerturbationEngine()
        self.results: List[StressTestResult] = []

        self._init_database()

    def _init_database(self):
        """Initialize stress testing database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Stress test results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_results (
                    test_id TEXT PRIMARY KEY,
                    test_type TEXT NOT NULL,
                    perturbation_params TEXT, -- JSON
                    subject_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    baseline_metrics TEXT, -- JSON
                    stressed_metrics TEXT, -- JSON
                    performance_drop TEXT, -- JSON
                    passed_thresholds BOOLEAN,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    explanation_available BOOLEAN DEFAULT FALSE
                )
            """)

            # Stress test campaigns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    campaign_name TEXT NOT NULL,
                    config_params TEXT, -- JSON
                    total_tests INTEGER,
                    passed_tests INTEGER,
                    failed_tests INTEGER,
                    overall_pass_rate REAL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled'))
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stress_results_model ON stress_test_results (model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stress_results_type ON stress_test_results (test_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stress_results_passed ON stress_test_results (passed_thresholds)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stress_campaigns_status ON stress_test_campaigns (status)")

    def run_comprehensive_stress_test(self, model, test_data: List[Tuple[np.ndarray, str, int]],
                                    model_id: str, campaign_name: str = None) -> str:
        """Run comprehensive stress testing campaign"""
        campaign_id = str(uuid.uuid4())
        if not campaign_name:
            campaign_name = f"Stress_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize campaign
        total_tests = self._calculate_total_tests(test_data)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO stress_test_campaigns
                (campaign_id, campaign_name, config_params, total_tests, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (campaign_id, campaign_name, json.dumps(asdict(self.config)),
                  total_tests, datetime.now(), 'running'))

        try:
            logger.info(f"Starting stress test campaign {campaign_name} with {total_tests} tests")

            # Run all stress tests
            all_results = []

            # Noise stress tests
            all_results.extend(self._run_noise_stress_tests(model, test_data, model_id))

            # Electrode dropout tests
            all_results.extend(self._run_dropout_stress_tests(model, test_data, model_id))

            # Duration variation tests
            all_results.extend(self._run_duration_stress_tests(model, test_data, model_id))

            # Amplitude variation tests
            all_results.extend(self._run_amplitude_stress_tests(model, test_data, model_id))

            # Filter corruption tests
            all_results.extend(self._run_filter_stress_tests(model, test_data, model_id))

            # Sampling rate tests
            all_results.extend(self._run_sampling_stress_tests(model, test_data, model_id))

            # Store results and update campaign
            self._store_results(all_results)

            passed_tests = sum(1 for r in all_results if r.passed_thresholds)
            failed_tests = len(all_results) - passed_tests
            pass_rate = passed_tests / len(all_results) if all_results else 0

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE stress_test_campaigns
                    SET passed_tests = ?, failed_tests = ?, overall_pass_rate = ?,
                        completed_at = ?, status = ?
                    WHERE campaign_id = ?
                """, (passed_tests, failed_tests, pass_rate, datetime.now(), 'completed', campaign_id))

            logger.info(f"Stress test campaign completed: {passed_tests}/{len(all_results)} tests passed ({pass_rate:.1%})")

            return campaign_id

        except Exception as e:
            logger.error(f"Stress test campaign failed: {e}")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE stress_test_campaigns SET status = ?, completed_at = ?
                    WHERE campaign_id = ?
                """, ('failed', datetime.now(), campaign_id))

            raise

    def _calculate_total_tests(self, test_data: List) -> int:
        """Calculate total number of tests to be run"""
        n_subjects = len(test_data)

        total = 0
        total += len(self.config.noise_levels) * len(self.config.noise_types) * n_subjects
        total += len(self.config.dropout_rates) * len(self.config.dropout_patterns) * n_subjects
        total += len(self.config.duration_factors) * len(self.config.truncation_positions) * n_subjects
        total += len(self.config.amplitude_factors) * n_subjects
        total += len(self.config.filter_corruptions) * n_subjects
        total += len(self.config.sampling_rates) * n_subjects

        return total

    def _run_noise_stress_tests(self, model, test_data: List, model_id: str) -> List[StressTestResult]:
        """Run noise robustness tests"""
        results = []

        for eeg_data, subject_id, true_label in test_data:
            # Get baseline performance
            baseline_metrics = self._evaluate_model(model, eeg_data, true_label)

            for noise_type in self.config.noise_types:
                for snr_db in self.config.noise_levels:
                    start_time = datetime.now()

                    # Apply noise perturbation
                    noisy_data = self.perturbation_engine.add_noise(eeg_data, noise_type, snr_db)

                    # Evaluate stressed model
                    stressed_metrics = self._evaluate_model(model, noisy_data, true_label)

                    # Calculate performance drop
                    perf_drop = {
                        key: baseline_metrics[key] - stressed_metrics[key]
                        for key in baseline_metrics.keys()
                    }

                    # Check thresholds
                    passed = (
                        stressed_metrics['accuracy'] >= self.config.min_accuracy_threshold and
                        perf_drop['accuracy'] <= self.config.max_performance_drop
                    )

                    execution_time = (datetime.now() - start_time).total_seconds()

                    result = StressTestResult(
                        test_id=str(uuid.uuid4()),
                        test_type=f"noise_{noise_type}",
                        perturbation_params={'noise_type': noise_type, 'snr_db': snr_db},
                        subject_id=subject_id,
                        baseline_metrics=baseline_metrics,
                        stressed_metrics=stressed_metrics,
                        performance_drop=perf_drop,
                        passed_thresholds=passed,
                        execution_time=execution_time,
                        timestamp=datetime.now(),
                        model_id=model_id
                    )

                    results.append(result)

        return results

    def _run_dropout_stress_tests(self, model, test_data: List, model_id: str) -> List[StressTestResult]:
        """Run electrode dropout tests"""
        results = []

        for eeg_data, subject_id, true_label in test_data:
            baseline_metrics = self._evaluate_model(model, eeg_data, true_label)

            for dropout_rate in self.config.dropout_rates:
                for pattern in self.config.dropout_patterns:
                    start_time = datetime.now()

                    # Apply electrode dropout
                    dropout_data = self.perturbation_engine.dropout_electrodes(
                        eeg_data, dropout_rate, pattern
                    )

                    stressed_metrics = self._evaluate_model(model, dropout_data, true_label)

                    perf_drop = {
                        key: baseline_metrics[key] - stressed_metrics[key]
                        for key in baseline_metrics.keys()
                    }

                    passed = (
                        stressed_metrics['accuracy'] >= self.config.min_accuracy_threshold and
                        perf_drop['accuracy'] <= self.config.max_performance_drop
                    )

                    execution_time = (datetime.now() - start_time).total_seconds()

                    result = StressTestResult(
                        test_id=str(uuid.uuid4()),
                        test_type="electrode_dropout",
                        perturbation_params={'dropout_rate': dropout_rate, 'pattern': pattern},
                        subject_id=subject_id,
                        baseline_metrics=baseline_metrics,
                        stressed_metrics=stressed_metrics,
                        performance_drop=perf_drop,
                        passed_thresholds=passed,
                        execution_time=execution_time,
                        timestamp=datetime.now(),
                        model_id=model_id
                    )

                    results.append(result)

        return results

    def _run_duration_stress_tests(self, model, test_data: List, model_id: str) -> List[StressTestResult]:
        """Run duration variation tests"""
        results = []

        for eeg_data, subject_id, true_label in test_data:
            baseline_metrics = self._evaluate_model(model, eeg_data, true_label)

            for factor in self.config.duration_factors:
                for position in self.config.truncation_positions:
                    start_time = datetime.now()

                    # Apply duration variation
                    duration_data = self.perturbation_engine.vary_duration(
                        eeg_data, factor, position
                    )

                    stressed_metrics = self._evaluate_model(model, duration_data, true_label)

                    perf_drop = {
                        key: baseline_metrics[key] - stressed_metrics[key]
                        for key in baseline_metrics.keys()
                    }

                    passed = (
                        stressed_metrics['accuracy'] >= self.config.min_accuracy_threshold and
                        perf_drop['accuracy'] <= self.config.max_performance_drop
                    )

                    execution_time = (datetime.now() - start_time).total_seconds()

                    result = StressTestResult(
                        test_id=str(uuid.uuid4()),
                        test_type="duration_variation",
                        perturbation_params={'factor': factor, 'position': position},
                        subject_id=subject_id,
                        baseline_metrics=baseline_metrics,
                        stressed_metrics=stressed_metrics,
                        performance_drop=perf_drop,
                        passed_thresholds=passed,
                        execution_time=execution_time,
                        timestamp=datetime.now(),
                        model_id=model_id
                    )

                    results.append(result)

        return results

    def _run_amplitude_stress_tests(self, model, test_data: List, model_id: str) -> List[StressTestResult]:
        """Run amplitude scaling tests"""
        results = []

        for eeg_data, subject_id, true_label in test_data:
            baseline_metrics = self._evaluate_model(model, eeg_data, true_label)

            for factor in self.config.amplitude_factors:
                start_time = datetime.now()

                # Apply amplitude scaling
                scaled_data = self.perturbation_engine.scale_amplitude(eeg_data, factor)

                stressed_metrics = self._evaluate_model(model, scaled_data, true_label)

                perf_drop = {
                    key: baseline_metrics[key] - stressed_metrics[key]
                    for key in baseline_metrics.keys()
                }

                passed = (
                    stressed_metrics['accuracy'] >= self.config.min_accuracy_threshold and
                    perf_drop['accuracy'] <= self.config.max_performance_drop
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                result = StressTestResult(
                    test_id=str(uuid.uuid4()),
                    test_type="amplitude_scaling",
                    perturbation_params={'factor': factor},
                    subject_id=subject_id,
                    baseline_metrics=baseline_metrics,
                    stressed_metrics=stressed_metrics,
                    performance_drop=perf_drop,
                    passed_thresholds=passed,
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                    model_id=model_id
                )

                results.append(result)

        return results

    def _run_filter_stress_tests(self, model, test_data: List, model_id: str) -> List[StressTestResult]:
        """Run filter corruption tests"""
        results = []

        for eeg_data, subject_id, true_label in test_data:
            baseline_metrics = self._evaluate_model(model, eeg_data, true_label)

            for corruption in self.config.filter_corruptions:
                start_time = datetime.now()

                # Apply filter corruption
                filtered_data = self.perturbation_engine.apply_filter_corruption(eeg_data, corruption)

                stressed_metrics = self._evaluate_model(model, filtered_data, true_label)

                perf_drop = {
                    key: baseline_metrics[key] - stressed_metrics[key]
                    for key in baseline_metrics.keys()
                }

                passed = (
                    stressed_metrics['accuracy'] >= self.config.min_accuracy_threshold and
                    perf_drop['accuracy'] <= self.config.max_performance_drop
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                result = StressTestResult(
                    test_id=str(uuid.uuid4()),
                    test_type="filter_corruption",
                    perturbation_params=corruption,
                    subject_id=subject_id,
                    baseline_metrics=baseline_metrics,
                    stressed_metrics=stressed_metrics,
                    performance_drop=perf_drop,
                    passed_thresholds=passed,
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                    model_id=model_id
                )

                results.append(result)

        return results

    def _run_sampling_stress_tests(self, model, test_data: List, model_id: str) -> List[StressTestResult]:
        """Run sampling rate variation tests"""
        results = []

        for eeg_data, subject_id, true_label in test_data:
            baseline_metrics = self._evaluate_model(model, eeg_data, true_label)

            for target_rate in self.config.sampling_rates:
                start_time = datetime.now()

                # Apply resampling
                resampled_data = self.perturbation_engine.resample_data(eeg_data, target_rate)

                stressed_metrics = self._evaluate_model(model, resampled_data, true_label)

                perf_drop = {
                    key: baseline_metrics[key] - stressed_metrics[key]
                    for key in baseline_metrics.keys()
                }

                passed = (
                    stressed_metrics['accuracy'] >= self.config.min_accuracy_threshold and
                    perf_drop['accuracy'] <= self.config.max_performance_drop
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                result = StressTestResult(
                    test_id=str(uuid.uuid4()),
                    test_type="sampling_rate",
                    perturbation_params={'target_rate': target_rate},
                    subject_id=subject_id,
                    baseline_metrics=baseline_metrics,
                    stressed_metrics=stressed_metrics,
                    performance_drop=perf_drop,
                    passed_thresholds=passed,
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                    model_id=model_id
                )

                results.append(result)

        return results

    def _evaluate_model(self, model, eeg_data: np.ndarray, true_label: int) -> Dict[str, float]:
        """Evaluate model performance on given EEG data"""
        try:
            # This would interface with your actual model prediction pipeline
            # For now, using placeholder implementation

            # Predict using the model
            if hasattr(model, 'predict'):
                prediction = model.predict(eeg_data.reshape(1, -1))
                probabilities = model.predict_proba(eeg_data.reshape(1, -1)) if hasattr(model, 'predict_proba') else None
            else:
                # Handle PyTorch models or other frameworks
                prediction = np.random.choice([0, 1])  # Placeholder
                probabilities = np.array([[0.4, 0.6]])  # Placeholder

            predicted_label = prediction[0] if hasattr(prediction, '__len__') else prediction
            confidence = probabilities.max() if probabilities is not None else 0.8

            # Calculate metrics
            accuracy = 1.0 if predicted_label == true_label else 0.0
            precision = accuracy  # For single sample
            recall = accuracy
            f1 = accuracy

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confidence': confidence,
                'predicted_label': int(predicted_label),
                'true_label': int(true_label)
            }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'confidence': 0.0,
                'predicted_label': -1,
                'true_label': int(true_label)
            }

    def _store_results(self, results: List[StressTestResult]):
        """Store stress test results in database"""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO stress_test_results (
                        test_id, test_type, perturbation_params, subject_id, model_id,
                        baseline_metrics, stressed_metrics, performance_drop,
                        passed_thresholds, execution_time, timestamp, explanation_available
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_id,
                    result.test_type,
                    json.dumps(result.perturbation_params),
                    result.subject_id,
                    result.model_id,
                    json.dumps(result.baseline_metrics),
                    json.dumps(result.stressed_metrics),
                    json.dumps(result.performance_drop),
                    result.passed_thresholds,
                    result.execution_time,
                    result.timestamp,
                    result.explanation_available
                ))

        self.results.extend(results)

    def get_campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of stress test campaign"""
        with sqlite3.connect(self.db_path) as conn:
            # Get campaign info
            cursor = conn.execute("""
                SELECT * FROM stress_test_campaigns WHERE campaign_id = ?
            """, (campaign_id,))

            campaign_row = cursor.fetchone()
            if not campaign_row:
                return {}

            # Get test results
            cursor = conn.execute("""
                SELECT test_type, passed_thresholds, performance_drop, stressed_metrics
                FROM stress_test_results WHERE test_id IN (
                    SELECT test_id FROM stress_test_results
                    WHERE timestamp BETWEEN ? AND ?
                )
            """, (campaign_row[6], campaign_row[7]))  # started_at, completed_at

            results = cursor.fetchall()

            # Analyze results by test type
            summary = {
                'campaign_id': campaign_id,
                'campaign_name': campaign_row[1],
                'total_tests': campaign_row[3],
                'passed_tests': campaign_row[4],
                'failed_tests': campaign_row[5],
                'overall_pass_rate': campaign_row[6],
                'by_test_type': {}
            }

            # Group by test type
            test_types = {}
            for row in results:
                test_type, passed, perf_drop_json, metrics_json = row

                if test_type not in test_types:
                    test_types[test_type] = {
                        'total': 0,
                        'passed': 0,
                        'performance_drops': [],
                        'accuracies': []
                    }

                test_types[test_type]['total'] += 1
                if passed:
                    test_types[test_type]['passed'] += 1

                perf_drop = json.loads(perf_drop_json)
                metrics = json.loads(metrics_json)

                test_types[test_type]['performance_drops'].append(perf_drop['accuracy'])
                test_types[test_type]['accuracies'].append(metrics['accuracy'])

            # Calculate statistics for each test type
            for test_type, data in test_types.items():
                summary['by_test_type'][test_type] = {
                    'pass_rate': data['passed'] / data['total'] if data['total'] > 0 else 0,
                    'total_tests': data['total'],
                    'passed_tests': data['passed'],
                    'mean_performance_drop': np.mean(data['performance_drops']),
                    'max_performance_drop': np.max(data['performance_drops']),
                    'mean_accuracy': np.mean(data['accuracies']),
                    'min_accuracy': np.min(data['accuracies'])
                }

            return summary