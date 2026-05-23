"""
Cross-Device Validation System
Tests model performance across different EEG acquisition systems and manufacturers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import json
import sqlite3
from datetime import datetime
import uuid
from scipy import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class DeviceProfile:
    """EEG device characteristics and specifications"""
    device_id: str
    manufacturer: str
    model_name: str
    sampling_rate: float
    n_channels: int
    channel_layout: str  # "10-20", "10-10", "custom"
    impedance_range: Tuple[float, float]  # kOhms
    noise_floor: float  # µV RMS
    frequency_response: Dict[str, float]  # bandwidth specifications
    adc_resolution: int  # bits
    common_mode_rejection: float  # dB
    input_range: Tuple[float, float]  # µV
    electrode_type: str  # "wet", "dry", "semi-dry"
    amplifier_type: str  # "active", "passive"
    filter_characteristics: Dict[str, Any]
    typical_artifacts: List[str]
    calibration_factors: Dict[str, float]

@dataclass
class CrossDeviceTestResult:
    """Results from cross-device validation test"""
    test_id: str
    source_device: str
    target_device: str
    subject_id: str
    model_id: str
    original_metrics: Dict[str, float]
    adapted_metrics: Dict[str, float]
    performance_degradation: Dict[str, float]
    adaptation_method: str
    adaptation_successful: bool
    execution_time: float
    timestamp: datetime

class DeviceSignatureAnalyzer:
    """Analyzes device-specific signal characteristics"""

    def __init__(self):
        self.known_devices = {}

    def register_device(self, profile: DeviceProfile):
        """Register a device profile"""
        self.known_devices[profile.device_id] = profile

    def analyze_device_signature(self, eeg_data: np.ndarray,
                                device_id: str = None) -> Dict[str, float]:
        """Extract device-specific signal characteristics"""
        n_channels, n_samples = eeg_data.shape

        # Spectral characteristics
        freqs, psd = signal.welch(eeg_data, fs=250, axis=1)

        # Noise floor estimation (high frequency content)
        high_freq_mask = freqs > 100
        noise_floor = np.median(psd[:, high_freq_mask])

        # Line noise detection (50/60 Hz)
        line_noise_50 = self._detect_line_noise(eeg_data, 50)
        line_noise_60 = self._detect_line_noise(eeg_data, 60)

        # Dynamic range
        dynamic_range = np.percentile(eeg_data, 99) - np.percentile(eeg_data, 1)

        # Channel correlation (measure of crosstalk)
        channel_corr = np.corrcoef(eeg_data)
        off_diagonal_corr = np.mean(np.abs(channel_corr[np.triu_indices_from(channel_corr, k=1)]))

        # Amplitude distribution characteristics
        amplitude_skewness = np.mean([self._calculate_skewness(ch) for ch in eeg_data])
        amplitude_kurtosis = np.mean([self._calculate_kurtosis(ch) for ch in eeg_data])

        # Frequency response estimation
        low_freq_power = np.mean(psd[:, (freqs >= 0.5) & (freqs <= 4)])
        alpha_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
        beta_power = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)])
        gamma_power = np.mean(psd[:, (freqs >= 30) & (freqs <= 100)])

        signature = {
            'noise_floor': float(noise_floor),
            'line_noise_50hz': float(line_noise_50),
            'line_noise_60hz': float(line_noise_60),
            'dynamic_range': float(dynamic_range),
            'channel_crosstalk': float(off_diagonal_corr),
            'amplitude_skewness': float(amplitude_skewness),
            'amplitude_kurtosis': float(amplitude_kurtosis),
            'low_freq_power': float(low_freq_power),
            'alpha_power': float(alpha_power),
            'beta_power': float(beta_power),
            'gamma_power': float(gamma_power),
            'spectral_centroid': float(np.mean([np.average(freqs, weights=psd[ch]) for ch in range(n_channels)])),
            'spectral_rolloff': float(np.mean([self._spectral_rolloff(freqs, psd[ch]) for ch in range(n_channels)]))
        }

        return signature

    def _detect_line_noise(self, eeg_data: np.ndarray, freq: float,
                          fs: float = 250) -> float:
        """Detect line noise at specific frequency"""
        freqs, psd = signal.welch(eeg_data, fs=fs, axis=1)
        freq_idx = np.argmin(np.abs(freqs - freq))

        # Get power in narrow band around target frequency
        band_width = 2  # Hz
        band_mask = (freqs >= freq - band_width/2) & (freqs <= freq + band_width/2)
        line_power = np.mean(psd[:, band_mask])

        # Compare to surrounding frequencies
        surround_mask = ((freqs >= freq - 10) & (freqs <= freq - 5)) | \
                       ((freqs >= freq + 5) & (freqs <= freq + 10))
        surround_power = np.mean(psd[:, surround_mask])

        return line_power / (surround_power + 1e-10)

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray,
                         threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency"""
        cumsum_psd = np.cumsum(psd)
        total_power = cumsum_psd[-1]

        rolloff_idx = np.where(cumsum_psd >= threshold * total_power)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]

class DeviceAdaptationEngine:
    """Adapts EEG signals between different device characteristics"""

    def __init__(self):
        self.adaptation_methods = {
            'spectral_matching': self._spectral_matching,
            'amplitude_scaling': self._amplitude_scaling,
            'noise_matching': self._noise_matching,
            'filter_matching': self._filter_matching,
            'combined_adaptation': self._combined_adaptation
        }

    def adapt_signal(self, eeg_data: np.ndarray,
                    source_profile: DeviceProfile,
                    target_profile: DeviceProfile,
                    method: str = 'combined_adaptation') -> np.ndarray:
        """Adapt EEG signal from source device characteristics to target device"""

        if method not in self.adaptation_methods:
            raise ValueError(f"Unknown adaptation method: {method}")

        return self.adaptation_methods[method](eeg_data, source_profile, target_profile)

    def _spectral_matching(self, eeg_data: np.ndarray,
                          source_profile: DeviceProfile,
                          target_profile: DeviceProfile) -> np.ndarray:
        """Match spectral characteristics between devices"""
        # Get frequency responses
        source_response = source_profile.frequency_response
        target_response = target_profile.frequency_response

        # Design compensation filter
        fs = source_profile.sampling_rate
        nyquist = fs / 2

        # Simple high/low-pass adaptation based on bandwidth differences
        if target_response.get('high_cutoff', 100) < source_response.get('high_cutoff', 100):
            # Target has lower bandwidth - apply low-pass filter
            cutoff = target_response.get('high_cutoff', 100)
            sos = signal.butter(4, cutoff / nyquist, btype='low', output='sos')
            adapted_data = signal.sosfilt(sos, eeg_data, axis=1)
        else:
            adapted_data = eeg_data.copy()

        if target_response.get('low_cutoff', 0.1) > source_response.get('low_cutoff', 0.1):
            # Target has higher high-pass cutoff
            cutoff = target_response.get('low_cutoff', 0.1)
            sos = signal.butter(4, cutoff / nyquist, btype='high', output='sos')
            adapted_data = signal.sosfilt(sos, adapted_data, axis=1)

        return adapted_data

    def _amplitude_scaling(self, eeg_data: np.ndarray,
                          source_profile: DeviceProfile,
                          target_profile: DeviceProfile) -> np.ndarray:
        """Scale amplitude based on device input ranges and gains"""
        source_range = source_profile.input_range[1] - source_profile.input_range[0]
        target_range = target_profile.input_range[1] - target_profile.input_range[0]

        # Apply calibration factors if available
        source_cal = source_profile.calibration_factors.get('amplitude', 1.0)
        target_cal = target_profile.calibration_factors.get('amplitude', 1.0)

        scaling_factor = (target_range / source_range) * (target_cal / source_cal)

        return eeg_data * scaling_factor

    def _noise_matching(self, eeg_data: np.ndarray,
                       source_profile: DeviceProfile,
                       target_profile: DeviceProfile) -> np.ndarray:
        """Add or remove noise to match target device noise floor"""
        source_noise = source_profile.noise_floor
        target_noise = target_profile.noise_floor

        if target_noise > source_noise:
            # Add noise to match target
            noise_to_add = np.sqrt(target_noise**2 - source_noise**2)
            noise = np.random.normal(0, noise_to_add, eeg_data.shape)
            return eeg_data + noise
        else:
            # Apply noise reduction (simple smoothing)
            # In practice, this would be more sophisticated
            smoothing_factor = source_noise / target_noise
            if smoothing_factor > 1:
                kernel_size = min(int(smoothing_factor), 5)
                kernel = np.ones(kernel_size) / kernel_size
                adapted_data = np.array([np.convolve(ch, kernel, mode='same')
                                       for ch in eeg_data])
                return adapted_data

        return eeg_data

    def _filter_matching(self, eeg_data: np.ndarray,
                        source_profile: DeviceProfile,
                        target_profile: DeviceProfile) -> np.ndarray:
        """Apply device-specific filtering characteristics"""
        adapted_data = eeg_data.copy()

        # Apply target device's typical filtering
        target_filters = target_profile.filter_characteristics
        fs = source_profile.sampling_rate
        nyquist = fs / 2

        for filter_name, filter_params in target_filters.items():
            if filter_name == 'notch_50hz' and filter_params.get('enabled', False):
                # Apply 50Hz notch filter
                sos = signal.iirnotch(50 / nyquist, 30, output='sos')
                adapted_data = signal.sosfilt(sos, adapted_data, axis=1)

            elif filter_name == 'notch_60hz' and filter_params.get('enabled', False):
                # Apply 60Hz notch filter
                sos = signal.iirnotch(60 / nyquist, 30, output='sos')
                adapted_data = signal.sosfilt(sos, adapted_data, axis=1)

            elif filter_name == 'highpass' and 'cutoff' in filter_params:
                cutoff = filter_params['cutoff'] / nyquist
                sos = signal.butter(4, cutoff, btype='high', output='sos')
                adapted_data = signal.sosfilt(sos, adapted_data, axis=1)

        return adapted_data

    def _combined_adaptation(self, eeg_data: np.ndarray,
                           source_profile: DeviceProfile,
                           target_profile: DeviceProfile) -> np.ndarray:
        """Apply combined adaptation strategy"""
        adapted_data = eeg_data.copy()

        # Apply adaptations in sequence
        adapted_data = self._spectral_matching(adapted_data, source_profile, target_profile)
        adapted_data = self._amplitude_scaling(adapted_data, source_profile, target_profile)
        adapted_data = self._noise_matching(adapted_data, source_profile, target_profile)
        adapted_data = self._filter_matching(adapted_data, source_profile, target_profile)

        return adapted_data

class CrossDeviceValidator:
    """Cross-device validation testing system"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.signature_analyzer = DeviceSignatureAnalyzer()
        self.adaptation_engine = DeviceAdaptationEngine()
        self.device_profiles = {}

        self._init_database()
        self._load_standard_devices()

    def _init_database(self):
        """Initialize cross-device validation database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Device profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_profiles (
                    device_id TEXT PRIMARY KEY,
                    manufacturer TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    specifications TEXT, -- JSON
                    signature_characteristics TEXT, -- JSON
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cross-device test results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_device_results (
                    test_id TEXT PRIMARY KEY,
                    source_device TEXT NOT NULL,
                    target_device TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    original_metrics TEXT, -- JSON
                    adapted_metrics TEXT, -- JSON
                    performance_degradation TEXT, -- JSON
                    adaptation_method TEXT,
                    adaptation_successful BOOLEAN,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Device compatibility matrix
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_compatibility (
                    compatibility_id TEXT PRIMARY KEY,
                    device_a TEXT NOT NULL,
                    device_b TEXT NOT NULL,
                    compatibility_score REAL,
                    adaptation_difficulty TEXT, -- 'easy', 'medium', 'hard', 'impossible'
                    recommended_method TEXT,
                    validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_a) REFERENCES device_profiles (device_id),
                    FOREIGN KEY (device_b) REFERENCES device_profiles (device_id)
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_device_source ON cross_device_results (source_device)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_device_target ON cross_device_results (target_device)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_device_model ON cross_device_results (model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_compatibility_devices ON device_compatibility (device_a, device_b)")

    def _load_standard_devices(self):
        """Load standard EEG device profiles"""
        standard_devices = [
            # BioSemi ActiveTwo
            DeviceProfile(
                device_id="biosemi_activetwo_64",
                manufacturer="BioSemi",
                model_name="ActiveTwo 64-channel",
                sampling_rate=2048.0,
                n_channels=64,
                channel_layout="10-20",
                impedance_range=(0, 50),  # kOhms
                noise_floor=0.8,  # µV RMS
                frequency_response={"low_cutoff": 0.016, "high_cutoff": 417},
                adc_resolution=24,
                common_mode_rejection=110,  # dB
                input_range=(-262144, 262143),  # µV
                electrode_type="wet",
                amplifier_type="active",
                filter_characteristics={
                    "anti_aliasing": {"enabled": True, "cutoff": 417},
                    "notch_50hz": {"enabled": False},
                    "notch_60hz": {"enabled": False}
                },
                typical_artifacts=["eye_movement", "muscle", "movement"],
                calibration_factors={"amplitude": 1.0, "offset": 0.0}
            ),

            # EGI HydroCel
            DeviceProfile(
                device_id="egi_hydrocel_128",
                manufacturer="EGI",
                model_name="HydroCel GSN 128",
                sampling_rate=1000.0,
                n_channels=128,
                channel_layout="10-10",
                impedance_range=(0, 100),
                noise_floor=1.2,
                frequency_response={"low_cutoff": 0.1, "high_cutoff": 400},
                adc_resolution=16,
                common_mode_rejection=90,
                input_range=(-200000, 200000),
                electrode_type="wet",
                amplifier_type="active",
                filter_characteristics={
                    "highpass": {"cutoff": 0.1},
                    "lowpass": {"cutoff": 400},
                    "notch_60hz": {"enabled": True}
                },
                typical_artifacts=["bridge", "salt_bridge", "movement"],
                calibration_factors={"amplitude": 0.976, "offset": 0.0}
            ),

            # Emotiv EPOC X
            DeviceProfile(
                device_id="emotiv_epocx_14",
                manufacturer="Emotiv",
                model_name="EPOC X",
                sampling_rate=256.0,
                n_channels=14,
                channel_layout="10-20",
                impedance_range=(0, 10),
                noise_floor=2.5,
                frequency_response={"low_cutoff": 0.2, "high_cutoff": 45},
                adc_resolution=16,
                common_mode_rejection=70,
                input_range=(-8192, 8191),
                electrode_type="semi-dry",
                amplifier_type="active",
                filter_characteristics={
                    "highpass": {"cutoff": 0.2},
                    "lowpass": {"cutoff": 45},
                    "notch_50hz": {"enabled": True},
                    "notch_60hz": {"enabled": True}
                },
                typical_artifacts=["contact_impedance", "motion", "wireless_interference"],
                calibration_factors={"amplitude": 1.024, "offset": 4096}
            ),

            # g.tec g.HIamp
            DeviceProfile(
                device_id="gtec_ghiamp_256",
                manufacturer="g.tec",
                model_name="g.HIamp",
                sampling_rate=38400.0,
                n_channels=256,
                channel_layout="custom",
                impedance_range=(0, 1000),
                noise_floor=0.3,
                frequency_response={"low_cutoff": 0.01, "high_cutoff": 8000},
                adc_resolution=24,
                common_mode_rejection=120,
                input_range=(-250000, 250000),
                electrode_type="wet",
                amplifier_type="active",
                filter_characteristics={
                    "anti_aliasing": {"enabled": True, "cutoff": 8000},
                    "configurable_filters": True
                },
                typical_artifacts=["electrode_polarization", "amplifier_saturation"],
                calibration_factors={"amplitude": 1.0, "offset": 0.0}
            )
        ]

        # Register standard devices
        for device in standard_devices:
            self.register_device(device)

    def register_device(self, profile: DeviceProfile):
        """Register a device profile"""
        self.device_profiles[profile.device_id] = profile
        self.signature_analyzer.register_device(profile)

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO device_profiles
                (device_id, manufacturer, model_name, specifications, signature_characteristics)
                VALUES (?, ?, ?, ?, ?)
            """, (
                profile.device_id,
                profile.manufacturer,
                profile.model_name,
                json.dumps(asdict(profile)),
                json.dumps({})  # Will be populated during validation
            ))

    def run_cross_device_validation(self, model, test_data: List[Tuple[np.ndarray, str, int, str]],
                                  model_id: str, adaptation_methods: List[str] = None) -> str:
        """Run comprehensive cross-device validation"""

        if adaptation_methods is None:
            adaptation_methods = ['spectral_matching', 'amplitude_scaling', 'combined_adaptation']

        campaign_id = str(uuid.uuid4())
        results = []

        logger.info(f"Starting cross-device validation campaign {campaign_id}")

        # Get all device pairs
        device_ids = list(self.device_profiles.keys())
        device_pairs = [(src, tgt) for src in device_ids for tgt in device_ids if src != tgt]

        for source_device_id, target_device_id in device_pairs:
            source_profile = self.device_profiles[source_device_id]
            target_profile = self.device_profiles[target_device_id]

            logger.info(f"Testing {source_profile.manufacturer} {source_profile.model_name} → "
                       f"{target_profile.manufacturer} {target_profile.model_name}")

            for eeg_data, subject_id, true_label, original_device in test_data:
                # Skip if original device doesn't match source
                if original_device != source_device_id:
                    continue

                for method in adaptation_methods:
                    start_time = datetime.now()

                    try:
                        # Get baseline performance (original device)
                        original_metrics = self._evaluate_model(model, eeg_data, true_label)

                        # Adapt signal to target device
                        adapted_data = self.adaptation_engine.adapt_signal(
                            eeg_data, source_profile, target_profile, method
                        )

                        # Evaluate on adapted signal
                        adapted_metrics = self._evaluate_model(model, adapted_data, true_label)

                        # Calculate performance degradation
                        degradation = {
                            key: original_metrics[key] - adapted_metrics[key]
                            for key in original_metrics.keys()
                            if isinstance(original_metrics[key], (int, float))
                        }

                        # Check if adaptation was successful
                        adaptation_successful = (
                            adapted_metrics['accuracy'] >= 0.5 and  # Better than chance
                            degradation['accuracy'] < 0.3  # Less than 30% drop
                        )

                        execution_time = (datetime.now() - start_time).total_seconds()

                        result = CrossDeviceTestResult(
                            test_id=str(uuid.uuid4()),
                            source_device=source_device_id,
                            target_device=target_device_id,
                            subject_id=subject_id,
                            model_id=model_id,
                            original_metrics=original_metrics,
                            adapted_metrics=adapted_metrics,
                            performance_degradation=degradation,
                            adaptation_method=method,
                            adaptation_successful=adaptation_successful,
                            execution_time=execution_time,
                            timestamp=datetime.now()
                        )

                        results.append(result)

                    except Exception as e:
                        logger.error(f"Cross-device adaptation failed: {e}")

        # Store results
        self._store_results(results)

        # Update compatibility matrix
        self._update_compatibility_matrix(results)

        logger.info(f"Cross-device validation completed: {len(results)} tests")

        return campaign_id

    def _evaluate_model(self, model, eeg_data: np.ndarray, true_label: int) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # This would interface with your actual model prediction pipeline
            if hasattr(model, 'predict'):
                prediction = model.predict(eeg_data.reshape(1, -1))
                probabilities = model.predict_proba(eeg_data.reshape(1, -1)) if hasattr(model, 'predict_proba') else None
            else:
                # Placeholder for other model types
                prediction = np.random.choice([0, 1])
                probabilities = np.array([[0.4, 0.6]])

            predicted_label = prediction[0] if hasattr(prediction, '__len__') else prediction
            confidence = probabilities.max() if probabilities is not None else 0.8

            accuracy = 1.0 if predicted_label == true_label else 0.0

            return {
                'accuracy': accuracy,
                'confidence': confidence,
                'predicted_label': int(predicted_label),
                'true_label': int(true_label)
            }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'confidence': 0.0,
                'predicted_label': -1,
                'true_label': int(true_label)
            }

    def _store_results(self, results: List[CrossDeviceTestResult]):
        """Store cross-device validation results"""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO cross_device_results (
                        test_id, source_device, target_device, subject_id, model_id,
                        original_metrics, adapted_metrics, performance_degradation,
                        adaptation_method, adaptation_successful, execution_time, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_id,
                    result.source_device,
                    result.target_device,
                    result.subject_id,
                    result.model_id,
                    json.dumps(result.original_metrics),
                    json.dumps(result.adapted_metrics),
                    json.dumps(result.performance_degradation),
                    result.adaptation_method,
                    result.adaptation_successful,
                    result.execution_time,
                    result.timestamp
                ))

    def _update_compatibility_matrix(self, results: List[CrossDeviceTestResult]):
        """Update device compatibility matrix based on results"""
        # Group results by device pair
        device_pairs = {}
        for result in results:
            pair_key = (result.source_device, result.target_device)
            if pair_key not in device_pairs:
                device_pairs[pair_key] = []
            device_pairs[pair_key].append(result)

        with sqlite3.connect(self.db_path) as conn:
            for (source_device, target_device), pair_results in device_pairs.items():
                # Calculate compatibility metrics
                successful_adaptations = sum(1 for r in pair_results if r.adaptation_successful)
                total_tests = len(pair_results)
                compatibility_score = successful_adaptations / total_tests if total_tests > 0 else 0

                # Determine adaptation difficulty
                avg_degradation = np.mean([
                    r.performance_degradation.get('accuracy', 1.0)
                    for r in pair_results
                ])

                if compatibility_score > 0.8 and avg_degradation < 0.1:
                    difficulty = 'easy'
                elif compatibility_score > 0.6 and avg_degradation < 0.2:
                    difficulty = 'medium'
                elif compatibility_score > 0.3:
                    difficulty = 'hard'
                else:
                    difficulty = 'impossible'

                # Find best adaptation method
                method_performance = {}
                for result in pair_results:
                    method = result.adaptation_method
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(result.adapted_metrics['accuracy'])

                best_method = max(method_performance.keys(),
                                key=lambda m: np.mean(method_performance[m]))

                # Store compatibility info
                compatibility_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT OR REPLACE INTO device_compatibility (
                        compatibility_id, device_a, device_b, compatibility_score,
                        adaptation_difficulty, recommended_method, validation_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    compatibility_id,
                    source_device,
                    target_device,
                    compatibility_score,
                    difficulty,
                    best_method,
                    datetime.now()
                ))

    def get_compatibility_report(self, device_a: str = None, device_b: str = None) -> Dict[str, Any]:
        """Get compatibility report between devices"""
        with sqlite3.connect(self.db_path) as conn:
            if device_a and device_b:
                # Specific device pair
                cursor = conn.execute("""
                    SELECT * FROM device_compatibility
                    WHERE (device_a = ? AND device_b = ?) OR (device_a = ? AND device_b = ?)
                """, (device_a, device_b, device_b, device_a))

                result = cursor.fetchone()
                if result:
                    return {
                        'device_a': result[1],
                        'device_b': result[2],
                        'compatibility_score': result[3],
                        'adaptation_difficulty': result[4],
                        'recommended_method': result[5],
                        'validation_date': result[6]
                    }
                else:
                    return {'error': 'No compatibility data found for this device pair'}

            else:
                # Full compatibility matrix
                cursor = conn.execute("""
                    SELECT device_a, device_b, compatibility_score, adaptation_difficulty
                    FROM device_compatibility
                    ORDER BY compatibility_score DESC
                """)

                matrix = {}
                for row in cursor.fetchall():
                    device_a, device_b, score, difficulty = row
                    if device_a not in matrix:
                        matrix[device_a] = {}
                    matrix[device_a][device_b] = {
                        'score': score,
                        'difficulty': difficulty
                    }

                return {'compatibility_matrix': matrix}

    def generate_device_recommendations(self, target_accuracy: float = 0.8) -> Dict[str, Any]:
        """Generate device recommendations based on compatibility"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT device_a, device_b, compatibility_score, adaptation_difficulty,
                       recommended_method
                FROM device_compatibility
                WHERE compatibility_score >= ?
                ORDER BY compatibility_score DESC
            """, (target_accuracy,))

            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'source_device': row[0],
                    'target_device': row[1],
                    'compatibility_score': row[2],
                    'adaptation_difficulty': row[3],
                    'recommended_method': row[4]
                })

            return {'recommendations': recommendations}