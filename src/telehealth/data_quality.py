"""
Phase X-A: Real-Time Data Quality Monitoring for Remote Sessions
Advanced quality control and validation for distributed EEG neurofeedback
"""

import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from enum import Enum
import statistics
from scipy import signal
from scipy.stats import zscore
import warnings

from .device_integration import HomeEEGDeviceManager, SignalQuality
from .telehealth_workflows import TelehealthWorkflowManager, RemoteSessionStatus
from .remote_monitoring import RemotePatientMonitor, AlertSeverity

class QualityMetricType(Enum):
    """Types of quality metrics monitored"""
    SIGNAL_QUALITY = "signal_quality"
    ARTIFACT_DETECTION = "artifact_detection"
    IMPEDANCE_MONITORING = "impedance_monitoring"
    CONNECTIVITY_STABILITY = "connectivity_stability"
    DATA_COMPLETENESS = "data_completeness"
    PROTOCOL_ADHERENCE = "protocol_adherence"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    CROSS_SESSION_RELIABILITY = "cross_session_reliability"

class QualityThresholdLevel(Enum):
    """Quality threshold severity levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

class ArtifactType(Enum):
    """Types of EEG artifacts detected"""
    EYE_BLINK = "eye_blink"
    EYE_MOVEMENT = "eye_movement"
    MUSCLE_TENSION = "muscle_tension"
    ELECTRODE_POP = "electrode_pop"
    LINE_NOISE = "line_noise"
    MOVEMENT_ARTIFACT = "movement_artifact"
    CARDIAC_ARTIFACT = "cardiac_artifact"
    RESPIRATION_ARTIFACT = "respiration_artifact"
    ENVIRONMENTAL_NOISE = "environmental_noise"

@dataclass
class QualityThreshold:
    """Quality threshold definition"""
    metric_type: QualityMetricType
    excellent_min: float
    good_min: float
    acceptable_min: float
    poor_max: float
    units: str
    description: str

@dataclass
class ArtifactDetection:
    """Artifact detection result"""
    artifact_type: ArtifactType
    timestamp: datetime
    duration_seconds: float
    severity: float
    affected_channels: List[str]
    confidence: float
    auto_corrected: bool = False
    correction_applied: Optional[str] = None

@dataclass
class SignalQualityAssessment:
    """Comprehensive signal quality assessment"""
    session_id: str
    participant_id: str
    timestamp: datetime
    overall_quality_score: float
    channel_quality_scores: Dict[str, float]
    impedance_values: Dict[str, float]
    signal_to_noise_ratio: float
    frequency_analysis: Dict[str, Any]
    artifact_summary: Dict[ArtifactType, int]
    data_completeness_percentage: float
    quality_trend: List[float]
    recommendations: List[str]
    threshold_violations: List[Dict[str, Any]]

@dataclass
class RealTimeQualityMetrics:
    """Real-time quality metrics collection"""
    session_id: str
    collection_start: datetime
    latest_update: datetime
    sampling_rate: float
    buffer_size_samples: int
    quality_assessments: List[SignalQualityAssessment]
    cumulative_artifacts: Dict[ArtifactType, int]
    session_statistics: Dict[str, float]
    quality_alerts_triggered: int
    intervention_events: List[Dict[str, Any]]

class RemoteDataQualityMonitor:
    """
    Real-time data quality monitoring system for remote EEG sessions
    Provides continuous quality assessment and automated interventions
    """

    def __init__(self, config_path: str = None):
        """Initialize remote data quality monitor"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Core services
        self.device_manager = HomeEEGDeviceManager()
        self.workflow_manager = TelehealthWorkflowManager()
        self.patient_monitor = RemotePatientMonitor()

        # Quality monitoring state
        self.active_monitoring: Dict[str, RealTimeQualityMetrics] = {}
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.artifact_detectors = self._initialize_artifact_detectors()
        self.quality_buffers: Dict[str, Dict[str, np.ndarray]] = {}

        # Real-time processing
        self.processing_queues: Dict[str, asyncio.Queue] = {}
        self.quality_processors: Dict[str, asyncio.Task] = {}

        # Machine learning models for quality assessment
        self.quality_models = self._initialize_quality_models()

        # Performance metrics
        self.monitoring_statistics = {
            'sessions_monitored': 0,
            'quality_interventions': 0,
            'artifacts_detected': 0,
            'average_processing_latency': 0.0
        }

        self.logger.info("Remote Data Quality Monitor initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load data quality monitoring configuration"""
        default_config = {
            'quality_assessment_interval_seconds': 5,
            'buffer_duration_seconds': 30,
            'artifact_detection_enabled': True,
            'real_time_feedback_enabled': True,
            'quality_intervention_threshold': 0.6,
            'automatic_correction_enabled': True,
            'machine_learning_enabled': True,
            'cross_session_validation': True,
            'quality_thresholds': {
                'signal_quality': {
                    'excellent': 0.9,
                    'good': 0.75,
                    'acceptable': 0.6,
                    'poor': 0.4
                },
                'impedance_kohm': {
                    'excellent': 5.0,
                    'good': 15.0,
                    'acceptable': 30.0,
                    'poor': 50.0
                },
                'signal_to_noise_db': {
                    'excellent': 20.0,
                    'good': 15.0,
                    'acceptable': 10.0,
                    'poor': 5.0
                },
                'data_completeness': {
                    'excellent': 0.98,
                    'good': 0.95,
                    'acceptable': 0.9,
                    'poor': 0.8
                }
            },
            'artifact_thresholds': {
                'eye_blink_threshold': 100.0,  # microvolts
                'muscle_threshold': 50.0,
                'line_noise_threshold': 10.0,
                'movement_threshold': 75.0
            },
            'frequency_analysis': {
                'bands': {
                    'delta': [1, 4],
                    'theta': [4, 8],
                    'alpha': [8, 12],
                    'beta': [12, 30],
                    'gamma': [30, 40]
                },
                'relative_power_analysis': True,
                'peak_frequency_detection': True
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup quality monitoring logging"""
        logger = logging.getLogger('data_quality_monitor')
        logger.setLevel(logging.INFO)

        log_dir = Path('logs/data_quality')
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / f'quality_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _initialize_quality_thresholds(self) -> Dict[QualityMetricType, QualityThreshold]:
        """Initialize quality threshold definitions"""
        thresholds = {}

        # Signal quality thresholds
        signal_config = self.config['quality_thresholds']['signal_quality']
        thresholds[QualityMetricType.SIGNAL_QUALITY] = QualityThreshold(
            metric_type=QualityMetricType.SIGNAL_QUALITY,
            excellent_min=signal_config['excellent'],
            good_min=signal_config['good'],
            acceptable_min=signal_config['acceptable'],
            poor_max=signal_config['poor'],
            units='normalized_score',
            description='Overall signal quality score (0.0-1.0)'
        )

        # Impedance thresholds
        impedance_config = self.config['quality_thresholds']['impedance_kohm']
        thresholds[QualityMetricType.IMPEDANCE_MONITORING] = QualityThreshold(
            metric_type=QualityMetricType.IMPEDANCE_MONITORING,
            excellent_min=0.0,
            good_min=0.0,
            acceptable_min=0.0,
            poor_max=impedance_config['poor'],
            units='kiloohms',
            description='Electrode impedance levels'
        )

        # Data completeness thresholds
        completeness_config = self.config['quality_thresholds']['data_completeness']
        thresholds[QualityMetricType.DATA_COMPLETENESS] = QualityThreshold(
            metric_type=QualityMetricType.DATA_COMPLETENESS,
            excellent_min=completeness_config['excellent'],
            good_min=completeness_config['good'],
            acceptable_min=completeness_config['acceptable'],
            poor_max=completeness_config['poor'],
            units='percentage',
            description='Percentage of expected data received'
        )

        return thresholds

    def _initialize_artifact_detectors(self) -> Dict[ArtifactType, Callable]:
        """Initialize artifact detection algorithms"""
        detectors = {}

        # Eye blink detector
        detectors[ArtifactType.EYE_BLINK] = self._detect_eye_blinks

        # Muscle artifact detector
        detectors[ArtifactType.MUSCLE_TENSION] = self._detect_muscle_artifacts

        # Line noise detector
        detectors[ArtifactType.LINE_NOISE] = self._detect_line_noise

        # Movement artifact detector
        detectors[ArtifactType.MOVEMENT_ARTIFACT] = self._detect_movement_artifacts

        # Electrode pop detector
        detectors[ArtifactType.ELECTRODE_POP] = self._detect_electrode_pops

        return detectors

    def _initialize_quality_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for quality assessment"""
        # In production, load trained ML models
        # For now, return placeholder models
        return {
            'signal_quality_classifier': None,
            'artifact_detector': None,
            'quality_predictor': None
        }

    async def start_session_monitoring(self,
                                     session_id: str,
                                     participant_id: str,
                                     sampling_rate: float = 256.0) -> bool:
        """Start real-time quality monitoring for session"""
        try:
            if session_id in self.active_monitoring:
                self.logger.warning(f"Quality monitoring already active for session {session_id}")
                return True

            # Initialize quality metrics collection
            quality_metrics = RealTimeQualityMetrics(
                session_id=session_id,
                collection_start=datetime.now(),
                latest_update=datetime.now(),
                sampling_rate=sampling_rate,
                buffer_size_samples=int(sampling_rate * self.config['buffer_duration_seconds']),
                quality_assessments=[],
                cumulative_artifacts={artifact: 0 for artifact in ArtifactType},
                session_statistics={},
                quality_alerts_triggered=0,
                intervention_events=[]
            )

            self.active_monitoring[session_id] = quality_metrics

            # Initialize data buffers
            self.quality_buffers[session_id] = {
                'eeg_data': np.array([]),
                'timestamps': np.array([]),
                'channel_names': []
            }

            # Create processing queue for real-time data
            self.processing_queues[session_id] = asyncio.Queue()

            # Start quality processing task
            self.quality_processors[session_id] = asyncio.create_task(
                self._process_quality_continuous(session_id)
            )

            self.logger.info(f"Quality monitoring started for session {session_id}")
            self.monitoring_statistics['sessions_monitored'] += 1

            return True

        except Exception as e:
            self.logger.error(f"Failed to start quality monitoring: {e}")
            return False

    async def stop_session_monitoring(self, session_id: str) -> Dict[str, Any]:
        """Stop quality monitoring and generate final report"""
        try:
            if session_id not in self.active_monitoring:
                self.logger.warning(f"No active monitoring for session {session_id}")
                return {}

            # Stop processing task
            if session_id in self.quality_processors:
                self.quality_processors[session_id].cancel()
                del self.quality_processors[session_id]

            # Generate final quality report
            final_report = await self._generate_session_quality_report(session_id)

            # Cleanup resources
            del self.active_monitoring[session_id]
            if session_id in self.quality_buffers:
                del self.quality_buffers[session_id]
            if session_id in self.processing_queues:
                del self.processing_queues[session_id]

            self.logger.info(f"Quality monitoring stopped for session {session_id}")
            return final_report

        except Exception as e:
            self.logger.error(f"Failed to stop quality monitoring: {e}")
            return {}

    async def ingest_real_time_data(self,
                                   session_id: str,
                                   eeg_data: np.ndarray,
                                   channel_names: List[str],
                                   timestamp: datetime) -> bool:
        """Ingest real-time EEG data for quality assessment"""
        try:
            if session_id not in self.active_monitoring:
                return False

            # Add to processing queue
            data_packet = {
                'timestamp': timestamp,
                'eeg_data': eeg_data,
                'channel_names': channel_names
            }

            await self.processing_queues[session_id].put(data_packet)
            return True

        except Exception as e:
            self.logger.error(f"Real-time data ingestion failed: {e}")
            return False

    async def _process_quality_continuous(self, session_id: str):
        """Continuous quality processing for session"""
        try:
            metrics = self.active_monitoring[session_id]
            buffer_data = self.quality_buffers[session_id]

            quality_interval = self.config['quality_assessment_interval_seconds']
            last_assessment = datetime.now()

            while session_id in self.active_monitoring:
                try:
                    # Get data from queue with timeout
                    data_packet = await asyncio.wait_for(
                        self.processing_queues[session_id].get(),
                        timeout=1.0
                    )

                    # Update buffer
                    await self._update_data_buffer(session_id, data_packet)

                    # Perform quality assessment if interval elapsed
                    if (datetime.now() - last_assessment).seconds >= quality_interval:
                        await self._perform_quality_assessment(session_id)
                        last_assessment = datetime.now()

                except asyncio.TimeoutError:
                    # No new data, check for stale connection
                    if (datetime.now() - metrics.latest_update).seconds > 30:
                        await self._handle_data_gap(session_id)

        except asyncio.CancelledError:
            self.logger.info(f"Quality processing cancelled for session {session_id}")
        except Exception as e:
            self.logger.error(f"Quality processing failed: {e}")

    async def _update_data_buffer(self, session_id: str, data_packet: Dict[str, Any]):
        """Update data buffer with new EEG data"""
        try:
            buffer_data = self.quality_buffers[session_id]
            metrics = self.active_monitoring[session_id]

            timestamp = data_packet['timestamp']
            eeg_data = data_packet['eeg_data']
            channel_names = data_packet['channel_names']

            # Initialize channel names if first packet
            if len(buffer_data['channel_names']) == 0:
                buffer_data['channel_names'] = channel_names

            # Ensure data shape consistency
            if len(channel_names) != len(buffer_data['channel_names']):
                self.logger.warning(f"Channel count mismatch in session {session_id}")
                return

            # Add to buffer
            if buffer_data['eeg_data'].size == 0:
                buffer_data['eeg_data'] = eeg_data.reshape(1, -1)
                buffer_data['timestamps'] = np.array([timestamp.timestamp()])
            else:
                buffer_data['eeg_data'] = np.vstack([buffer_data['eeg_data'], eeg_data.reshape(1, -1)])
                buffer_data['timestamps'] = np.append(buffer_data['timestamps'], timestamp.timestamp())

            # Maintain buffer size limit
            max_samples = metrics.buffer_size_samples
            if buffer_data['eeg_data'].shape[0] > max_samples:
                excess = buffer_data['eeg_data'].shape[0] - max_samples
                buffer_data['eeg_data'] = buffer_data['eeg_data'][excess:]
                buffer_data['timestamps'] = buffer_data['timestamps'][excess:]

            metrics.latest_update = timestamp

        except Exception as e:
            self.logger.error(f"Buffer update failed: {e}")

    async def _perform_quality_assessment(self, session_id: str):
        """Perform comprehensive quality assessment"""
        try:
            start_time = datetime.now()

            buffer_data = self.quality_buffers[session_id]
            metrics = self.active_monitoring[session_id]

            if buffer_data['eeg_data'].size == 0:
                return

            # Calculate signal quality metrics
            quality_scores = await self._calculate_signal_quality(session_id)

            # Detect artifacts
            artifacts = await self._detect_artifacts(session_id)

            # Assess data completeness
            completeness = await self._assess_data_completeness(session_id)

            # Frequency analysis
            frequency_analysis = await self._perform_frequency_analysis(session_id)

            # Generate recommendations
            recommendations = await self._generate_quality_recommendations(
                session_id, quality_scores, artifacts, completeness
            )

            # Check threshold violations
            violations = await self._check_threshold_violations(session_id, quality_scores)

            # Create quality assessment
            assessment = SignalQualityAssessment(
                session_id=session_id,
                participant_id=metrics.session_id.split('_')[1] if '_' in metrics.session_id else '',
                timestamp=datetime.now(),
                overall_quality_score=quality_scores['overall'],
                channel_quality_scores=quality_scores['channels'],
                impedance_values=quality_scores.get('impedance', {}),
                signal_to_noise_ratio=quality_scores.get('snr', 0.0),
                frequency_analysis=frequency_analysis,
                artifact_summary={artifact_type: len([a for a in artifacts if a.artifact_type == artifact_type]) for artifact_type in ArtifactType},
                data_completeness_percentage=completeness * 100,
                quality_trend=self._calculate_quality_trend(session_id),
                recommendations=recommendations,
                threshold_violations=violations
            )

            # Add to metrics collection
            metrics.quality_assessments.append(assessment)

            # Update cumulative artifacts
            for artifact in artifacts:
                metrics.cumulative_artifacts[artifact.artifact_type] += 1

            # Trigger interventions if needed
            if violations:
                await self._trigger_quality_interventions(session_id, assessment)

            # Update processing latency
            processing_latency = (datetime.now() - start_time).total_seconds()
            self.monitoring_statistics['average_processing_latency'] = (
                self.monitoring_statistics['average_processing_latency'] * 0.9 +
                processing_latency * 0.1
            )

            self.logger.debug(f"Quality assessment completed for session {session_id}: {quality_scores['overall']:.3f}")

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")

    async def _calculate_signal_quality(self, session_id: str) -> Dict[str, Any]:
        """Calculate comprehensive signal quality metrics"""
        try:
            buffer_data = self.quality_buffers[session_id]
            eeg_data = buffer_data['eeg_data']
            channel_names = buffer_data['channel_names']

            if eeg_data.size == 0:
                return {'overall': 0.0, 'channels': {}}

            quality_scores = {'channels': {}}

            # Calculate per-channel quality
            channel_qualities = []
            for i, channel in enumerate(channel_names):
                if i < eeg_data.shape[1]:
                    channel_data = eeg_data[:, i]

                    # Signal quality metrics
                    variance = np.var(channel_data)
                    std_dev = np.std(channel_data)

                    # Check for saturation
                    saturation_ratio = np.sum(np.abs(channel_data) > 200) / len(channel_data)

                    # Check for flatline
                    flatline_ratio = np.sum(np.abs(np.diff(channel_data)) < 1.0) / (len(channel_data) - 1)

                    # Calculate signal-to-noise ratio estimate
                    # High frequency content as noise proxy
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            high_freq = signal.filtfilt(
                                *signal.butter(4, [30, 40], btype='band', fs=256),
                                channel_data
                            )
                            low_freq = signal.filtfilt(
                                *signal.butter(4, [1, 30], btype='band', fs=256),
                                channel_data
                            )

                            signal_power = np.var(low_freq)
                            noise_power = np.var(high_freq)
                            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
                        except:
                            snr = 10.0  # Default SNR

                    # Combine metrics into quality score
                    quality_score = 1.0

                    # Penalize saturation
                    quality_score *= (1.0 - saturation_ratio)

                    # Penalize flatline
                    quality_score *= (1.0 - flatline_ratio)

                    # Reward good SNR
                    quality_score *= min(1.0, snr / 20.0)

                    # Penalize extreme variance
                    if std_dev > 100 or std_dev < 1:
                        quality_score *= 0.5

                    quality_scores['channels'][channel] = max(0.0, min(1.0, quality_score))
                    channel_qualities.append(quality_score)

            # Overall quality as average of channels
            quality_scores['overall'] = np.mean(channel_qualities) if channel_qualities else 0.0

            # Add SNR estimate
            quality_scores['snr'] = snr if 'snr' in locals() else 10.0

            return quality_scores

        except Exception as e:
            self.logger.error(f"Signal quality calculation failed: {e}")
            return {'overall': 0.0, 'channels': {}}

    async def _detect_artifacts(self, session_id: str) -> List[ArtifactDetection]:
        """Detect artifacts in EEG data"""
        artifacts = []

        try:
            buffer_data = self.quality_buffers[session_id]
            eeg_data = buffer_data['eeg_data']
            timestamps = buffer_data['timestamps']

            if eeg_data.size == 0:
                return artifacts

            # Run each artifact detector
            for artifact_type, detector in self.artifact_detectors.items():
                try:
                    detected_artifacts = detector(eeg_data, timestamps, buffer_data['channel_names'])
                    artifacts.extend(detected_artifacts)
                except Exception as e:
                    self.logger.warning(f"Artifact detector {artifact_type} failed: {e}")

            # Update monitoring statistics
            self.monitoring_statistics['artifacts_detected'] += len(artifacts)

            return artifacts

        except Exception as e:
            self.logger.error(f"Artifact detection failed: {e}")
            return []

    def _detect_eye_blinks(self, eeg_data: np.ndarray, timestamps: np.ndarray, channels: List[str]) -> List[ArtifactDetection]:
        """Detect eye blink artifacts"""
        artifacts = []

        try:
            # Find frontal channels
            frontal_channels = [i for i, ch in enumerate(channels)
                              if any(frontal in ch.upper() for frontal in ['FP1', 'FP2', 'AF7', 'AF8'])]

            if not frontal_channels:
                return artifacts

            threshold = self.config['artifact_thresholds']['eye_blink_threshold']

            for ch_idx in frontal_channels:
                if ch_idx < eeg_data.shape[1]:
                    channel_data = eeg_data[:, ch_idx]

                    # Detect large amplitude deflections
                    blink_indices = np.where(np.abs(channel_data) > threshold)[0]

                    if len(blink_indices) > 0:
                        # Group consecutive samples into blink events
                        blink_groups = []
                        current_group = [blink_indices[0]]

                        for i in range(1, len(blink_indices)):
                            if blink_indices[i] - blink_indices[i-1] <= 5:  # Within 5 samples
                                current_group.append(blink_indices[i])
                            else:
                                blink_groups.append(current_group)
                                current_group = [blink_indices[i]]
                        blink_groups.append(current_group)

                        # Create artifact detections
                        for group in blink_groups:
                            if len(group) >= 3:  # Minimum duration
                                start_idx = group[0]
                                end_idx = group[-1]

                                artifact = ArtifactDetection(
                                    artifact_type=ArtifactType.EYE_BLINK,
                                    timestamp=datetime.fromtimestamp(timestamps[start_idx]),
                                    duration_seconds=(end_idx - start_idx) / 256.0,  # Assume 256 Hz
                                    severity=np.max(np.abs(channel_data[group])) / threshold,
                                    affected_channels=[channels[ch_idx]],
                                    confidence=0.8
                                )
                                artifacts.append(artifact)

        except Exception as e:
            self.logger.warning(f"Eye blink detection failed: {e}")

        return artifacts

    def _detect_muscle_artifacts(self, eeg_data: np.ndarray, timestamps: np.ndarray, channels: List[str]) -> List[ArtifactDetection]:
        """Detect muscle tension artifacts"""
        artifacts = []

        try:
            threshold = self.config['artifact_thresholds']['muscle_threshold']

            # High frequency activity indicates muscle artifacts
            for ch_idx, channel in enumerate(channels):
                if ch_idx < eeg_data.shape[1]:
                    channel_data = eeg_data[:, ch_idx]

                    # Calculate high frequency power (20-50 Hz band)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            high_freq = signal.filtfilt(
                                *signal.butter(4, [20, 50], btype='band', fs=256),
                                channel_data
                            )

                            # Moving window RMS
                            window_size = int(256 * 1.0)  # 1 second windows
                            rms_values = []

                            for i in range(0, len(high_freq) - window_size, window_size // 2):
                                window_data = high_freq[i:i+window_size]
                                rms = np.sqrt(np.mean(window_data**2))
                                rms_values.append(rms)

                            # Detect periods above threshold
                            muscle_periods = np.where(np.array(rms_values) > threshold)[0]

                            for period_idx in muscle_periods:
                                start_time = timestamps[period_idx * window_size // 2]

                                artifact = ArtifactDetection(
                                    artifact_type=ArtifactType.MUSCLE_TENSION,
                                    timestamp=datetime.fromtimestamp(start_time),
                                    duration_seconds=1.0,
                                    severity=rms_values[period_idx] / threshold,
                                    affected_channels=[channel],
                                    confidence=0.7
                                )
                                artifacts.append(artifact)

                        except Exception:
                            pass  # Skip if filtering fails

        except Exception as e:
            self.logger.warning(f"Muscle artifact detection failed: {e}")

        return artifacts

    def _detect_line_noise(self, eeg_data: np.ndarray, timestamps: np.ndarray, channels: List[str]) -> List[ArtifactDetection]:
        """Detect line noise artifacts (50/60 Hz)"""
        artifacts = []

        try:
            threshold = self.config['artifact_thresholds']['line_noise_threshold']

            for ch_idx, channel in enumerate(channels):
                if ch_idx < eeg_data.shape[1]:
                    channel_data = eeg_data[:, ch_idx]

                    # FFT to detect line noise
                    freqs = np.fft.fftfreq(len(channel_data), 1/256.0)
                    fft_data = np.abs(np.fft.fft(channel_data))

                    # Check power at 50 Hz and 60 Hz
                    for line_freq in [50, 60]:
                        freq_idx = np.argmin(np.abs(freqs - line_freq))
                        line_power = fft_data[freq_idx]

                        # Compare to surrounding frequencies
                        surrounding_indices = np.arange(max(0, freq_idx-5), min(len(fft_data), freq_idx+6))
                        surrounding_indices = surrounding_indices[surrounding_indices != freq_idx]
                        avg_surrounding = np.mean(fft_data[surrounding_indices])

                        if line_power > avg_surrounding * threshold:
                            artifact = ArtifactDetection(
                                artifact_type=ArtifactType.LINE_NOISE,
                                timestamp=datetime.fromtimestamp(timestamps[0]),
                                duration_seconds=len(timestamps) / 256.0,
                                severity=line_power / avg_surrounding,
                                affected_channels=[channel],
                                confidence=0.9
                            )
                            artifacts.append(artifact)

        except Exception as e:
            self.logger.warning(f"Line noise detection failed: {e}")

        return artifacts

    def _detect_movement_artifacts(self, eeg_data: np.ndarray, timestamps: np.ndarray, channels: List[str]) -> List[ArtifactDetection]:
        """Detect movement artifacts"""
        artifacts = []

        try:
            threshold = self.config['artifact_thresholds']['movement_threshold']

            # Movement artifacts typically affect multiple channels simultaneously
            for ch_idx, channel in enumerate(channels):
                if ch_idx < eeg_data.shape[1]:
                    channel_data = eeg_data[:, ch_idx]

                    # Calculate gradient (rate of change)
                    gradient = np.abs(np.gradient(channel_data))

                    # Find periods of high gradient (rapid changes)
                    movement_indices = np.where(gradient > threshold)[0]

                    if len(movement_indices) > 0:
                        # Group consecutive samples
                        movement_groups = []
                        current_group = [movement_indices[0]]

                        for i in range(1, len(movement_indices)):
                            if movement_indices[i] - movement_indices[i-1] <= 10:
                                current_group.append(movement_indices[i])
                            else:
                                movement_groups.append(current_group)
                                current_group = [movement_indices[i]]
                        movement_groups.append(current_group)

                        for group in movement_groups:
                            if len(group) >= 5:  # Minimum duration
                                start_idx = group[0]
                                end_idx = group[-1]

                                artifact = ArtifactDetection(
                                    artifact_type=ArtifactType.MOVEMENT_ARTIFACT,
                                    timestamp=datetime.fromtimestamp(timestamps[start_idx]),
                                    duration_seconds=(end_idx - start_idx) / 256.0,
                                    severity=np.max(gradient[group]) / threshold,
                                    affected_channels=[channel],
                                    confidence=0.6
                                )
                                artifacts.append(artifact)

        except Exception as e:
            self.logger.warning(f"Movement artifact detection failed: {e}")

        return artifacts

    def _detect_electrode_pops(self, eeg_data: np.ndarray, timestamps: np.ndarray, channels: List[str]) -> List[ArtifactDetection]:
        """Detect electrode pop artifacts"""
        artifacts = []

        try:
            # Electrode pops are sudden, large amplitude changes
            for ch_idx, channel in enumerate(channels):
                if ch_idx < eeg_data.shape[1]:
                    channel_data = eeg_data[:, ch_idx]

                    # Calculate z-score to find outliers
                    z_scores = np.abs(zscore(channel_data))

                    # Find extreme outliers (z-score > 4)
                    pop_indices = np.where(z_scores > 4)[0]

                    for pop_idx in pop_indices:
                        artifact = ArtifactDetection(
                            artifact_type=ArtifactType.ELECTRODE_POP,
                            timestamp=datetime.fromtimestamp(timestamps[pop_idx]),
                            duration_seconds=1.0 / 256.0,  # Single sample
                            severity=z_scores[pop_idx] / 4.0,
                            affected_channels=[channel],
                            confidence=0.8
                        )
                        artifacts.append(artifact)

        except Exception as e:
            self.logger.warning(f"Electrode pop detection failed: {e}")

        return artifacts

    async def _assess_data_completeness(self, session_id: str) -> float:
        """Assess data completeness for session"""
        try:
            metrics = self.active_monitoring[session_id]

            # Calculate expected vs actual data points
            elapsed_seconds = (datetime.now() - metrics.collection_start).total_seconds()
            expected_samples = int(elapsed_seconds * metrics.sampling_rate)

            buffer_data = self.quality_buffers[session_id]
            actual_samples = buffer_data['eeg_data'].shape[0] if buffer_data['eeg_data'].size > 0 else 0

            # Calculate completeness ratio
            if expected_samples > 0:
                completeness = min(1.0, actual_samples / expected_samples)
            else:
                completeness = 1.0

            return completeness

        except Exception as e:
            self.logger.error(f"Data completeness assessment failed: {e}")
            return 0.0

    async def _perform_frequency_analysis(self, session_id: str) -> Dict[str, Any]:
        """Perform frequency domain analysis"""
        try:
            buffer_data = self.quality_buffers[session_id]
            eeg_data = buffer_data['eeg_data']

            if eeg_data.size == 0:
                return {}

            freq_config = self.config['frequency_analysis']
            bands = freq_config['bands']

            analysis = {
                'band_powers': {},
                'relative_powers': {},
                'peak_frequencies': {},
                'spectral_quality': {}
            }

            # Analyze first channel as representative
            if eeg_data.shape[1] > 0:
                channel_data = eeg_data[:, 0]

                # Calculate power spectral density
                freqs, psd = signal.welch(channel_data, fs=256.0, nperseg=min(256, len(channel_data)//4))

                # Calculate band powers
                total_power = np.trapz(psd, freqs)

                for band_name, (low_freq, high_freq) in bands.items():
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.trapz(psd[freq_mask], freqs[freq_mask])

                    analysis['band_powers'][band_name] = band_power
                    analysis['relative_powers'][band_name] = band_power / total_power if total_power > 0 else 0

                    # Find peak frequency in band
                    if np.any(freq_mask):
                        peak_idx = np.argmax(psd[freq_mask])
                        peak_freq = freqs[freq_mask][peak_idx]
                        analysis['peak_frequencies'][band_name] = peak_freq

                # Spectral quality metrics
                analysis['spectral_quality'] = {
                    'total_power': total_power,
                    'spectral_entropy': -np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd)),
                    'dominant_frequency': freqs[np.argmax(psd)]
                }

            return analysis

        except Exception as e:
            self.logger.error(f"Frequency analysis failed: {e}")
            return {}

    async def _generate_quality_recommendations(self,
                                              session_id: str,
                                              quality_scores: Dict[str, Any],
                                              artifacts: List[ArtifactDetection],
                                              completeness: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        try:
            overall_quality = quality_scores.get('overall', 0.0)

            # Signal quality recommendations
            if overall_quality < 0.6:
                recommendations.append("Signal quality is poor. Check electrode placement and impedance.")
            elif overall_quality < 0.8:
                recommendations.append("Signal quality could be improved. Ensure good electrode contact.")

            # Artifact-specific recommendations
            artifact_counts = {}
            for artifact in artifacts:
                artifact_counts[artifact.artifact_type] = artifact_counts.get(artifact.artifact_type, 0) + 1

            if artifact_counts.get(ArtifactType.EYE_BLINK, 0) > 5:
                recommendations.append("Excessive eye blinks detected. Ask participant to minimize eye movements.")

            if artifact_counts.get(ArtifactType.MUSCLE_TENSION, 0) > 3:
                recommendations.append("Muscle tension artifacts detected. Ensure participant is relaxed.")

            if artifact_counts.get(ArtifactType.LINE_NOISE, 0) > 0:
                recommendations.append("Electrical interference detected. Check power sources and grounding.")

            if artifact_counts.get(ArtifactType.MOVEMENT_ARTIFACT, 0) > 3:
                recommendations.append("Movement artifacts detected. Ask participant to remain still.")

            # Data completeness recommendations
            if completeness < 0.9:
                recommendations.append("Data loss detected. Check device connection and signal stability.")

            # Channel-specific recommendations
            channel_qualities = quality_scores.get('channels', {})
            poor_channels = [ch for ch, quality in channel_qualities.items() if quality < 0.5]

            if poor_channels:
                recommendations.append(f"Poor signal quality on channels: {', '.join(poor_channels)}. Check electrode contact.")

            # Default recommendation if quality is good
            if not recommendations and overall_quality > 0.8:
                recommendations.append("Signal quality is good. Continue current session.")

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")

        return recommendations

    async def _check_threshold_violations(self,
                                        session_id: str,
                                        quality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quality threshold violations"""
        violations = []

        try:
            overall_quality = quality_scores.get('overall', 0.0)

            # Check overall signal quality
            signal_threshold = self.quality_thresholds[QualityMetricType.SIGNAL_QUALITY]
            if overall_quality < signal_threshold.acceptable_min:
                severity = AlertSeverity.CRITICAL if overall_quality < signal_threshold.poor_max else AlertSeverity.WARNING

                violations.append({
                    'metric_type': QualityMetricType.SIGNAL_QUALITY.value,
                    'current_value': overall_quality,
                    'threshold_level': signal_threshold.acceptable_min,
                    'severity': severity.value,
                    'description': f"Signal quality {overall_quality:.3f} below acceptable threshold {signal_threshold.acceptable_min}"
                })

            # Check data completeness
            metrics = self.active_monitoring[session_id]
            completeness = await self._assess_data_completeness(session_id)
            completeness_threshold = self.quality_thresholds[QualityMetricType.DATA_COMPLETENESS]

            if completeness < completeness_threshold.acceptable_min:
                severity = AlertSeverity.CRITICAL if completeness < completeness_threshold.poor_max else AlertSeverity.WARNING

                violations.append({
                    'metric_type': QualityMetricType.DATA_COMPLETENESS.value,
                    'current_value': completeness,
                    'threshold_level': completeness_threshold.acceptable_min,
                    'severity': severity.value,
                    'description': f"Data completeness {completeness:.3f} below acceptable threshold {completeness_threshold.acceptable_min}"
                })

        except Exception as e:
            self.logger.error(f"Threshold violation check failed: {e}")

        return violations

    def _calculate_quality_trend(self, session_id: str) -> List[float]:
        """Calculate quality trend over recent assessments"""
        try:
            metrics = self.active_monitoring[session_id]
            recent_assessments = metrics.quality_assessments[-10:]  # Last 10 assessments

            trend = [assessment.overall_quality_score for assessment in recent_assessments]
            return trend

        except Exception as e:
            self.logger.error(f"Quality trend calculation failed: {e}")
            return []

    async def _trigger_quality_interventions(self,
                                           session_id: str,
                                           assessment: SignalQualityAssessment):
        """Trigger automated quality interventions"""
        try:
            metrics = self.active_monitoring[session_id]

            for violation in assessment.threshold_violations:
                intervention_event = {
                    'timestamp': datetime.now().isoformat(),
                    'violation_type': violation['metric_type'],
                    'severity': violation['severity'],
                    'intervention_actions': []
                }

                # Trigger appropriate interventions
                if violation['metric_type'] == QualityMetricType.SIGNAL_QUALITY.value:
                    if violation['severity'] == AlertSeverity.CRITICAL.value:
                        # Pause session and notify
                        intervention_event['intervention_actions'].extend([
                            'pause_session',
                            'notify_technician',
                            'send_quality_guidance'
                        ])
                        await self._pause_session_for_quality(session_id)
                    else:
                        # Send guidance
                        intervention_event['intervention_actions'].append('send_quality_guidance')
                        await self._send_quality_guidance(session_id, assessment)

                elif violation['metric_type'] == QualityMetricType.DATA_COMPLETENESS.value:
                    # Check connectivity
                    intervention_event['intervention_actions'].extend([
                        'check_connectivity',
                        'notify_technical_support'
                    ])
                    await self._check_device_connectivity(session_id)

                metrics.intervention_events.append(intervention_event)
                metrics.quality_alerts_triggered += 1

            # Update monitoring statistics
            self.monitoring_statistics['quality_interventions'] += len(assessment.threshold_violations)

        except Exception as e:
            self.logger.error(f"Quality intervention trigger failed: {e}")

    async def _pause_session_for_quality(self, session_id: str):
        """Pause session due to quality issues"""
        try:
            # Update workflow manager
            if hasattr(self.workflow_manager, 'active_sessions') and session_id in self.workflow_manager.active_sessions:
                session = self.workflow_manager.active_sessions[session_id]
                session.current_status = RemoteSessionStatus.SESSION_PAUSED

            self.logger.warning(f"Session {session_id} paused due to critical quality issues")

        except Exception as e:
            self.logger.error(f"Session pause failed: {e}")

    async def _send_quality_guidance(self, session_id: str, assessment: SignalQualityAssessment):
        """Send quality improvement guidance to participant"""
        try:
            # Create guidance message based on assessment
            guidance_parts = []

            if assessment.overall_quality_score < 0.6:
                guidance_parts.append("Signal quality needs improvement:")
                guidance_parts.extend(assessment.recommendations)

            guidance_message = "\n".join(guidance_parts)

            # In production, send via appropriate communication channel
            self.logger.info(f"Quality guidance sent for session {session_id}: {guidance_message[:100]}...")

        except Exception as e:
            self.logger.error(f"Quality guidance sending failed: {e}")

    async def _check_device_connectivity(self, session_id: str):
        """Check device connectivity for session"""
        try:
            # Get device status from device manager
            # This would check actual device connectivity
            self.logger.info(f"Checking device connectivity for session {session_id}")

        except Exception as e:
            self.logger.error(f"Device connectivity check failed: {e}")

    async def _handle_data_gap(self, session_id: str):
        """Handle data gap or connection loss"""
        try:
            metrics = self.active_monitoring[session_id]

            gap_event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'data_gap',
                'duration_seconds': (datetime.now() - metrics.latest_update).total_seconds()
            }

            metrics.intervention_events.append(gap_event)

            # Trigger connectivity check
            await self._check_device_connectivity(session_id)

            self.logger.warning(f"Data gap detected in session {session_id}")

        except Exception as e:
            self.logger.error(f"Data gap handling failed: {e}")

    async def _generate_session_quality_report(self, session_id: str) -> Dict[str, Any]:
        """Generate final quality report for session"""
        try:
            if session_id not in self.active_monitoring:
                return {}

            metrics = self.active_monitoring[session_id]

            # Calculate session statistics
            quality_scores = [assessment.overall_quality_score for assessment in metrics.quality_assessments]

            session_stats = {
                'average_quality': statistics.mean(quality_scores) if quality_scores else 0.0,
                'minimum_quality': min(quality_scores) if quality_scores else 0.0,
                'maximum_quality': max(quality_scores) if quality_scores else 0.0,
                'quality_stability': 1.0 - (statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0),
                'total_assessments': len(metrics.quality_assessments),
                'total_artifacts': sum(metrics.cumulative_artifacts.values()),
                'quality_alerts': metrics.quality_alerts_triggered,
                'intervention_events': len(metrics.intervention_events)
            }

            # Artifact breakdown
            artifact_breakdown = {
                artifact_type.value: count
                for artifact_type, count in metrics.cumulative_artifacts.items()
                if count > 0
            }

            # Quality trend analysis
            quality_trend = quality_scores[-20:] if len(quality_scores) >= 20 else quality_scores
            trend_direction = "stable"
            if len(quality_trend) > 5:
                early_avg = statistics.mean(quality_trend[:len(quality_trend)//2])
                late_avg = statistics.mean(quality_trend[len(quality_trend)//2:])

                if late_avg > early_avg + 0.1:
                    trend_direction = "improving"
                elif late_avg < early_avg - 0.1:
                    trend_direction = "declining"

            return {
                'session_id': session_id,
                'monitoring_duration_seconds': (datetime.now() - metrics.collection_start).total_seconds(),
                'session_statistics': session_stats,
                'artifact_breakdown': artifact_breakdown,
                'quality_trend_direction': trend_direction,
                'final_recommendations': self._generate_final_recommendations(session_stats, artifact_breakdown),
                'data_completeness': await self._assess_data_completeness(session_id),
                'report_generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Session quality report generation failed: {e}")
            return {}

    def _generate_final_recommendations(self,
                                      session_stats: Dict[str, float],
                                      artifact_breakdown: Dict[str, int]) -> List[str]:
        """Generate final session recommendations"""
        recommendations = []

        try:
            avg_quality = session_stats.get('average_quality', 0.0)

            # Overall quality recommendations
            if avg_quality >= 0.9:
                recommendations.append("Excellent session quality maintained throughout.")
            elif avg_quality >= 0.75:
                recommendations.append("Good session quality with minor issues.")
            elif avg_quality >= 0.6:
                recommendations.append("Acceptable session quality but room for improvement.")
            else:
                recommendations.append("Session quality was poor and may affect results.")

            # Artifact-specific recommendations
            if artifact_breakdown.get('eye_blink', 0) > 10:
                recommendations.append("Consider additional eye movement training for future sessions.")

            if artifact_breakdown.get('muscle_tension', 0) > 5:
                recommendations.append("Implement relaxation techniques before future sessions.")

            if artifact_breakdown.get('line_noise', 0) > 0:
                recommendations.append("Review environmental setup to reduce electrical interference.")

            # Stability recommendations
            quality_stability = session_stats.get('quality_stability', 1.0)
            if quality_stability < 0.8:
                recommendations.append("Work on maintaining consistent signal quality throughout sessions.")

        except Exception as e:
            self.logger.error(f"Final recommendations generation failed: {e}")

        return recommendations

    async def get_real_time_quality_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current real-time quality status"""
        try:
            if session_id not in self.active_monitoring:
                return None

            metrics = self.active_monitoring[session_id]

            # Get latest assessment
            latest_assessment = metrics.quality_assessments[-1] if metrics.quality_assessments else None

            return {
                'session_id': session_id,
                'monitoring_active': True,
                'latest_quality_score': latest_assessment.overall_quality_score if latest_assessment else 0.0,
                'latest_assessment_time': latest_assessment.timestamp.isoformat() if latest_assessment else None,
                'total_assessments': len(metrics.quality_assessments),
                'active_alerts': metrics.quality_alerts_triggered,
                'cumulative_artifacts': {k.value: v for k, v in metrics.cumulative_artifacts.items()},
                'data_completeness': await self._assess_data_completeness(session_id),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Real-time quality status failed: {e}")
            return None

    async def get_quality_monitoring_summary(self) -> Dict[str, Any]:
        """Get overall quality monitoring summary"""
        return {
            'active_sessions': len(self.active_monitoring),
            'total_sessions_monitored': self.monitoring_statistics['sessions_monitored'],
            'total_quality_interventions': self.monitoring_statistics['quality_interventions'],
            'total_artifacts_detected': self.monitoring_statistics['artifacts_detected'],
            'average_processing_latency_ms': self.monitoring_statistics['average_processing_latency'] * 1000,
            'quality_threshold_summary': {
                threshold_type.value: {
                    'excellent_min': threshold.excellent_min,
                    'acceptable_min': threshold.acceptable_min,
                    'units': threshold.units
                }
                for threshold_type, threshold in self.quality_thresholds.items()
            },
            'last_updated': datetime.now().isoformat()
        }