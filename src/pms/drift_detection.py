"""
Advanced drift detection algorithms for post-market surveillance
Implements PSI, CUSUM, and other statistical methods for monitoring model performance
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import logging

logger = logging.getLogger(__name__)

class DriftDetector(ABC):
    """Base class for drift detection algorithms"""

    @abstractmethod
    def detect_drift(self, baseline_data: np.ndarray, current_data: np.ndarray) -> bool:
        """Detect if drift has occurred between baseline and current data"""
        pass

class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) for detecting covariate drift

    PSI ranges:
    - 0.0 to 0.1: No significant population change
    - 0.1 to 0.25: Some minor population change
    - > 0.25: Major population shift, further investigation required
    """

    def __init__(self, n_bins: int = 10, bin_method: str = 'quantile'):
        self.n_bins = n_bins
        self.bin_method = bin_method

    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Calculate PSI between baseline and current populations"""
        if baseline.ndim == 1:
            return self._calculate_psi_univariate(baseline, current)
        else:
            # For multivariate data, calculate PSI for each feature and aggregate
            psi_scores = []
            for i in range(baseline.shape[1]):
                psi_score = self._calculate_psi_univariate(baseline[:, i], current[:, i])
                psi_scores.append(psi_score)
            return np.mean(psi_scores)

    def _calculate_psi_univariate(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Calculate PSI for a single feature"""
        try:
            # Create bins based on baseline distribution
            if self.bin_method == 'quantile':
                bin_edges = np.quantile(baseline, np.linspace(0, 1, self.n_bins + 1))
            else:  # equal width
                bin_edges = np.linspace(baseline.min(), baseline.max(), self.n_bins + 1)

            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:  # Need at least 2 bins
                return 0.0

            # Calculate distributions
            baseline_dist, _ = np.histogram(baseline, bins=bin_edges, density=True)
            current_dist, _ = np.histogram(current, bins=bin_edges, density=True)

            # Normalize to get proportions
            baseline_prop = baseline_dist / baseline_dist.sum()
            current_prop = current_dist / current_dist.sum()

            # Avoid division by zero
            baseline_prop = np.where(baseline_prop == 0, 1e-8, baseline_prop)
            current_prop = np.where(current_prop == 0, 1e-8, current_prop)

            # Calculate PSI
            psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

            return psi

        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) detector for concept drift
    Detects shifts in the mean of a time series
    """

    def __init__(self, threshold: float = 5.0, drift_magnitude: float = 0.5):
        self.threshold = threshold
        self.drift_magnitude = drift_magnitude

    def detect_drift(self, data: np.ndarray) -> List[int]:
        """
        Detect drift points using CUSUM algorithm
        Returns list of indices where drift was detected
        """
        if len(data) < 10:  # Need minimum data points
            return []

        try:
            # Calculate CUSUM statistics
            mean_data = np.mean(data[:50]) if len(data) >= 50 else np.mean(data)

            cumsum_pos = np.zeros(len(data))
            cumsum_neg = np.zeros(len(data))

            drift_points = []

            for i in range(1, len(data)):
                # Positive CUSUM (detecting upward drift)
                cumsum_pos[i] = max(0, cumsum_pos[i-1] + (data[i] - mean_data - self.drift_magnitude))

                # Negative CUSUM (detecting downward drift)
                cumsum_neg[i] = max(0, cumsum_neg[i-1] - (data[i] - mean_data + self.drift_magnitude))

                # Check for drift
                if cumsum_pos[i] > self.threshold or cumsum_neg[i] > self.threshold:
                    drift_points.append(i)
                    # Reset CUSUM after detection
                    cumsum_pos[i] = 0
                    cumsum_neg[i] = 0

            return drift_points

        except Exception as e:
            logger.error(f"Error in CUSUM drift detection: {e}")
            return []

class KolmogorovSmirnovDetector(DriftDetector):
    """
    Kolmogorov-Smirnov test for distribution drift detection
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect_drift(self, baseline_data: np.ndarray, current_data: np.ndarray) -> bool:
        """Detect drift using KS test"""
        try:
            if baseline_data.ndim == 1:
                statistic, p_value = ks_2samp(baseline_data, current_data)
                return p_value < self.alpha
            else:
                # For multivariate data, test each feature
                p_values = []
                for i in range(baseline_data.shape[1]):
                    _, p_val = ks_2samp(baseline_data[:, i], current_data[:, i])
                    p_values.append(p_val)

                # Use Bonferroni correction for multiple testing
                adjusted_alpha = self.alpha / baseline_data.shape[1]
                return any(p < adjusted_alpha for p in p_values)

        except Exception as e:
            logger.error(f"Error in KS drift detection: {e}")
            return False

class DataDriftMonitor:
    """
    Comprehensive data drift monitoring with multiple detection methods
    """

    def __init__(self, config: Dict = None):
        default_config = {
            'psi_threshold': 0.25,
            'ks_alpha': 0.05,
            'cusum_threshold': 5.0,
            'cusum_drift_magnitude': 0.5,
            'min_baseline_samples': 100,
            'min_current_samples': 50
        }

        self.config = {**default_config, **(config or {})}

        self.psi_detector = PopulationStabilityIndex()
        self.ks_detector = KolmogorovSmirnovDetector(alpha=self.config['ks_alpha'])
        self.cusum_detector = CUSUMDetector(
            threshold=self.config['cusum_threshold'],
            drift_magnitude=self.config['cusum_drift_magnitude']
        )

    def comprehensive_drift_assessment(self, baseline_data: np.ndarray,
                                     current_data: np.ndarray,
                                     feature_names: List[str] = None) -> Dict:
        """
        Perform comprehensive drift assessment using multiple methods
        """
        if len(baseline_data) < self.config['min_baseline_samples']:
            return {
                'status': 'insufficient_baseline_data',
                'baseline_samples': len(baseline_data),
                'required_samples': self.config['min_baseline_samples']
            }

        if len(current_data) < self.config['min_current_samples']:
            return {
                'status': 'insufficient_current_data',
                'current_samples': len(current_data),
                'required_samples': self.config['min_current_samples']
            }

        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'baseline_samples': len(baseline_data),
            'current_samples': len(current_data),
            'methods': {}
        }

        try:
            # PSI Analysis
            psi_score = self.psi_detector.calculate_psi(baseline_data, current_data)
            psi_drift = psi_score > self.config['psi_threshold']

            results['methods']['psi'] = {
                'score': float(psi_score),
                'threshold': self.config['psi_threshold'],
                'drift_detected': psi_drift,
                'interpretation': self._interpret_psi(psi_score)
            }

            # Kolmogorov-Smirnov test
            ks_drift = self.ks_detector.detect_drift(baseline_data, current_data)
            results['methods']['kolmogorov_smirnov'] = {
                'drift_detected': ks_drift,
                'alpha': self.config['ks_alpha']
            }

            # Feature-wise analysis if multivariate
            if baseline_data.ndim > 1:
                feature_analysis = self._analyze_feature_drift(
                    baseline_data, current_data, feature_names
                )
                results['feature_analysis'] = feature_analysis

            # Overall drift assessment
            drift_methods_positive = sum([
                psi_drift,
                ks_drift
            ])

            results['overall_assessment'] = {
                'drift_detected': drift_methods_positive >= 1,
                'confidence': 'high' if drift_methods_positive >= 2 else 'medium' if drift_methods_positive == 1 else 'low',
                'methods_detecting_drift': drift_methods_positive,
                'recommendation': self._get_recommendation(drift_methods_positive, psi_score)
            }

        except Exception as e:
            logger.error(f"Error in comprehensive drift assessment: {e}")
            results['error'] = str(e)

        return results

    def monitor_prediction_drift(self, prediction_history: np.ndarray,
                               window_size: int = 100) -> Dict:
        """Monitor prediction drift using CUSUM"""
        if len(prediction_history) < window_size * 2:
            return {'status': 'insufficient_data', 'required_samples': window_size * 2}

        try:
            # Use sliding window approach
            baseline_window = prediction_history[-window_size*2:-window_size]
            current_window = prediction_history[-window_size:]

            # CUSUM detection on recent predictions
            drift_points = self.cusum_detector.detect_drift(prediction_history)

            # Statistical comparison
            baseline_mean = np.mean(baseline_window)
            current_mean = np.mean(current_window)

            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(baseline_window, current_window)

            return {
                'timestamp': pd.Timestamp.now().isoformat(),
                'baseline_mean': float(baseline_mean),
                'current_mean': float(current_mean),
                'mean_difference': float(current_mean - baseline_mean),
                'cusum_drift_points': len(drift_points),
                'recent_drift_points': [p for p in drift_points if p >= len(prediction_history) - window_size],
                't_test': {
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                },
                'drift_detected': len([p for p in drift_points if p >= len(prediction_history) - window_size]) > 0
            }

        except Exception as e:
            logger.error(f"Error in prediction drift monitoring: {e}")
            return {'error': str(e)}

    def _analyze_feature_drift(self, baseline_data: np.ndarray,
                              current_data: np.ndarray,
                              feature_names: List[str] = None) -> Dict:
        """Analyze drift for individual features"""
        n_features = baseline_data.shape[1]
        feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        feature_results = {}

        for i, feature_name in enumerate(feature_names):
            baseline_feature = baseline_data[:, i]
            current_feature = current_data[:, i]

            # PSI for this feature
            psi_score = self.psi_detector._calculate_psi_univariate(
                baseline_feature, current_feature
            )

            # KS test for this feature
            ks_stat, ks_p = ks_2samp(baseline_feature, current_feature)

            # Basic statistics
            baseline_stats = {
                'mean': float(np.mean(baseline_feature)),
                'std': float(np.std(baseline_feature)),
                'median': float(np.median(baseline_feature))
            }

            current_stats = {
                'mean': float(np.mean(current_feature)),
                'std': float(np.std(current_feature)),
                'median': float(np.median(current_feature))
            }

            feature_results[feature_name] = {
                'psi_score': float(psi_score),
                'psi_drift': psi_score > self.config['psi_threshold'],
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'ks_drift': ks_p < self.config['ks_alpha'],
                'baseline_stats': baseline_stats,
                'current_stats': current_stats,
                'mean_shift': float(current_stats['mean'] - baseline_stats['mean']),
                'std_change': float(current_stats['std'] - baseline_stats['std'])
            }

        return feature_results

    def _interpret_psi(self, psi_score: float) -> str:
        """Interpret PSI score"""
        if psi_score < 0.1:
            return "No significant population change"
        elif psi_score < 0.25:
            return "Some minor population change"
        else:
            return "Major population shift - investigation required"

    def _get_recommendation(self, drift_methods_positive: int, psi_score: float) -> str:
        """Get recommendation based on drift analysis"""
        if drift_methods_positive >= 2:
            return "Strong evidence of drift - immediate model retraining recommended"
        elif drift_methods_positive == 1:
            if psi_score > 0.3:
                return "Moderate drift detected - monitor closely and consider retraining"
            else:
                return "Minor drift detected - continue monitoring"
        else:
            return "No significant drift detected - continue normal operations"

class PerformanceDriftDetector:
    """
    Detect performance drift using statistical process control methods
    """

    def __init__(self, baseline_performance: float, control_limits: Tuple[float, float] = None):
        self.baseline_performance = baseline_performance

        if control_limits:
            self.lower_limit, self.upper_limit = control_limits
        else:
            # Default 3-sigma control limits (assuming normal distribution)
            self.lower_limit = baseline_performance - 0.05  # 5% below baseline
            self.upper_limit = baseline_performance + 0.05  # 5% above baseline

    def detect_performance_drift(self, recent_performance: List[float],
                               window_size: int = 10) -> Dict:
        """
        Detect performance drift using control charts and trend analysis
        """
        if len(recent_performance) < window_size:
            return {
                'status': 'insufficient_data',
                'required_samples': window_size,
                'current_samples': len(recent_performance)
            }

        recent_values = np.array(recent_performance[-window_size:])

        # Control chart analysis
        out_of_control = np.sum((recent_values < self.lower_limit) |
                               (recent_values > self.upper_limit))

        # Trend analysis
        x = np.arange(len(recent_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)

        # Performance stability
        cv = np.std(recent_values) / np.mean(recent_values) if np.mean(recent_values) != 0 else 0

        # Current vs baseline comparison
        current_mean = np.mean(recent_values)
        performance_change = current_mean - self.baseline_performance

        return {
            'timestamp': pd.Timestamp.now().isoformat(),
            'baseline_performance': self.baseline_performance,
            'current_mean_performance': float(current_mean),
            'performance_change': float(performance_change),
            'control_limits': {
                'lower': self.lower_limit,
                'upper': self.upper_limit
            },
            'out_of_control_points': int(out_of_control),
            'trend_analysis': {
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'significant_trend': p_value < 0.05
            },
            'stability_metrics': {
                'coefficient_of_variation': float(cv),
                'stable': cv < 0.1  # Less than 10% variation
            },
            'drift_assessment': {
                'drift_detected': out_of_control > 0 or abs(performance_change) > 0.05,
                'severity': self._assess_drift_severity(out_of_control, performance_change)
            }
        }

    def _assess_drift_severity(self, out_of_control: int, performance_change: float) -> str:
        """Assess the severity of performance drift"""
        if out_of_control >= 3 or abs(performance_change) > 0.1:
            return "high"
        elif out_of_control >= 1 or abs(performance_change) > 0.05:
            return "medium"
        else:
            return "low"