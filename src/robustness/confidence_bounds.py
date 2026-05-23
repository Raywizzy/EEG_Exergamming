"""
Statistical Confidence Bounds Engine
Calculates rigorous confidence intervals and statistical bounds for robustness metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import json
import sqlite3
from datetime import datetime
from scipy import stats
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceInterval:
    """Statistical confidence interval"""
    metric_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str  # 'bootstrap', 'normal', 'binomial', 'empirical'
    sample_size: int
    std_error: Optional[float] = None

@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    effect_size: Optional[float]
    confidence_interval: Optional[ConfidenceInterval]
    interpretation: str
    assumptions_met: bool

@dataclass
class RobustnessStatistics:
    """Comprehensive robustness statistics"""
    metric_name: str
    baseline_stats: Dict[str, float]
    stressed_stats: Dict[str, float]
    degradation_stats: Dict[str, float]
    confidence_intervals: List[ConfidenceInterval]
    statistical_tests: List[StatisticalTest]
    risk_assessment: Dict[str, Any]

class StatisticalConfidenceEngine:
    """Engine for calculating statistical confidence bounds and significance tests"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.confidence_levels = [0.90, 0.95, 0.99]

        self._init_database()

    def _init_database(self):
        """Initialize statistical analysis database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Confidence intervals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confidence_intervals (
                    interval_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    point_estimate REAL NOT NULL,
                    lower_bound REAL NOT NULL,
                    upper_bound REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    method TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    std_error REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Statistical tests table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statistical_tests (
                    test_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    statistic REAL NOT NULL,
                    p_value REAL NOT NULL,
                    critical_value REAL,
                    effect_size REAL,
                    interpretation TEXT,
                    assumptions_met BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Risk assessments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    risk_type TEXT NOT NULL, -- 'performance_degradation', 'failure_probability', 'outlier_detection'
                    risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high', 'critical')),
                    risk_score REAL,
                    risk_factors TEXT, -- JSON
                    mitigation_strategies TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence_analysis ON confidence_intervals (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tests_analysis ON statistical_tests (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_analysis ON risk_assessments (analysis_id)")

    def calculate_robustness_statistics(self,
                                      baseline_metrics: List[Dict[str, float]],
                                      stressed_metrics: List[Dict[str, float]],
                                      analysis_id: str,
                                      test_conditions: Dict[str, Any]) -> Dict[str, RobustnessStatistics]:
        """Calculate comprehensive robustness statistics with confidence bounds"""

        if len(baseline_metrics) != len(stressed_metrics):
            raise ValueError("Baseline and stressed metrics must have same length")

        results = {}

        # Get all metric names
        metric_names = set()
        for metrics in baseline_metrics + stressed_metrics:
            metric_names.update(metrics.keys())

        for metric_name in metric_names:
            if metric_name in ['predicted_label', 'true_label']:
                continue  # Skip non-numeric metrics

            try:
                # Extract metric values
                baseline_values = [m.get(metric_name, 0) for m in baseline_metrics]
                stressed_values = [m.get(metric_name, 0) for m in stressed_metrics]
                degradation_values = [b - s for b, s in zip(baseline_values, stressed_values)]

                # Calculate basic statistics
                baseline_stats = self._calculate_descriptive_stats(baseline_values)
                stressed_stats = self._calculate_descriptive_stats(stressed_values)
                degradation_stats = self._calculate_descriptive_stats(degradation_values)

                # Calculate confidence intervals
                confidence_intervals = []

                # Baseline confidence intervals
                for conf_level in self.confidence_levels:
                    ci_baseline = self._calculate_confidence_interval(
                        baseline_values, conf_level, f"{metric_name}_baseline"
                    )
                    ci_stressed = self._calculate_confidence_interval(
                        stressed_values, conf_level, f"{metric_name}_stressed"
                    )
                    ci_degradation = self._calculate_confidence_interval(
                        degradation_values, conf_level, f"{metric_name}_degradation"
                    )

                    confidence_intervals.extend([ci_baseline, ci_stressed, ci_degradation])

                # Statistical tests
                statistical_tests = []

                # Paired t-test for before/after comparison
                if len(baseline_values) > 1:
                    t_test = self._paired_t_test(baseline_values, stressed_values, metric_name)
                    statistical_tests.append(t_test)

                # Wilcoxon signed-rank test (non-parametric alternative)
                wilcoxon_test = self._wilcoxon_test(baseline_values, stressed_values, metric_name)
                statistical_tests.append(wilcoxon_test)

                # Effect size calculation
                effect_size_test = self._calculate_effect_size(baseline_values, stressed_values, metric_name)
                statistical_tests.append(effect_size_test)

                # Risk assessment
                risk_assessment = self._assess_performance_risk(
                    baseline_values, stressed_values, degradation_values, metric_name, test_conditions
                )

                # Create robustness statistics object
                robustness_stats = RobustnessStatistics(
                    metric_name=metric_name,
                    baseline_stats=baseline_stats,
                    stressed_stats=stressed_stats,
                    degradation_stats=degradation_stats,
                    confidence_intervals=confidence_intervals,
                    statistical_tests=statistical_tests,
                    risk_assessment=risk_assessment
                )

                results[metric_name] = robustness_stats

                # Store in database
                self._store_statistical_analysis(analysis_id, robustness_stats)

            except Exception as e:
                logger.error(f"Failed to calculate statistics for {metric_name}: {e}")

        return results

    def _calculate_descriptive_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        values_array = np.array(values)

        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'q25': float(np.percentile(values_array, 25)),
            'q75': float(np.percentile(values_array, 75)),
            'iqr': float(np.percentile(values_array, 75) - np.percentile(values_array, 25)),
            'skewness': float(stats.skew(values_array)),
            'kurtosis': float(stats.kurtosis(values_array)),
            'cv': float(np.std(values_array) / np.mean(values_array)) if np.mean(values_array) != 0 else float('inf')
        }

    def _calculate_confidence_interval(self, values: List[float],
                                     confidence_level: float,
                                     metric_name: str) -> ConfidenceInterval:
        """Calculate confidence interval using appropriate method"""
        values_array = np.array(values)
        n = len(values_array)
        mean = np.mean(values_array)

        if n < 2:
            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=mean,
                lower_bound=mean,
                upper_bound=mean,
                confidence_level=confidence_level,
                method="insufficient_data",
                sample_size=n
            )

        # Choose method based on sample size and data characteristics
        if n >= 30:
            # Large sample: use normal approximation
            std_error = np.std(values_array, ddof=1) / np.sqrt(n)
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            margin_error = z_critical * std_error

            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=mean,
                lower_bound=mean - margin_error,
                upper_bound=mean + margin_error,
                confidence_level=confidence_level,
                method="normal_approximation",
                sample_size=n,
                std_error=std_error
            )

        elif n >= 5:
            # Small sample: use t-distribution
            std_error = np.std(values_array, ddof=1) / np.sqrt(n)
            alpha = 1 - confidence_level
            df = n - 1
            t_critical = stats.t.ppf(1 - alpha/2, df)
            margin_error = t_critical * std_error

            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=mean,
                lower_bound=mean - margin_error,
                upper_bound=mean + margin_error,
                confidence_level=confidence_level,
                method="t_distribution",
                sample_size=n,
                std_error=std_error
            )
        else:
            # Very small sample: use bootstrap
            return self._bootstrap_confidence_interval(values_array, confidence_level, metric_name)

    def _bootstrap_confidence_interval(self, values: np.ndarray,
                                     confidence_level: float,
                                     metric_name: str,
                                     n_bootstrap: int = 10000) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval"""

        def statistic(x):
            return np.mean(x)

        # Use scipy.stats.bootstrap for modern approach
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        try:
            res = bootstrap((values,), statistic, n_resamples=n_bootstrap,
                          confidence_level=confidence_level, random_state=rng)

            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=np.mean(values),
                lower_bound=res.confidence_interval.low,
                upper_bound=res.confidence_interval.high,
                confidence_level=confidence_level,
                method="bootstrap",
                sample_size=len(values)
            )
        except Exception as e:
            logger.warning(f"Bootstrap failed, using percentile method: {e}")

            # Fallback to manual bootstrap
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = resample(values, n_samples=len(values), random_state=None)
                bootstrap_means.append(np.mean(bootstrap_sample))

            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100

            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=np.mean(values),
                lower_bound=np.percentile(bootstrap_means, lower_percentile),
                upper_bound=np.percentile(bootstrap_means, upper_percentile),
                confidence_level=confidence_level,
                method="bootstrap_percentile",
                sample_size=len(values)
            )

    def _paired_t_test(self, baseline: List[float], stressed: List[float],
                      metric_name: str) -> StatisticalTest:
        """Perform paired t-test"""
        baseline_array = np.array(baseline)
        stressed_array = np.array(stressed)

        # Check assumptions
        differences = baseline_array - stressed_array
        n = len(differences)

        # Normality test (Shapiro-Wilk for small samples)
        if n > 3:
            _, p_normality = stats.shapiro(differences)
            assumptions_met = p_normality > 0.05
        else:
            assumptions_met = False

        # Perform test
        t_statistic, p_value = stats.ttest_rel(baseline_array, stressed_array)

        # Calculate effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        effect_size = mean_diff / std_diff if std_diff > 0 else 0

        # Critical value
        df = n - 1
        alpha = 0.05
        critical_value = stats.t.ppf(1 - alpha/2, df)

        # Interpretation
        if p_value < 0.001:
            interpretation = "Highly significant difference (p < 0.001)"
        elif p_value < 0.01:
            interpretation = "Very significant difference (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "Significant difference (p < 0.05)"
        else:
            interpretation = "No significant difference (p ≥ 0.05)"

        return StatisticalTest(
            test_name=f"paired_t_test_{metric_name}",
            statistic=t_statistic,
            p_value=p_value,
            critical_value=critical_value,
            effect_size=effect_size,
            confidence_interval=None,  # Could add CI for mean difference
            interpretation=interpretation,
            assumptions_met=assumptions_met
        )

    def _wilcoxon_test(self, baseline: List[float], stressed: List[float],
                      metric_name: str) -> StatisticalTest:
        """Perform Wilcoxon signed-rank test (non-parametric)"""
        baseline_array = np.array(baseline)
        stressed_array = np.array(stressed)

        try:
            statistic, p_value = stats.wilcoxon(baseline_array, stressed_array,
                                              alternative='two-sided')

            # Effect size (r = Z / sqrt(N))
            n = len(baseline_array)
            z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
            effect_size = abs(z_score) / np.sqrt(n)

            if p_value < 0.001:
                interpretation = "Highly significant difference (p < 0.001, non-parametric)"
            elif p_value < 0.01:
                interpretation = "Very significant difference (p < 0.01, non-parametric)"
            elif p_value < 0.05:
                interpretation = "Significant difference (p < 0.05, non-parametric)"
            else:
                interpretation = "No significant difference (p ≥ 0.05, non-parametric)"

            return StatisticalTest(
                test_name=f"wilcoxon_signed_rank_{metric_name}",
                statistic=statistic,
                p_value=p_value,
                critical_value=None,
                effect_size=effect_size,
                confidence_interval=None,
                interpretation=interpretation,
                assumptions_met=True  # Non-parametric test has fewer assumptions
            )

        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return StatisticalTest(
                test_name=f"wilcoxon_signed_rank_{metric_name}",
                statistic=0,
                p_value=1.0,
                critical_value=None,
                effect_size=0,
                confidence_interval=None,
                interpretation="Test failed to run",
                assumptions_met=False
            )

    def _calculate_effect_size(self, baseline: List[float], stressed: List[float],
                             metric_name: str) -> StatisticalTest:
        """Calculate effect size (Cohen's d)"""
        baseline_array = np.array(baseline)
        stressed_array = np.array(stressed)

        mean1 = np.mean(baseline_array)
        mean2 = np.mean(stressed_array)
        std1 = np.std(baseline_array, ddof=1)
        std2 = np.std(stressed_array, ddof=1)
        n1 = len(baseline_array)
        n2 = len(stressed_array)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "Negligible effect size (|d| < 0.2)"
        elif abs_d < 0.5:
            interpretation = "Small effect size (0.2 ≤ |d| < 0.5)"
        elif abs_d < 0.8:
            interpretation = "Medium effect size (0.5 ≤ |d| < 0.8)"
        else:
            interpretation = "Large effect size (|d| ≥ 0.8)"

        return StatisticalTest(
            test_name=f"cohens_d_{metric_name}",
            statistic=cohens_d,
            p_value=None,  # Effect size doesn't have p-value
            critical_value=None,
            effect_size=cohens_d,
            confidence_interval=None,
            interpretation=interpretation,
            assumptions_met=True
        )

    def _assess_performance_risk(self, baseline: List[float], stressed: List[float],
                               degradation: List[float], metric_name: str,
                               test_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess performance risk based on degradation patterns"""

        degradation_array = np.array(degradation)
        baseline_array = np.array(baseline)

        # Risk factors
        risk_factors = {}

        # 1. Average degradation risk
        mean_degradation = np.mean(degradation_array)
        risk_factors['mean_degradation'] = mean_degradation

        # 2. Worst-case degradation
        max_degradation = np.max(degradation_array)
        risk_factors['max_degradation'] = max_degradation

        # 3. Proportion of severe degradations (>20% drop)
        severe_degradations = np.sum(degradation_array > 0.2) / len(degradation_array)
        risk_factors['severe_degradation_rate'] = severe_degradations

        # 4. Variability in degradation
        degradation_cv = np.std(degradation_array) / (np.mean(degradation_array) + 1e-10)
        risk_factors['degradation_variability'] = degradation_cv

        # 5. Failure rate (performance below threshold)
        if metric_name == 'accuracy':
            failure_threshold = 0.5  # Below chance level
            failure_rate = np.sum(np.array(stressed) < failure_threshold) / len(stressed)
            risk_factors['failure_rate'] = failure_rate
        else:
            failure_rate = 0

        # Calculate overall risk score (0-1 scale)
        risk_weights = {
            'mean_degradation': 0.3,
            'max_degradation': 0.2,
            'severe_degradation_rate': 0.3,
            'degradation_variability': 0.1,
            'failure_rate': 0.1
        }

        risk_score = 0
        for factor, weight in risk_weights.items():
            normalized_factor = min(risk_factors.get(factor, 0), 1.0)  # Cap at 1.0
            risk_score += weight * normalized_factor

        # Determine risk level
        if risk_score < 0.2:
            risk_level = 'low'
        elif risk_score < 0.4:
            risk_level = 'medium'
        elif risk_score < 0.7:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        # Mitigation strategies
        mitigation_strategies = []

        if mean_degradation > 0.1:
            mitigation_strategies.append("Consider additional training data for robustness")

        if severe_degradations > 0.2:
            mitigation_strategies.append("Implement input validation and preprocessing")

        if degradation_cv > 0.5:
            mitigation_strategies.append("Investigate causes of inconsistent performance")

        if failure_rate > 0.05:
            mitigation_strategies.append("Add safety mechanisms for low-confidence predictions")

        # Test condition specific risks
        condition_type = test_conditions.get('test_type', 'unknown')
        if condition_type == 'noise' and mean_degradation > 0.15:
            mitigation_strategies.append("Implement noise reduction preprocessing")
        elif condition_type == 'electrode_dropout' and mean_degradation > 0.2:
            mitigation_strategies.append("Add electrode interpolation methods")

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mitigation_strategies': mitigation_strategies,
            'assessment_summary': f"{risk_level.title()} risk with {mean_degradation:.1%} average degradation"
        }

    def _store_statistical_analysis(self, analysis_id: str, stats: RobustnessStatistics):
        """Store statistical analysis results in database"""
        with sqlite3.connect(self.db_path) as conn:
            # Store confidence intervals
            for ci in stats.confidence_intervals:
                conn.execute("""
                    INSERT INTO confidence_intervals (
                        interval_id, analysis_id, metric_name, point_estimate,
                        lower_bound, upper_bound, confidence_level, method,
                        sample_size, std_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{analysis_id}_{ci.metric_name}_{ci.confidence_level}",
                    analysis_id,
                    ci.metric_name,
                    ci.point_estimate,
                    ci.lower_bound,
                    ci.upper_bound,
                    ci.confidence_level,
                    ci.method,
                    ci.sample_size,
                    ci.std_error
                ))

            # Store statistical tests
            for test in stats.statistical_tests:
                conn.execute("""
                    INSERT INTO statistical_tests (
                        test_id, analysis_id, test_name, statistic, p_value,
                        critical_value, effect_size, interpretation, assumptions_met
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{analysis_id}_{test.test_name}",
                    analysis_id,
                    test.test_name,
                    test.statistic,
                    test.p_value or 0,
                    test.critical_value,
                    test.effect_size,
                    test.interpretation,
                    test.assumptions_met
                ))

            # Store risk assessment
            risk = stats.risk_assessment
            conn.execute("""
                INSERT INTO risk_assessments (
                    assessment_id, analysis_id, risk_type, risk_level,
                    risk_score, risk_factors, mitigation_strategies
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{analysis_id}_{stats.metric_name}_risk",
                analysis_id,
                'performance_degradation',
                risk['risk_level'],
                risk['risk_score'],
                json.dumps(risk['risk_factors']),
                json.dumps(risk['mitigation_strategies'])
            ))

    def generate_statistical_report(self, analysis_id: str) -> Dict[str, Any]:
        """Generate comprehensive statistical report"""
        with sqlite3.connect(self.db_path) as conn:
            # Get confidence intervals
            cursor = conn.execute("""
                SELECT metric_name, confidence_level, point_estimate,
                       lower_bound, upper_bound, method, sample_size
                FROM confidence_intervals
                WHERE analysis_id = ?
                ORDER BY metric_name, confidence_level
            """, (analysis_id,))

            confidence_data = {}
            for row in cursor.fetchall():
                metric, conf_level, point_est, lower, upper, method, n = row
                if metric not in confidence_data:
                    confidence_data[metric] = []
                confidence_data[metric].append({
                    'confidence_level': conf_level,
                    'point_estimate': point_est,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'method': method,
                    'sample_size': n
                })

            # Get statistical tests
            cursor = conn.execute("""
                SELECT test_name, statistic, p_value, effect_size, interpretation
                FROM statistical_tests
                WHERE analysis_id = ?
                ORDER BY test_name
            """, (analysis_id,))

            test_results = []
            for row in cursor.fetchall():
                test_results.append({
                    'test_name': row[0],
                    'statistic': row[1],
                    'p_value': row[2],
                    'effect_size': row[3],
                    'interpretation': row[4]
                })

            # Get risk assessments
            cursor = conn.execute("""
                SELECT risk_type, risk_level, risk_score, risk_factors, mitigation_strategies
                FROM risk_assessments
                WHERE analysis_id = ?
            """, (analysis_id,))

            risk_assessments = []
            for row in cursor.fetchall():
                risk_assessments.append({
                    'risk_type': row[0],
                    'risk_level': row[1],
                    'risk_score': row[2],
                    'risk_factors': json.loads(row[3]) if row[3] else {},
                    'mitigation_strategies': json.loads(row[4]) if row[4] else []
                })

            return {
                'analysis_id': analysis_id,
                'confidence_intervals': confidence_data,
                'statistical_tests': test_results,
                'risk_assessments': risk_assessments,
                'generated_at': datetime.now().isoformat()
            }

    def plot_confidence_intervals(self, analysis_id: str, metric_name: str,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot confidence intervals for a specific metric"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT confidence_level, point_estimate, lower_bound, upper_bound, method
                FROM confidence_intervals
                WHERE analysis_id = ? AND metric_name = ?
                ORDER BY confidence_level
            """, (analysis_id, metric_name))

            data = cursor.fetchall()

        if not data:
            raise ValueError(f"No confidence interval data found for {metric_name}")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot confidence intervals
        confidence_levels = [row[0] for row in data]
        point_estimates = [row[1] for row in data]
        lower_bounds = [row[2] for row in data]
        upper_bounds = [row[3] for row in data]
        methods = [row[4] for row in data]

        # Create error bars
        errors_lower = [pe - lb for pe, lb in zip(point_estimates, lower_bounds)]
        errors_upper = [ub - pe for pe, ub in zip(point_estimates, upper_bounds)]

        bars = ax.errorbar(confidence_levels, point_estimates,
                          yerr=[errors_lower, errors_upper],
                          fmt='o-', capsize=5, capthick=2, linewidth=2)

        ax.set_xlabel('Confidence Level')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
        ax.set_title(f'Confidence Intervals for {metric_name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

        # Add method annotations
        for i, (cl, pe, method) in enumerate(zip(confidence_levels, point_estimates, methods)):
            ax.annotate(method, (cl, pe), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig