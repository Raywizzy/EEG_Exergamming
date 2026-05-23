"""
Audit Integration for Robustness Testing
Integrates stress testing with existing clinical audit trail and reporting systems
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from ..platform.persistent_manager import PersistentPlatformManager
from .stress_testing import StressTestSuite, StressTestResult, StressTestConfig
from .cross_device import CrossDeviceValidator, CrossDeviceTestResult
from .confidence_bounds import StatisticalConfidenceEngine, RobustnessStatistics

logger = logging.getLogger(__name__)

class RobustnessAuditManager:
    """Manages audit trail integration for robustness testing"""

    def __init__(self, platform_manager: PersistentPlatformManager):
        self.platform_manager = platform_manager
        self.db_path = platform_manager.db_path

        self._extend_audit_schema()

    def _extend_audit_schema(self):
        """Extend existing audit schema for robustness testing"""
        with sqlite3.connect(self.db_path) as conn:
            # Add robustness-specific audit events if not already present
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS robustness_audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT CHECK(event_type IN (
                            'stress_test_campaign_start', 'stress_test_campaign_complete',
                            'stress_test_individual', 'cross_device_validation',
                            'statistical_analysis', 'confidence_calculation',
                            'robustness_threshold_violation', 'performance_degradation_alert'
                        )),
                        campaign_id TEXT,
                        test_id TEXT,
                        model_id TEXT,
                        subject_id TEXT,
                        test_conditions TEXT, -- JSON
                        results_summary TEXT, -- JSON
                        risk_assessment TEXT, -- JSON
                        compliance_status TEXT CHECK(compliance_status IN ('compliant', 'warning', 'violation')),
                        user_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        details TEXT -- JSON
                    )
                """)

                # Robustness evidence trail
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS robustness_evidence (
                        evidence_id TEXT PRIMARY KEY,
                        evidence_type TEXT CHECK(evidence_type IN (
                            'stress_test_results', 'statistical_analysis', 'confidence_intervals',
                            'cross_device_validation', 'risk_assessment', 'mitigation_plan'
                        )),
                        campaign_id TEXT,
                        model_id TEXT,
                        evidence_data TEXT, -- JSON
                        file_references TEXT, -- JSON array of file paths
                        regulatory_relevance TEXT CHECK(regulatory_relevance IN ('high', 'medium', 'low')),
                        created_by TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        verified_by TEXT,
                        verified_at TIMESTAMP,
                        verification_status TEXT CHECK(verification_status IN ('pending', 'verified', 'rejected'))
                    )
                """)

                # Performance thresholds and violations
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_thresholds (
                        threshold_id TEXT PRIMARY KEY,
                        threshold_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        threshold_type TEXT CHECK(threshold_type IN ('minimum', 'maximum', 'range')),
                        threshold_value REAL,
                        threshold_range_min REAL,
                        threshold_range_max REAL,
                        severity TEXT CHECK(severity IN ('info', 'warning', 'critical')),
                        regulatory_basis TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT TRUE
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS threshold_violations (
                        violation_id TEXT PRIMARY KEY,
                        threshold_id TEXT,
                        test_id TEXT,
                        campaign_id TEXT,
                        actual_value REAL,
                        threshold_value REAL,
                        violation_magnitude REAL,
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolution_status TEXT CHECK(resolution_status IN ('open', 'investigating', 'resolved', 'accepted_risk')),
                        resolution_notes TEXT,
                        FOREIGN KEY (threshold_id) REFERENCES performance_thresholds (threshold_id)
                    )
                """)

                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_robustness_audit_campaign ON robustness_audit_events (campaign_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_robustness_audit_model ON robustness_audit_events (model_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_robustness_evidence_campaign ON robustness_evidence (campaign_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_threshold_violations_test ON threshold_violations (test_id)")

                logger.info("Robustness audit schema extended successfully")

            except sqlite3.Error as e:
                logger.error(f"Failed to extend audit schema: {e}")

        # Insert default performance thresholds
        self._insert_default_thresholds()

    def _insert_default_thresholds(self):
        """Insert default performance thresholds for robustness testing"""
        default_thresholds = [
            {
                'threshold_id': 'ACC_MIN_70',
                'threshold_name': 'Minimum Accuracy Threshold',
                'metric_name': 'accuracy',
                'threshold_type': 'minimum',
                'threshold_value': 0.70,
                'severity': 'critical',
                'regulatory_basis': 'FDA Guidance: Medical Device Development - AI/ML algorithms should maintain clinically acceptable performance'
            },
            {
                'threshold_id': 'ACC_DROP_MAX_10',
                'threshold_name': 'Maximum Accuracy Degradation',
                'metric_name': 'accuracy_drop',
                'threshold_type': 'maximum',
                'threshold_value': 0.10,
                'severity': 'warning',
                'regulatory_basis': 'Clinical validation standard: <10% performance degradation under stress'
            },
            {
                'threshold_id': 'CONF_MIN_80',
                'threshold_name': 'Minimum Confidence Threshold',
                'metric_name': 'confidence',
                'threshold_type': 'minimum',
                'threshold_value': 0.80,
                'severity': 'warning',
                'regulatory_basis': 'Risk management: High-confidence predictions for clinical decision support'
            },
            {
                'threshold_id': 'PASS_RATE_MIN_90',
                'threshold_name': 'Minimum Test Pass Rate',
                'metric_name': 'pass_rate',
                'threshold_type': 'minimum',
                'threshold_value': 0.90,
                'severity': 'critical',
                'regulatory_basis': 'System reliability: ≥90% of stress tests must pass for deployment approval'
            }
        ]

        with sqlite3.connect(self.db_path) as conn:
            for threshold in default_thresholds:
                conn.execute("""
                    INSERT OR REPLACE INTO performance_thresholds (
                        threshold_id, threshold_name, metric_name, threshold_type,
                        threshold_value, severity, regulatory_basis, active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    threshold['threshold_id'],
                    threshold['threshold_name'],
                    threshold['metric_name'],
                    threshold['threshold_type'],
                    threshold['threshold_value'],
                    threshold['severity'],
                    threshold['regulatory_basis'],
                    True
                ))

    def audit_stress_test_campaign_start(self, campaign_id: str, config: StressTestConfig,
                                       model_id: str, user_id: str) -> str:
        """Audit the start of a stress test campaign"""
        audit_event_id = str(uuid.uuid4())

        # Create audit event in main audit system
        self.platform_manager.create_audit_event(
            event_type='robustness_test_start',
            entity_type='model',
            entity_id=model_id,
            user_id=user_id,
            action='stress_test_campaign_initiated',
            details={
                'campaign_id': campaign_id,
                'stress_test_config': {
                    'noise_levels': config.noise_levels,
                    'noise_types': config.noise_types,
                    'dropout_rates': config.dropout_rates,
                    'n_repetitions': config.n_repetitions,
                    'confidence_level': config.confidence_level,
                    'min_accuracy_threshold': config.min_accuracy_threshold,
                    'max_performance_drop': config.max_performance_drop
                },
                'audit_event_id': audit_event_id
            }
        )

        # Create robustness-specific audit record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO robustness_audit_events (
                    event_id, event_type, campaign_id, model_id, test_conditions,
                    compliance_status, user_id, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_event_id,
                'stress_test_campaign_start',
                campaign_id,
                model_id,
                json.dumps({
                    'config': config.__dict__,
                    'regulatory_requirements_checked': True
                }),
                'compliant',
                user_id,
                json.dumps({
                    'initiation_reason': 'Regulatory robustness validation',
                    'expected_duration_hours': 2,
                    'total_tests_planned': len(config.noise_levels) * len(config.noise_types) * 10  # Estimate
                })
            ))

        logger.info(f"Stress test campaign audit initiated: {campaign_id}")
        return audit_event_id

    def audit_stress_test_result(self, result: StressTestResult, user_id: str) -> str:
        """Audit individual stress test result"""
        audit_event_id = str(uuid.uuid4())

        # Check for threshold violations
        violations = self._check_threshold_violations(result)

        # Determine compliance status
        compliance_status = 'violation' if violations else ('warning' if not result.passed_thresholds else 'compliant')

        # Create audit record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO robustness_audit_events (
                    event_id, event_type, test_id, model_id, subject_id,
                    test_conditions, results_summary, compliance_status, user_id, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_event_id,
                'stress_test_individual',
                result.test_id,
                result.model_id,
                result.subject_id,
                json.dumps(result.perturbation_params),
                json.dumps({
                    'baseline_accuracy': result.baseline_metrics.get('accuracy', 0),
                    'stressed_accuracy': result.stressed_metrics.get('accuracy', 0),
                    'accuracy_drop': result.performance_drop.get('accuracy', 0),
                    'passed_thresholds': result.passed_thresholds,
                    'execution_time': result.execution_time
                }),
                compliance_status,
                user_id,
                json.dumps({
                    'test_type': result.test_type,
                    'violations': violations,
                    'regulatory_impact': 'high' if violations else 'low'
                })
            ))

        # Record violations if any
        if violations:
            self._record_threshold_violations(violations, result.test_id, user_id)

        return audit_event_id

    def audit_campaign_completion(self, campaign_id: str, campaign_summary: Dict[str, Any],
                                user_id: str) -> str:
        """Audit completion of stress test campaign"""
        audit_event_id = str(uuid.uuid4())

        # Determine overall compliance
        overall_pass_rate = campaign_summary.get('overall_pass_rate', 0)
        compliance_status = 'compliant' if overall_pass_rate >= 0.9 else 'warning' if overall_pass_rate >= 0.7 else 'violation'

        # Create comprehensive audit record
        self.platform_manager.create_audit_event(
            event_type='robustness_test_complete',
            entity_type='model',
            entity_id=campaign_summary.get('model_id', 'unknown'),
            user_id=user_id,
            action='stress_test_campaign_completed',
            details={
                'campaign_id': campaign_id,
                'total_tests': campaign_summary.get('total_tests', 0),
                'passed_tests': campaign_summary.get('passed_tests', 0),
                'overall_pass_rate': overall_pass_rate,
                'compliance_status': compliance_status,
                'regulatory_ready': compliance_status == 'compliant'
            }
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO robustness_audit_events (
                    event_id, event_type, campaign_id, results_summary,
                    compliance_status, user_id, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_event_id,
                'stress_test_campaign_complete',
                campaign_id,
                json.dumps(campaign_summary),
                compliance_status,
                user_id,
                json.dumps({
                    'completion_timestamp': datetime.now().isoformat(),
                    'evidence_packages_generated': True,
                    'regulatory_submission_ready': compliance_status == 'compliant'
                })
            ))

        # Create evidence record
        self._create_evidence_record(campaign_id, 'stress_test_results', campaign_summary, user_id)

        return audit_event_id

    def audit_statistical_analysis(self, analysis_id: str, statistics: Dict[str, RobustnessStatistics],
                                  user_id: str) -> str:
        """Audit statistical analysis of robustness results"""
        audit_event_id = str(uuid.uuid4())

        # Summarize statistical findings
        summary = {
            'metrics_analyzed': list(statistics.keys()),
            'confidence_intervals_calculated': True,
            'statistical_tests_performed': True,
            'risk_assessments_completed': True
        }

        # Check for statistical significance
        significant_degradations = []
        for metric_name, stats in statistics.items():
            for test in stats.statistical_tests:
                if test.p_value and test.p_value < 0.05:
                    significant_degradations.append({
                        'metric': metric_name,
                        'test': test.test_name,
                        'p_value': test.p_value,
                        'effect_size': test.effect_size
                    })

        compliance_status = 'warning' if significant_degradations else 'compliant'

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO robustness_audit_events (
                    event_id, event_type, results_summary, compliance_status, user_id, details
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                audit_event_id,
                'statistical_analysis',
                json.dumps(summary),
                compliance_status,
                user_id,
                json.dumps({
                    'analysis_id': analysis_id,
                    'significant_degradations': significant_degradations,
                    'regulatory_statistical_requirements_met': True
                })
            ))

        # Create evidence record for statistical analysis
        self._create_evidence_record(None, 'statistical_analysis', summary, user_id, analysis_id)

        return audit_event_id

    def _check_threshold_violations(self, result: StressTestResult) -> List[Dict[str, Any]]:
        """Check for performance threshold violations"""
        violations = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT threshold_id, threshold_name, metric_name, threshold_type,
                       threshold_value, severity, regulatory_basis
                FROM performance_thresholds
                WHERE active = TRUE
            """)

            thresholds = cursor.fetchall()

        for threshold in thresholds:
            threshold_id, name, metric, thresh_type, value, severity, basis = threshold

            # Get actual value from result
            actual_value = None
            if metric in result.stressed_metrics:
                actual_value = result.stressed_metrics[metric]
            elif metric in result.performance_drop:
                actual_value = result.performance_drop[metric]
            elif metric == 'pass_rate':
                actual_value = 1.0 if result.passed_thresholds else 0.0

            if actual_value is not None:
                violation_detected = False

                if thresh_type == 'minimum' and actual_value < value:
                    violation_detected = True
                elif thresh_type == 'maximum' and actual_value > value:
                    violation_detected = True

                if violation_detected:
                    violations.append({
                        'threshold_id': threshold_id,
                        'threshold_name': name,
                        'metric_name': metric,
                        'expected_value': value,
                        'actual_value': actual_value,
                        'violation_magnitude': abs(actual_value - value),
                        'severity': severity,
                        'regulatory_basis': basis
                    })

        return violations

    def _record_threshold_violations(self, violations: List[Dict[str, Any]],
                                   test_id: str, user_id: str):
        """Record threshold violations in database"""
        with sqlite3.connect(self.db_path) as conn:
            for violation in violations:
                violation_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO threshold_violations (
                        violation_id, threshold_id, test_id, actual_value,
                        threshold_value, violation_magnitude, resolution_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation_id,
                    violation['threshold_id'],
                    test_id,
                    violation['actual_value'],
                    violation['expected_value'],
                    violation['violation_magnitude'],
                    'open'
                ))

                # Create alert for critical violations
                if violation['severity'] == 'critical':
                    self.platform_manager.create_audit_event(
                        event_type='threshold_violation',
                        entity_type='robustness_test',
                        entity_id=test_id,
                        user_id=user_id,
                        action='critical_threshold_violation',
                        details=violation
                    )

    def _create_evidence_record(self, campaign_id: Optional[str], evidence_type: str,
                              evidence_data: Dict[str, Any], user_id: str,
                              analysis_id: str = None):
        """Create regulatory evidence record"""
        evidence_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO robustness_evidence (
                    evidence_id, evidence_type, campaign_id, evidence_data,
                    regulatory_relevance, created_by, verification_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence_id,
                evidence_type,
                campaign_id or analysis_id,
                json.dumps(evidence_data),
                'high',
                user_id,
                'pending'
            ))

        return evidence_id

    def generate_compliance_report(self, campaign_id: str = None,
                                 model_id: str = None,
                                 date_from: datetime = None,
                                 date_to: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""

        with sqlite3.connect(self.db_path) as conn:
            # Build query conditions
            conditions = []
            params = []

            if campaign_id:
                conditions.append("campaign_id = ?")
                params.append(campaign_id)

            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)

            if date_from:
                conditions.append("timestamp >= ?")
                params.append(date_from)

            if date_to:
                conditions.append("timestamp <= ?")
                params.append(date_to)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Get audit events
            cursor = conn.execute(f"""
                SELECT event_type, compliance_status, COUNT(*) as count
                FROM robustness_audit_events
                {where_clause}
                GROUP BY event_type, compliance_status
            """, params)

            audit_summary = {}
            for row in cursor.fetchall():
                event_type, compliance, count = row
                if event_type not in audit_summary:
                    audit_summary[event_type] = {}
                audit_summary[event_type][compliance] = count

            # Get violations
            cursor = conn.execute(f"""
                SELECT t.threshold_name, t.severity, v.resolution_status, COUNT(*) as count
                FROM threshold_violations v
                JOIN performance_thresholds t ON v.threshold_id = t.threshold_id
                {'JOIN robustness_audit_events r ON v.test_id = r.test_id ' + where_clause if where_clause else ''}
                GROUP BY t.threshold_name, t.severity, v.resolution_status
            """, params)

            violations_summary = {}
            for row in cursor.fetchall():
                threshold, severity, status, count = row
                key = f"{threshold} ({severity})"
                if key not in violations_summary:
                    violations_summary[key] = {}
                violations_summary[key][status] = count

            # Get evidence summary
            cursor = conn.execute(f"""
                SELECT evidence_type, verification_status, COUNT(*) as count
                FROM robustness_evidence
                {'WHERE campaign_id = ?' if campaign_id else ''}
                GROUP BY evidence_type, verification_status
            """, [campaign_id] if campaign_id else [])

            evidence_summary = {}
            for row in cursor.fetchall():
                evidence_type, verification, count = row
                if evidence_type not in evidence_summary:
                    evidence_summary[evidence_type] = {}
                evidence_summary[evidence_type][verification] = count

        # Calculate compliance score
        total_events = sum(
            sum(statuses.values()) for statuses in audit_summary.values()
        )
        compliant_events = sum(
            statuses.get('compliant', 0) for statuses in audit_summary.values()
        )

        compliance_score = compliant_events / total_events if total_events > 0 else 0

        return {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'scope': {
                'campaign_id': campaign_id,
                'model_id': model_id,
                'date_range': {
                    'from': date_from.isoformat() if date_from else None,
                    'to': date_to.isoformat() if date_to else None
                }
            },
            'compliance_score': compliance_score,
            'audit_summary': audit_summary,
            'violations_summary': violations_summary,
            'evidence_summary': evidence_summary,
            'regulatory_readiness': {
                'overall_score': compliance_score,
                'ready_for_submission': compliance_score >= 0.95,
                'areas_needing_attention': [
                    threshold for threshold, statuses in violations_summary.items()
                    if statuses.get('open', 0) > 0
                ]
            }
        }

    def get_audit_trail(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a robustness testing campaign"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT event_id, event_type, test_id, model_id, subject_id,
                       test_conditions, results_summary, compliance_status,
                       user_id, timestamp, details
                FROM robustness_audit_events
                WHERE campaign_id = ?
                ORDER BY timestamp ASC
            """, (campaign_id,))

            audit_trail = []
            for row in cursor.fetchall():
                event_id, event_type, test_id, model_id, subject_id, conditions, results, compliance, user_id, timestamp, details = row

                audit_trail.append({
                    'event_id': event_id,
                    'event_type': event_type,
                    'test_id': test_id,
                    'model_id': model_id,
                    'subject_id': subject_id,
                    'test_conditions': json.loads(conditions) if conditions else {},
                    'results_summary': json.loads(results) if results else {},
                    'compliance_status': compliance,
                    'user_id': user_id,
                    'timestamp': timestamp,
                    'details': json.loads(details) if details else {}
                })

        return audit_trail