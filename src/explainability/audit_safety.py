"""
Audit and Safety Features for Explainability Layer
Ensures regulatory compliance, data integrity, and safety for clinical deployment
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for explanation quality"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

class AuditEventType(Enum):
    """Types of audit events"""
    EXPLANATION_GENERATED = "explanation_generated"
    EXPLANATION_ACCESSED = "explanation_accessed"
    EXPLANATION_EXPORTED = "explanation_exported"
    EXPLANATION_MODIFIED = "explanation_modified"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    DATA_INTEGRITY_VIOLATION = "data_integrity_violation"
    THRESHOLD_VIOLATION = "threshold_violation"
    MODEL_DRIFT_DETECTED = "model_drift_detected"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

@dataclass
class SafetyThresholds:
    """Safety thresholds for explanation quality"""
    min_confidence: float = 0.5
    max_attribution_variance: float = 1.0
    min_attribution_magnitude: float = 0.01
    max_attention_entropy: float = 5.0
    min_feature_coverage: float = 0.1
    max_explanation_time: float = 300.0  # seconds
    consistency_tolerance: float = 0.1

@dataclass
class ExplanationQualityMetrics:
    """Quality metrics for explanations"""
    confidence_score: float
    attribution_variance: float
    attribution_magnitude: float
    attention_entropy: Optional[float]
    feature_coverage: float
    explanation_time: float
    consistency_score: Optional[float]
    stability_score: Optional[float]
    faithfulness_score: Optional[float]

@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    explanation_id: Optional[str]
    model_id: Optional[str]
    subject_id: Optional[str]
    details: Dict[str, Any]
    severity: str
    ip_address: Optional[str]
    session_id: Optional[str]

class ExplanationSafetyValidator:
    """Validates explanation safety and quality"""

    def __init__(self, thresholds: SafetyThresholds = None):
        self.thresholds = thresholds or SafetyThresholds()

    def validate_explanation(self, explanation: Dict[str, Any]) -> Tuple[bool, List[str], ExplanationQualityMetrics]:
        """Validate explanation against safety thresholds"""
        violations = []

        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(explanation)

        # Check confidence threshold
        if metrics.confidence_score < self.thresholds.min_confidence:
            violations.append(f"Low confidence: {metrics.confidence_score:.3f} < {self.thresholds.min_confidence}")

        # Check attribution variance
        if metrics.attribution_variance > self.thresholds.max_attribution_variance:
            violations.append(f"High attribution variance: {metrics.attribution_variance:.3f} > {self.thresholds.max_attribution_variance}")

        # Check attribution magnitude
        if metrics.attribution_magnitude < self.thresholds.min_attribution_magnitude:
            violations.append(f"Low attribution magnitude: {metrics.attribution_magnitude:.3f} < {self.thresholds.min_attribution_magnitude}")

        # Check attention entropy (if applicable)
        if metrics.attention_entropy is not None and metrics.attention_entropy > self.thresholds.max_attention_entropy:
            violations.append(f"High attention entropy: {metrics.attention_entropy:.3f} > {self.thresholds.max_attention_entropy}")

        # Check feature coverage
        if metrics.feature_coverage < self.thresholds.min_feature_coverage:
            violations.append(f"Low feature coverage: {metrics.feature_coverage:.3f} < {self.thresholds.min_feature_coverage}")

        # Check explanation time
        if metrics.explanation_time > self.thresholds.max_explanation_time:
            violations.append(f"Explanation timeout: {metrics.explanation_time:.1f}s > {self.thresholds.max_explanation_time}s")

        is_safe = len(violations) == 0

        return is_safe, violations, metrics

    def _calculate_quality_metrics(self, explanation: Dict[str, Any]) -> ExplanationQualityMetrics:
        """Calculate quality metrics from explanation"""
        confidence_score = explanation.get('prediction', {}).get('confidence', 0.0)

        # Initialize metrics
        attribution_variance = 0.0
        attribution_magnitude = 0.0
        attention_entropy = None
        feature_coverage = 0.0

        # Calculate attribution metrics
        if 'attributions' in explanation:
            attr_data = explanation['attributions']

            if 'feature_attributions' in attr_data:
                all_attributions = []
                for method, values in attr_data['feature_attributions'].items():
                    if isinstance(values, np.ndarray):
                        all_attributions.extend(values.flatten())
                    elif isinstance(values, list):
                        all_attributions.extend(values)

                if all_attributions:
                    all_attributions = np.array(all_attributions)
                    attribution_variance = float(np.var(all_attributions))
                    attribution_magnitude = float(np.mean(np.abs(all_attributions)))

                    # Feature coverage (proportion of non-zero attributions)
                    feature_coverage = float(np.mean(np.abs(all_attributions) > 1e-6))

        elif 'attribution' in explanation:
            # For CNN/Grad-CAM explanations
            if 'cam' in explanation['attribution']:
                cam = explanation['attribution']['cam']
                if isinstance(cam, np.ndarray):
                    attribution_variance = float(np.var(cam))
                    attribution_magnitude = float(np.mean(np.abs(cam)))
                    feature_coverage = float(np.mean(np.abs(cam) > 1e-6))

        # Calculate attention entropy (for transformer explanations)
        if 'attention' in explanation:
            attention_data = explanation['attention']
            if 'statistics' in attention_data:
                attention_entropy = attention_data['statistics'].get('attention_entropy')

        return ExplanationQualityMetrics(
            confidence_score=confidence_score,
            attribution_variance=attribution_variance,
            attribution_magnitude=attribution_magnitude,
            attention_entropy=attention_entropy,
            feature_coverage=feature_coverage,
            explanation_time=0.0,  # Would be set during generation
            consistency_score=None,  # Would require multiple runs
            stability_score=None,    # Would require perturbation testing
            faithfulness_score=None  # Would require model occlusion tests
        )

    def get_safety_level(self, violations: List[str]) -> SafetyLevel:
        """Determine safety level based on violations"""
        if not violations:
            return SafetyLevel.HIGH
        elif len(violations) == 1:
            return SafetyLevel.MEDIUM
        elif len(violations) <= 3:
            return SafetyLevel.LOW
        else:
            return SafetyLevel.CRITICAL

class ExplanationAuditor:
    """Comprehensive auditing system for explanations"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.validator = ExplanationSafetyValidator()
        self._init_audit_database()

    def _init_audit_database(self):
        """Initialize audit database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Audit events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT NOT NULL,
                    explanation_id TEXT,
                    model_id TEXT,
                    subject_id TEXT,
                    details TEXT, -- JSON
                    severity TEXT CHECK(severity IN ('info', 'warning', 'error', 'critical')),
                    ip_address TEXT,
                    session_id TEXT
                )
            """)

            # Safety violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_safety_violations (
                    violation_id TEXT PRIMARY KEY,
                    explanation_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    violation_message TEXT NOT NULL,
                    severity TEXT CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
                )
            """)

            # Data integrity checks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_integrity_checks (
                    check_id TEXT PRIMARY KEY,
                    explanation_id TEXT NOT NULL,
                    check_type TEXT NOT NULL, -- 'hash_verification', 'format_validation', 'completeness_check'
                    check_status TEXT CHECK(check_status IN ('passed', 'failed', 'warning')),
                    check_details TEXT, -- JSON
                    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
                )
            """)

            # Model drift monitoring
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_drift_monitoring (
                    drift_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    baseline_period_start TIMESTAMP,
                    baseline_period_end TIMESTAMP,
                    current_period_start TIMESTAMP,
                    current_period_end TIMESTAMP,
                    drift_score REAL,
                    drift_threshold REAL,
                    drift_detected BOOLEAN,
                    drift_type TEXT, -- 'confidence_drift', 'attribution_drift', 'attention_drift'
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON explanation_audit_events (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_user ON explanation_audit_events (user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_explanation ON explanation_audit_events (explanation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_explanation ON explanation_safety_violations (explanation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_severity ON explanation_safety_violations (severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_integrity_explanation ON explanation_integrity_checks (explanation_id)")

    def audit_explanation_generation(self, explanation: Dict[str, Any],
                                   user_id: str, execution_time: float,
                                   session_id: Optional[str] = None,
                                   ip_address: Optional[str] = None) -> str:
        """Audit explanation generation with safety validation"""
        explanation_id = explanation.get('metadata', {}).get('explanation_id', str(uuid.uuid4()))

        # Validate explanation safety
        is_safe, violations, metrics = self.validator.validate_explanation(explanation)
        safety_level = self.validator.get_safety_level(violations)

        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.EXPLANATION_GENERATED,
            timestamp=datetime.now(),
            user_id=user_id,
            explanation_id=explanation_id,
            model_id=explanation.get('metadata', {}).get('model_id'),
            subject_id=explanation.get('subject_id'),
            details={
                'execution_time': execution_time,
                'safety_level': safety_level.value,
                'quality_metrics': asdict(metrics),
                'violations_count': len(violations)
            },
            severity='info' if is_safe else 'warning',
            ip_address=ip_address,
            session_id=session_id
        )

        self._store_audit_event(event)

        # Store safety violations if any
        if violations:
            self._store_safety_violations(explanation_id, violations, safety_level)

        # Perform integrity checks
        self._perform_integrity_checks(explanation_id, explanation)

        return event.event_id

    def audit_explanation_access(self, explanation_id: str, user_id: str,
                               session_id: Optional[str] = None,
                               ip_address: Optional[str] = None) -> str:
        """Audit explanation access"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.EXPLANATION_ACCESSED,
            timestamp=datetime.now(),
            user_id=user_id,
            explanation_id=explanation_id,
            model_id=None,
            subject_id=None,
            details={'access_method': 'api'},
            severity='info',
            ip_address=ip_address,
            session_id=session_id
        )

        self._store_audit_event(event)
        return event.event_id

    def audit_explanation_export(self, explanation_id: str, export_format: str,
                               user_id: str, export_path: Optional[str] = None,
                               session_id: Optional[str] = None,
                               ip_address: Optional[str] = None) -> str:
        """Audit explanation export"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.EXPLANATION_EXPORTED,
            timestamp=datetime.now(),
            user_id=user_id,
            explanation_id=explanation_id,
            model_id=None,
            subject_id=None,
            details={
                'export_format': export_format,
                'export_path': export_path
            },
            severity='info',
            ip_address=ip_address,
            session_id=session_id
        )

        self._store_audit_event(event)
        return event.event_id

    def detect_model_drift(self, model_id: str, explanation_ids: List[str],
                          drift_threshold: float = 0.1) -> bool:
        """Detect drift in model explanations"""
        # This would implement statistical tests for drift detection
        # For now, return placeholder implementation

        drift_id = str(uuid.uuid4())
        current_time = datetime.now()

        # Placeholder drift detection (would use actual statistical tests)
        drift_score = np.random.random()  # Replace with real calculation
        drift_detected = drift_score > drift_threshold

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO explanation_drift_monitoring (
                    drift_id, model_id, current_period_start, current_period_end,
                    drift_score, drift_threshold, drift_detected, drift_type, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                drift_id, model_id, current_time - timedelta(days=7), current_time,
                drift_score, drift_threshold, drift_detected, 'confidence_drift', current_time
            ))

        if drift_detected:
            # Create audit event for drift detection
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.MODEL_DRIFT_DETECTED,
                timestamp=current_time,
                user_id='system',
                explanation_id=None,
                model_id=model_id,
                subject_id=None,
                details={
                    'drift_score': drift_score,
                    'drift_threshold': drift_threshold,
                    'drift_id': drift_id
                },
                severity='warning',
                ip_address=None,
                session_id=None
            )
            self._store_audit_event(event)

        return drift_detected

    def get_audit_report(self, start_date: datetime, end_date: datetime,
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        with sqlite3.connect(self.db_path) as conn:
            # Get event counts by type
            cursor = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM explanation_audit_events
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY event_type
            """, (start_date, end_date))

            event_counts = dict(cursor.fetchall())

            # Get safety violations
            cursor = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM explanation_safety_violations
                WHERE detected_at BETWEEN ? AND ?
                GROUP BY severity
            """, (start_date, end_date))

            violation_counts = dict(cursor.fetchall())

            # Get integrity check results
            cursor = conn.execute("""
                SELECT check_status, COUNT(*) as count
                FROM explanation_integrity_checks
                WHERE performed_at BETWEEN ? AND ?
                GROUP BY check_status
            """, (start_date, end_date))

            integrity_counts = dict(cursor.fetchall())

            # Get drift detection results
            cursor = conn.execute("""
                SELECT drift_detected, COUNT(*) as count
                FROM explanation_drift_monitoring
                WHERE detected_at BETWEEN ? AND ?
                GROUP BY drift_detected
            """, (start_date, end_date))

            drift_counts = dict(cursor.fetchall())

        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'event_summary': {
                'total_events': sum(event_counts.values()),
                'by_type': event_counts
            },
            'safety_violations': {
                'total_violations': sum(violation_counts.values()),
                'by_severity': violation_counts
            },
            'integrity_checks': {
                'total_checks': sum(integrity_counts.values()),
                'by_status': integrity_counts
            },
            'drift_detection': {
                'total_checks': sum(drift_counts.values()),
                'drift_detected': drift_counts.get(True, 0),
                'no_drift': drift_counts.get(False, 0)
            },
            'compliance_score': self._calculate_compliance_score(violation_counts, integrity_counts)
        }

        return report

    def _store_audit_event(self, event: AuditEvent):
        """Store audit event in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO explanation_audit_events (
                    event_id, event_type, timestamp, user_id, explanation_id,
                    model_id, subject_id, details, severity, ip_address, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp,
                event.user_id,
                event.explanation_id,
                event.model_id,
                event.subject_id,
                json.dumps(event.details),
                event.severity,
                event.ip_address,
                event.session_id
            ))

    def _store_safety_violations(self, explanation_id: str, violations: List[str],
                               safety_level: SafetyLevel):
        """Store safety violations"""
        with sqlite3.connect(self.db_path) as conn:
            for violation in violations:
                violation_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO explanation_safety_violations (
                        violation_id, explanation_id, violation_type, violation_message, severity
                    ) VALUES (?, ?, ?, ?, ?)
                """, (violation_id, explanation_id, 'threshold_violation', violation, safety_level.value))

    def _perform_integrity_checks(self, explanation_id: str, explanation: Dict[str, Any]):
        """Perform data integrity checks"""
        checks = []

        # Hash verification
        if 'metadata' in explanation and 'data_hash' in explanation['metadata']:
            checks.append(('hash_verification', 'passed', {'hash_present': True}))
        else:
            checks.append(('hash_verification', 'failed', {'hash_present': False}))

        # Format validation
        required_fields = ['subject_id', 'prediction']
        missing_fields = [field for field in required_fields if field not in explanation]

        if missing_fields:
            checks.append(('format_validation', 'failed', {'missing_fields': missing_fields}))
        else:
            checks.append(('format_validation', 'passed', {'complete': True}))

        # Completeness check
        has_attribution = 'attributions' in explanation or 'attribution' in explanation or 'attention' in explanation
        checks.append(('completeness_check', 'passed' if has_attribution else 'warning',
                      {'has_attribution': has_attribution}))

        # Store checks
        with sqlite3.connect(self.db_path) as conn:
            for check_type, status, details in checks:
                check_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO explanation_integrity_checks (
                        check_id, explanation_id, check_type, check_status, check_details
                    ) VALUES (?, ?, ?, ?, ?)
                """, (check_id, explanation_id, check_type, status, json.dumps(details)))

    def _calculate_compliance_score(self, violation_counts: Dict, integrity_counts: Dict) -> float:
        """Calculate overall compliance score"""
        total_violations = sum(violation_counts.values())
        total_integrity_checks = sum(integrity_counts.values())

        if total_integrity_checks == 0:
            return 1.0

        # Weight violations by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0}
        weighted_violations = sum(
            violation_counts.get(severity, 0) * weight
            for severity, weight in severity_weights.items()
        )

        # Calculate score (higher is better)
        max_possible_violations = total_integrity_checks
        if max_possible_violations == 0:
            return 1.0

        score = 1.0 - (weighted_violations / max_possible_violations)
        return max(0.0, min(1.0, score))

@contextmanager
def explanation_audit_context(auditor: ExplanationAuditor, user_id: str,
                             session_id: Optional[str] = None,
                             ip_address: Optional[str] = None):
    """Context manager for audited explanation operations"""
    start_time = datetime.now()

    try:
        yield auditor
    except Exception as e:
        # Log error event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SAFETY_CHECK_FAILED,
            timestamp=datetime.now(),
            user_id=user_id,
            explanation_id=None,
            model_id=None,
            subject_id=None,
            details={'error': str(e), 'execution_time': (datetime.now() - start_time).total_seconds()},
            severity='error',
            ip_address=ip_address,
            session_id=session_id
        )
        auditor._store_audit_event(event)
        raise
    finally:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Explanation operation completed in {execution_time:.2f}s")