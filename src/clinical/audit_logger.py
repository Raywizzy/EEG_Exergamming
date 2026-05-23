#!/usr/bin/env python3
"""
Clinical Audit Logger - Phase VI
Regulatory-compliant logging system for clinical deployment

Features:
- Complete audit trail for FDA/EMA compliance
- Real-time logging with timestamps
- Structured logging for clinical review
- Error tracking and reporting
- Performance monitoring
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
import time

@dataclass
class AuditEvent:
    """Structured audit event for clinical compliance."""
    event_id: str
    timestamp: str
    event_type: str  # 'processing', 'qc', 'prediction', 'error', 'system'
    subject_id: str
    user_id: str
    session_id: str

    # Event details
    action: str
    component: str
    status: str  # 'started', 'completed', 'failed', 'warning'

    # Clinical context
    trial_id: Optional[str] = None
    site_id: Optional[str] = None

    # Technical details
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

    # Performance metrics
    processing_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    # Compliance fields
    pipeline_version: str = "Core15+_v1.0"
    audit_version: str = "v1.0_Phase_VI"
    regulatory_flags: Optional[List[str]] = None


class ClinicalAuditLogger:
    """Clinical-grade audit logging system."""

    def __init__(self, log_dir: str = "clinical_audit_logs",
                 trial_id: str = None, site_id: str = None):
        """Initialize clinical audit logger.

        Args:
            log_dir: Directory for audit logs
            trial_id: Clinical trial identifier
            site_id: Site identifier
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.trial_id = trial_id
        self.site_id = site_id
        self.session_id = self._generate_session_id()

        # Setup logging infrastructure
        self._setup_audit_logging()

        # Event tracking
        self.events = []
        self.event_lock = threading.Lock()

        # Performance tracking
        self.performance_metrics = {
            'session_start': datetime.now(),
            'events_logged': 0,
            'errors_logged': 0,
            'warnings_logged': 0,
            'processing_times': [],
            'memory_usage': []
        }

        # Log session start
        self.log_system_event(
            action="audit_session_start",
            component="audit_logger",
            status="started",
            parameters={
                'trial_id': self.trial_id,
                'site_id': self.site_id,
                'session_id': self.session_id,
                'log_directory': str(self.log_dir)
            }
        )

    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_component = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"SESSION_{timestamp}_{random_component}"

    def _setup_audit_logging(self):
        """Setup structured audit logging."""
        # Create audit log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audit_file = self.log_dir / f"clinical_audit_{timestamp}.log"

        # Setup logger
        self.logger = logging.getLogger('ClinicalAudit')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler for audit trail
        file_handler = logging.FileHandler(audit_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [AUDIT] %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - [AUDIT] %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create JSON audit log file
        self.json_log_file = self.log_dir / f"clinical_audit_{timestamp}.json"

        self.logger.info(f"Clinical audit logging initialized")
        self.logger.info(f"Audit file: {audit_file}")
        self.logger.info(f"JSON log: {self.json_log_file}")

    def log_processing_event(self, subject_id: str, action: str, component: str,
                           status: str, user_id: str = "system",
                           parameters: Dict[str, Any] = None,
                           results: Dict[str, Any] = None,
                           processing_time_ms: float = None,
                           memory_usage_mb: float = None) -> str:
        """Log a processing event.

        Args:
            subject_id: Subject identifier
            action: Action performed (e.g., 'preprocess', 'feature_extract')
            component: Component performing action (e.g., 'preprocessor')
            status: Event status ('started', 'completed', 'failed', 'warning')
            user_id: User identifier
            parameters: Processing parameters
            results: Processing results
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in MB

        Returns:
            Event ID for tracking
        """
        return self._log_event(
            event_type="processing",
            subject_id=subject_id,
            action=action,
            component=component,
            status=status,
            user_id=user_id,
            parameters=parameters,
            results=results,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb
        )

    def log_qc_event(self, subject_id: str, qc_results: Dict[str, Any],
                     user_id: str = "system") -> str:
        """Log a quality control event."""
        return self._log_event(
            event_type="qc",
            subject_id=subject_id,
            action="quality_control_assessment",
            component="qc_manager",
            status="completed" if qc_results.get('overall_pass', False) else "warning",
            user_id=user_id,
            results=qc_results,
            regulatory_flags=["quality_control_critical"] if not qc_results.get('overall_pass', False) else None
        )

    def log_prediction_event(self, subject_id: str, prediction: Dict[str, Any],
                           confidence: float, user_id: str = "system") -> str:
        """Log a prediction/classification event."""
        regulatory_flags = []

        if confidence < 0.7:
            regulatory_flags.append("low_confidence_prediction")
        if prediction.get('predicted_class') == 'PD_REAL':
            regulatory_flags.append("positive_prediction")

        return self._log_event(
            event_type="prediction",
            subject_id=subject_id,
            action="classification_prediction",
            component="ensemble_classifier",
            status="completed",
            user_id=user_id,
            results={
                'prediction': prediction,
                'confidence': confidence,
                'clinical_significance': 'high' if confidence > 0.8 else 'moderate'
            },
            regulatory_flags=regulatory_flags if regulatory_flags else None
        )

    def log_error_event(self, subject_id: str, component: str, error_type: str,
                       error_message: str, error_details: Dict[str, Any] = None,
                       user_id: str = "system") -> str:
        """Log an error event."""
        return self._log_event(
            event_type="error",
            subject_id=subject_id,
            action=f"error_{error_type}",
            component=component,
            status="failed",
            user_id=user_id,
            error_details={
                'error_type': error_type,
                'error_message': error_message,
                'error_details': error_details or {}
            },
            regulatory_flags=["processing_error"]
        )

    def log_system_event(self, action: str, component: str, status: str,
                        parameters: Dict[str, Any] = None,
                        user_id: str = "system") -> str:
        """Log a system-level event."""
        return self._log_event(
            event_type="system",
            subject_id="SYSTEM",
            action=action,
            component=component,
            status=status,
            user_id=user_id,
            parameters=parameters
        )

    def _log_event(self, event_type: str, subject_id: str, action: str,
                   component: str, status: str, user_id: str = "system",
                   **kwargs) -> str:
        """Internal method to log structured event."""

        # Generate event ID
        event_id = self._generate_event_id()

        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            subject_id=subject_id,
            user_id=user_id,
            session_id=self.session_id,
            action=action,
            component=component,
            status=status,
            trial_id=self.trial_id,
            site_id=self.site_id,
            **kwargs
        )

        # Thread-safe event logging
        with self.event_lock:
            self.events.append(event)
            self._write_event_to_log(event)
            self._update_performance_metrics(event)

        # Log to standard logger
        log_level = self._get_log_level(status)
        log_message = self._format_log_message(event)
        self.logger.log(log_level, log_message)

        return event_id

    def _generate_event_id(self) -> str:
        """Generate unique event identifier."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        event_count = len(self.events)
        return f"EVT_{timestamp}_{event_count:04d}"

    def _write_event_to_log(self, event: AuditEvent):
        """Write event to JSON audit log."""
        try:
            # Append to JSON log file
            with open(self.json_log_file, 'a') as f:
                json.dump(asdict(event), f)
                f.write('\n')  # Newline-delimited JSON
        except Exception as e:
            self.logger.error(f"Failed to write audit event to JSON log: {str(e)}")

    def _update_performance_metrics(self, event: AuditEvent):
        """Update performance tracking metrics."""
        self.performance_metrics['events_logged'] += 1

        if event.status == 'failed':
            self.performance_metrics['errors_logged'] += 1
        elif event.status == 'warning':
            self.performance_metrics['warnings_logged'] += 1

        if event.processing_time_ms:
            self.performance_metrics['processing_times'].append(event.processing_time_ms)

        if event.memory_usage_mb:
            self.performance_metrics['memory_usage'].append(event.memory_usage_mb)

    def _get_log_level(self, status: str) -> int:
        """Get logging level based on event status."""
        level_map = {
            'started': logging.INFO,
            'completed': logging.INFO,
            'warning': logging.WARNING,
            'failed': logging.ERROR
        }
        return level_map.get(status, logging.INFO)

    def _format_log_message(self, event: AuditEvent) -> str:
        """Format event for standard log output."""
        return (f"{event.event_type.upper()}: {event.subject_id} - "
                f"{event.component}.{event.action} - {event.status.upper()} "
                f"[{event.event_id}]")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current audit session."""
        session_duration = (datetime.now() - self.performance_metrics['session_start']).total_seconds()

        # Calculate processing statistics
        processing_times = self.performance_metrics['processing_times']
        memory_usage = self.performance_metrics['memory_usage']

        return {
            'session_info': {
                'session_id': self.session_id,
                'trial_id': self.trial_id,
                'site_id': self.site_id,
                'start_time': self.performance_metrics['session_start'].isoformat(),
                'duration_seconds': session_duration
            },
            'event_statistics': {
                'total_events': self.performance_metrics['events_logged'],
                'errors': self.performance_metrics['errors_logged'],
                'warnings': self.performance_metrics['warnings_logged'],
                'success_rate': self._calculate_success_rate()
            },
            'performance_metrics': {
                'avg_processing_time_ms': sum(processing_times) / len(processing_times) if processing_times else 0,
                'max_processing_time_ms': max(processing_times) if processing_times else 0,
                'avg_memory_usage_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'max_memory_usage_mb': max(memory_usage) if memory_usage else 0
            },
            'compliance_metrics': {
                'audit_events_logged': len(self.events),
                'regulatory_flags_raised': self._count_regulatory_flags(),
                'audit_log_integrity': self._verify_audit_integrity()
            }
        }

    def _calculate_success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.performance_metrics['events_logged'] == 0:
            return 1.0

        failed_events = self.performance_metrics['errors_logged']
        total_events = self.performance_metrics['events_logged']

        return (total_events - failed_events) / total_events

    def _count_regulatory_flags(self) -> int:
        """Count regulatory flags across all events."""
        flag_count = 0
        for event in self.events:
            if event.regulatory_flags:
                flag_count += len(event.regulatory_flags)
        return flag_count

    def _verify_audit_integrity(self) -> bool:
        """Verify audit log integrity."""
        try:
            # Check if JSON log file exists and is readable
            if not self.json_log_file.exists():
                return False

            # Verify event count matches
            with open(self.json_log_file, 'r') as f:
                line_count = sum(1 for line in f if line.strip())

            return line_count == len(self.events)

        except:
            return False

    def generate_audit_report(self, output_dir: Path = None) -> str:
        """Generate comprehensive audit report for regulatory compliance."""
        if output_dir is None:
            output_dir = self.log_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate report timestamp
        report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Collect report data
        session_summary = self.get_session_summary()

        # Generate subject-level summaries
        subject_summaries = self._generate_subject_summaries()

        # Generate compliance checklist
        compliance_checklist = self._generate_compliance_checklist()

        # Create comprehensive report
        audit_report = {
            'report_header': {
                'report_type': 'Clinical Audit Report',
                'generation_timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'trial_id': self.trial_id,
                'site_id': self.site_id,
                'pipeline_version': 'Core15+_v1.0',
                'audit_version': 'v1.0_Phase_VI'
            },
            'session_summary': session_summary,
            'subject_summaries': subject_summaries,
            'compliance_checklist': compliance_checklist,
            'regulatory_findings': self._generate_regulatory_findings(),
            'recommendations': self._generate_recommendations()
        }

        # Save audit report
        report_file = output_dir / f"clinical_audit_report_{report_timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(audit_report, f, indent=2)

        # Generate human-readable summary
        summary_file = output_dir / f"audit_summary_{report_timestamp}.txt"
        self._generate_readable_summary(audit_report, summary_file)

        self.logger.info(f"Audit report generated: {report_file}")
        return str(report_file)

    def _generate_subject_summaries(self) -> Dict[str, Any]:
        """Generate per-subject processing summaries."""
        subject_data = {}

        for event in self.events:
            if event.subject_id != "SYSTEM":
                if event.subject_id not in subject_data:
                    subject_data[event.subject_id] = {
                        'events': [],
                        'processing_successful': True,
                        'qc_passed': None,
                        'prediction_confidence': None,
                        'regulatory_flags': []
                    }

                subject_data[event.subject_id]['events'].append({
                    'timestamp': event.timestamp,
                    'action': event.action,
                    'component': event.component,
                    'status': event.status
                })

                if event.status == 'failed':
                    subject_data[event.subject_id]['processing_successful'] = False

                if event.event_type == 'qc' and event.results:
                    subject_data[event.subject_id]['qc_passed'] = event.results.get('overall_pass')

                if event.event_type == 'prediction' and event.results:
                    subject_data[event.subject_id]['prediction_confidence'] = event.results.get('confidence')

                if event.regulatory_flags:
                    subject_data[event.subject_id]['regulatory_flags'].extend(event.regulatory_flags)

        return subject_data

    def _generate_compliance_checklist(self) -> Dict[str, bool]:
        """Generate regulatory compliance checklist."""
        return {
            'audit_logging_enabled': True,
            'event_timestamps_recorded': all(event.timestamp for event in self.events),
            'processing_parameters_logged': any(event.parameters for event in self.events),
            'qc_results_documented': any(event.event_type == 'qc' for event in self.events),
            'error_handling_documented': any(event.event_type == 'error' for event in self.events),
            'user_actions_traced': all(event.user_id for event in self.events),
            'performance_metrics_captured': any(event.processing_time_ms for event in self.events),
            'audit_log_integrity_verified': self._verify_audit_integrity()
        }

    def _generate_regulatory_findings(self) -> List[Dict[str, str]]:
        """Generate regulatory findings and concerns."""
        findings = []

        # Check for high error rates
        error_rate = self.performance_metrics['errors_logged'] / max(1, self.performance_metrics['events_logged'])
        if error_rate > 0.1:  # More than 10% errors
            findings.append({
                'finding': 'High error rate detected',
                'severity': 'HIGH',
                'description': f'Error rate: {error_rate:.1%}, exceeds 10% threshold',
                'recommendation': 'Review processing pipeline for systematic issues'
            })

        # Check for regulatory flags
        regulatory_flags = self._count_regulatory_flags()
        if regulatory_flags > 0:
            findings.append({
                'finding': 'Regulatory flags raised',
                'severity': 'MEDIUM',
                'description': f'{regulatory_flags} regulatory flags raised during processing',
                'recommendation': 'Review flagged events for compliance concerns'
            })

        # Check for QC failures
        qc_failures = sum(1 for event in self.events
                         if event.event_type == 'qc' and
                         event.results and not event.results.get('overall_pass', True))
        if qc_failures > 0:
            findings.append({
                'finding': 'Quality control failures',
                'severity': 'HIGH',
                'description': f'{qc_failures} subjects failed quality control',
                'recommendation': 'Review QC failures for data quality issues'
            })

        return findings

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit findings."""
        recommendations = [
            "✅ Maintain complete audit trail for all processing steps",
            "✅ Regular review of QC metrics and thresholds",
            "✅ Monitor processing performance against benchmarks",
            "✅ Document all regulatory flags and their resolution"
        ]

        # Add specific recommendations based on findings
        error_rate = self.performance_metrics['errors_logged'] / max(1, self.performance_metrics['events_logged'])
        if error_rate > 0.05:
            recommendations.append("⚠️ Review error handling procedures - error rate elevated")

        if self._count_regulatory_flags() > 0:
            recommendations.append("⚠️ Address regulatory flags before clinical deployment")

        return recommendations

    def _generate_readable_summary(self, audit_report: Dict, output_file: Path):
        """Generate human-readable audit summary."""
        with open(output_file, 'w') as f:
            f.write("CLINICAL AUDIT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Header information
            header = audit_report['report_header']
            f.write(f"Report Generated: {header['generation_timestamp']}\n")
            f.write(f"Session ID: {header['session_id']}\n")
            f.write(f"Trial ID: {header.get('trial_id', 'N/A')}\n")
            f.write(f"Site ID: {header.get('site_id', 'N/A')}\n")
            f.write(f"Pipeline Version: {header['pipeline_version']}\n\n")

            # Session statistics
            session = audit_report['session_summary']
            f.write("SESSION STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Events: {session['event_statistics']['total_events']}\n")
            f.write(f"Errors: {session['event_statistics']['errors']}\n")
            f.write(f"Warnings: {session['event_statistics']['warnings']}\n")
            f.write(f"Success Rate: {session['event_statistics']['success_rate']:.1%}\n\n")

            # Performance metrics
            perf = session['performance_metrics']
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Avg Processing Time: {perf['avg_processing_time_ms']:.1f}ms\n")
            f.write(f"Max Processing Time: {perf['max_processing_time_ms']:.1f}ms\n")
            f.write(f"Avg Memory Usage: {perf['avg_memory_usage_mb']:.1f}MB\n\n")

            # Compliance checklist
            f.write("COMPLIANCE CHECKLIST\n")
            f.write("-" * 20 + "\n")
            compliance = audit_report['compliance_checklist']
            for check, status in compliance.items():
                status_str = "✅ PASS" if status else "❌ FAIL"
                f.write(f"{check}: {status_str}\n")
            f.write("\n")

            # Regulatory findings
            if audit_report['regulatory_findings']:
                f.write("REGULATORY FINDINGS\n")
                f.write("-" * 20 + "\n")
                for finding in audit_report['regulatory_findings']:
                    f.write(f"[{finding['severity']}] {finding['finding']}\n")
                    f.write(f"Description: {finding['description']}\n")
                    f.write(f"Recommendation: {finding['recommendation']}\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for rec in audit_report['recommendations']:
                f.write(f"{rec}\n")

    def close_session(self):
        """Close audit session and finalize logs."""
        self.log_system_event(
            action="audit_session_end",
            component="audit_logger",
            status="completed",
            parameters=self.get_session_summary()
        )

        # Generate final audit report
        report_file = self.generate_audit_report()

        self.logger.info(f"Audit session closed: {self.session_id}")
        self.logger.info(f"Final audit report: {report_file}")

        return report_file