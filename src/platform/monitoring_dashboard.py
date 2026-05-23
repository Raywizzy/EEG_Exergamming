#!/usr/bin/env python3
"""
Clinical Platform Monitoring Dashboard
Phase VI-B: Real-time system monitoring and configuration management
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import logging
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemAlert:
    """System alert for monitoring dashboard."""
    alert_id: str
    level: AlertLevel
    component: str
    message: str
    details: Dict
    timestamp: datetime
    resolved: bool = False

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring."""
    timestamp: datetime
    jobs_per_hour: float
    avg_processing_time: float
    queue_length: int
    success_rate: float
    qc_pass_rate: float
    error_rate: float
    system_load: float

class MonitoringDashboard:
    """Real-time monitoring dashboard for clinical platform."""

    def __init__(self, db_path: str = "data/eeg_platform.db", update_interval: int = 30):
        """Initialize monitoring dashboard."""
        self.db_path = Path(db_path)
        self.update_interval = update_interval
        self.monitoring_active = False
        self.monitor_thread = None

        # Alert management
        self.alerts = []
        self.alert_history = []

        # Performance tracking
        self.performance_history = []
        self.max_history_length = 1000

        # Alert thresholds
        self.thresholds = {
            'queue_length_warning': 10,
            'queue_length_critical': 25,
            'error_rate_warning': 0.10,
            'error_rate_critical': 0.25,
            'processing_time_warning': 300,  # 5 minutes
            'processing_time_critical': 600,  # 10 minutes
            'qc_pass_rate_warning': 0.80,
            'qc_pass_rate_critical': 0.70
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def get_db_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ================================================================================
    # REAL-TIME MONITORING
    # ================================================================================

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        conn = self.get_db_connection()
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'job_statistics': self._get_job_statistics(conn),
                'performance_metrics': self._get_performance_metrics(conn),
                'trial_status': self._get_trial_status(conn),
                'qc_metrics': self._get_qc_metrics(conn),
                'system_health': self._get_system_health(conn),
                'recent_alerts': self._get_recent_alerts(5),
                'configuration': self._get_system_configuration(conn)
            }
            return status
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
        finally:
            conn.close()

    def _get_job_statistics(self, conn: sqlite3.Connection) -> Dict:
        """Get job processing statistics."""
        # Current queue status
        cursor = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM jobs
            GROUP BY status
        """)
        job_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Recent activity (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor = conn.execute("""
            SELECT COUNT(*) as completed_24h
            FROM jobs
            WHERE status = 'completed' AND completed_at >= ?
        """, (yesterday,))
        completed_24h = cursor.fetchone()[0]

        # Average processing time
        cursor = conn.execute("""
            SELECT AVG(duration_seconds) as avg_duration
            FROM jobs
            WHERE status = 'completed' AND duration_seconds IS NOT NULL
        """)
        avg_duration = cursor.fetchone()[0] or 0

        return {
            'queue_counts': job_counts,
            'completed_24h': completed_24h,
            'avg_processing_time_seconds': avg_duration,
            'total_jobs': sum(job_counts.values())
        }

    def _get_performance_metrics(self, conn: sqlite3.Connection) -> Dict:
        """Get system performance metrics."""
        # Success rate
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
            FROM jobs
            WHERE submitted_at >= datetime('now', '-24 hours')
        """)
        row = cursor.fetchone()
        total_jobs = row[0]
        success_rate = row[1] / total_jobs if total_jobs > 0 else 1.0
        error_rate = row[2] / total_jobs if total_jobs > 0 else 0.0

        # QC metrics
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_results,
                COUNT(CASE WHEN qc_pass = 1 THEN 1 END) as qc_passed,
                AVG(qc_score) as avg_qc_score
            FROM results r
            JOIN jobs j ON r.job_id = j.job_id
            WHERE j.completed_at >= datetime('now', '-24 hours')
        """)
        qc_row = cursor.fetchone()
        qc_total = qc_row[0]
        qc_pass_rate = qc_row[1] / qc_total if qc_total > 0 else 1.0
        avg_qc_score = qc_row[2] or 0.0

        return {
            'success_rate': success_rate,
            'error_rate': error_rate,
            'qc_pass_rate': qc_pass_rate,
            'avg_qc_score': avg_qc_score,
            'jobs_last_24h': total_jobs
        }

    def _get_trial_status(self, conn: sqlite3.Connection) -> Dict:
        """Get trial enrollment and status summary."""
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_trials,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_trials,
                COUNT(CASE WHEN status = 'planning' THEN 1 END) as planning_trials,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_trials
            FROM trials
        """)
        trial_stats = dict(cursor.fetchone())

        # Enrollment statistics
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_subjects,
                COUNT(CASE WHEN status = 'enrolled' THEN 1 END) as enrolled_subjects,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_subjects
            FROM subjects
        """)
        enrollment_stats = dict(cursor.fetchone())

        return {
            'trial_statistics': trial_stats,
            'enrollment_statistics': enrollment_stats
        }

    def _get_qc_metrics(self, conn: sqlite3.Connection) -> Dict:
        """Get quality control metrics."""
        cursor = conn.execute("""
            SELECT
                signal_quality,
                COUNT(*) as count,
                AVG(qc_score) as avg_score
            FROM results
            WHERE signal_quality IS NOT NULL
            GROUP BY signal_quality
        """)
        quality_distribution = {}
        for row in cursor.fetchall():
            quality_distribution[row[0]] = {
                'count': row[1],
                'avg_score': row[2]
            }

        # Recent QC trends
        cursor = conn.execute("""
            SELECT
                DATE(r.created_at) as date,
                COUNT(*) as total_results,
                COUNT(CASE WHEN r.qc_pass = 1 THEN 1 END) as passed,
                AVG(r.qc_score) as avg_score
            FROM results r
            JOIN jobs j ON r.job_id = j.job_id
            WHERE r.created_at >= datetime('now', '-7 days')
            GROUP BY DATE(r.created_at)
            ORDER BY date
        """)
        qc_trends = [dict(row) for row in cursor.fetchall()]

        return {
            'quality_distribution': quality_distribution,
            'qc_trends': qc_trends
        }

    def _get_system_health(self, conn: sqlite3.Connection) -> Dict:
        """Get system health indicators."""
        # Database health
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        # Recent audit events
        cursor = conn.execute("""
            SELECT COUNT(*) FROM audit_events
            WHERE timestamp >= datetime('now', '-1 hour')
        """)
        recent_events = cursor.fetchone()[0]

        # Error events
        cursor = conn.execute("""
            SELECT COUNT(*) FROM audit_events
            WHERE success = 0 AND timestamp >= datetime('now', '-24 hours')
        """)
        error_events = cursor.fetchone()[0]

        return {
            'database_tables': table_count,
            'recent_audit_events': recent_events,
            'error_events_24h': error_events,
            'database_path': str(self.db_path),
            'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }

    def _get_system_configuration(self, conn: sqlite3.Connection) -> Dict:
        """Get current system configuration."""
        cursor = conn.execute("""
            SELECT config_category, config_key, config_value, config_type
            FROM system_config
            WHERE is_sensitive = 0
            ORDER BY config_category, config_key
        """)

        config = {}
        for row in cursor.fetchall():
            category = row[0]
            if category not in config:
                config[category] = {}

            value = row[2]
            if row[3] == 'integer':
                value = int(value)
            elif row[3] == 'float':
                value = float(value)
            elif row[3] == 'boolean':
                value = value.lower() == 'true'
            elif row[3] == 'json':
                value = json.loads(value)

            config[category][row[1]] = value

        return config

    # ================================================================================
    # ALERT MANAGEMENT
    # ================================================================================

    def check_system_alerts(self) -> List[SystemAlert]:
        """Check for system alerts based on current metrics."""
        conn = self.get_db_connection()
        try:
            alerts = []
            job_stats = self._get_job_statistics(conn)
            perf_metrics = self._get_performance_metrics(conn)

            # Check queue length
            queue_length = job_stats['queue_counts'].get('queued', 0)
            if queue_length >= self.thresholds['queue_length_critical']:
                alerts.append(SystemAlert(
                    alert_id=f"queue_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    component="job_queue",
                    message=f"Critical queue backlog: {queue_length} jobs",
                    details={'queue_length': queue_length, 'threshold': self.thresholds['queue_length_critical']},
                    timestamp=datetime.now()
                ))
            elif queue_length >= self.thresholds['queue_length_warning']:
                alerts.append(SystemAlert(
                    alert_id=f"queue_warning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    component="job_queue",
                    message=f"Queue backlog warning: {queue_length} jobs",
                    details={'queue_length': queue_length, 'threshold': self.thresholds['queue_length_warning']},
                    timestamp=datetime.now()
                ))

            # Check error rate
            error_rate = perf_metrics['error_rate']
            if error_rate >= self.thresholds['error_rate_critical']:
                alerts.append(SystemAlert(
                    alert_id=f"error_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    component="job_processing",
                    message=f"Critical error rate: {error_rate:.1%}",
                    details={'error_rate': error_rate, 'threshold': self.thresholds['error_rate_critical']},
                    timestamp=datetime.now()
                ))
            elif error_rate >= self.thresholds['error_rate_warning']:
                alerts.append(SystemAlert(
                    alert_id=f"error_warning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    component="job_processing",
                    message=f"Error rate warning: {error_rate:.1%}",
                    details={'error_rate': error_rate, 'threshold': self.thresholds['error_rate_warning']},
                    timestamp=datetime.now()
                ))

            # Check QC pass rate
            qc_pass_rate = perf_metrics['qc_pass_rate']
            if qc_pass_rate <= self.thresholds['qc_pass_rate_critical']:
                alerts.append(SystemAlert(
                    alert_id=f"qc_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    component="quality_control",
                    message=f"Critical QC pass rate: {qc_pass_rate:.1%}",
                    details={'qc_pass_rate': qc_pass_rate, 'threshold': self.thresholds['qc_pass_rate_critical']},
                    timestamp=datetime.now()
                ))
            elif qc_pass_rate <= self.thresholds['qc_pass_rate_warning']:
                alerts.append(SystemAlert(
                    alert_id=f"qc_warning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    component="quality_control",
                    message=f"QC pass rate warning: {qc_pass_rate:.1%}",
                    details={'qc_pass_rate': qc_pass_rate, 'threshold': self.thresholds['qc_pass_rate_warning']},
                    timestamp=datetime.now()
                ))

            # Check processing time
            avg_time = job_stats['avg_processing_time_seconds']
            if avg_time >= self.thresholds['processing_time_critical']:
                alerts.append(SystemAlert(
                    alert_id=f"time_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    component="job_processing",
                    message=f"Critical processing time: {avg_time:.1f}s",
                    details={'avg_time': avg_time, 'threshold': self.thresholds['processing_time_critical']},
                    timestamp=datetime.now()
                ))
            elif avg_time >= self.thresholds['processing_time_warning']:
                alerts.append(SystemAlert(
                    alert_id=f"time_warning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    component="job_processing",
                    message=f"Processing time warning: {avg_time:.1f}s",
                    details={'avg_time': avg_time, 'threshold': self.thresholds['processing_time_warning']},
                    timestamp=datetime.now()
                ))

            return alerts

        except Exception as e:
            self.logger.error(f"Failed to check alerts: {e}")
            return []
        finally:
            conn.close()

    def _get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return [
            {
                'alert_id': alert.alert_id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved
            }
            for alert in self.alerts[-limit:]
        ]

    # ================================================================================
    # CONFIGURATION MANAGEMENT
    # ================================================================================

    def update_configuration(self, category: str, key: str, value: Any,
                           user_id: str = "system") -> bool:
        """Update system configuration."""
        conn = self.get_db_connection()
        try:
            # Determine value type
            if isinstance(value, bool):
                config_type = "boolean"
                config_value = str(value).lower()
            elif isinstance(value, int):
                config_type = "integer"
                config_value = str(value)
            elif isinstance(value, float):
                config_type = "float"
                config_value = str(value)
            elif isinstance(value, (dict, list)):
                config_type = "json"
                config_value = json.dumps(value)
            else:
                config_type = "string"
                config_value = str(value)

            # Update configuration
            cursor = conn.execute("""
                UPDATE system_config
                SET config_value = ?, config_type = ?, updated_at = CURRENT_TIMESTAMP, updated_by = ?
                WHERE config_category = ? AND config_key = ?
            """, (config_value, config_type, user_id, category, key))

            if cursor.rowcount == 0:
                # Insert new configuration
                config_id = f"{category}_{key}_{datetime.now().strftime('%Y%m%d')}"
                conn.execute("""
                    INSERT INTO system_config
                    (config_id, config_category, config_key, config_value, config_type,
                     description, is_sensitive, updated_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_id, category, key, config_value, config_type,
                    f"Configuration for {key}", False, user_id
                ))

            conn.commit()

            # Log configuration change
            conn.execute("""
                INSERT INTO audit_events
                (event_type, entity_type, entity_id, user_id, action, details, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'config_change', 'system', f"{category}.{key}", user_id,
                'update_configuration',
                json.dumps({'old_value': 'unknown', 'new_value': value, 'category': category}),
                True
            ))
            conn.commit()

            self.logger.info(f"Updated configuration {category}.{key} = {value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
        finally:
            conn.close()

    def get_configuration_history(self, category: str = None, key: str = None,
                                limit: int = 100) -> List[Dict]:
        """Get configuration change history."""
        conn = self.get_db_connection()
        try:
            query = """
                SELECT * FROM audit_events
                WHERE event_type = 'config_change'
            """
            params = []

            if category:
                query += " AND details LIKE ?"
                params.append(f"%{category}%")

            if key:
                query += " AND entity_id LIKE ?"
                params.append(f"%.{key}")

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get configuration history: {e}")
            return []
        finally:
            conn.close()

    # ================================================================================
    # MONITORING THREAD
    # ================================================================================

    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Started monitoring thread")

    def stop_monitoring(self):
        """Stop background monitoring thread."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Stopped monitoring thread")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for alerts
                new_alerts = self.check_system_alerts()
                for alert in new_alerts:
                    # Check if this type of alert already exists
                    existing_alerts = [a for a in self.alerts
                                     if a.component == alert.component and
                                        a.level == alert.level and
                                        not a.resolved]

                    if not existing_alerts:
                        self.alerts.append(alert)
                        self.logger.warning(f"New alert: {alert.message}")

                # Collect performance metrics
                status = self.get_system_status()
                if 'error' not in status:
                    perf_metrics = PerformanceMetrics(
                        timestamp=datetime.now(),
                        jobs_per_hour=status['job_statistics']['completed_24h'] / 24.0,
                        avg_processing_time=status['job_statistics']['avg_processing_time_seconds'],
                        queue_length=status['job_statistics']['queue_counts'].get('queued', 0),
                        success_rate=status['performance_metrics']['success_rate'],
                        qc_pass_rate=status['performance_metrics']['qc_pass_rate'],
                        error_rate=status['performance_metrics']['error_rate'],
                        system_load=0.5  # Mock system load
                    )

                    self.performance_history.append(perf_metrics)
                    if len(self.performance_history) > self.max_history_length:
                        self.performance_history = self.performance_history[-self.max_history_length:]

                time.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)

    def get_performance_history(self, hours: int = 24) -> List[Dict]:
        """Get performance history for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff]

        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'jobs_per_hour': m.jobs_per_hour,
                'avg_processing_time': m.avg_processing_time,
                'queue_length': m.queue_length,
                'success_rate': m.success_rate,
                'qc_pass_rate': m.qc_pass_rate,
                'error_rate': m.error_rate,
                'system_load': m.system_load
            }
            for m in recent_metrics
        ]

def main():
    """Demo monitoring dashboard."""
    print("="*60)
    print("CLINICAL PLATFORM MONITORING DASHBOARD")
    print("Phase VI-B: Real-time System Monitoring")
    print("="*60)

    dashboard = MonitoringDashboard(update_interval=5)  # 5 second updates for demo

    try:
        # Start monitoring
        dashboard.start_monitoring()
        print("🔍 Started real-time monitoring")

        # Monitor for a short period
        for i in range(6):
            time.sleep(5)

            # Get current status
            status = dashboard.get_system_status()

            print(f"\n📊 System Status Update {i+1}:")
            print(f"   Jobs: {status['job_statistics']['queue_counts']}")
            print(f"   Performance: {status['performance_metrics']['success_rate']:.1%} success rate")
            print(f"   QC: {status['performance_metrics']['qc_pass_rate']:.1%} pass rate")

            # Check for alerts
            if status['recent_alerts']:
                print(f"   🚨 Alerts: {len(status['recent_alerts'])} active")
                for alert in status['recent_alerts'][-2:]:  # Show last 2
                    print(f"      {alert['level'].upper()}: {alert['message']}")

        # Test configuration update
        print("\n⚙️  Testing configuration management...")
        success = dashboard.update_configuration(
            "monitoring", "update_interval", 60, "demo_user"
        )
        print(f"   Configuration update: {'✅ Success' if success else '❌ Failed'}")

        # Get configuration history
        history = dashboard.get_configuration_history(limit=5)
        print(f"   Configuration history: {len(history)} recent changes")

        # Get performance history
        perf_history = dashboard.get_performance_history(hours=1)
        print(f"   Performance history: {len(perf_history)} data points")

        print("\n✅ Monitoring dashboard demonstration completed")

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted by user")

    finally:
        dashboard.stop_monitoring()
        print("🛑 Monitoring stopped")

if __name__ == "__main__":
    main()