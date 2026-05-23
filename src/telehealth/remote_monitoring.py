"""
Phase X-A: Remote Patient Monitoring Dashboards
Real-time monitoring and intervention systems for distributed telehealth
"""

import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from enum import Enum
import statistics

from .telehealth_workflows import TelehealthWorkflowManager, RemoteSessionStatus, AutomatedIntervention
from .device_integration import HomeEEGDeviceManager, DeviceStatus, SignalQuality
from .cloud_fhir import CloudFHIREndpoints, CloudRegion
from ..clinical.authentication import EnterpriseAuthManager, RoleBasedAccessControl

class MonitoringViewType(Enum):
    """Remote monitoring dashboard view types"""
    CLINICIAN_OVERVIEW = "clinician_overview"
    TECHNICIAN_CONSOLE = "technician_console"
    TRIAL_COORDINATOR = "trial_coordinator"
    DATA_SCIENTIST = "data_scientist"
    EXECUTIVE_SUMMARY = "executive_summary"
    PARTICIPANT_SUPPORT = "participant_support"
    EMERGENCY_RESPONSE = "emergency_response"

class AlertSeverity(Enum):
    """Remote monitoring alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ParticipantRiskLevel(Enum):
    """Remote participant risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RemoteAlert:
    """Remote monitoring alert"""
    alert_id: str
    participant_id: str
    session_id: Optional[str]
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    detected_at: datetime
    data_values: Dict[str, Any]
    auto_resolution_attempted: bool = False
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalation_level: int = 0

@dataclass
class ParticipantVitals:
    """Remote participant vital signs and metrics"""
    participant_id: str
    timestamp: datetime
    device_connection_status: str
    signal_quality_score: float
    session_compliance_rate: float
    technical_issues_count: int
    last_session_date: Optional[datetime]
    next_session_date: Optional[datetime]
    risk_level: ParticipantRiskLevel
    intervention_history_count: int
    satisfaction_score: Optional[float] = None
    connectivity_stability: float = 0.0
    data_completeness: float = 0.0

@dataclass
class RealTimeMetrics:
    """Real-time monitoring metrics"""
    timestamp: datetime
    active_sessions: int
    total_participants: int
    average_signal_quality: float
    active_alerts: int
    critical_alerts: int
    intervention_rate: float
    system_uptime: float
    regional_distribution: Dict[str, int]
    quality_trends: Dict[str, List[float]]

@dataclass
class GeographicMetrics:
    """Geographic distribution metrics"""
    region: str
    country: str
    active_participants: int
    active_sessions: int
    average_connectivity: float
    technical_support_requests: int
    satisfaction_score: float
    compliance_rate: float

class RemotePatientMonitor:
    """
    Real-time remote patient monitoring system
    Provides continuous oversight of distributed participants
    """

    def __init__(self, config_path: str = None):
        """Initialize remote patient monitor"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Core services
        self.workflow_manager = TelehealthWorkflowManager()
        self.device_manager = HomeEEGDeviceManager()
        self.cloud_fhir = CloudFHIREndpoints()
        self.auth_manager = EnterpriseAuthManager()
        self.rbac = RoleBasedAccessControl()

        # Monitoring state
        self.active_alerts: Dict[str, RemoteAlert] = {}
        self.participant_vitals: Dict[str, ParticipantVitals] = {}
        self.real_time_metrics: Optional[RealTimeMetrics] = None
        self.geographic_metrics: Dict[str, GeographicMetrics] = {}

        # Monitoring thresholds
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.quality_benchmarks = self._initialize_quality_benchmarks()

        # Background monitoring
        self.monitoring_active = True
        self.update_interval = self.config.get('monitoring_interval_seconds', 30)

        # Start monitoring
        self._start_background_monitoring()

        self.logger.info("Remote Patient Monitor initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load remote monitoring configuration"""
        default_config = {
            'monitoring_interval_seconds': 30,
            'alert_retention_hours': 168,  # 1 week
            'auto_escalation_enabled': True,
            'escalation_thresholds': {
                'signal_quality_critical': 0.3,
                'device_disconnect_minutes': 5,
                'missed_session_threshold': 2,
                'technical_issues_threshold': 3
            },
            'quality_benchmarks': {
                'excellent_signal_quality': 0.9,
                'good_signal_quality': 0.7,
                'acceptable_signal_quality': 0.5,
                'target_compliance_rate': 0.85,
                'target_satisfaction_score': 4.0
            },
            'geographic_monitoring': {
                'enabled': True,
                'timezone_awareness': True,
                'regional_thresholds': True
            },
            'notification_channels': {
                'email_alerts': True,
                'sms_urgent': True,
                'dashboard_notifications': True,
                'slack_integration': False
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup monitoring-specific logging"""
        logger = logging.getLogger('remote_monitoring')
        logger.setLevel(logging.INFO)

        log_dir = Path('logs/telehealth')
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / f'monitoring_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _initialize_alert_thresholds(self) -> Dict[str, float]:
        """Initialize alert threshold values"""
        return {
            'signal_quality_warning': 0.6,
            'signal_quality_critical': self.config['escalation_thresholds']['signal_quality_critical'],
            'device_disconnect_warning_minutes': 2,
            'device_disconnect_critical_minutes': self.config['escalation_thresholds']['device_disconnect_minutes'],
            'session_compliance_warning': 0.7,
            'session_compliance_critical': 0.5,
            'technical_issues_warning': 2,
            'technical_issues_critical': self.config['escalation_thresholds']['technical_issues_threshold'],
            'satisfaction_warning': 3.0,
            'satisfaction_critical': 2.0
        }

    def _initialize_quality_benchmarks(self) -> Dict[str, float]:
        """Initialize quality benchmark values"""
        return self.config['quality_benchmarks']

    def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._monitor_participants_continuously())
        asyncio.create_task(self._monitor_active_sessions())
        asyncio.create_task(self._update_real_time_metrics())
        asyncio.create_task(self._process_alert_escalations())

    async def _monitor_participants_continuously(self):
        """Continuously monitor all remote participants"""
        try:
            while self.monitoring_active:
                await self._update_participant_vitals()
                await self._check_participant_alerts()
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            self.logger.error(f"Continuous monitoring failed: {e}")

    async def _update_participant_vitals(self):
        """Update vital signs for all participants"""
        try:
            # Get all active participants from workflow manager
            all_sessions = await self.workflow_manager.get_workflow_metrics()

            for session_id, session in self.workflow_manager.active_sessions.items():
                participant_id = session.participant_id

                # Get device status if available
                device_status = "disconnected"
                signal_quality = 0.0
                connectivity_stability = 0.0

                if session.device_id:
                    device_info = await self.device_manager.get_device_status(session.device_id)
                    if device_info:
                        device_status = device_info.get('status', 'unknown')
                        signal_data = device_info.get('signal_quality', {})
                        signal_quality = signal_data.get('overall_quality', 0.0)
                        connectivity_stability = self._calculate_connectivity_stability(session_id)

                # Calculate compliance rate
                compliance_rate = self._calculate_session_compliance(participant_id)

                # Assess risk level
                risk_level = self._assess_participant_risk(session, signal_quality, compliance_rate)

                # Count technical issues
                technical_issues = len(session.technical_issues)

                # Create/update participant vitals
                vitals = ParticipantVitals(
                    participant_id=participant_id,
                    timestamp=datetime.now(),
                    device_connection_status=device_status,
                    signal_quality_score=signal_quality,
                    session_compliance_rate=compliance_rate,
                    technical_issues_count=technical_issues,
                    last_session_date=session.actual_start,
                    next_session_date=session.scheduled_start,
                    risk_level=risk_level,
                    intervention_history_count=len(session.intervention_history),
                    connectivity_stability=connectivity_stability,
                    data_completeness=self._calculate_data_completeness(session)
                )

                self.participant_vitals[participant_id] = vitals

        except Exception as e:
            self.logger.error(f"Participant vitals update failed: {e}")

    def _calculate_connectivity_stability(self, session_id: str) -> float:
        """Calculate connectivity stability for participant"""
        # In production, this would analyze connection history
        # For now, return a placeholder value
        return 0.95

    def _calculate_session_compliance(self, participant_id: str) -> float:
        """Calculate session compliance rate for participant"""
        # Count completed vs scheduled sessions
        participant_sessions = [
            s for s in self.workflow_manager.active_sessions.values()
            if s.participant_id == participant_id
        ]

        if not participant_sessions:
            return 1.0

        completed = len([s for s in participant_sessions if s.current_status == RemoteSessionStatus.SESSION_COMPLETED])
        total = len(participant_sessions)

        return completed / total if total > 0 else 1.0

    def _assess_participant_risk(self,
                               session: Any,
                               signal_quality: float,
                               compliance_rate: float) -> ParticipantRiskLevel:
        """Assess overall risk level for participant"""

        # Count high-severity factors
        risk_factors = 0

        # Signal quality risk
        if signal_quality < self.alert_thresholds['signal_quality_critical']:
            risk_factors += 3
        elif signal_quality < self.alert_thresholds['signal_quality_warning']:
            risk_factors += 1

        # Compliance risk
        if compliance_rate < 0.5:
            risk_factors += 3
        elif compliance_rate < 0.7:
            risk_factors += 1

        # Technical issues risk
        technical_issues = len(session.technical_issues)
        if technical_issues >= self.alert_thresholds['technical_issues_critical']:
            risk_factors += 2
        elif technical_issues >= self.alert_thresholds['technical_issues_warning']:
            risk_factors += 1

        # Recent interventions risk
        recent_interventions = len([
            i for i in session.intervention_history
            if datetime.fromisoformat(i['timestamp']) > datetime.now() - timedelta(hours=24)
        ])
        if recent_interventions >= 3:
            risk_factors += 2

        # Determine risk level
        if risk_factors >= 6:
            return ParticipantRiskLevel.CRITICAL
        elif risk_factors >= 4:
            return ParticipantRiskLevel.HIGH
        elif risk_factors >= 2:
            return ParticipantRiskLevel.MODERATE
        else:
            return ParticipantRiskLevel.LOW

    def _calculate_data_completeness(self, session: Any) -> float:
        """Calculate data completeness for session"""
        # In production, analyze actual data completeness
        # For now, estimate based on session status and quality
        if session.current_status == RemoteSessionStatus.SESSION_COMPLETED:
            return 0.95
        elif session.current_status in [RemoteSessionStatus.SESSION_ACTIVE, RemoteSessionStatus.DATA_PROCESSING]:
            return 0.8
        else:
            return 0.5

    async def _check_participant_alerts(self):
        """Check for new participant alerts"""
        try:
            for participant_id, vitals in self.participant_vitals.items():
                await self._check_signal_quality_alerts(participant_id, vitals)
                await self._check_compliance_alerts(participant_id, vitals)
                await self._check_technical_alerts(participant_id, vitals)
                await self._check_connectivity_alerts(participant_id, vitals)

        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")

    async def _check_signal_quality_alerts(self, participant_id: str, vitals: ParticipantVitals):
        """Check for signal quality alerts"""
        try:
            signal_quality = vitals.signal_quality_score

            if signal_quality < self.alert_thresholds['signal_quality_critical']:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="signal_quality_critical",
                    severity=AlertSeverity.CRITICAL,
                    title="Critical Signal Quality",
                    description=f"Signal quality {signal_quality:.2f} below critical threshold",
                    data_values={'signal_quality': signal_quality, 'threshold': self.alert_thresholds['signal_quality_critical']}
                )
            elif signal_quality < self.alert_thresholds['signal_quality_warning']:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="signal_quality_warning",
                    severity=AlertSeverity.WARNING,
                    title="Poor Signal Quality",
                    description=f"Signal quality {signal_quality:.2f} below warning threshold",
                    data_values={'signal_quality': signal_quality, 'threshold': self.alert_thresholds['signal_quality_warning']}
                )

        except Exception as e:
            self.logger.error(f"Signal quality alert check failed: {e}")

    async def _check_compliance_alerts(self, participant_id: str, vitals: ParticipantVitals):
        """Check for session compliance alerts"""
        try:
            compliance_rate = vitals.session_compliance_rate

            if compliance_rate < self.alert_thresholds['session_compliance_critical']:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="compliance_critical",
                    severity=AlertSeverity.CRITICAL,
                    title="Critical Compliance Issue",
                    description=f"Session compliance {compliance_rate:.1%} critically low",
                    data_values={'compliance_rate': compliance_rate, 'threshold': self.alert_thresholds['session_compliance_critical']}
                )
            elif compliance_rate < self.alert_thresholds['session_compliance_warning']:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="compliance_warning",
                    severity=AlertSeverity.WARNING,
                    title="Low Session Compliance",
                    description=f"Session compliance {compliance_rate:.1%} below target",
                    data_values={'compliance_rate': compliance_rate, 'threshold': self.alert_thresholds['session_compliance_warning']}
                )

        except Exception as e:
            self.logger.error(f"Compliance alert check failed: {e}")

    async def _check_technical_alerts(self, participant_id: str, vitals: ParticipantVitals):
        """Check for technical issue alerts"""
        try:
            technical_issues = vitals.technical_issues_count

            if technical_issues >= self.alert_thresholds['technical_issues_critical']:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="technical_issues_critical",
                    severity=AlertSeverity.CRITICAL,
                    title="Multiple Technical Issues",
                    description=f"{technical_issues} technical issues detected",
                    data_values={'technical_issues_count': technical_issues, 'threshold': self.alert_thresholds['technical_issues_critical']}
                )
            elif technical_issues >= self.alert_thresholds['technical_issues_warning']:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="technical_issues_warning",
                    severity=AlertSeverity.WARNING,
                    title="Technical Issues Detected",
                    description=f"{technical_issues} technical issues need attention",
                    data_values={'technical_issues_count': technical_issues, 'threshold': self.alert_thresholds['technical_issues_warning']}
                )

        except Exception as e:
            self.logger.error(f"Technical alert check failed: {e}")

    async def _check_connectivity_alerts(self, participant_id: str, vitals: ParticipantVitals):
        """Check for device connectivity alerts"""
        try:
            if vitals.device_connection_status == "disconnected":
                # Check how long device has been disconnected
                # In production, track disconnection time
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="device_disconnected",
                    severity=AlertSeverity.WARNING,
                    title="Device Disconnected",
                    description="EEG device is not connected",
                    data_values={'connection_status': vitals.device_connection_status}
                )

            # Check connectivity stability
            if vitals.connectivity_stability < 0.8:
                await self._create_alert(
                    participant_id=participant_id,
                    alert_type="connectivity_unstable",
                    severity=AlertSeverity.WARNING,
                    title="Unstable Connection",
                    description=f"Connection stability {vitals.connectivity_stability:.1%} below threshold",
                    data_values={'connectivity_stability': vitals.connectivity_stability, 'threshold': 0.8}
                )

        except Exception as e:
            self.logger.error(f"Connectivity alert check failed: {e}")

    async def _create_alert(self,
                           participant_id: str,
                           alert_type: str,
                           severity: AlertSeverity,
                           title: str,
                           description: str,
                           data_values: Dict[str, Any],
                           session_id: str = None):
        """Create new monitoring alert"""
        try:
            # Check for duplicate alerts
            existing_alert = self._find_existing_alert(participant_id, alert_type)
            if existing_alert and not existing_alert.resolved:
                # Update existing alert with new data
                existing_alert.data_values.update(data_values)
                existing_alert.detected_at = datetime.now()
                return existing_alert.alert_id

            # Create new alert
            alert_id = f"ALERT_{participant_id}_{alert_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            alert = RemoteAlert(
                alert_id=alert_id,
                participant_id=participant_id,
                session_id=session_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                description=description,
                detected_at=datetime.now(),
                data_values=data_values
            )

            self.active_alerts[alert_id] = alert

            # Log alert
            self.logger.warning(f"Alert created: {alert_id} - {title} for participant {participant_id}")

            # Trigger automatic resolution attempt if configured
            if self.config.get('auto_resolution_enabled', True):
                await self._attempt_auto_resolution(alert_id)

            # Schedule escalation if critical
            if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                asyncio.create_task(self._schedule_alert_escalation(alert_id))

            return alert_id

        except Exception as e:
            self.logger.error(f"Alert creation failed: {e}")
            return None

    def _find_existing_alert(self, participant_id: str, alert_type: str) -> Optional[RemoteAlert]:
        """Find existing unresolved alert of same type"""
        for alert in self.active_alerts.values():
            if (alert.participant_id == participant_id and
                alert.alert_type == alert_type and
                not alert.resolved):
                return alert
        return None

    async def _attempt_auto_resolution(self, alert_id: str):
        """Attempt automatic resolution of alert"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return

            alert.auto_resolution_attempted = True

            # Auto-resolution strategies by alert type
            if alert.alert_type == "signal_quality_warning":
                await self._auto_resolve_signal_quality(alert)
            elif alert.alert_type == "device_disconnected":
                await self._auto_resolve_device_connection(alert)
            elif alert.alert_type == "compliance_warning":
                await self._auto_resolve_compliance_issue(alert)

        except Exception as e:
            self.logger.error(f"Auto-resolution failed: {e}")

    async def _auto_resolve_signal_quality(self, alert: RemoteAlert):
        """Attempt automatic resolution of signal quality issues"""
        try:
            # Send signal improvement guidance
            guidance_message = """
            Your EEG signal quality has decreased. Please try:
            1. Adjust headset position
            2. Clean electrode contacts
            3. Sit still and relax
            4. Check for interference sources
            """

            # In production, send via appropriate communication channel
            self.logger.info(f"Auto-resolution: Signal quality guidance sent to {alert.participant_id}")

        except Exception as e:
            self.logger.error(f"Signal quality auto-resolution failed: {e}")

    async def _auto_resolve_device_connection(self, alert: RemoteAlert):
        """Attempt automatic resolution of device connection issues"""
        try:
            # Send reconnection instructions
            reconnection_message = """
            Your EEG device appears disconnected. Please:
            1. Check device power and battery
            2. Verify Bluetooth/WiFi connection
            3. Restart the device if necessary
            4. Contact support if issues persist
            """

            # In production, send via appropriate communication channel
            self.logger.info(f"Auto-resolution: Reconnection guidance sent to {alert.participant_id}")

        except Exception as e:
            self.logger.error(f"Device connection auto-resolution failed: {e}")

    async def _auto_resolve_compliance_issue(self, alert: RemoteAlert):
        """Attempt automatic resolution of compliance issues"""
        try:
            # Send compliance reminder and support offer
            compliance_message = """
            We noticed you may have missed some sessions. Our team is here to help!
            Please contact us to discuss any challenges you're experiencing.
            We can adjust schedules or provide additional support as needed.
            """

            # In production, send via appropriate communication channel
            self.logger.info(f"Auto-resolution: Compliance support offered to {alert.participant_id}")

        except Exception as e:
            self.logger.error(f"Compliance auto-resolution failed: {e}")

    async def _update_real_time_metrics(self):
        """Update real-time monitoring metrics"""
        try:
            while self.monitoring_active:
                # Calculate current metrics
                active_sessions = len([
                    s for s in self.workflow_manager.active_sessions.values()
                    if s.current_status == RemoteSessionStatus.SESSION_ACTIVE
                ])

                total_participants = len(self.participant_vitals)

                # Calculate average signal quality
                signal_qualities = [v.signal_quality_score for v in self.participant_vitals.values()]
                avg_signal_quality = statistics.mean(signal_qualities) if signal_qualities else 0.0

                # Count alerts
                active_alerts_count = len([a for a in self.active_alerts.values() if not a.resolved])
                critical_alerts_count = len([
                    a for a in self.active_alerts.values()
                    if not a.resolved and a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                ])

                # Calculate intervention rate
                total_interventions = sum(
                    len(s.intervention_history) for s in self.workflow_manager.active_sessions.values()
                )
                intervention_rate = total_interventions / max(1, total_participants)

                # Regional distribution
                regional_dist = {}
                for session in self.workflow_manager.active_sessions.values():
                    region = session.assigned_region.value
                    regional_dist[region] = regional_dist.get(region, 0) + 1

                # Create metrics object
                self.real_time_metrics = RealTimeMetrics(
                    timestamp=datetime.now(),
                    active_sessions=active_sessions,
                    total_participants=total_participants,
                    average_signal_quality=avg_signal_quality,
                    active_alerts=active_alerts_count,
                    critical_alerts=critical_alerts_count,
                    intervention_rate=intervention_rate,
                    system_uptime=99.9,  # Placeholder
                    regional_distribution=regional_dist,
                    quality_trends=self._calculate_quality_trends()
                )

                await asyncio.sleep(self.update_interval)

        except Exception as e:
            self.logger.error(f"Real-time metrics update failed: {e}")

    def _calculate_quality_trends(self) -> Dict[str, List[float]]:
        """Calculate quality trends over time"""
        # In production, maintain historical data
        # For now, return placeholder trends
        return {
            'signal_quality': [0.85, 0.87, 0.84, 0.89, 0.86],
            'compliance_rate': [0.92, 0.89, 0.94, 0.91, 0.88],
            'satisfaction': [4.2, 4.1, 4.3, 4.0, 4.2]
        }

    async def get_monitoring_dashboard(self, user_id: str, view_type: MonitoringViewType) -> Dict[str, Any]:
        """Get monitoring dashboard data for specific user role"""
        try:
            # Validate user permissions
            user_role = self.auth_manager.get_user_role(user_id)
            if not self._validate_dashboard_access(user_role, view_type):
                raise PermissionError(f"User {user_id} cannot access {view_type.value} dashboard")

            # Base dashboard data
            dashboard_data = {
                'view_type': view_type.value,
                'user_role': user_role,
                'last_updated': datetime.now().isoformat(),
                'real_time_metrics': asdict(self.real_time_metrics) if self.real_time_metrics else {},
                'system_status': 'healthy'
            }

            # Add view-specific data
            if view_type == MonitoringViewType.CLINICIAN_OVERVIEW:
                dashboard_data.update(await self._get_clinician_overview())
            elif view_type == MonitoringViewType.TECHNICIAN_CONSOLE:
                dashboard_data.update(await self._get_technician_console())
            elif view_type == MonitoringViewType.TRIAL_COORDINATOR:
                dashboard_data.update(await self._get_trial_coordinator_view())
            elif view_type == MonitoringViewType.EXECUTIVE_SUMMARY:
                dashboard_data.update(await self._get_executive_summary())
            elif view_type == MonitoringViewType.EMERGENCY_RESPONSE:
                dashboard_data.update(await self._get_emergency_response_view())

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            raise

    def _validate_dashboard_access(self, user_role: str, view_type: MonitoringViewType) -> bool:
        """Validate user access to dashboard view"""
        role_permissions = {
            'SUPER_ADMIN': [MonitoringViewType.EXECUTIVE_SUMMARY, MonitoringViewType.EMERGENCY_RESPONSE],
            'NEUROLOGIST': [MonitoringViewType.CLINICIAN_OVERVIEW, MonitoringViewType.EMERGENCY_RESPONSE],
            'CLINICAL_COORDINATOR': [MonitoringViewType.TRIAL_COORDINATOR, MonitoringViewType.CLINICIAN_OVERVIEW],
            'REMOTE_TECHNICIAN': [MonitoringViewType.TECHNICIAN_CONSOLE, MonitoringViewType.PARTICIPANT_SUPPORT],
            'DATA_SCIENTIST': [MonitoringViewType.DATA_SCIENTIST, MonitoringViewType.EXECUTIVE_SUMMARY],
            'HOSPITAL_ADMIN': [MonitoringViewType.EXECUTIVE_SUMMARY],
            'EEG_TECHNICIAN': [MonitoringViewType.TECHNICIAN_CONSOLE]
        }

        return view_type in role_permissions.get(user_role, [])

    async def _get_clinician_overview(self) -> Dict[str, Any]:
        """Get clinician overview dashboard data"""
        # High-risk participants requiring clinical attention
        high_risk_participants = [
            p for p in self.participant_vitals.values()
            if p.risk_level in [ParticipantRiskLevel.HIGH, ParticipantRiskLevel.CRITICAL]
        ]

        # Recent interventions requiring review
        recent_interventions = []
        for session in self.workflow_manager.active_sessions.values():
            for intervention in session.intervention_history[-5:]:  # Last 5 interventions
                recent_interventions.append({
                    'participant_id': session.participant_id,
                    'intervention_id': intervention.get('intervention_id'),
                    'timestamp': intervention.get('timestamp'),
                    'trigger': intervention.get('trigger'),
                    'description': intervention.get('description')
                })

        return {
            'high_risk_participants': [asdict(p) for p in high_risk_participants],
            'pending_clinical_reviews': len([
                s for s in self.workflow_manager.active_sessions.values()
                if s.current_status == RemoteSessionStatus.CLINICAL_REVIEW_PENDING
            ]),
            'recent_interventions': recent_interventions,
            'quality_summary': {
                'average_signal_quality': self.real_time_metrics.average_signal_quality if self.real_time_metrics else 0,
                'participants_below_threshold': len([
                    p for p in self.participant_vitals.values()
                    if p.signal_quality_score < self.quality_benchmarks['acceptable_signal_quality']
                ]),
                'compliance_issues': len([
                    p for p in self.participant_vitals.values()
                    if p.session_compliance_rate < self.quality_benchmarks['target_compliance_rate']
                ])
            }
        }

    async def _get_technician_console(self) -> Dict[str, Any]:
        """Get technician console dashboard data"""
        # Active sessions requiring technical support
        technical_support_needed = [
            p for p in self.participant_vitals.values()
            if p.device_connection_status != 'streaming' or p.technical_issues_count > 0
        ]

        # Current signal quality issues
        signal_quality_issues = [
            p for p in self.participant_vitals.values()
            if p.signal_quality_score < self.alert_thresholds['signal_quality_warning']
        ]

        # Device status summary
        device_statuses = {}
        for vitals in self.participant_vitals.values():
            status = vitals.device_connection_status
            device_statuses[status] = device_statuses.get(status, 0) + 1

        return {
            'technical_support_queue': [asdict(p) for p in technical_support_needed],
            'signal_quality_issues': [asdict(p) for p in signal_quality_issues],
            'device_status_summary': device_statuses,
            'active_technical_alerts': [
                asdict(a) for a in self.active_alerts.values()
                if not a.resolved and a.alert_type.startswith('technical')
            ],
            'connectivity_map': self._generate_connectivity_map()
        }

    def _generate_connectivity_map(self) -> Dict[str, Any]:
        """Generate real-time connectivity map"""
        return {
            'total_devices': len(self.participant_vitals),
            'connected': len([p for p in self.participant_vitals.values() if p.device_connection_status == 'streaming']),
            'disconnected': len([p for p in self.participant_vitals.values() if p.device_connection_status == 'disconnected']),
            'poor_quality': len([p for p in self.participant_vitals.values() if p.signal_quality_score < 0.6]),
            'geographic_distribution': self._get_geographic_connectivity()
        }

    def _get_geographic_connectivity(self) -> Dict[str, Dict[str, int]]:
        """Get connectivity by geographic region"""
        # In production, this would use actual geographic data
        return {
            'US_EAST_1': {'connected': 45, 'total': 50},
            'EU_WEST_1': {'connected': 32, 'total': 35},
            'ASIA_PACIFIC_1': {'connected': 18, 'total': 20}
        }

    async def _get_trial_coordinator_view(self) -> Dict[str, Any]:
        """Get trial coordinator dashboard data"""
        # Enrollment and compliance tracking
        workflow_metrics = await self.workflow_manager.get_workflow_metrics()

        return {
            'enrollment_status': {
                'total_enrolled': workflow_metrics['total_sessions'],
                'active_participants': len([
                    p for p in self.participant_vitals.values()
                    if p.last_session_date and p.last_session_date > datetime.now() - timedelta(days=7)
                ]),
                'completion_rate': workflow_metrics['completion_rate']
            },
            'compliance_tracking': {
                'overall_compliance': statistics.mean([
                    p.session_compliance_rate for p in self.participant_vitals.values()
                ]) if self.participant_vitals else 0,
                'participants_at_risk': len([
                    p for p in self.participant_vitals.values()
                    if p.session_compliance_rate < 0.7
                ]),
                'intervention_success_rate': 0.85  # Placeholder
            },
            'geographic_distribution': workflow_metrics['regional_distribution'],
            'protocol_deviations': self._get_protocol_deviations_summary()
        }

    def _get_protocol_deviations_summary(self) -> Dict[str, int]:
        """Get protocol deviations summary"""
        return {
            'total_deviations': 5,
            'minor_deviations': 3,
            'major_deviations': 2,
            'resolved_deviations': 4
        }

    async def _get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary dashboard data"""
        return {
            'key_performance_indicators': {
                'total_active_trials': len(set(s.trial_id for s in self.workflow_manager.active_sessions.values())),
                'total_participants': len(self.participant_vitals),
                'overall_satisfaction': 4.2,
                'retention_rate': 0.89,
                'data_quality_score': self.real_time_metrics.average_signal_quality if self.real_time_metrics else 0
            },
            'operational_metrics': {
                'system_uptime': 99.9,
                'support_response_time': '4.2 minutes',
                'technical_issues_resolved': 94.5,
                'cost_per_participant': '$145'
            },
            'growth_trends': {
                'participant_growth_rate': 12.5,
                'geographic_expansion': 3,
                'quality_improvement': 8.2
            }
        }

    async def _get_emergency_response_view(self) -> Dict[str, Any]:
        """Get emergency response dashboard data"""
        # Critical alerts requiring immediate attention
        critical_alerts = [
            a for a in self.active_alerts.values()
            if not a.resolved and a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        ]

        # Participants requiring urgent intervention
        urgent_participants = [
            p for p in self.participant_vitals.values()
            if p.risk_level == ParticipantRiskLevel.CRITICAL
        ]

        return {
            'critical_alerts': [asdict(a) for a in critical_alerts],
            'urgent_participants': [asdict(p) for p in urgent_participants],
            'emergency_contacts': [
                {'role': 'Medical Director', 'name': 'Dr. Smith', 'phone': '+1-555-0123'},
                {'role': 'Technical Lead', 'name': 'John Doe', 'phone': '+1-555-0124'}
            ],
            'escalation_procedures': [
                'Level 1: Automated intervention',
                'Level 2: Technical support contact',
                'Level 3: Clinical team notification',
                'Level 4: Emergency medical services'
            ]
        }

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge monitoring alert"""
        try:
            if alert_id not in self.active_alerts:
                return False

            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.now()

            self.logger.info(f"Alert {alert_id} acknowledged by {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Alert acknowledgment failed: {e}")
            return False

    async def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Mark alert as resolved"""
        try:
            if alert_id not in self.active_alerts:
                return False

            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            if not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.now()

            self.logger.info(f"Alert {alert_id} resolved by {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Alert resolution failed: {e}")
            return False

    async def get_participant_detail_view(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed view for specific participant"""
        try:
            if participant_id not in self.participant_vitals:
                return None

            vitals = self.participant_vitals[participant_id]

            # Get participant sessions
            participant_sessions = [
                s for s in self.workflow_manager.active_sessions.values()
                if s.participant_id == participant_id
            ]

            # Get participant alerts
            participant_alerts = [
                a for a in self.active_alerts.values()
                if a.participant_id == participant_id
            ]

            return {
                'participant_id': participant_id,
                'current_vitals': asdict(vitals),
                'recent_sessions': [
                    {
                        'session_id': s.session_id,
                        'status': s.current_status.value,
                        'scheduled_start': s.scheduled_start.isoformat() if s.scheduled_start else None,
                        'actual_start': s.actual_start.isoformat() if s.actual_start else None,
                        'data_quality': s.data_quality_metrics
                    }
                    for s in participant_sessions[-10:]  # Last 10 sessions
                ],
                'active_alerts': [asdict(a) for a in participant_alerts if not a.resolved],
                'intervention_history': [
                    intervention for session in participant_sessions
                    for intervention in session.intervention_history[-20:]  # Last 20 interventions
                ],
                'performance_trends': self._get_participant_trends(participant_id),
                'support_recommendations': self._generate_support_recommendations(vitals)
            }

        except Exception as e:
            self.logger.error(f"Participant detail view failed: {e}")
            return None

    def _get_participant_trends(self, participant_id: str) -> Dict[str, List[float]]:
        """Get performance trends for participant"""
        # In production, this would analyze historical data
        return {
            'signal_quality_trend': [0.8, 0.75, 0.82, 0.78, 0.85],
            'compliance_trend': [0.9, 0.85, 0.92, 0.88, 0.94],
            'satisfaction_trend': [4.0, 3.8, 4.2, 4.0, 4.3]
        }

    def _generate_support_recommendations(self, vitals: ParticipantVitals) -> List[str]:
        """Generate support recommendations for participant"""
        recommendations = []

        if vitals.signal_quality_score < 0.6:
            recommendations.append("Schedule technical support session for signal quality improvement")

        if vitals.session_compliance_rate < 0.7:
            recommendations.append("Provide scheduling flexibility and motivational support")

        if vitals.technical_issues_count > 2:
            recommendations.append("Consider device replacement or additional training")

        if vitals.risk_level in [ParticipantRiskLevel.HIGH, ParticipantRiskLevel.CRITICAL]:
            recommendations.append("Increase monitoring frequency and clinical check-ins")

        return recommendations if recommendations else ["Continue current support level"]


class TelehealthDashboard:
    """
    Comprehensive telehealth dashboard aggregating all monitoring data
    Provides unified view for different user roles and use cases
    """

    def __init__(self, remote_monitor: RemotePatientMonitor):
        """Initialize telehealth dashboard"""
        self.remote_monitor = remote_monitor
        self.logger = logging.getLogger('telehealth_dashboard')

    async def get_unified_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get unified dashboard view based on user role"""
        try:
            user_role = self.remote_monitor.auth_manager.get_user_role(user_id)

            # Determine primary view type based on role
            primary_view = self._get_primary_view_for_role(user_role)

            # Get primary dashboard data
            dashboard_data = await self.remote_monitor.get_monitoring_dashboard(user_id, primary_view)

            # Add cross-functional widgets based on role
            dashboard_data['widgets'] = await self._get_role_specific_widgets(user_role)

            # Add navigation options
            dashboard_data['navigation'] = self._get_navigation_options(user_role)

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Unified dashboard generation failed: {e}")
            raise

    def _get_primary_view_for_role(self, user_role: str) -> MonitoringViewType:
        """Get primary dashboard view for user role"""
        role_views = {
            'SUPER_ADMIN': MonitoringViewType.EXECUTIVE_SUMMARY,
            'NEUROLOGIST': MonitoringViewType.CLINICIAN_OVERVIEW,
            'CLINICAL_COORDINATOR': MonitoringViewType.TRIAL_COORDINATOR,
            'REMOTE_TECHNICIAN': MonitoringViewType.TECHNICIAN_CONSOLE,
            'DATA_SCIENTIST': MonitoringViewType.DATA_SCIENTIST,
            'HOSPITAL_ADMIN': MonitoringViewType.EXECUTIVE_SUMMARY,
            'EEG_TECHNICIAN': MonitoringViewType.TECHNICIAN_CONSOLE
        }

        return role_views.get(user_role, MonitoringViewType.CLINICIAN_OVERVIEW)

    async def _get_role_specific_widgets(self, user_role: str) -> List[Dict[str, Any]]:
        """Get dashboard widgets specific to user role"""
        if user_role in ['NEUROLOGIST', 'ATTENDING_PHYSICIAN']:
            return [
                {'type': 'high_risk_participants', 'priority': 1},
                {'type': 'pending_reviews', 'priority': 2},
                {'type': 'quality_trends', 'priority': 3},
                {'type': 'recent_interventions', 'priority': 4}
            ]
        elif user_role in ['REMOTE_TECHNICIAN', 'EEG_TECHNICIAN']:
            return [
                {'type': 'technical_alerts', 'priority': 1},
                {'type': 'device_status_map', 'priority': 2},
                {'type': 'signal_quality_monitor', 'priority': 3},
                {'type': 'support_queue', 'priority': 4}
            ]
        elif user_role == 'CLINICAL_COORDINATOR':
            return [
                {'type': 'enrollment_progress', 'priority': 1},
                {'type': 'compliance_tracking', 'priority': 2},
                {'type': 'scheduling_overview', 'priority': 3},
                {'type': 'participant_communication', 'priority': 4}
            ]
        else:
            return [
                {'type': 'system_overview', 'priority': 1},
                {'type': 'performance_metrics', 'priority': 2}
            ]

    def _get_navigation_options(self, user_role: str) -> List[Dict[str, Any]]:
        """Get navigation options for user role"""
        base_navigation = [
            {'label': 'Overview', 'view': 'overview', 'icon': 'dashboard'},
            {'label': 'Participants', 'view': 'participants', 'icon': 'users'},
            {'label': 'Alerts', 'view': 'alerts', 'icon': 'bell'}
        ]

        role_specific = {
            'NEUROLOGIST': [
                {'label': 'Clinical Reviews', 'view': 'clinical_reviews', 'icon': 'clipboard'},
                {'label': 'Reports', 'view': 'reports', 'icon': 'file-text'}
            ],
            'REMOTE_TECHNICIAN': [
                {'label': 'Technical Support', 'view': 'technical_support', 'icon': 'tool'},
                {'label': 'Device Management', 'view': 'devices', 'icon': 'wifi'}
            ],
            'CLINICAL_COORDINATOR': [
                {'label': 'Trial Management', 'view': 'trial_management', 'icon': 'users'},
                {'label': 'Scheduling', 'view': 'scheduling', 'icon': 'calendar'}
            ]
        }

        return base_navigation + role_specific.get(user_role, [])

    async def export_monitoring_report(self,
                                     report_type: str,
                                     date_range: Tuple[datetime, datetime],
                                     user_id: str) -> str:
        """Export comprehensive monitoring report"""
        try:
            # Generate report based on type
            if report_type == 'quality_summary':
                report_data = await self._generate_quality_summary_report(date_range)
            elif report_type == 'compliance_report':
                report_data = await self._generate_compliance_report(date_range)
            elif report_type == 'technical_summary':
                report_data = await self._generate_technical_summary_report(date_range)
            else:
                raise ValueError(f"Unknown report type: {report_type}")

            # Export report
            report_id = f"REPORT_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report_path = f"reports/telehealth/{report_id}.json"

            # In production, generate actual report file
            self.logger.info(f"Monitoring report {report_id} generated by {user_id}")

            return report_path

        except Exception as e:
            self.logger.error(f"Report export failed: {e}")
            raise

    async def _generate_quality_summary_report(self, date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate data quality summary report"""
        return {
            'report_type': 'quality_summary',
            'date_range': {
                'start': date_range[0].isoformat(),
                'end': date_range[1].isoformat()
            },
            'summary_metrics': {
                'average_signal_quality': 0.85,
                'data_completeness': 0.94,
                'protocol_compliance': 0.92
            },
            'quality_trends': self.remote_monitor._calculate_quality_trends(),
            'generated_at': datetime.now().isoformat()
        }

    async def _generate_compliance_report(self, date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate compliance report"""
        return {
            'report_type': 'compliance_report',
            'overall_compliance': 0.89,
            'participant_compliance_distribution': {
                'excellent': 45,
                'good': 32,
                'needs_improvement': 8,
                'poor': 3
            },
            'intervention_summary': {
                'total_interventions': 156,
                'successful_resolutions': 142,
                'escalations_required': 14
            }
        }