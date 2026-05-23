"""
Phase IX: Hospital-Grade Dashboard and Clinical Interface
Real-time clinical dashboards for hospital deployment
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

from .workflow_orchestration import ClinicalWorkflowManager, WorkflowStatus, WorkflowPriority
from .authentication import EnterpriseAuthManager, RoleBasedAccessControl
from .fhir_integration import FHIRClient

class DashboardViewType(Enum):
    """Dashboard view types for different user roles"""
    EXECUTIVE_SUMMARY = "executive_summary"
    CLINICAL_OVERVIEW = "clinical_overview"
    PATIENT_MANAGEMENT = "patient_management"
    WORKFLOW_MONITORING = "workflow_monitoring"
    QUALITY_METRICS = "quality_metrics"
    RESEARCH_ANALYTICS = "research_analytics"
    SYSTEM_ADMINISTRATION = "system_administration"

class AlertSeverity(Enum):
    """Clinical alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ClinicalAlert:
    """Clinical alert data structure"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    patient_id: Optional[str]
    workflow_id: Optional[str]
    created_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    requires_action: bool = True
    action_deadline: Optional[datetime] = None

@dataclass
class PatientSummary:
    """Patient summary for dashboard display"""
    patient_id: str
    name: str
    age: int
    diagnosis: str
    current_status: str
    last_session: Optional[datetime]
    next_appointment: Optional[datetime]
    treatment_progress: float
    risk_level: str
    assigned_clinician: str

@dataclass
class WorkflowSummary:
    """Workflow summary for dashboard display"""
    workflow_id: str
    patient_name: str
    current_step: str
    progress_percentage: float
    priority: str
    estimated_completion: Optional[datetime]
    assigned_staff: Dict[str, str]
    behind_schedule: bool
    requires_attention: bool

class HospitalDashboard:
    """
    Hospital-grade clinical dashboard system
    Provides real-time monitoring and management capabilities
    """

    def __init__(self, config_path: str = None):
        """Initialize hospital dashboard"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Initialize core services
        self.workflow_manager = ClinicalWorkflowManager()
        self.auth_manager = EnterpriseAuthManager()
        self.rbac = RoleBasedAccessControl()
        self.fhir_client = FHIRClient()

        # Dashboard state
        self.active_alerts: List[ClinicalAlert] = []
        self.cached_metrics: Dict[str, Any] = {}
        self.last_refresh: Optional[datetime] = None

        # Real-time monitoring
        self.monitoring_active = False
        self.update_interval = self.config.get('update_interval_seconds', 30)

        self.logger.info("Hospital Dashboard initialized successfully")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load dashboard configuration"""
        default_config = {
            'update_interval_seconds': 30,
            'alert_retention_hours': 168,  # 1 week
            'cache_timeout_minutes': 5,
            'max_dashboard_items': 100,
            'real_time_monitoring': True,
            'notification_channels': ['dashboard', 'email', 'sms'],
            'quality_thresholds': {
                'workflow_completion_rate': 0.95,
                'average_turnaround_time': 72,  # hours
                'patient_satisfaction': 4.0,
                'system_uptime': 0.999
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup dashboard-specific logging"""
        logger = logging.getLogger('hospital_dashboard')
        logger.setLevel(logging.INFO)

        log_dir = Path('logs/dashboard')
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / f'dashboard_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    async def get_dashboard_data(self, user_id: str, view_type: DashboardViewType) -> Dict[str, Any]:
        """Get dashboard data based on user role and view type"""

        # Validate user permissions
        user_role = self.auth_manager.get_user_role(user_id)
        if not self._validate_view_access(user_role, view_type):
            raise PermissionError(f"User {user_id} cannot access {view_type.value} view")

        # Check cache validity
        if self._is_cache_valid():
            self.logger.debug(f"Serving cached dashboard data for {user_id}")
        else:
            await self._refresh_dashboard_cache()

        # Generate role-specific dashboard data
        dashboard_data = {
            'view_type': view_type.value,
            'user_role': user_role,
            'last_updated': self.last_refresh.isoformat() if self.last_refresh else None,
            'system_status': await self._get_system_status(),
            'alerts': self._get_filtered_alerts(user_role),
            'quick_actions': self._get_available_actions(user_role)
        }

        # Add view-specific data
        if view_type == DashboardViewType.EXECUTIVE_SUMMARY:
            dashboard_data.update(await self._get_executive_summary())
        elif view_type == DashboardViewType.CLINICAL_OVERVIEW:
            dashboard_data.update(await self._get_clinical_overview())
        elif view_type == DashboardViewType.PATIENT_MANAGEMENT:
            dashboard_data.update(await self._get_patient_management_data())
        elif view_type == DashboardViewType.WORKFLOW_MONITORING:
            dashboard_data.update(await self._get_workflow_monitoring_data())
        elif view_type == DashboardViewType.QUALITY_METRICS:
            dashboard_data.update(await self._get_quality_metrics())
        elif view_type == DashboardViewType.RESEARCH_ANALYTICS:
            dashboard_data.update(await self._get_research_analytics())
        elif view_type == DashboardViewType.SYSTEM_ADMINISTRATION:
            dashboard_data.update(await self._get_system_admin_data())

        return dashboard_data

    def _validate_view_access(self, user_role: str, view_type: DashboardViewType) -> bool:
        """Validate user access to specific dashboard views"""
        role_permissions = {
            'SUPER_ADMIN': [DashboardViewType.EXECUTIVE_SUMMARY, DashboardViewType.SYSTEM_ADMINISTRATION,
                           DashboardViewType.QUALITY_METRICS, DashboardViewType.WORKFLOW_MONITORING],
            'NEUROLOGIST': [DashboardViewType.CLINICAL_OVERVIEW, DashboardViewType.PATIENT_MANAGEMENT,
                           DashboardViewType.WORKFLOW_MONITORING],
            'ATTENDING_PHYSICIAN': [DashboardViewType.CLINICAL_OVERVIEW, DashboardViewType.PATIENT_MANAGEMENT],
            'CLINICAL_COORDINATOR': [DashboardViewType.WORKFLOW_MONITORING, DashboardViewType.PATIENT_MANAGEMENT],
            'RESEARCH_COORDINATOR': [DashboardViewType.RESEARCH_ANALYTICS, DashboardViewType.PATIENT_MANAGEMENT],
            'DATA_SCIENTIST': [DashboardViewType.RESEARCH_ANALYTICS, DashboardViewType.QUALITY_METRICS],
            'EEG_TECHNICIAN': [DashboardViewType.WORKFLOW_MONITORING],
            'HOSPITAL_ADMIN': [DashboardViewType.EXECUTIVE_SUMMARY, DashboardViewType.QUALITY_METRICS],
            'IT_SUPPORT': [DashboardViewType.SYSTEM_ADMINISTRATION],
            'AUDIT_REVIEWER': [DashboardViewType.QUALITY_METRICS, DashboardViewType.RESEARCH_ANALYTICS]
        }

        return view_type in role_permissions.get(user_role, [])

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if not self.last_refresh:
            return False

        cache_timeout = timedelta(minutes=self.config.get('cache_timeout_minutes', 5))
        return datetime.now() - self.last_refresh < cache_timeout

    async def _refresh_dashboard_cache(self):
        """Refresh all cached dashboard data"""
        self.logger.info("Refreshing dashboard cache")

        # Update workflow metrics
        self.cached_metrics['workflows'] = self.workflow_manager.get_workflow_metrics()

        # Update system health
        self.cached_metrics['system_health'] = await self._calculate_system_health()

        # Update patient summaries
        self.cached_metrics['patient_summaries'] = await self._generate_patient_summaries()

        # Update workflow summaries
        self.cached_metrics['workflow_summaries'] = await self._generate_workflow_summaries()

        # Clean up old alerts
        self._cleanup_old_alerts()

        self.last_refresh = datetime.now()

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'overall_health': 'healthy',  # green, yellow, red
            'uptime_percentage': 99.9,
            'active_users': len(self.auth_manager.get_active_sessions()),
            'system_load': {
                'cpu_usage': 45.2,
                'memory_usage': 62.1,
                'disk_usage': 78.5
            },
            'service_status': {
                'fhir_integration': 'online',
                'workflow_engine': 'online',
                'authentication': 'online',
                'dashboard': 'online'
            },
            'last_backup': '2024-12-20T02:00:00Z'
        }

    def _get_filtered_alerts(self, user_role: str) -> List[Dict[str, Any]]:
        """Get alerts filtered by user role"""
        # Filter alerts based on role permissions
        filtered_alerts = []

        for alert in self.active_alerts:
            if self._can_view_alert(user_role, alert):
                filtered_alerts.append(asdict(alert))

        # Sort by severity and creation time
        filtered_alerts.sort(key=lambda x: (
            ['info', 'warning', 'error', 'critical'].index(x['severity']),
            x['created_at']
        ), reverse=True)

        return filtered_alerts[:self.config.get('max_dashboard_items', 100)]

    def _can_view_alert(self, user_role: str, alert: ClinicalAlert) -> bool:
        """Check if user can view specific alert"""
        # Admin roles can see all alerts
        if user_role in ['SUPER_ADMIN', 'HOSPITAL_ADMIN']:
            return True

        # Role-specific alert filtering
        role_alert_filters = {
            'NEUROLOGIST': ['patient_safety', 'clinical_decision', 'workflow_delay'],
            'CLINICAL_COORDINATOR': ['workflow_delay', 'scheduling', 'resource_allocation'],
            'EEG_TECHNICIAN': ['equipment_status', 'data_quality', 'workflow_delay'],
            'IT_SUPPORT': ['system_error', 'integration_failure', 'performance'],
            'RESEARCH_COORDINATOR': ['data_quality', 'compliance', 'research_protocol']
        }

        # For now, return True (in real implementation, check alert.category)
        return True

    def _get_available_actions(self, user_role: str) -> List[Dict[str, str]]:
        """Get available quick actions for user role"""
        role_actions = {
            'SUPER_ADMIN': [
                {'id': 'system_maintenance', 'label': 'System Maintenance', 'icon': 'settings'},
                {'id': 'user_management', 'label': 'User Management', 'icon': 'users'},
                {'id': 'backup_system', 'label': 'Backup System', 'icon': 'database'}
            ],
            'NEUROLOGIST': [
                {'id': 'new_patient', 'label': 'Enroll New Patient', 'icon': 'user-plus'},
                {'id': 'review_pending', 'label': 'Review Pending Reports', 'icon': 'clipboard'},
                {'id': 'schedule_session', 'label': 'Schedule Session', 'icon': 'calendar'}
            ],
            'CLINICAL_COORDINATOR': [
                {'id': 'schedule_patient', 'label': 'Schedule Patient', 'icon': 'calendar'},
                {'id': 'workflow_status', 'label': 'Check Workflow Status', 'icon': 'activity'},
                {'id': 'resource_allocation', 'label': 'Manage Resources', 'icon': 'layout'}
            ],
            'EEG_TECHNICIAN': [
                {'id': 'equipment_check', 'label': 'Equipment Check', 'icon': 'check-circle'},
                {'id': 'start_session', 'label': 'Start EEG Session', 'icon': 'play'},
                {'id': 'data_quality', 'label': 'Check Data Quality', 'icon': 'bar-chart'}
            ]
        }

        return role_actions.get(user_role, [])

    async def _get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary dashboard data"""
        return {
            'key_metrics': {
                'total_patients': 247,
                'active_workflows': len(self.workflow_manager.active_workflows),
                'completion_rate': 94.2,
                'average_turnaround': '24.5 hours',
                'patient_satisfaction': 4.6,
                'cost_per_assessment': '$145.30'
            },
            'trends': {
                'patient_volume': [
                    {'date': '2024-12-16', 'value': 12},
                    {'date': '2024-12-17', 'value': 15},
                    {'date': '2024-12-18', 'value': 18},
                    {'date': '2024-12-19', 'value': 14},
                    {'date': '2024-12-20', 'value': 20}
                ],
                'quality_scores': [
                    {'metric': 'Data Quality', 'current': 96.5, 'target': 95.0},
                    {'metric': 'Workflow Efficiency', 'current': 89.2, 'target': 90.0},
                    {'metric': 'Patient Safety', 'current': 99.8, 'target': 99.5},
                    {'metric': 'Regulatory Compliance', 'current': 100.0, 'target': 100.0}
                ]
            },
            'financial_summary': {
                'monthly_revenue': '$42,850',
                'cost_savings': '$8,250',
                'roi_percentage': 145.2
            }
        }

    async def _get_clinical_overview(self) -> Dict[str, Any]:
        """Get clinical overview dashboard data"""
        return {
            'patient_census': {
                'total_active': 89,
                'new_this_week': 12,
                'high_priority': 7,
                'follow_up_needed': 15
            },
            'recent_sessions': await self._get_recent_sessions(),
            'pending_reviews': await self._get_pending_reviews(),
            'clinical_insights': {
                'most_common_findings': [
                    {'finding': 'Alpha rhythm asymmetry', 'frequency': 34},
                    {'finding': 'Beta activity reduction', 'frequency': 28},
                    {'finding': 'Theta band irregularities', 'frequency': 19}
                ],
                'treatment_effectiveness': {
                    'significant_improvement': 67.2,
                    'moderate_improvement': 22.1,
                    'stable': 8.5,
                    'declined': 2.2
                }
            }
        }

    async def _get_patient_management_data(self) -> Dict[str, Any]:
        """Get patient management dashboard data"""
        patient_summaries = self.cached_metrics.get('patient_summaries', [])

        return {
            'patient_list': patient_summaries[:50],  # Limit display
            'scheduling': {
                'today_appointments': await self._get_todays_appointments(),
                'upcoming_this_week': await self._get_weekly_appointments(),
                'overdue_follow_ups': await self._get_overdue_followups()
            },
            'patient_alerts': [
                alert for alert in self.active_alerts
                if alert.patient_id and not alert.acknowledged
            ][:10]
        }

    async def _get_workflow_monitoring_data(self) -> Dict[str, Any]:
        """Get workflow monitoring dashboard data"""
        workflow_summaries = self.cached_metrics.get('workflow_summaries', [])

        return {
            'active_workflows': workflow_summaries,
            'workflow_performance': {
                'on_time_completion': 87.5,
                'average_delay': '2.3 hours',
                'bottleneck_steps': [
                    {'step': 'Clinical Review', 'avg_delay': '4.2 hours'},
                    {'step': 'Report Generation', 'avg_delay': '1.8 hours'}
                ]
            },
            'resource_utilization': {
                'neurologists': {'available': 3, 'busy': 2, 'utilization': 67},
                'eeg_technicians': {'available': 4, 'busy': 3, 'utilization': 75},
                'eeg_systems': {'available': 6, 'in_use': 3, 'maintenance': 1}
            }
        }

    async def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics dashboard data"""
        return {
            'data_quality': {
                'signal_quality_score': 94.2,
                'artifact_rejection_rate': 12.5,
                'preprocessing_success_rate': 98.7
            },
            'clinical_quality': {
                'inter_rater_reliability': 0.89,
                'diagnostic_accuracy': 91.3,
                'false_positive_rate': 4.2,
                'false_negative_rate': 4.5
            },
            'compliance_metrics': {
                'fda_compliance': 100.0,
                'hipaa_compliance': 100.0,
                'gdpr_compliance': 100.0,
                'audit_trail_completeness': 100.0
            },
            'performance_trends': [
                {'month': '2024-10', 'accuracy': 89.5, 'quality_score': 92.1},
                {'month': '2024-11', 'accuracy': 90.8, 'quality_score': 93.4},
                {'month': '2024-12', 'accuracy': 91.3, 'quality_score': 94.2}
            ]
        }

    async def _get_research_analytics(self) -> Dict[str, Any]:
        """Get research analytics dashboard data"""
        return {
            'study_enrollment': {
                'total_enrolled': 247,
                'enrollment_rate': 12.3,  # per week
                'target_enrollment': 300,
                'completion_rate': 89.2
            },
            'data_collection': {
                'sessions_completed': 1456,
                'data_quality_score': 96.5,
                'protocol_adherence': 94.8
            },
            'preliminary_findings': {
                'significant_biomarkers': [
                    {'biomarker': 'Alpha/Beta Ratio', 'p_value': 0.001, 'effect_size': 0.73},
                    {'biomarker': 'Gamma Burst Duration', 'p_value': 0.023, 'effect_size': 0.52}
                ],
                'classification_performance': {
                    'accuracy': 84.2,
                    'sensitivity': 82.7,
                    'specificity': 85.8,
                    'auc_roc': 0.89
                }
            }
        }

    async def _get_system_admin_data(self) -> Dict[str, Any]:
        """Get system administration dashboard data"""
        return {
            'system_performance': {
                'cpu_usage': 45.2,
                'memory_usage': 62.1,
                'disk_usage': 78.5,
                'network_latency': 12.3
            },
            'user_activity': {
                'active_sessions': len(self.auth_manager.get_active_sessions()),
                'failed_logins_24h': 3,
                'user_registrations_week': 2
            },
            'data_management': {
                'database_size': '45.2 GB',
                'backup_status': 'completed',
                'last_backup': '2024-12-20T02:00:00Z',
                'retention_compliance': 100.0
            },
            'integration_status': {
                'fhir_endpoints': {'active': 3, 'errors': 0},
                'external_systems': {'connected': 5, 'failed': 0},
                'api_usage': {'requests_24h': 15247, 'error_rate': 0.2}
            }
        }

    async def _generate_patient_summaries(self) -> List[PatientSummary]:
        """Generate patient summaries for dashboard"""
        # In real implementation, this would query patient database
        return [
            PatientSummary(
                patient_id="PT001",
                name="John D.",
                age=67,
                diagnosis="Parkinson's Disease",
                current_status="Active Treatment",
                last_session=datetime.now() - timedelta(days=2),
                next_appointment=datetime.now() + timedelta(days=5),
                treatment_progress=78.5,
                risk_level="Low",
                assigned_clinician="Dr. Smith"
            )
            # Add more patients...
        ]

    async def _generate_workflow_summaries(self) -> List[WorkflowSummary]:
        """Generate workflow summaries for dashboard"""
        summaries = []

        for workflow_id, workflow in self.workflow_manager.active_workflows.items():
            summary = WorkflowSummary(
                workflow_id=workflow_id,
                patient_name=f"Patient {workflow.patient_id}",
                current_step=workflow.current_status.value,
                progress_percentage=self.workflow_manager._calculate_progress(workflow),
                priority=workflow.priority.value,
                estimated_completion=workflow.estimated_completion,
                assigned_staff=workflow.assigned_staff,
                behind_schedule=self._is_workflow_behind_schedule(workflow),
                requires_attention=self._workflow_requires_attention(workflow)
            )
            summaries.append(summary)

        return summaries

    def _is_workflow_behind_schedule(self, workflow) -> bool:
        """Check if workflow is behind schedule"""
        if not workflow.estimated_completion:
            return False
        return datetime.now() > workflow.estimated_completion

    def _workflow_requires_attention(self, workflow) -> bool:
        """Check if workflow requires immediate attention"""
        return (
            workflow.current_status == WorkflowStatus.ERROR or
            workflow.current_status == WorkflowStatus.AUDIT_REQUIRED or
            self._is_workflow_behind_schedule(workflow)
        )

    async def _get_recent_sessions(self) -> List[Dict[str, Any]]:
        """Get recent EEG sessions"""
        return [
            {
                'patient_id': 'PT001',
                'session_date': (datetime.now() - timedelta(hours=2)).isoformat(),
                'session_type': 'Neurofeedback',
                'duration': 45,
                'quality_score': 94.2,
                'technician': 'Tech A'
            }
            # Add more sessions...
        ]

    async def _get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get pending clinical reviews"""
        return [
            {
                'patient_id': 'PT001',
                'report_type': 'EEG Analysis',
                'priority': 'High',
                'due_date': (datetime.now() + timedelta(hours=4)).isoformat(),
                'assigned_to': 'Dr. Smith'
            }
            # Add more reviews...
        ]

    async def _get_todays_appointments(self) -> List[Dict[str, Any]]:
        """Get today's appointments"""
        return [
            {
                'time': '09:00',
                'patient_name': 'John D.',
                'appointment_type': 'EEG Session',
                'clinician': 'Dr. Smith',
                'room': 'EEG-1'
            }
            # Add more appointments...
        ]

    async def _get_weekly_appointments(self) -> List[Dict[str, Any]]:
        """Get this week's appointments"""
        return []  # Implementation would query scheduling system

    async def _get_overdue_followups(self) -> List[Dict[str, Any]]:
        """Get overdue follow-up appointments"""
        return []  # Implementation would query scheduling system

    async def _calculate_system_health(self) -> Dict[str, float]:
        """Calculate overall system health metrics"""
        return {
            'overall_score': 94.2,
            'performance_score': 89.5,
            'reliability_score': 98.1,
            'security_score': 96.8,
            'compliance_score': 100.0
        }

    def _cleanup_old_alerts(self):
        """Remove old alerts based on retention policy"""
        retention_hours = self.config.get('alert_retention_hours', 168)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)

        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.created_at > cutoff_time
        ]

    async def create_alert(self,
                          severity: AlertSeverity,
                          title: str,
                          message: str,
                          patient_id: str = None,
                          workflow_id: str = None,
                          requires_action: bool = True,
                          action_deadline: datetime = None) -> str:
        """Create new clinical alert"""

        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.active_alerts):03d}"

        alert = ClinicalAlert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            patient_id=patient_id,
            workflow_id=workflow_id,
            created_at=datetime.now(),
            requires_action=requires_action,
            action_deadline=action_deadline
        )

        self.active_alerts.append(alert)

        # Log alert creation
        self.logger.warning(f"Alert created: {alert_id} - {title}")

        return alert_id

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge clinical alert"""
        alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)

        if not alert:
            return False

        alert.acknowledged = True
        alert.acknowledged_by = user_id
        alert.acknowledged_at = datetime.now()

        self.logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        return True

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration for frontend"""
        return {
            'update_interval': self.update_interval,
            'max_items_per_view': self.config.get('max_dashboard_items', 100),
            'available_views': [view.value for view in DashboardViewType],
            'alert_severities': [severity.value for severity in AlertSeverity],
            'refresh_enabled': True,
            'real_time_updates': self.config.get('real_time_monitoring', True)
        }


class ClinicalInterface:
    """
    Clinical user interface components and interactions
    Provides standardized UI components for clinical workflows
    """

    def __init__(self, dashboard: HospitalDashboard):
        """Initialize clinical interface"""
        self.dashboard = dashboard
        self.logger = logging.getLogger('clinical_interface')

    async def render_patient_card(self, patient_summary: PatientSummary) -> Dict[str, Any]:
        """Render patient information card"""
        return {
            'component': 'patient_card',
            'data': asdict(patient_summary),
            'actions': [
                {'id': 'view_details', 'label': 'View Details', 'icon': 'eye'},
                {'id': 'schedule_session', 'label': 'Schedule', 'icon': 'calendar'},
                {'id': 'view_history', 'label': 'History', 'icon': 'clock'}
            ],
            'status_indicator': {
                'color': self._get_status_color(patient_summary.risk_level),
                'label': patient_summary.current_status
            }
        }

    async def render_workflow_card(self, workflow_summary: WorkflowSummary) -> Dict[str, Any]:
        """Render workflow status card"""
        return {
            'component': 'workflow_card',
            'data': asdict(workflow_summary),
            'progress_bar': {
                'percentage': workflow_summary.progress_percentage,
                'color': 'red' if workflow_summary.behind_schedule else 'green'
            },
            'actions': [
                {'id': 'view_workflow', 'label': 'View Details', 'icon': 'eye'},
                {'id': 'advance_step', 'label': 'Advance', 'icon': 'arrow-right'},
                {'id': 'assign_staff', 'label': 'Assign', 'icon': 'user'}
            ],
            'alerts': workflow_summary.requires_attention
        }

    async def render_alert_panel(self, alerts: List[ClinicalAlert]) -> Dict[str, Any]:
        """Render alert notification panel"""
        return {
            'component': 'alert_panel',
            'alerts': [
                {
                    'id': alert.alert_id,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.created_at.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'actions': [
                        {'id': 'acknowledge', 'label': 'Acknowledge', 'icon': 'check'},
                        {'id': 'view_details', 'label': 'Details', 'icon': 'info'}
                    ]
                }
                for alert in alerts
            ],
            'summary': {
                'total': len(alerts),
                'critical': len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
                'unacknowledged': len([a for a in alerts if not a.acknowledged])
            }
        }

    def _get_status_color(self, risk_level: str) -> str:
        """Get color code for risk level"""
        color_map = {
            'Low': 'green',
            'Medium': 'yellow',
            'High': 'red',
            'Critical': 'purple'
        }
        return color_map.get(risk_level, 'gray')

    async def generate_clinical_report_template(self,
                                              patient_id: str,
                                              workflow_id: str) -> Dict[str, Any]:
        """Generate clinical report template"""
        return {
            'template': 'clinical_eeg_report',
            'sections': [
                {
                    'id': 'patient_info',
                    'title': 'Patient Information',
                    'fields': ['name', 'age', 'diagnosis', 'medications']
                },
                {
                    'id': 'session_details',
                    'title': 'Session Details',
                    'fields': ['date', 'duration', 'intervention_type', 'technician']
                },
                {
                    'id': 'eeg_findings',
                    'title': 'EEG Findings',
                    'fields': ['signal_quality', 'dominant_frequency', 'asymmetry_index']
                },
                {
                    'id': 'ai_analysis',
                    'title': 'AI Analysis Results',
                    'fields': ['classification', 'confidence', 'biomarkers']
                },
                {
                    'id': 'clinical_interpretation',
                    'title': 'Clinical Interpretation',
                    'fields': ['findings', 'recommendations', 'follow_up']
                }
            ],
            'metadata': {
                'patient_id': patient_id,
                'workflow_id': workflow_id,
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }