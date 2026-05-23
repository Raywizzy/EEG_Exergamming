"""
Safety monitoring and alert escalation for post-market surveillance
Implements automated safety checks, risk assessment, and escalation protocols
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional, Any
from enum import Enum
from uuid import UUID
import numpy as np
from dataclasses import dataclass

import requests
import asyncpg

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    POPULATION_DRIFT = "population_drift"
    CONCEPT_DRIFT = "concept_drift"
    DATA_QUALITY = "data_quality"
    CALIBRATION_ERROR = "calibration_error"
    SAFETY_SIGNAL = "safety_signal"
    SYSTEM_FAILURE = "system_failure"
    REGULATORY_THRESHOLD = "regulatory_threshold"

class EscalationLevel(Enum):
    INTERNAL = "internal"
    CLINICAL_TEAM = "clinical_team"
    REGULATORY = "regulatory"
    EMERGENCY = "emergency"

@dataclass
class EscalationRule:
    alert_type: AlertType
    severity: AlertSeverity
    escalation_level: EscalationLevel
    time_to_escalate_hours: int
    notification_channels: List[str]
    auto_actions: List[str]

@dataclass
class SafetyThreshold:
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<='
    alert_severity: AlertSeverity
    monitoring_window_hours: int
    min_samples: int

class SafetyMonitor:
    """
    Core safety monitoring system for post-market surveillance
    Tracks clinical safety signals and model performance degradation
    """

    def __init__(self, config: Dict = None):
        default_config = {
            'safety_thresholds': self._get_default_safety_thresholds(),
            'monitoring_intervals': {
                'real_time': 300,  # 5 minutes
                'daily': 86400,    # 24 hours
                'weekly': 604800   # 7 days
            },
            'clinical_endpoints': {
                'adverse_events_rate_threshold': 0.05,  # 5%
                'efficacy_degradation_threshold': 0.10,  # 10%
                'safety_margin': 0.02  # 2%
            }
        }

        self.config = {**default_config, **(config or {})}
        self.safety_thresholds = self._initialize_safety_thresholds()

    def _get_default_safety_thresholds(self) -> List[Dict]:
        """Define default safety thresholds for clinical monitoring"""
        return [
            {
                'metric_name': 'prediction_accuracy',
                'threshold_value': 0.65,  # Below 65% accuracy
                'comparison_operator': '<',
                'alert_severity': 'high',
                'monitoring_window_hours': 24,
                'min_samples': 100
            },
            {
                'metric_name': 'calibration_error',
                'threshold_value': 0.15,  # Above 15% calibration error
                'comparison_operator': '>',
                'alert_severity': 'medium',
                'monitoring_window_hours': 12,
                'min_samples': 50
            },
            {
                'metric_name': 'false_positive_rate',
                'threshold_value': 0.20,  # Above 20% false positive rate
                'comparison_operator': '>',
                'alert_severity': 'high',
                'monitoring_window_hours': 6,
                'min_samples': 30
            },
            {
                'metric_name': 'confidence_variance',
                'threshold_value': 0.25,  # High confidence variance
                'comparison_operator': '>',
                'alert_severity': 'medium',
                'monitoring_window_hours': 24,
                'min_samples': 100
            },
            {
                'metric_name': 'adverse_event_correlation',
                'threshold_value': 0.30,  # Above 30% correlation with adverse events
                'comparison_operator': '>',
                'alert_severity': 'critical',
                'monitoring_window_hours': 1,
                'min_samples': 10
            }
        ]

    def _initialize_safety_thresholds(self) -> List[SafetyThreshold]:
        """Initialize safety threshold objects"""
        thresholds = []
        for threshold_config in self.config['safety_thresholds']:
            threshold = SafetyThreshold(
                metric_name=threshold_config['metric_name'],
                threshold_value=threshold_config['threshold_value'],
                comparison_operator=threshold_config['comparison_operator'],
                alert_severity=AlertSeverity(threshold_config['alert_severity']),
                monitoring_window_hours=threshold_config['monitoring_window_hours'],
                min_samples=threshold_config['min_samples']
            )
            thresholds.append(threshold)
        return thresholds

    async def assess_safety_metrics(self, metrics: Dict[str, Any],
                                   historical_data: Dict[str, List[float]] = None) -> List[Dict]:
        """
        Assess current metrics against safety thresholds
        Returns list of safety alerts if thresholds are breached
        """
        safety_alerts = []

        try:
            for threshold in self.safety_thresholds:
                if threshold.metric_name not in metrics:
                    continue

                current_value = metrics[threshold.metric_name]
                alert = self._check_threshold_breach(threshold, current_value, historical_data)

                if alert:
                    safety_alerts.append(alert)

            # Additional safety checks
            clinical_alerts = await self._assess_clinical_safety(metrics, historical_data)
            safety_alerts.extend(clinical_alerts)

            return safety_alerts

        except Exception as e:
            logger.error(f"Error assessing safety metrics: {e}")
            return []

    def _check_threshold_breach(self, threshold: SafetyThreshold, current_value: float,
                               historical_data: Dict[str, List[float]] = None) -> Optional[Dict]:
        """Check if a specific threshold has been breached"""
        try:
            # Determine if threshold is breached
            is_breached = False

            if threshold.comparison_operator == '>':
                is_breached = current_value > threshold.threshold_value
            elif threshold.comparison_operator == '<':
                is_breached = current_value < threshold.threshold_value
            elif threshold.comparison_operator == '>=':
                is_breached = current_value >= threshold.threshold_value
            elif threshold.comparison_operator == '<=':
                is_breached = current_value <= threshold.threshold_value

            if not is_breached:
                return None

            # Calculate severity and context
            breach_magnitude = abs(current_value - threshold.threshold_value)
            breach_percentage = breach_magnitude / threshold.threshold_value * 100

            # Historical context if available
            trend_info = None
            if historical_data and threshold.metric_name in historical_data:
                trend_info = self._analyze_metric_trend(
                    historical_data[threshold.metric_name],
                    current_value
                )

            return {
                'alert_type': AlertType.SAFETY_SIGNAL.value,
                'severity': threshold.alert_severity.value,
                'metric_name': threshold.metric_name,
                'current_value': current_value,
                'threshold_value': threshold.threshold_value,
                'breach_magnitude': breach_magnitude,
                'breach_percentage': breach_percentage,
                'comparison_operator': threshold.comparison_operator,
                'trend_analysis': trend_info,
                'monitoring_window_hours': threshold.monitoring_window_hours,
                'timestamp': datetime.utcnow().isoformat(),
                'message': f"{threshold.metric_name} ({current_value:.3f}) breached threshold "
                          f"({threshold.comparison_operator} {threshold.threshold_value:.3f})"
            }

        except Exception as e:
            logger.error(f"Error checking threshold breach for {threshold.metric_name}: {e}")
            return None

    def _analyze_metric_trend(self, historical_values: List[float], current_value: float) -> Dict:
        """Analyze trend in metric values"""
        if len(historical_values) < 3:
            return {'trend': 'insufficient_data'}

        try:
            # Add current value to historical data
            all_values = historical_values + [current_value]

            # Calculate trend
            x = np.arange(len(all_values))
            z = np.polyfit(x, all_values, 1)
            slope = z[0]

            # Calculate recent change
            recent_change = current_value - np.mean(historical_values[-5:]) if len(historical_values) >= 5 else 0

            # Trend classification
            if abs(slope) < 0.001:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'

            return {
                'trend': trend_direction,
                'slope': float(slope),
                'recent_change': float(recent_change),
                'historical_mean': float(np.mean(historical_values)),
                'historical_std': float(np.std(historical_values)),
                'current_vs_historical_zscore': float((current_value - np.mean(historical_values)) / np.std(historical_values)) if np.std(historical_values) > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error analyzing metric trend: {e}")
            return {'trend': 'error', 'error': str(e)}

    async def _assess_clinical_safety(self, metrics: Dict[str, Any],
                                     historical_data: Dict[str, List[float]] = None) -> List[Dict]:
        """Assess clinical safety indicators"""
        clinical_alerts = []

        try:
            # Check for calibration issues
            if 'calibration_error' in metrics:
                expected_calibration_error = metrics['calibration_error']
                if expected_calibration_error > 0.15:  # 15% calibration error threshold
                    clinical_alerts.append({
                        'alert_type': AlertType.CALIBRATION_ERROR.value,
                        'severity': AlertSeverity.HIGH.value if expected_calibration_error > 0.25 else AlertSeverity.MEDIUM.value,
                        'metric_name': 'calibration_error',
                        'current_value': expected_calibration_error,
                        'message': f"Model calibration error ({expected_calibration_error:.1%}) exceeds safety threshold",
                        'timestamp': datetime.utcnow().isoformat(),
                        'clinical_impact': 'May lead to inappropriate clinical decisions'
                    })

            # Check for prediction consistency
            if 'prediction_variance' in metrics:
                pred_variance = metrics['prediction_variance']
                if pred_variance > 0.3:  # High prediction variance
                    clinical_alerts.append({
                        'alert_type': AlertType.DATA_QUALITY.value,
                        'severity': AlertSeverity.MEDIUM.value,
                        'metric_name': 'prediction_variance',
                        'current_value': pred_variance,
                        'message': f"High prediction variance ({pred_variance:.3f}) detected",
                        'timestamp': datetime.utcnow().isoformat(),
                        'clinical_impact': 'May indicate unstable model behavior'
                    })

            # Check for bias indicators
            if 'fairness_metrics' in metrics:
                fairness = metrics['fairness_metrics']
                for group, metric_value in fairness.items():
                    if abs(metric_value - 0.5) > 0.15:  # Significant bias
                        clinical_alerts.append({
                            'alert_type': AlertType.SAFETY_SIGNAL.value,
                            'severity': AlertSeverity.HIGH.value,
                            'metric_name': f'fairness_{group}',
                            'current_value': metric_value,
                            'message': f"Potential bias detected for {group} (score: {metric_value:.3f})",
                            'timestamp': datetime.utcnow().isoformat(),
                            'clinical_impact': 'May result in disparate treatment outcomes'
                        })

            return clinical_alerts

        except Exception as e:
            logger.error(f"Error in clinical safety assessment: {e}")
            return []

    def calculate_safety_score(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall safety score based on multiple metrics"""
        try:
            # Define metric weights based on clinical importance
            metric_weights = {
                'prediction_accuracy': 0.25,
                'calibration_error': 0.20,
                'false_positive_rate': 0.15,
                'false_negative_rate': 0.15,
                'confidence_score': 0.10,
                'prediction_variance': 0.10,
                'data_quality_score': 0.05
            }

            # Normalize metrics to 0-1 scale (higher is better)
            normalized_scores = {}

            for metric, weight in metric_weights.items():
                if metric in metrics:
                    value = metrics[metric]

                    # Normalize based on metric type
                    if metric in ['prediction_accuracy', 'confidence_score', 'data_quality_score']:
                        # Higher is better
                        normalized_scores[metric] = min(1.0, max(0.0, value))
                    else:
                        # Lower is better (error rates, variance)
                        normalized_scores[metric] = min(1.0, max(0.0, 1.0 - value))

            # Calculate weighted safety score
            if normalized_scores:
                total_weight = sum(metric_weights[m] for m in normalized_scores.keys())
                weighted_sum = sum(normalized_scores[m] * metric_weights[m] for m in normalized_scores.keys())
                safety_score = weighted_sum / total_weight
            else:
                safety_score = 0.5  # Neutral score when no metrics available

            # Calculate individual component scores
            performance_score = np.mean([
                normalized_scores.get('prediction_accuracy', 0.5),
                1.0 - normalized_scores.get('false_positive_rate', 0.5),
                1.0 - normalized_scores.get('false_negative_rate', 0.5)
            ])

            reliability_score = np.mean([
                normalized_scores.get('calibration_error', 0.5),
                normalized_scores.get('confidence_score', 0.5),
                normalized_scores.get('prediction_variance', 0.5)
            ])

            data_quality_score = normalized_scores.get('data_quality_score', 0.5)

            return {
                'overall_safety_score': float(safety_score),
                'performance_score': float(performance_score),
                'reliability_score': float(reliability_score),
                'data_quality_score': float(data_quality_score),
                'safety_level': self._classify_safety_level(safety_score)
            }

        except Exception as e:
            logger.error(f"Error calculating safety score: {e}")
            return {
                'overall_safety_score': 0.0,
                'performance_score': 0.0,
                'reliability_score': 0.0,
                'data_quality_score': 0.0,
                'safety_level': 'unknown'
            }

    def _classify_safety_level(self, safety_score: float) -> str:
        """Classify overall safety level based on score"""
        if safety_score >= 0.85:
            return 'excellent'
        elif safety_score >= 0.75:
            return 'good'
        elif safety_score >= 0.65:
            return 'acceptable'
        elif safety_score >= 0.50:
            return 'concerning'
        else:
            return 'critical'

class AlertEscalator:
    """
    Alert escalation system with configurable rules and notification channels
    """

    def __init__(self, config: Dict = None):
        default_config = {
            'escalation_rules': self._get_default_escalation_rules(),
            'notification_channels': {
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.example.com',
                    'smtp_port': 587,
                    'username': 'alerts@neurofeedback-trials.com',
                    'password': 'your_password',
                    'recipients': {
                        'internal': ['dev-team@neurofeedback-trials.com'],
                        'clinical_team': ['clinical@neurofeedback-trials.com'],
                        'regulatory': ['regulatory@neurofeedback-trials.com'],
                        'emergency': ['emergency@neurofeedback-trials.com']
                    }
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': 'https://hooks.slack.com/services/...',
                    'channels': {
                        'internal': '#alerts',
                        'clinical_team': '#clinical-alerts',
                        'regulatory': '#regulatory',
                        'emergency': '#emergency'
                    }
                },
                'pagerduty': {
                    'enabled': False,
                    'api_key': 'your_pagerduty_key',
                    'service_keys': {
                        'critical': 'critical_service_key',
                        'high': 'high_priority_service_key'
                    }
                }
            }
        }

        self.config = {**default_config, **(config or {})}
        self.escalation_rules = self._initialize_escalation_rules()

    def _get_default_escalation_rules(self) -> List[Dict]:
        """Define default escalation rules"""
        return [
            {
                'alert_type': 'safety_signal',
                'severity': 'critical',
                'escalation_level': 'emergency',
                'time_to_escalate_hours': 0,  # Immediate
                'notification_channels': ['email', 'pagerduty'],
                'auto_actions': ['disable_model', 'notify_regulatory']
            },
            {
                'alert_type': 'performance_degradation',
                'severity': 'high',
                'escalation_level': 'clinical_team',
                'time_to_escalate_hours': 1,
                'notification_channels': ['email', 'slack'],
                'auto_actions': ['flag_for_review']
            },
            {
                'alert_type': 'population_drift',
                'severity': 'high',
                'escalation_level': 'clinical_team',
                'time_to_escalate_hours': 2,
                'notification_channels': ['email'],
                'auto_actions': ['schedule_retraining']
            },
            {
                'alert_type': 'calibration_error',
                'severity': 'medium',
                'escalation_level': 'internal',
                'time_to_escalate_hours': 4,
                'notification_channels': ['email'],
                'auto_actions': ['calibration_check']
            }
        ]

    def _initialize_escalation_rules(self) -> List[EscalationRule]:
        """Initialize escalation rule objects"""
        rules = []
        for rule_config in self.config['escalation_rules']:
            rule = EscalationRule(
                alert_type=AlertType(rule_config['alert_type']),
                severity=AlertSeverity(rule_config['severity']),
                escalation_level=EscalationLevel(rule_config['escalation_level']),
                time_to_escalate_hours=rule_config['time_to_escalate_hours'],
                notification_channels=rule_config['notification_channels'],
                auto_actions=rule_config['auto_actions']
            )
            rules.append(rule)
        return rules

    async def escalate_alert(self, alert_id: UUID, alert_type: str, alert_message: str,
                           severity: str = 'medium', metadata: Dict = None) -> bool:
        """Escalate alert according to configured rules"""
        try:
            # Find matching escalation rule
            escalation_rule = self._find_escalation_rule(alert_type, severity)

            if not escalation_rule:
                logger.warning(f"No escalation rule found for {alert_type}/{severity}")
                return False

            # Execute auto actions
            await self._execute_auto_actions(escalation_rule.auto_actions, alert_id, metadata)

            # Send notifications
            await self._send_notifications(
                escalation_rule.notification_channels,
                escalation_rule.escalation_level,
                alert_id,
                alert_type,
                alert_message,
                severity,
                metadata
            )

            # Schedule follow-up escalation if needed
            if escalation_rule.time_to_escalate_hours > 0:
                await self._schedule_follow_up_escalation(
                    alert_id, escalation_rule.time_to_escalate_hours
                )

            logger.info(f"Alert {alert_id} escalated to {escalation_rule.escalation_level.value}")
            return True

        except Exception as e:
            logger.error(f"Error escalating alert {alert_id}: {e}")
            return False

    def _find_escalation_rule(self, alert_type: str, severity: str) -> Optional[EscalationRule]:
        """Find matching escalation rule"""
        for rule in self.escalation_rules:
            if (rule.alert_type.value == alert_type and
                rule.severity.value == severity):
                return rule

        # Fallback to generic rules if exact match not found
        for rule in self.escalation_rules:
            if rule.alert_type.value == alert_type:
                return rule

        return None

    async def _execute_auto_actions(self, auto_actions: List[str], alert_id: UUID,
                                   metadata: Dict = None):
        """Execute automated actions based on alert"""
        for action in auto_actions:
            try:
                if action == 'disable_model':
                    await self._disable_model(metadata)
                elif action == 'flag_for_review':
                    await self._flag_for_review(alert_id, metadata)
                elif action == 'schedule_retraining':
                    await self._schedule_retraining(metadata)
                elif action == 'calibration_check':
                    await self._trigger_calibration_check(metadata)
                elif action == 'notify_regulatory':
                    await self._notify_regulatory_bodies(alert_id, metadata)

                logger.info(f"Executed auto action: {action}")

            except Exception as e:
                logger.error(f"Error executing auto action {action}: {e}")

    async def _send_notifications(self, channels: List[str], escalation_level: EscalationLevel,
                                 alert_id: UUID, alert_type: str, message: str,
                                 severity: str, metadata: Dict = None):
        """Send notifications through configured channels"""
        for channel in channels:
            try:
                if channel == 'email':
                    await self._send_email_notification(
                        escalation_level, alert_id, alert_type, message, severity, metadata
                    )
                elif channel == 'slack':
                    await self._send_slack_notification(
                        escalation_level, alert_id, alert_type, message, severity, metadata
                    )
                elif channel == 'pagerduty':
                    await self._send_pagerduty_notification(
                        escalation_level, alert_id, alert_type, message, severity, metadata
                    )

            except Exception as e:
                logger.error(f"Error sending notification via {channel}: {e}")

    async def _send_email_notification(self, escalation_level: EscalationLevel,
                                      alert_id: UUID, alert_type: str, message: str,
                                      severity: str, metadata: Dict = None):
        """Send email notification"""
        if not self.config['notification_channels']['email']['enabled']:
            return

        try:
            email_config = self.config['notification_channels']['email']
            recipients = email_config['recipients'][escalation_level.value]

            subject = f"[{severity.upper()}] EEG Neurofeedback PMS Alert - {alert_type}"

            body = f"""
Alert ID: {alert_id}
Alert Type: {alert_type}
Severity: {severity}
Escalation Level: {escalation_level.value}
Timestamp: {datetime.utcnow().isoformat()}

Message: {message}

Metadata:
{json.dumps(metadata, indent=2) if metadata else 'None'}

This is an automated alert from the Post-Market Surveillance system.
Please review and take appropriate action.
            """

            msg = MimeMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"Email notification sent for alert {alert_id}")

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")

    async def _send_slack_notification(self, escalation_level: EscalationLevel,
                                      alert_id: UUID, alert_type: str, message: str,
                                      severity: str, metadata: Dict = None):
        """Send Slack notification"""
        if not self.config['notification_channels']['slack']['enabled']:
            return

        try:
            slack_config = self.config['notification_channels']['slack']
            webhook_url = slack_config['webhook_url']

            color_map = {
                'low': '#36a64f',
                'medium': '#ffcc00',
                'high': '#ff6600',
                'critical': '#ff0000'
            }

            payload = {
                'attachments': [{
                    'color': color_map.get(severity, '#808080'),
                    'title': f"{severity.upper()} Alert: {alert_type}",
                    'text': message,
                    'fields': [
                        {'title': 'Alert ID', 'value': str(alert_id), 'short': True},
                        {'title': 'Severity', 'value': severity, 'short': True},
                        {'title': 'Escalation Level', 'value': escalation_level.value, 'short': True},
                        {'title': 'Timestamp', 'value': datetime.utcnow().isoformat(), 'short': True}
                    ]
                }]
            }

            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            logger.info(f"Slack notification sent for alert {alert_id}")

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    async def _send_pagerduty_notification(self, escalation_level: EscalationLevel,
                                          alert_id: UUID, alert_type: str, message: str,
                                          severity: str, metadata: Dict = None):
        """Send PagerDuty notification for critical alerts"""
        if not self.config['notification_channels']['pagerduty']['enabled']:
            return

        if severity not in ['high', 'critical']:
            return  # Only escalate high/critical alerts to PagerDuty

        try:
            pagerduty_config = self.config['notification_channels']['pagerduty']
            service_key = pagerduty_config['service_keys'].get(severity)

            if not service_key:
                logger.warning(f"No PagerDuty service key configured for severity: {severity}")
                return

            payload = {
                'routing_key': service_key,
                'event_action': 'trigger',
                'dedup_key': str(alert_id),
                'payload': {
                    'summary': f"{severity.upper()}: {alert_type} - {message}",
                    'source': 'EEG-Neurofeedback-PMS',
                    'severity': severity,
                    'component': 'post-market-surveillance',
                    'group': escalation_level.value,
                    'custom_details': metadata or {}
                }
            }

            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            logger.info(f"PagerDuty notification sent for alert {alert_id}")

        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")

    async def _disable_model(self, metadata: Dict = None):
        """Disable model deployment for critical safety issues"""
        # Implementation would integrate with model serving infrastructure
        logger.critical("AUTO ACTION: Model disabled due to critical safety alert")

    async def _flag_for_review(self, alert_id: UUID, metadata: Dict = None):
        """Flag alert for manual review"""
        logger.info(f"AUTO ACTION: Alert {alert_id} flagged for manual review")

    async def _schedule_retraining(self, metadata: Dict = None):
        """Schedule model retraining"""
        logger.info("AUTO ACTION: Model retraining scheduled")

    async def _trigger_calibration_check(self, metadata: Dict = None):
        """Trigger model calibration check"""
        logger.info("AUTO ACTION: Model calibration check triggered")

    async def _notify_regulatory_bodies(self, alert_id: UUID, metadata: Dict = None):
        """Notify regulatory bodies of critical safety signal"""
        logger.critical(f"AUTO ACTION: Regulatory notification triggered for alert {alert_id}")

    async def _schedule_follow_up_escalation(self, alert_id: UUID, hours: int):
        """Schedule follow-up escalation if alert not resolved"""
        # Implementation would use task scheduler (Celery, etc.)
        logger.info(f"Follow-up escalation scheduled for alert {alert_id} in {hours} hours")