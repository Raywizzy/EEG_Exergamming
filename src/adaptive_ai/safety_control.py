"""
Safety-First Control Stack for Adaptive Clinical AI
Implements hard safety gates, rate limits, and confidence-weighted control
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from uuid import uuid4

logger = logging.getLogger(__name__)

class SafetyState(Enum):
    """Safety states for adaptive controller"""
    IDLE = "IDLE"                    # Not delivering feedback
    ARMED = "ARMED"                  # Ready to deliver, gates checking
    DELIVERING = "DELIVERING"        # Actively delivering feedback
    PAUSED = "PAUSED"               # Temporarily paused
    ROLLBACK = "ROLLBACK"           # Rolling back to safe state
    EMERGENCY_STOP = "EMERGENCY_STOP"  # Emergency halt

class QualityStatus(Enum):
    """Signal quality status"""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

@dataclass
class QualityMetrics:
    """Signal quality assessment metrics"""
    status: QualityStatus
    impedance_ok: bool
    artifact_rate: float
    coverage_percentage: float
    motion_level: float
    power_line_interference: float

    def is_acceptable(self) -> bool:
        """Check if quality meets minimum standards"""
        return (self.status == QualityStatus.GREEN and
                self.impedance_ok and
                self.artifact_rate < 0.15 and
                self.coverage_percentage >= 85.0)

@dataclass
class BiometricFeatures:
    """Real-time biomarker features"""
    beta_power: float
    alpha_beta_ratio: float
    coherence_m1: float
    theta_alpha_ratio: float
    gamma_power: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for processing"""
        return {
            'beta_power': self.beta_power,
            'alpha_beta_ratio': self.alpha_beta_ratio,
            'coherence_m1': self.coherence_m1,
            'theta_alpha_ratio': self.theta_alpha_ratio,
            'gamma_power': self.gamma_power
        }

@dataclass
class VitalSigns:
    """Physiological monitoring"""
    heart_rate: Optional[float] = None
    emg_activity: Optional[float] = None
    skin_conductance: Optional[float] = None
    respiration_rate: Optional[float] = None

    def has_adverse_signals(self) -> bool:
        """Check for adverse physiological signals"""
        if self.heart_rate and (self.heart_rate > 120 or self.heart_rate < 50):
            return True
        if self.emg_activity and self.emg_activity > 0.20:  # High muscle tension
            return True
        return False

@dataclass
class FeedbackParameters:
    """Neurofeedback control parameters"""
    gain: float
    threshold: float
    frequency_band: str = "beta"
    modulation_type: str = "amplitude"

    def is_within_bounds(self) -> bool:
        """Check if parameters are within safe bounds"""
        return (0.6 <= self.gain <= 1.4 and
                0.4 <= self.threshold <= 0.9)

@dataclass
class SafetyLimits:
    """Configurable safety limits"""
    min_confidence: float = 0.75
    max_gain_delta_per_10s: float = 0.10
    min_gain: float = 0.6
    max_gain: float = 1.4
    min_threshold: float = 0.4
    max_threshold: float = 0.9
    max_dose_per_20min: float = 9.0
    max_artifact_rate: float = 0.15
    min_coverage: float = 85.0
    rollback_confidence_threshold: float = 0.6

class SafetyGateManager:
    """
    Manages safety gates and constraints for adaptive control
    Implements hard safety limits with fail-safe behavior
    """

    def __init__(self, limits: SafetyLimits = None):
        self.limits = limits or SafetyLimits()
        self.dose_history = []  # (timestamp, dose_increment)
        self.gain_history = []  # (timestamp, gain_value)
        self.last_gate_check = datetime.utcnow()

    def check_all_gates(self,
                       features: BiometricFeatures,
                       quality: QualityMetrics,
                       vitals: VitalSigns,
                       model_confidence: float,
                       current_feedback: FeedbackParameters,
                       proposed_feedback: FeedbackParameters) -> Tuple[bool, Dict[str, Any]]:
        """
        Check all safety gates

        Returns:
            Tuple of (gates_pass, gate_status_dict)
        """
        gate_results = {}
        all_pass = True

        # Gate 1: Signal Quality
        quality_pass = self._check_quality_gate(quality)
        gate_results['quality'] = {
            'pass': quality_pass,
            'status': quality.status.value,
            'details': asdict(quality)
        }
        all_pass &= quality_pass

        # Gate 2: Model Confidence
        confidence_pass = self._check_confidence_gate(model_confidence)
        gate_results['confidence'] = {
            'pass': confidence_pass,
            'value': model_confidence,
            'threshold': self.limits.min_confidence
        }
        all_pass &= confidence_pass

        # Gate 3: Parameter Bounds
        bounds_pass = self._check_parameter_bounds(proposed_feedback)
        gate_results['parameter_bounds'] = {
            'pass': bounds_pass,
            'proposed_gain': proposed_feedback.gain,
            'proposed_threshold': proposed_feedback.threshold,
            'bounds': {
                'gain_range': [self.limits.min_gain, self.limits.max_gain],
                'threshold_range': [self.limits.min_threshold, self.limits.max_threshold]
            }
        }
        all_pass &= bounds_pass

        # Gate 4: Rate Limits
        rate_pass = self._check_rate_limits(current_feedback, proposed_feedback)
        gate_results['rate_limits'] = {
            'pass': rate_pass,
            'delta_gain': abs(proposed_feedback.gain - current_feedback.gain),
            'max_allowed_delta': self.limits.max_gain_delta_per_10s
        }
        all_pass &= rate_pass

        # Gate 5: Dose Limits
        dose_pass = self._check_dose_limits()
        gate_results['dose_limits'] = {
            'pass': dose_pass,
            'current_dose_20min': self._calculate_dose_window(),
            'max_dose': self.limits.max_dose_per_20min
        }
        all_pass &= dose_pass

        # Gate 6: Vital Signs
        vitals_pass = self._check_vitals_gate(vitals)
        gate_results['vitals'] = {
            'pass': vitals_pass,
            'adverse_signals': vitals.has_adverse_signals(),
            'details': asdict(vitals)
        }
        all_pass &= vitals_pass

        # Update history
        self._update_histories(proposed_feedback)

        return all_pass, gate_results

    def _check_quality_gate(self, quality: QualityMetrics) -> bool:
        """Check signal quality gate"""
        return quality.is_acceptable()

    def _check_confidence_gate(self, confidence: float) -> bool:
        """Check model confidence gate"""
        return confidence >= self.limits.min_confidence

    def _check_parameter_bounds(self, feedback: FeedbackParameters) -> bool:
        """Check parameter bounds gate"""
        return feedback.is_within_bounds()

    def _check_rate_limits(self, current: FeedbackParameters, proposed: FeedbackParameters) -> bool:
        """Check rate of change limits"""
        delta_gain = abs(proposed.gain - current.gain)
        return delta_gain <= self.limits.max_gain_delta_per_10s

    def _check_dose_limits(self) -> bool:
        """Check cumulative dose limits"""
        current_dose = self._calculate_dose_window()
        return current_dose <= self.limits.max_dose_per_20min

    def _check_vitals_gate(self, vitals: VitalSigns) -> bool:
        """Check vital signs gate"""
        return not vitals.has_adverse_signals()

    def _calculate_dose_window(self, window_minutes: int = 20) -> float:
        """Calculate cumulative dose in rolling window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_doses = [dose for timestamp, dose in self.dose_history
                       if timestamp >= cutoff_time]
        return sum(recent_doses)

    def _update_histories(self, feedback: FeedbackParameters):
        """Update dose and gain histories"""
        now = datetime.utcnow()

        # Calculate dose increment (simplified model)
        dose_increment = feedback.gain * 0.1  # Normalized dose units
        self.dose_history.append((now, dose_increment))

        # Update gain history
        self.gain_history.append((now, feedback.gain))

        # Cleanup old history (keep 24 hours)
        cutoff = now - timedelta(hours=24)
        self.dose_history = [(t, d) for t, d in self.dose_history if t >= cutoff]
        self.gain_history = [(t, g) for t, g in self.gain_history if t >= cutoff]

    def should_rollback(self, confidence: float, quality: QualityMetrics,
                       vitals: VitalSigns, pms_drift_flag: str = "LOW") -> bool:
        """Determine if automatic rollback is needed"""
        rollback_triggers = [
            confidence < self.limits.rollback_confidence_threshold,
            quality.status in [QualityStatus.AMBER, QualityStatus.RED],
            vitals.has_adverse_signals(),
            pms_drift_flag in ["HIGH", "CRITICAL"]
        ]

        return any(rollback_triggers)

    def get_safety_summary(self) -> Dict[str, Any]:
        """Get current safety status summary"""
        return {
            'dose_window_20min': self._calculate_dose_window(),
            'dose_limit': self.limits.max_dose_per_20min,
            'recent_gains': [g for _, g in self.gain_history[-10:]],
            'safety_limits': asdict(self.limits),
            'last_check': self.last_gate_check.isoformat()
        }

class AdaptiveController:
    """
    PID-lite controller with confidence weighting and safety projection
    """

    def __init__(self, config: Dict = None):
        default_config = {
            'kp': 0.5,          # Proportional gain
            'ki': 0.1,          # Integral gain
            'kd': 0.05,         # Derivative gain
            'target_beta_power': 0.65,
            'target_alpha_beta': 0.8,
            'confidence_weight_threshold': 0.75,
            'exploration_enabled': False,  # Disabled in clinical mode
            'bias_factor_drift_med': 0.8,  # Down-bias if drift detected
            'bias_factor_drift_high': 0.6
        }

        self.config = {**default_config, **(config or {})}
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.safety_gates = SafetyGateManager()

    def compute_adaptive_action(self,
                               features: BiometricFeatures,
                               current_feedback: FeedbackParameters,
                               model_confidence: float,
                               quality: QualityMetrics,
                               vitals: VitalSigns,
                               pms_drift_flag: str = "LOW") -> Tuple[FeedbackParameters, Dict[str, Any]]:
        """
        Compute next adaptive action with safety projection

        Returns:
            Tuple of (new_feedback_parameters, decision_metadata)
        """
        try:
            # Calculate control error
            target_biomarker = self.config['target_beta_power']
            current_biomarker = features.beta_power
            error = target_biomarker - current_biomarker

            # PID calculation
            proportional = self.config['kp'] * error
            self.integral_error += error
            integral = self.config['ki'] * self.integral_error
            derivative = self.config['kd'] * (error - self.previous_error)

            raw_action = proportional + integral + derivative
            self.previous_error = error

            # Confidence weighting
            confidence_weight = self._calculate_confidence_weight(model_confidence)
            weighted_action = raw_action * confidence_weight

            # Drift bias adjustment
            drift_bias = self._get_drift_bias_factor(pms_drift_flag)
            biased_action = weighted_action * drift_bias

            # Project to safe parameter space
            proposed_gain = current_feedback.gain + biased_action
            proposed_threshold = current_feedback.threshold

            proposed_feedback = FeedbackParameters(
                gain=proposed_gain,
                threshold=proposed_threshold,
                frequency_band=current_feedback.frequency_band,
                modulation_type=current_feedback.modulation_type
            )

            # Safety gate check
            gates_pass, gate_status = self.safety_gates.check_all_gates(
                features, quality, vitals, model_confidence,
                current_feedback, proposed_feedback
            )

            # Project to safe set if needed
            if not gates_pass:
                proposed_feedback = self._project_to_safe_set(
                    current_feedback, proposed_feedback, gate_status
                )

                # Re-check after projection
                gates_pass, gate_status = self.safety_gates.check_all_gates(
                    features, quality, vitals, model_confidence,
                    current_feedback, proposed_feedback
                )

            # Decision metadata
            decision_metadata = {
                'error': error,
                'pid_components': {
                    'proportional': proportional,
                    'integral': integral,
                    'derivative': derivative,
                    'raw_action': raw_action
                },
                'confidence_weight': confidence_weight,
                'drift_bias_factor': drift_bias,
                'weighted_action': weighted_action,
                'biased_action': biased_action,
                'gates_status': gate_status,
                'gates_pass': gates_pass,
                'safety_projection_applied': not gates_pass,
                'timestamp': datetime.utcnow().isoformat()
            }

            return proposed_feedback, decision_metadata

        except Exception as e:
            logger.error(f"Error in adaptive controller: {e}")
            # Return safe fallback
            return current_feedback, {
                'error': str(e),
                'fallback_applied': True,
                'timestamp': datetime.utcnow().isoformat()
            }

    def _calculate_confidence_weight(self, confidence: float) -> float:
        """Calculate confidence-based action weighting"""
        tau = self.config['confidence_weight_threshold']
        if confidence < tau:
            return 0.0
        return min(1.0, (confidence - tau) / (1.0 - tau))

    def _get_drift_bias_factor(self, drift_flag: str) -> float:
        """Get bias factor based on PMS drift detection"""
        if drift_flag == "HIGH":
            return self.config['bias_factor_drift_high']
        elif drift_flag == "MEDIUM":
            return self.config['bias_factor_drift_med']
        else:
            return 1.0

    def _project_to_safe_set(self,
                            current: FeedbackParameters,
                            proposed: FeedbackParameters,
                            gate_status: Dict) -> FeedbackParameters:
        """Project proposed parameters to safe constraint set"""
        safe_gain = proposed.gain
        safe_threshold = proposed.threshold

        # Bound constraints
        if not gate_status.get('parameter_bounds', {}).get('pass', True):
            safe_gain = np.clip(proposed.gain,
                              self.safety_gates.limits.min_gain,
                              self.safety_gates.limits.max_gain)
            safe_threshold = np.clip(proposed.threshold,
                                   self.safety_gates.limits.min_threshold,
                                   self.safety_gates.limits.max_threshold)

        # Rate constraints
        if not gate_status.get('rate_limits', {}).get('pass', True):
            max_delta = self.safety_gates.limits.max_gain_delta_per_10s
            if proposed.gain > current.gain:
                safe_gain = min(proposed.gain, current.gain + max_delta)
            else:
                safe_gain = max(proposed.gain, current.gain - max_delta)

        # If other gates fail, maintain current parameters
        if not gate_status.get('quality', {}).get('pass', True) or \
           not gate_status.get('confidence', {}).get('pass', True) or \
           not gate_status.get('vitals', {}).get('pass', True):
            safe_gain = current.gain
            safe_threshold = current.threshold

        return FeedbackParameters(
            gain=safe_gain,
            threshold=safe_threshold,
            frequency_band=proposed.frequency_band,
            modulation_type=proposed.modulation_type
        )

    def reset_controller_state(self):
        """Reset controller internal state"""
        self.integral_error = 0.0
        self.previous_error = 0.0
        logger.info("Adaptive controller state reset")

class SupervisorSystem:
    """
    Human-in-the-loop supervisor with override capabilities
    """

    def __init__(self):
        self.override_active = False
        self.override_reason = None
        self.override_timestamp = None
        self.override_user = None
        self.cooldown_until = None

    def clinician_override(self, action: str, user_id: str, reason: str = None) -> Dict[str, Any]:
        """
        Process clinician override command

        Args:
            action: 'pause', 'resume', 'rollback', 'emergency_stop'
            user_id: Clinician identifier
            reason: Override reason

        Returns:
            Override result status
        """
        try:
            valid_actions = ['pause', 'resume', 'rollback', 'emergency_stop']
            if action not in valid_actions:
                return {
                    'success': False,
                    'error': f'Invalid action. Must be one of: {valid_actions}'
                }

            self.override_active = True
            self.override_reason = reason
            self.override_timestamp = datetime.utcnow()
            self.override_user = user_id

            if action == 'rollback':
                # Set cooldown period
                self.cooldown_until = datetime.utcnow() + timedelta(minutes=5)
            elif action == 'emergency_stop':
                # Extended cooldown for emergency stop
                self.cooldown_until = datetime.utcnow() + timedelta(minutes=15)

            logger.warning(f"Clinician override activated: {action} by {user_id} - {reason}")

            return {
                'success': True,
                'action': action,
                'user': user_id,
                'timestamp': self.override_timestamp.isoformat(),
                'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None
            }

        except Exception as e:
            logger.error(f"Error processing clinician override: {e}")
            return {'success': False, 'error': str(e)}

    def is_in_cooldown(self) -> bool:
        """Check if system is in cooldown period"""
        if not self.cooldown_until:
            return False
        return datetime.utcnow() < self.cooldown_until

    def can_resume_adaptive(self) -> bool:
        """Check if adaptive control can be resumed"""
        return not self.is_in_cooldown() and not self.override_active

    def clear_override(self, user_id: str) -> Dict[str, Any]:
        """Clear active override"""
        if self.is_in_cooldown():
            return {
                'success': False,
                'error': f'System in cooldown until {self.cooldown_until.isoformat()}'
            }

        self.override_active = False
        self.override_reason = None
        self.override_timestamp = None

        logger.info(f"Override cleared by {user_id}")
        return {
            'success': True,
            'cleared_by': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_supervisor_status(self) -> Dict[str, Any]:
        """Get current supervisor status"""
        return {
            'override_active': self.override_active,
            'override_reason': self.override_reason,
            'override_timestamp': self.override_timestamp.isoformat() if self.override_timestamp else None,
            'override_user': self.override_user,
            'in_cooldown': self.is_in_cooldown(),
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'can_resume': self.can_resume_adaptive()
        }

# Safety validation functions
def validate_safety_configuration(limits: SafetyLimits) -> List[str]:
    """Validate safety configuration parameters"""
    issues = []

    if limits.min_confidence < 0.5 or limits.min_confidence > 0.95:
        issues.append("min_confidence should be between 0.5 and 0.95")

    if limits.max_gain_delta_per_10s <= 0 or limits.max_gain_delta_per_10s > 0.5:
        issues.append("max_gain_delta_per_10s should be between 0 and 0.5")

    if limits.min_gain >= limits.max_gain:
        issues.append("min_gain must be less than max_gain")

    if limits.max_dose_per_20min <= 0:
        issues.append("max_dose_per_20min must be positive")

    return issues

def create_safe_fallback_parameters() -> FeedbackParameters:
    """Create safe fallback feedback parameters"""
    return FeedbackParameters(
        gain=1.0,  # Neutral gain
        threshold=0.6,  # Conservative threshold
        frequency_band="beta",
        modulation_type="amplitude"
    )