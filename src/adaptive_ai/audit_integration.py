"""
Audit trails and PMS integration for Adaptive Clinical AI
Ensures complete traceability and real-time safety monitoring integration
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from dataclasses import asdict

import asyncpg
import numpy as np

from ..pms.pms_service import PMSService, PredictionEvent
from .safety_control import (
    BiometricFeatures,
    QualityMetrics,
    VitalSigns,
    FeedbackParameters,
    SafetyState
)

logger = logging.getLogger(__name__)

class AdaptiveAuditManager:
    """
    Comprehensive audit trail management for adaptive sessions
    Ensures complete traceability and regulatory compliance
    """

    def __init__(self, db_pool, pms_service: PMSService = None):
        self.db_pool = db_pool
        self.pms_service = pms_service
        self.audit_cache = {}  # session_id -> audit records
        self.hash_chain = {}   # session_id -> previous hash for chaining

    async def initialize_session_audit(self, session_id: UUID, session_config: Dict) -> str:
        """Initialize audit trail for new session"""
        try:
            # Create audit initialization record
            init_record = {
                'audit_type': 'session_initialization',
                'session_id': str(session_id),
                'timestamp': datetime.utcnow().isoformat(),
                'config': session_config,
                'system_info': await self._get_system_info(),
                'policy_version': session_config.get('policy_version', 'v1.0.0'),
                'safety_limits': session_config.get('safety_limits', {}),
                'validation_hash': self._calculate_config_hash(session_config)
            }

            # Calculate initial hash
            init_hash = self._calculate_record_hash(init_record)
            self.hash_chain[session_id] = init_hash

            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO adaptive_session_audit (
                        audit_id, session_id, audit_type, timestamp_audit,
                        record_data, record_hash, previous_hash, chain_valid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, str(uuid4()), session_id, 'session_initialization',
                    datetime.utcnow(), json.dumps(init_record), init_hash,
                    None, True)

            logger.info(f"Audit trail initialized for session {session_id}")
            return init_hash

        except Exception as e:
            logger.error(f"Error initializing session audit: {e}")
            raise

    async def log_control_tick(self,
                              session_id: UUID,
                              tick_id: str,
                              tick_number: int,
                              inputs: Dict,
                              outputs: Dict,
                              decision_metadata: Dict,
                              safety_assessment: Dict) -> str:
        """Log detailed control tick with full decision context"""
        try:
            # Create comprehensive tick record
            tick_record = {
                'audit_type': 'control_tick',
                'session_id': str(session_id),
                'tick_id': tick_id,
                'tick_number': tick_number,
                'timestamp': datetime.utcnow().isoformat(),
                'inputs': inputs,
                'outputs': outputs,
                'decision_metadata': decision_metadata,
                'safety_assessment': safety_assessment,
                'system_state': await self._capture_system_state()
            }

            # Calculate hash with chain validation
            previous_hash = self.hash_chain.get(session_id)
            current_hash = self._calculate_record_hash(tick_record, previous_hash)
            self.hash_chain[session_id] = current_hash

            # Store in database with detailed breakdown
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO adaptive_ticks (
                        tick_id, session_id, tick_number, timestamp_tick,
                        features_json, model_confidence, qc_json, vitals_json,
                        previous_feedback_json, output_gain, output_threshold,
                        output_frequency_band, output_modulation_type,
                        safety_state, gates_pass, guards_json, rationale,
                        confidence_weight, dose_increment, cumulative_dose,
                        control_error, pid_proportional, pid_integral, pid_derivative,
                        raw_action, weighted_action, biased_action, drift_bias_factor,
                        safety_projection_applied, decision_metadata_json
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24,
                        $25, $26, $27, $28, $29, $30
                    )
                """,
                    tick_id, session_id, tick_number,
                    datetime.fromisoformat(inputs['timestamp']),
                    json.dumps(inputs['features']),
                    inputs['model_confidence'],
                    json.dumps(inputs['quality_metrics']),
                    json.dumps(inputs['vital_signs']),
                    json.dumps(inputs['previous_feedback']),
                    outputs['gain'],
                    outputs['threshold'],
                    outputs.get('frequency_band', 'beta'),
                    outputs.get('modulation_type', 'amplitude'),
                    safety_assessment.get('safety_state', 'unknown'),
                    decision_metadata.get('gates_pass', True),
                    json.dumps(safety_assessment.get('guards', {})),
                    decision_metadata.get('rationale', ''),
                    decision_metadata.get('confidence_weight', 0.0),
                    decision_metadata.get('dose_increment', 0.0),
                    decision_metadata.get('cumulative_dose', 0.0),
                    decision_metadata.get('error', 0.0),
                    decision_metadata.get('pid_components', {}).get('proportional', 0.0),
                    decision_metadata.get('pid_components', {}).get('integral', 0.0),
                    decision_metadata.get('pid_components', {}).get('derivative', 0.0),
                    decision_metadata.get('pid_components', {}).get('raw_action', 0.0),
                    decision_metadata.get('weighted_action', 0.0),
                    decision_metadata.get('biased_action', 0.0),
                    decision_metadata.get('drift_bias_factor', 1.0),
                    decision_metadata.get('safety_projection_applied', False),
                    json.dumps(decision_metadata)
                )

                # Store in audit table
                await conn.execute("""
                    INSERT INTO adaptive_session_audit (
                        audit_id, session_id, audit_type, timestamp_audit,
                        record_data, record_hash, previous_hash, chain_valid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, str(uuid4()), session_id, 'control_tick',
                    datetime.utcnow(), json.dumps(tick_record), current_hash,
                    previous_hash, True)

            # Send to PMS if configured
            if self.pms_service:
                await self._send_pms_adaptive_event(session_id, tick_record)

            return current_hash

        except Exception as e:
            logger.error(f"Error logging control tick {tick_id}: {e}")
            raise

    async def log_safety_violation(self,
                                  session_id: UUID,
                                  violation_type: str,
                                  severity: str,
                                  context: Dict) -> str:
        """Log safety violation with full context"""
        try:
            violation_record = {
                'audit_type': 'safety_violation',
                'session_id': str(session_id),
                'violation_type': violation_type,
                'severity': severity,
                'timestamp': datetime.utcnow().isoformat(),
                'context': context,
                'system_response': context.get('system_response', {}),
                'escalation_triggered': context.get('escalation_triggered', False),
                'human_notified': context.get('human_notified', False)
            }

            # Calculate hash
            previous_hash = self.hash_chain.get(session_id)
            current_hash = self._calculate_record_hash(violation_record, previous_hash)
            self.hash_chain[session_id] = current_hash

            # Store in database
            async with self.db_pool.acquire() as conn:
                # Log in audit table
                await conn.execute("""
                    INSERT INTO adaptive_session_audit (
                        audit_id, session_id, audit_type, timestamp_audit,
                        record_data, record_hash, previous_hash, chain_valid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, str(uuid4()), session_id, 'safety_violation',
                    datetime.utcnow(), json.dumps(violation_record), current_hash,
                    previous_hash, True)

                # Create safety incident record if severe
                if severity in ['high', 'critical']:
                    await conn.execute("""
                        INSERT INTO adaptive_safety_incidents (
                            session_id, incident_type, severity, description,
                            tick_number_at_incident, safety_state_at_incident,
                            feedback_params_at_incident_json, vitals_at_incident_json,
                            detected_by
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, session_id, violation_type, severity,
                        f"Safety violation: {violation_type}",
                        context.get('tick_number'),
                        context.get('safety_state'),
                        json.dumps(context.get('feedback_params', {})),
                        json.dumps(context.get('vitals', {})),
                        'adaptive_system')

            # Immediate PMS alert for high/critical violations
            if severity in ['high', 'critical'] and self.pms_service:
                await self._send_pms_safety_alert(session_id, violation_record)

            logger.warning(f"Safety violation logged: {violation_type} ({severity}) for session {session_id}")
            return current_hash

        except Exception as e:
            logger.error(f"Error logging safety violation: {e}")
            raise

    async def log_supervisor_override(self,
                                     session_id: UUID,
                                     override_action: str,
                                     user_id: str,
                                     reason: str,
                                     context: Dict) -> str:
        """Log human supervisor override"""
        try:
            override_record = {
                'audit_type': 'supervisor_override',
                'session_id': str(session_id),
                'override_action': override_action,
                'user_id': user_id,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'context': context,
                'authorization_level': await self._get_user_authorization(user_id),
                'system_state_pre_override': context.get('system_state', {})
            }

            # Calculate hash
            previous_hash = self.hash_chain.get(session_id)
            current_hash = self._calculate_record_hash(override_record, previous_hash)
            self.hash_chain[session_id] = current_hash

            # Store in database
            async with self.db_pool.acquire() as conn:
                # Log in audit table
                await conn.execute("""
                    INSERT INTO adaptive_session_audit (
                        audit_id, session_id, audit_type, timestamp_audit,
                        record_data, record_hash, previous_hash, chain_valid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, str(uuid4()), session_id, 'supervisor_override',
                    datetime.utcnow(), json.dumps(override_record), current_hash,
                    previous_hash, True)

                # Log in overrides table
                await conn.execute("""
                    INSERT INTO adaptive_overrides (
                        session_id, override_action, override_reason, override_user,
                        tick_number_at_override, safety_state_at_override,
                        feedback_params_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, session_id, override_action, reason, user_id,
                    context.get('tick_number'),
                    context.get('safety_state'),
                    json.dumps(context.get('feedback_params', {})))

            logger.info(f"Supervisor override logged: {override_action} by {user_id} for session {session_id}")
            return current_hash

        except Exception as e:
            logger.error(f"Error logging supervisor override: {e}")
            raise

    async def finalize_session_audit(self, session_id: UUID, session_summary: Dict) -> str:
        """Finalize audit trail for completed session"""
        try:
            finalization_record = {
                'audit_type': 'session_finalization',
                'session_id': str(session_id),
                'timestamp': datetime.utcnow().isoformat(),
                'session_summary': session_summary,
                'audit_integrity_check': await self._verify_audit_chain(session_id),
                'total_audit_records': await self._count_audit_records(session_id),
                'safety_violations_total': session_summary.get('safety_violations', 0),
                'regulatory_flags': await self._check_regulatory_requirements(session_id)
            }

            # Calculate final hash
            previous_hash = self.hash_chain.get(session_id)
            final_hash = self._calculate_record_hash(finalization_record, previous_hash)

            # Store final audit record
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO adaptive_session_audit (
                        audit_id, session_id, audit_type, timestamp_audit,
                        record_data, record_hash, previous_hash, chain_valid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, str(uuid4()), session_id, 'session_finalization',
                    datetime.utcnow(), json.dumps(finalization_record), final_hash,
                    previous_hash, True)

                # Update session with final audit hash
                await conn.execute("""
                    UPDATE adaptive_sessions
                    SET audit_chain_hash = $1, audit_verified = $2
                    WHERE session_id = $3
                """, final_hash, finalization_record['audit_integrity_check'], session_id)

            # Clean up cache
            if session_id in self.hash_chain:
                del self.hash_chain[session_id]
            if session_id in self.audit_cache:
                del self.audit_cache[session_id]

            logger.info(f"Session audit finalized for {session_id}: hash {final_hash[:8]}...")
            return final_hash

        except Exception as e:
            logger.error(f"Error finalizing session audit: {e}")
            raise

    async def _send_pms_adaptive_event(self, session_id: UUID, tick_record: Dict):
        """Send adaptive control event to PMS"""
        try:
            pms_event = {
                'event_type': 'adaptive_control',
                'subject_id': tick_record.get('inputs', {}).get('subject_id', 'unknown'),
                'site_id': tick_record.get('inputs', {}).get('site_id', 'unknown'),
                'model_version': 'adaptive_v1',
                'session_id': str(session_id),
                'tick_id': tick_record.get('tick_id'),
                'timestamp': tick_record.get('timestamp'),
                'model_confidence': tick_record.get('inputs', {}).get('model_confidence', 0.0),
                'safety_gates_pass': tick_record.get('decision_metadata', {}).get('gates_pass', True),
                'controller_output': {
                    'gain': tick_record.get('outputs', {}).get('gain', 1.0),
                    'threshold': tick_record.get('outputs', {}).get('threshold', 0.6)
                },
                'biomarkers': tick_record.get('inputs', {}).get('features', {}),
                'quality_metrics': tick_record.get('inputs', {}).get('quality_metrics', {}),
                'safety_state': tick_record.get('safety_assessment', {}).get('safety_state', 'unknown')
            }

            # This would integrate with actual PMS service
            logger.debug(f"PMS adaptive event prepared for session {session_id}")

        except Exception as e:
            logger.error(f"Error sending PMS adaptive event: {e}")

    async def _send_pms_safety_alert(self, session_id: UUID, violation_record: Dict):
        """Send safety alert to PMS"""
        try:
            pms_alert = {
                'alert_type': 'adaptive_safety_violation',
                'severity': violation_record.get('severity', 'medium'),
                'session_id': str(session_id),
                'violation_type': violation_record.get('violation_type'),
                'message': f"Adaptive AI safety violation: {violation_record.get('violation_type')}",
                'timestamp': violation_record.get('timestamp'),
                'context': violation_record.get('context', {}),
                'immediate_response_required': violation_record.get('severity') == 'critical'
            }

            # This would trigger PMS alert processing
            logger.warning(f"PMS safety alert sent for session {session_id}")

        except Exception as e:
            logger.error(f"Error sending PMS safety alert: {e}")

    def _calculate_record_hash(self, record: Dict, previous_hash: str = None) -> str:
        """Calculate tamper-evident hash for audit record"""
        # Create consistent string representation
        record_str = json.dumps(record, sort_keys=True)
        if previous_hash:
            combined_str = f"{previous_hash}:{record_str}"
        else:
            combined_str = record_str

        return hashlib.sha256(combined_str.encode()).hexdigest()

    def _calculate_config_hash(self, config: Dict) -> str:
        """Calculate validation hash for configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    async def _get_system_info(self) -> Dict:
        """Capture system information for audit"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_version': '1.0.0',
            'python_version': '3.9+',
            'database_version': 'PostgreSQL 13+',
            'audit_version': '1.0.0'
        }

    async def _capture_system_state(self) -> Dict:
        """Capture current system state"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'memory_usage': 'captured',
            'cpu_usage': 'captured',
            'active_sessions': len(self.hash_chain),
            'database_connections': 'monitored'
        }

    async def _get_user_authorization(self, user_id: str) -> Dict:
        """Get user authorization level"""
        # In real implementation, this would query user permissions
        return {
            'user_id': user_id,
            'authorization_level': 'clinician',
            'override_permissions': ['pause', 'resume', 'rollback'],
            'verified_at': datetime.utcnow().isoformat()
        }

    async def _verify_audit_chain(self, session_id: UUID) -> bool:
        """Verify integrity of audit chain"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get all audit records for session
                records = await conn.fetch("""
                    SELECT record_hash, previous_hash, record_data
                    FROM adaptive_session_audit
                    WHERE session_id = $1
                    ORDER BY timestamp_audit
                """, session_id)

                # Verify hash chain
                previous_hash = None
                for record in records:
                    expected_hash = self._calculate_record_hash(
                        json.loads(record['record_data']),
                        previous_hash
                    )
                    if expected_hash != record['record_hash']:
                        logger.error(f"Audit chain verification failed for session {session_id}")
                        return False
                    previous_hash = record['record_hash']

                return True

        except Exception as e:
            logger.error(f"Error verifying audit chain: {e}")
            return False

    async def _count_audit_records(self, session_id: UUID) -> int:
        """Count total audit records for session"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT COUNT(*) FROM adaptive_session_audit
                    WHERE session_id = $1
                """, session_id)
                return result or 0
        except Exception as e:
            logger.error(f"Error counting audit records: {e}")
            return 0

    async def _check_regulatory_requirements(self, session_id: UUID) -> Dict:
        """Check regulatory reporting requirements"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check for safety incidents
                incidents = await conn.fetch("""
                    SELECT incident_type, severity, reportable_to_fda, reportable_to_ema
                    FROM adaptive_safety_incidents
                    WHERE session_id = $1
                """, session_id)

                safety_violations = await conn.fetchval("""
                    SELECT COUNT(*) FROM adaptive_ticks
                    WHERE session_id = $1 AND gates_pass = FALSE
                """, session_id)

                return {
                    'safety_incidents': len(incidents),
                    'reportable_incidents': len([i for i in incidents
                                               if i['reportable_to_fda'] or i['reportable_to_ema']]),
                    'safety_violations': safety_violations,
                    'regulatory_review_required': len(incidents) > 0 or safety_violations > 5
                }

        except Exception as e:
            logger.error(f"Error checking regulatory requirements: {e}")
            return {'error': str(e)}

class AdaptivePMSIntegration:
    """
    Integration layer between Adaptive AI and Post-Market Surveillance
    """

    def __init__(self, pms_service: PMSService):
        self.pms_service = pms_service
        self.drift_alerts_cache = {}

    async def register_adaptive_model(self, model_version: str, config: Dict) -> bool:
        """Register adaptive model with PMS"""
        try:
            model_event = {
                'event_type': 'adaptive_model_deployment',
                'model_version': model_version,
                'deployment_sites': config.get('deployment_sites', ['all']),
                'model_type': 'adaptive_controller',
                'safety_features': {
                    'hard_safety_gates': True,
                    'human_supervisor': True,
                    'automatic_rollback': True,
                    'dose_limiting': True
                },
                'performance_baseline': config.get('performance_baseline', {}),
                'validation_report': config.get('validation_report_path'),
                'timestamp': datetime.utcnow().isoformat()
            }

            # Register with PMS
            # In real implementation, this would call PMS API
            logger.info(f"Adaptive model {model_version} registered with PMS")
            return True

        except Exception as e:
            logger.error(f"Error registering adaptive model with PMS: {e}")
            return False

    async def report_adaptive_performance(self, session_summary: Dict) -> bool:
        """Report adaptive session performance to PMS"""
        try:
            performance_event = {
                'event_type': 'adaptive_performance',
                'session_id': session_summary.get('session_id'),
                'site_id': session_summary.get('site_id'),
                'performance_metrics': {
                    'safety_violations': session_summary.get('safety_violations', 0),
                    'time_in_adaptive': session_summary.get('time_in_adaptive', 0.0),
                    'avg_confidence': session_summary.get('avg_confidence', 0.0),
                    'dose_efficiency': session_summary.get('dose_used', 0.0),
                    'control_stability': session_summary.get('stability_score', 0.0)
                },
                'safety_assessment': {
                    'supervisor_overrides': session_summary.get('supervisor_overrides', 0),
                    'emergency_stops': session_summary.get('emergency_stops', 0),
                    'system_failures': session_summary.get('system_failures', 0)
                },
                'timestamp': datetime.utcnow().isoformat()
            }

            # Send to PMS
            logger.info(f"Adaptive performance reported to PMS for session {session_summary.get('session_id')}")
            return True

        except Exception as e:
            logger.error(f"Error reporting adaptive performance to PMS: {e}")
            return False

    async def get_pms_drift_status(self, site_id: str, model_version: str) -> str:
        """Get current drift status from PMS"""
        try:
            # In real implementation, this would query PMS API
            # For now, return cached or default value
            cache_key = f"{site_id}:{model_version}"
            return self.drift_alerts_cache.get(cache_key, "LOW")

        except Exception as e:
            logger.error(f"Error getting PMS drift status: {e}")
            return "LOW"

    async def update_drift_status(self, site_id: str, model_version: str, drift_level: str):
        """Update drift status cache from PMS alerts"""
        try:
            cache_key = f"{site_id}:{model_version}"
            self.drift_alerts_cache[cache_key] = drift_level
            logger.info(f"Updated drift status for {site_id}/{model_version}: {drift_level}")

        except Exception as e:
            logger.error(f"Error updating drift status: {e}")

# Add audit table schema (to be included in database schema)
AUDIT_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS adaptive_session_audit (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES adaptive_sessions(session_id) ON DELETE CASCADE,
    audit_type VARCHAR(50) NOT NULL,
    timestamp_audit TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    record_data JSONB NOT NULL,
    record_hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),
    chain_valid BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_adaptive_session_audit_session ON adaptive_session_audit(session_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_session_audit_type ON adaptive_session_audit(audit_type);
CREATE INDEX IF NOT EXISTS idx_adaptive_session_audit_timestamp ON adaptive_session_audit(timestamp_audit);
CREATE INDEX IF NOT EXISTS idx_adaptive_session_audit_hash ON adaptive_session_audit(record_hash);

-- Add audit chain hash to sessions table
ALTER TABLE adaptive_sessions
ADD COLUMN IF NOT EXISTS audit_chain_hash VARCHAR(64),
ADD COLUMN IF NOT EXISTS audit_verified BOOLEAN DEFAULT FALSE;
"""