"""
Adaptive Clinical AI FastAPI Service
REST endpoints for closed-loop neurofeedback with safety-first control
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib asynccontextmanager

from ..config.settings import DATABASE_CONFIG
from ..pms.pms_service import PMSService
from .safety_control import (
    AdaptiveController,
    SafetyGateManager,
    SupervisorSystem,
    BiometricFeatures,
    QualityMetrics,
    VitalSigns,
    FeedbackParameters,
    SafetyState,
    QualityStatus,
    SafetyLimits,
    create_safe_fallback_parameters
)

logger = logging.getLogger(__name__)

# Pydantic models for API
class SessionCreateRequest(BaseModel):
    trial_id: str
    subject_id: str
    site_id: str
    clinician_id: str
    target_biomarker: str = "beta_power"
    target_value: float = Field(0.65, ge=0.1, le=1.0)
    session_duration_minutes: int = Field(20, ge=5, le=60)
    mode: str = Field("shadow", regex="^(shadow|assist|active)$")

class TickInput(BaseModel):
    timestamp: datetime
    features: Dict[str, float] = Field(..., description="Biomarker features")
    model_confidence: float = Field(..., ge=0.0, le=1.0)
    quality_metrics: Dict[str, Any]
    vital_signs: Dict[str, Optional[float]]
    previous_feedback: Dict[str, float]

class TickOutput(BaseModel):
    gain: float
    threshold: float
    safety_state: str
    guards: Dict[str, bool]
    rationale: str
    audit_ref: str
    confidence_weight: float
    dose_remaining: float

class OverrideRequest(BaseModel):
    action: str = Field(..., regex="^(pause|resume|rollback|emergency_stop)$")
    reason: Optional[str] = None

class SessionSummary(BaseModel):
    session_id: UUID
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    total_ticks: int
    safety_violations: int
    avg_confidence: float
    time_in_adaptive: float
    dose_used: float

# FastAPI app
app = FastAPI(
    title="Adaptive Clinical AI API",
    description="Closed-loop neurofeedback with safety-first control",
    version="1.0.0"
)

# Global state
adaptive_sessions = {}  # session_id -> AdaptiveSession
db_pool = None
pms_service = None

class AdaptiveSession:
    """Manages state for a single adaptive session"""

    def __init__(self, session_id: UUID, config: Dict):
        self.session_id = session_id
        self.config = config
        self.controller = AdaptiveController(config.get('controller_params'))
        self.supervisor = SupervisorSystem()
        self.safety_state = SafetyState.IDLE
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.tick_count = 0
        self.safety_violations = 0
        self.confidence_history = []
        self.dose_used = 0.0
        self.audit_trail = []
        self.pms_drift_flag = "LOW"

        # Current state
        self.current_feedback = create_safe_fallback_parameters()
        self.last_features = None
        self.last_quality = None
        self.last_vitals = None

    async def process_tick(self, tick_input: TickInput) -> TickOutput:
        """Process a single control tick"""
        try:
            self.tick_count += 1
            tick_id = str(uuid4())

            # Parse inputs
            features = BiometricFeatures(
                beta_power=tick_input.features.get('beta_power', 0.5),
                alpha_beta_ratio=tick_input.features.get('alpha_beta_ratio', 0.8),
                coherence_m1=tick_input.features.get('coherence_m1', 0.3),
                theta_alpha_ratio=tick_input.features.get('theta_alpha_ratio', 0.6),
                gamma_power=tick_input.features.get('gamma_power', 0.2),
                timestamp=tick_input.timestamp
            )

            quality = QualityMetrics(
                status=QualityStatus(tick_input.quality_metrics.get('status', 'green')),
                impedance_ok=tick_input.quality_metrics.get('impedance_ok', True),
                artifact_rate=tick_input.quality_metrics.get('artifact_rate', 0.05),
                coverage_percentage=tick_input.quality_metrics.get('coverage_percentage', 95.0),
                motion_level=tick_input.quality_metrics.get('motion_level', 0.1),
                power_line_interference=tick_input.quality_metrics.get('power_line_interference', 0.02)
            )

            vitals = VitalSigns(
                heart_rate=tick_input.vital_signs.get('heart_rate'),
                emg_activity=tick_input.vital_signs.get('emg_activity'),
                skin_conductance=tick_input.vital_signs.get('skin_conductance'),
                respiration_rate=tick_input.vital_signs.get('respiration_rate')
            )

            # Update previous feedback from input
            self.current_feedback = FeedbackParameters(
                gain=tick_input.previous_feedback.get('gain', 1.0),
                threshold=tick_input.previous_feedback.get('threshold', 0.6),
                frequency_band=tick_input.previous_feedback.get('frequency_band', 'beta'),
                modulation_type=tick_input.previous_feedback.get('modulation_type', 'amplitude')
            )

            # Store current state
            self.last_features = features
            self.last_quality = quality
            self.last_vitals = vitals
            self.confidence_history.append(tick_input.model_confidence)

            # Check for supervisor overrides
            if self.supervisor.override_active:
                return await self._handle_override_state(tick_id)

            # Determine safety state
            self.safety_state = self._determine_safety_state(quality, vitals, tick_input.model_confidence)

            # Process based on mode and safety state
            if self.config['mode'] == 'shadow':
                # Shadow mode: compute but don't actuate
                proposed_feedback, decision_metadata = self.controller.compute_adaptive_action(
                    features, self.current_feedback, tick_input.model_confidence,
                    quality, vitals, self.pms_drift_flag
                )
                # Keep current parameters
                output_feedback = self.current_feedback
                decision_metadata['shadow_mode'] = True
                decision_metadata['proposed_gain'] = proposed_feedback.gain
                decision_metadata['proposed_threshold'] = proposed_feedback.threshold

            elif self.config['mode'] == 'assist':
                # Assist mode: compute and flag for clinician approval
                proposed_feedback, decision_metadata = self.controller.compute_adaptive_action(
                    features, self.current_feedback, tick_input.model_confidence,
                    quality, vitals, self.pms_drift_flag
                )
                # Return proposed parameters but require approval
                output_feedback = proposed_feedback
                decision_metadata['assist_mode'] = True
                decision_metadata['approval_required'] = True

            elif self.config['mode'] == 'active':
                # Active mode: compute and apply if safe
                if self.safety_state == SafetyState.DELIVERING:
                    proposed_feedback, decision_metadata = self.controller.compute_adaptive_action(
                        features, self.current_feedback, tick_input.model_confidence,
                        quality, vitals, self.pms_drift_flag
                    )
                    output_feedback = proposed_feedback
                else:
                    # Not safe to adapt, maintain current
                    output_feedback = self.current_feedback
                    decision_metadata = {
                        'safety_hold': True,
                        'safety_state': self.safety_state.value,
                        'timestamp': datetime.utcnow().isoformat()
                    }
            else:
                raise ValueError(f"Invalid mode: {self.config['mode']}")

            # Update dose tracking
            dose_increment = output_feedback.gain * 0.1  # Simplified dose model
            self.dose_used += dose_increment

            # Check for safety violations
            if not decision_metadata.get('gates_pass', True):
                self.safety_violations += 1

            # Create audit record
            audit_record = self._create_audit_record(
                tick_id, tick_input, output_feedback, decision_metadata
            )
            self.audit_trail.append(audit_record)

            # Generate rationale
            rationale = self._generate_rationale(decision_metadata, quality, tick_input.model_confidence)

            # Calculate remaining dose budget
            dose_remaining = max(0, 9.0 - self._calculate_dose_window())

            # Send to PMS if active mode
            if self.config['mode'] == 'active':
                await self._send_pms_event(tick_input, output_feedback, decision_metadata)

            return TickOutput(
                gain=output_feedback.gain,
                threshold=output_feedback.threshold,
                safety_state=self.safety_state.value,
                guards={
                    'quality_ok': quality.is_acceptable(),
                    'confidence_ok': tick_input.model_confidence >= 0.75,
                    'vitals_ok': not vitals.has_adverse_signals(),
                    'dose_ok': dose_remaining > 0
                },
                rationale=rationale,
                audit_ref=tick_id,
                confidence_weight=decision_metadata.get('confidence_weight', 0.0),
                dose_remaining=dose_remaining
            )

        except Exception as e:
            logger.error(f"Error processing tick for session {self.session_id}: {e}")
            # Return safe fallback
            return TickOutput(
                gain=1.0,
                threshold=0.6,
                safety_state=SafetyState.EMERGENCY_STOP.value,
                guards={'error': True},
                rationale=f"Error: {str(e)}",
                audit_ref=str(uuid4()),
                confidence_weight=0.0,
                dose_remaining=0.0
            )

    def _determine_safety_state(self, quality: QualityMetrics, vitals: VitalSigns, confidence: float) -> SafetyState:
        """Determine current safety state"""
        if self.supervisor.override_active:
            return SafetyState.PAUSED

        if (not quality.is_acceptable() or
            vitals.has_adverse_signals() or
            confidence < 0.6 or
            self.pms_drift_flag == "HIGH"):
            return SafetyState.ROLLBACK

        if confidence >= 0.75 and quality.is_acceptable() and not vitals.has_adverse_signals():
            return SafetyState.DELIVERING

        return SafetyState.ARMED

    async def _handle_override_state(self, tick_id: str) -> TickOutput:
        """Handle tick during supervisor override"""
        return TickOutput(
            gain=self.current_feedback.gain,
            threshold=self.current_feedback.threshold,
            safety_state=SafetyState.PAUSED.value,
            guards={'supervisor_override': True},
            rationale=f"Supervisor override active: {self.supervisor.override_reason}",
            audit_ref=tick_id,
            confidence_weight=0.0,
            dose_remaining=max(0, 9.0 - self.dose_used)
        )

    def _calculate_dose_window(self, window_minutes: int = 20) -> float:
        """Calculate dose in rolling window"""
        # Simplified calculation using recent audit trail
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_dose = sum(
            record.get('dose_increment', 0)
            for record in self.audit_trail
            if datetime.fromisoformat(record['timestamp']) >= cutoff_time
        )
        return recent_dose

    def _create_audit_record(self, tick_id: str, tick_input: TickInput,
                           output_feedback: FeedbackParameters, decision_metadata: Dict) -> Dict:
        """Create audit trail record"""
        return {
            'tick_id': tick_id,
            'session_id': str(self.session_id),
            'timestamp': tick_input.timestamp.isoformat(),
            'inputs': {
                'features': tick_input.features,
                'model_confidence': tick_input.model_confidence,
                'quality_metrics': tick_input.quality_metrics,
                'vital_signs': tick_input.vital_signs,
                'previous_feedback': tick_input.previous_feedback
            },
            'outputs': {
                'gain': output_feedback.gain,
                'threshold': output_feedback.threshold,
                'frequency_band': output_feedback.frequency_band,
                'modulation_type': output_feedback.modulation_type
            },
            'decision_metadata': decision_metadata,
            'safety_state': self.safety_state.value,
            'dose_increment': output_feedback.gain * 0.1,
            'cumulative_dose': self.dose_used,
            'tick_number': self.tick_count
        }

    def _generate_rationale(self, decision_metadata: Dict, quality: QualityMetrics, confidence: float) -> str:
        """Generate human-readable rationale for decision"""
        if decision_metadata.get('shadow_mode'):
            return f"Shadow mode: conf={confidence:.2f}, QC={quality.status.value}, would adjust gain to {decision_metadata.get('proposed_gain', 0):.2f}"

        if decision_metadata.get('assist_mode'):
            return f"Assist mode: conf={confidence:.2f}, QC={quality.status.value}, proposed gain={decision_metadata.get('gain', 0):.2f} (approval required)"

        if decision_metadata.get('safety_hold'):
            return f"Safety hold: {decision_metadata.get('safety_state', 'unknown')} state"

        if decision_metadata.get('fallback_applied'):
            return f"Error fallback applied: {decision_metadata.get('error', 'unknown error')}"

        # Normal adaptive rationale
        error = decision_metadata.get('error', 0)
        confidence_weight = decision_metadata.get('confidence_weight', 0)
        gates_pass = decision_metadata.get('gates_pass', False)

        if error > 0:
            direction = "increase"
        elif error < 0:
            direction = "decrease"
        else:
            direction = "maintain"

        return f"conf={confidence:.2f}, QC={quality.status.value}, error={error:.3f} → {direction} gain (weight={confidence_weight:.2f}, gates={'pass' if gates_pass else 'fail'})"

    async def _send_pms_event(self, tick_input: TickInput, feedback: FeedbackParameters, metadata: Dict):
        """Send event to PMS system"""
        try:
            if pms_service:
                pms_event = {
                    'event_type': 'adaptive_action',
                    'session_id': str(self.session_id),
                    'timestamp': tick_input.timestamp.isoformat(),
                    'model_confidence': tick_input.model_confidence,
                    'feedback_parameters': {
                        'gain': feedback.gain,
                        'threshold': feedback.threshold
                    },
                    'safety_gates_pass': metadata.get('gates_pass', True),
                    'dose_used': self.dose_used,
                    'safety_violations': self.safety_violations
                }
                # In real implementation, this would be async
                logger.info(f"PMS event logged for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error sending PMS event: {e}")

    def end_session(self) -> SessionSummary:
        """End the adaptive session and return summary"""
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds() / 60

        # Calculate time in adaptive mode
        adaptive_ticks = sum(1 for record in self.audit_trail
                           if record.get('safety_state') == SafetyState.DELIVERING.value)
        time_in_adaptive = (adaptive_ticks / max(1, self.tick_count)) * duration

        return SessionSummary(
            session_id=self.session_id,
            status='completed',
            start_time=self.start_time,
            end_time=self.end_time,
            total_ticks=self.tick_count,
            safety_violations=self.safety_violations,
            avg_confidence=np.mean(self.confidence_history) if self.confidence_history else 0.0,
            time_in_adaptive=time_in_adaptive,
            dose_used=self.dose_used
        )

# Database dependency
async def get_db_pool():
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database=DATABASE_CONFIG['database'],
            min_size=2,
            max_size=10
        )
    return db_pool

@app.on_event("startup")
async def startup_event():
    global pms_service
    db = await get_db_pool()
    pms_service = PMSService(db)
    logger.info("Adaptive AI service initialized")

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()
    logger.info("Adaptive AI service shutdown")

# API Endpoints

@app.post("/api/adapt/session", response_model=dict)
async def create_session(request: SessionCreateRequest, db_pool=Depends(get_db_pool)):
    """Create new adaptive session"""
    try:
        session_id = uuid4()

        # Validate mode
        if request.mode not in ['shadow', 'assist', 'active']:
            raise HTTPException(status_code=400, detail="Invalid mode")

        # Create session configuration
        session_config = {
            'trial_id': request.trial_id,
            'subject_id': request.subject_id,
            'site_id': request.site_id,
            'clinician_id': request.clinician_id,
            'target_biomarker': request.target_biomarker,
            'target_value': request.target_value,
            'session_duration_minutes': request.session_duration_minutes,
            'mode': request.mode,
            'controller_params': {
                'target_beta_power': request.target_value,
                'kp': 0.5,
                'ki': 0.1,
                'kd': 0.05
            }
        }

        # Create adaptive session
        session = AdaptiveSession(session_id, session_config)
        adaptive_sessions[session_id] = session

        # Store in database
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO adaptive_sessions (session_id, trial_id, subject_id, site_id,
                                             clinician_id, mode, target_biomarker, target_value,
                                             session_duration_minutes, started_at, status, config_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'active', $11)
            """, session_id, request.trial_id, request.subject_id, request.site_id,
                request.clinician_id, request.mode, request.target_biomarker,
                request.target_value, request.session_duration_minutes,
                session.start_time, json.dumps(session_config))

        logger.info(f"Created adaptive session {session_id} in {request.mode} mode")

        return {
            'session_id': str(session_id),
            'mode': request.mode,
            'status': 'created',
            'start_time': session.start_time.isoformat()
        }

    except Exception as e:
        logger.error(f"Error creating adaptive session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/adapt/{session_id}/tick", response_model=TickOutput)
async def process_tick(session_id: UUID, tick_input: TickInput):
    """Process control tick for adaptive session"""
    try:
        if session_id not in adaptive_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = adaptive_sessions[session_id]
        result = await session.process_tick(tick_input)

        return result

    except Exception as e:
        logger.error(f"Error processing tick for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/adapt/{session_id}/override", response_model=dict)
async def clinician_override(session_id: UUID, request: OverrideRequest,
                           clinician_id: str = "system"):
    """Process clinician override"""
    try:
        if session_id not in adaptive_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = adaptive_sessions[session_id]
        result = session.supervisor.clinician_override(
            request.action, clinician_id, request.reason
        )

        if result['success']:
            logger.warning(f"Override {request.action} applied to session {session_id}")

        return result

    except Exception as e:
        logger.error(f"Error processing override for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/adapt/{session_id}/end", response_model=SessionSummary)
async def end_session(session_id: UUID, db_pool=Depends(get_db_pool)):
    """End adaptive session"""
    try:
        if session_id not in adaptive_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = adaptive_sessions[session_id]
        summary = session.end_session()

        # Update database
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE adaptive_sessions
                SET ended_at = $1, status = 'completed', total_ticks = $2,
                    safety_violations = $3, avg_confidence = $4, dose_used = $5
                WHERE session_id = $6
            """, session.end_time, session.tick_count, session.safety_violations,
                np.mean(session.confidence_history) if session.confidence_history else 0.0,
                session.dose_used, session_id)

        # Remove from active sessions
        del adaptive_sessions[session_id]

        logger.info(f"Ended adaptive session {session_id}")
        return summary

    except Exception as e:
        logger.error(f"Error ending session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adapt/{session_id}/status", response_model=dict)
async def get_session_status(session_id: UUID):
    """Get current session status"""
    try:
        if session_id not in adaptive_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = adaptive_sessions[session_id]

        return {
            'session_id': str(session_id),
            'mode': session.config['mode'],
            'safety_state': session.safety_state.value,
            'tick_count': session.tick_count,
            'safety_violations': session.safety_violations,
            'dose_used': session.dose_used,
            'supervisor_status': session.supervisor.get_supervisor_status(),
            'current_feedback': {
                'gain': session.current_feedback.gain,
                'threshold': session.current_feedback.threshold
            }
        }

    except Exception as e:
        logger.error(f"Error getting session status {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adapt/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'active_sessions': len(adaptive_sessions),
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Adaptive Clinical AI'
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)