"""
Post-Market Surveillance (PMS) FastAPI Service
Core endpoints for model monitoring, safety alerting, and regulatory compliance
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
from scipy import stats
import pandas as pd

from ..config.settings import DATABASE_CONFIG, PMS_CONFIG
from .drift_detection import DriftDetector, PopulationStabilityIndex, CUSUMDetector
from .safety_monitoring import SafetyMonitor, AlertEscalator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Post-Market Surveillance API",
    description="Real-time model monitoring and safety alerting for EEG neurofeedback trials",
    version="1.0.0"
)

class PredictionEvent(BaseModel):
    subject_id: str = Field(..., description="Subject identifier")
    site_id: str = Field(..., description="Clinical site identifier")
    model_version: str = Field(..., description="Model version (e.g., 'v2.1.3')")
    prediction_score: float = Field(..., ge=0.0, le=1.0, description="Model prediction probability")
    prediction_class: str = Field(..., description="Predicted class (PD_REAL, PD_SHAM)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    features: Dict[str, float] = Field(..., description="Input features used for prediction")
    session_id: str = Field(..., description="EEG session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class GroundTruthEvent(BaseModel):
    subject_id: str
    site_id: str
    true_class: str = Field(..., description="Ground truth class (PD_REAL, PD_SHAM)")
    session_id: str
    clinical_outcome: Optional[str] = Field(None, description="Clinical outcome if available")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelUpdateEvent(BaseModel):
    model_version: str
    previous_version: str
    deployment_sites: List[str]
    performance_metrics: Dict[str, float]
    update_reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AlertResponse(BaseModel):
    alert_id: UUID
    alert_type: str
    severity: str
    message: str
    site_id: Optional[str]
    model_version: Optional[str]
    created_at: datetime
    status: str

class MetricsQuery(BaseModel):
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    site_id: Optional[str] = Field(None, description="Filter by site")
    model_version: Optional[str] = Field(None, description="Filter by model version")

async def get_db_pool():
    """Database connection pool dependency"""
    if not hasattr(app.state, 'db_pool'):
        app.state.db_pool = await asyncpg.create_pool(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database=DATABASE_CONFIG['database'],
            min_size=5,
            max_size=20
        )
    return app.state.db_pool

class PMSService:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.drift_detector = DriftDetector()
        self.safety_monitor = SafetyMonitor()
        self.alert_escalator = AlertEscalator()
        self.active_alerts = {}

    async def store_prediction_event(self, event: PredictionEvent) -> str:
        """Store prediction event and trigger monitoring"""
        async with self.db_pool.acquire() as conn:
            event_id = str(uuid4())

            # Store prediction event
            await conn.execute("""
                INSERT INTO pms_events (event_id, subject_id, site_id, model_version,
                                      event_type, payload_json, created_at)
                VALUES ($1, $2, $3, $4, 'prediction', $5, $6)
            """, event_id, event.subject_id, event.site_id, event.model_version,
                json.dumps({
                    'prediction_score': event.prediction_score,
                    'prediction_class': event.prediction_class,
                    'confidence_score': event.confidence_score,
                    'features': event.features,
                    'session_id': event.session_id
                }), event.timestamp)

            # Trigger real-time monitoring
            await self._trigger_realtime_monitoring(event, conn)

            return event_id

    async def store_ground_truth_event(self, event: GroundTruthEvent) -> str:
        """Store ground truth event and update performance metrics"""
        async with self.db_pool.acquire() as conn:
            event_id = str(uuid4())

            await conn.execute("""
                INSERT INTO pms_events (event_id, subject_id, site_id, model_version,
                                      event_type, payload_json, created_at)
                VALUES ($1, $2, $3, $4, 'ground_truth', $5, $6)
            """, event_id, event.subject_id, event.site_id, "current",
                json.dumps({
                    'true_class': event.true_class,
                    'session_id': event.session_id,
                    'clinical_outcome': event.clinical_outcome
                }), event.timestamp)

            # Update performance metrics
            await self._update_performance_metrics(event, conn)

            return event_id

    async def store_model_update_event(self, event: ModelUpdateEvent) -> str:
        """Store model update event and register new version"""
        async with self.db_pool.acquire() as conn:
            event_id = str(uuid4())

            # Store update event
            await conn.execute("""
                INSERT INTO pms_events (event_id, subject_id, site_id, model_version,
                                      event_type, payload_json, created_at)
                VALUES ($1, 'system', 'all', $2, 'model_update', $3, $4)
            """, event_id, event.model_version,
                json.dumps({
                    'previous_version': event.previous_version,
                    'deployment_sites': event.deployment_sites,
                    'performance_metrics': event.performance_metrics,
                    'update_reason': event.update_reason
                }), event.timestamp)

            # Register in model registry
            await conn.execute("""
                INSERT INTO pms_model_registry (model_version, deployment_date,
                                              deployment_sites, performance_baseline,
                                              status, metadata)
                VALUES ($1, $2, $3, $4, 'active', $5)
            """, event.model_version, event.timestamp, event.deployment_sites,
                json.dumps(event.performance_metrics),
                json.dumps({
                    'previous_version': event.previous_version,
                    'update_reason': event.update_reason
                }))

            return event_id

    async def get_daily_metrics(self, date: str = None, site_id: str = None,
                               model_version: str = None) -> Dict[str, Any]:
        """Get daily aggregated metrics"""
        if not date:
            date = datetime.utcnow().strftime('%Y-%m-%d')

        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM pms_daily_metrics
                WHERE metric_date = $1
            """
            params = [date]

            if site_id:
                query += " AND site_id = $2"
                params.append(site_id)
            if model_version:
                query += f" AND model_version = ${len(params) + 1}"
                params.append(model_version)

            rows = await conn.fetch(query, *params)

            metrics = {}
            for row in rows:
                key = f"{row['site_id']}_{row['model_version']}"
                metrics[key] = dict(row)

            return metrics

    async def get_active_alerts(self) -> List[AlertResponse]:
        """Get all active alerts"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT alert_id, alert_type, severity, message, site_id,
                       model_version, created_at, status
                FROM pms_safety_alerts
                WHERE status IN ('active', 'acknowledged')
                ORDER BY created_at DESC
            """)

            return [AlertResponse(**dict(row)) for row in rows]

    async def close_alert(self, alert_id: UUID, resolution_notes: str = None) -> bool:
        """Close an active alert"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE pms_safety_alerts
                SET status = 'resolved', resolved_at = $1, resolution_notes = $2
                WHERE alert_id = $3 AND status IN ('active', 'acknowledged')
            """, datetime.utcnow(), resolution_notes, alert_id)

            if result == "UPDATE 1":
                # Remove from active alerts cache
                self.active_alerts.pop(str(alert_id), None)
                return True
            return False

    async def _trigger_realtime_monitoring(self, event: PredictionEvent, conn):
        """Trigger real-time drift and safety monitoring"""
        try:
            # Get recent predictions for drift detection
            recent_predictions = await conn.fetch("""
                SELECT payload_json FROM pms_events
                WHERE event_type = 'prediction'
                  AND model_version = $1
                  AND site_id = $2
                  AND created_at >= $3
                ORDER BY created_at DESC
                LIMIT 1000
            """, event.model_version, event.site_id,
                datetime.utcnow() - timedelta(hours=24))

            if len(recent_predictions) < 100:  # Need minimum samples
                return

            # Extract features and predictions
            features_list = []
            predictions_list = []
            for row in recent_predictions:
                payload = json.loads(row['payload_json'])
                features_list.append(list(payload['features'].values()))
                predictions_list.append(payload['prediction_score'])

            features_array = np.array(features_list)
            predictions_array = np.array(predictions_list)

            # Check for drift
            drift_detected = await self._check_drift(
                features_array, predictions_array, event.model_version, event.site_id, conn
            )

            # Check for performance degradation
            await self._check_performance_degradation(
                predictions_array, event.model_version, event.site_id, conn
            )

        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}")

    async def _check_drift(self, features: np.ndarray, predictions: np.ndarray,
                          model_version: str, site_id: str, conn) -> bool:
        """Check for population or concept drift"""
        try:
            # Get baseline features
            baseline_data = await conn.fetch("""
                SELECT payload_json FROM pms_events
                WHERE event_type = 'prediction'
                  AND model_version = $1
                  AND site_id = $2
                  AND created_at BETWEEN $3 AND $4
                LIMIT 1000
            """, model_version, site_id,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow() - timedelta(days=7))

            if len(baseline_data) < 100:
                return False

            baseline_features = np.array([
                list(json.loads(row['payload_json'])['features'].values())
                for row in baseline_data
            ])

            # Population Stability Index
            psi_detector = PopulationStabilityIndex()
            psi_score = psi_detector.calculate_psi(baseline_features, features[-500:])

            if psi_score > 0.25:  # Significant drift threshold
                await self._create_alert(
                    alert_type="population_drift",
                    severity="high",
                    message=f"Population drift detected (PSI: {psi_score:.3f})",
                    site_id=site_id,
                    model_version=model_version,
                    conn=conn
                )
                return True

            # CUSUM for prediction drift
            cusum_detector = CUSUMDetector(threshold=5.0)
            drift_points = cusum_detector.detect_drift(predictions)

            if len(drift_points) > 0:
                await self._create_alert(
                    alert_type="concept_drift",
                    severity="medium",
                    message=f"Prediction drift detected at {len(drift_points)} points",
                    site_id=site_id,
                    model_version=model_version,
                    conn=conn
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False

    async def _check_performance_degradation(self, predictions: np.ndarray,
                                           model_version: str, site_id: str, conn):
        """Check for performance degradation"""
        try:
            # Get baseline performance
            baseline_metrics = await conn.fetchrow("""
                SELECT performance_baseline FROM pms_model_registry
                WHERE model_version = $1
            """, model_version)

            if not baseline_metrics:
                return

            baseline = json.loads(baseline_metrics['performance_baseline'])
            baseline_accuracy = baseline.get('accuracy', 0.75)

            # Calculate recent performance (if ground truth available)
            recent_performance = await conn.fetch("""
                WITH predictions AS (
                    SELECT subject_id, session_id,
                           (payload_json->>'prediction_class') as pred_class,
                           created_at
                    FROM pms_events
                    WHERE event_type = 'prediction'
                      AND model_version = $1
                      AND site_id = $2
                      AND created_at >= $3
                ),
                ground_truth AS (
                    SELECT subject_id, session_id,
                           (payload_json->>'true_class') as true_class,
                           created_at
                    FROM pms_events
                    WHERE event_type = 'ground_truth'
                      AND site_id = $2
                      AND created_at >= $3
                )
                SELECT p.pred_class, g.true_class
                FROM predictions p
                JOIN ground_truth g ON p.subject_id = g.subject_id
                                   AND p.session_id = g.session_id
            """, model_version, site_id, datetime.utcnow() - timedelta(days=7))

            if len(recent_performance) >= 20:  # Minimum samples for reliable estimate
                correct = sum(1 for row in recent_performance
                             if row['pred_class'] == row['true_class'])
                current_accuracy = correct / len(recent_performance)

                if current_accuracy < baseline_accuracy - 0.05:  # 5% degradation threshold
                    await self._create_alert(
                        alert_type="performance_degradation",
                        severity="high",
                        message=f"Accuracy dropped to {current_accuracy:.1%} (baseline: {baseline_accuracy:.1%})",
                        site_id=site_id,
                        model_version=model_version,
                        conn=conn
                    )

        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")

    async def _create_alert(self, alert_type: str, severity: str, message: str,
                           site_id: str = None, model_version: str = None, conn=None):
        """Create a new safety alert"""
        alert_id = uuid4()

        if conn is None:
            async with self.db_pool.acquire() as conn:
                await self._insert_alert(alert_id, alert_type, severity, message,
                                       site_id, model_version, conn)
        else:
            await self._insert_alert(alert_id, alert_type, severity, message,
                                   site_id, model_version, conn)

        # Add to active alerts cache
        self.active_alerts[str(alert_id)] = {
            'alert_id': alert_id,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'created_at': datetime.utcnow()
        }

        # Trigger escalation if high severity
        if severity == "high":
            await self.alert_escalator.escalate_alert(alert_id, alert_type, message)

    async def _insert_alert(self, alert_id: UUID, alert_type: str, severity: str,
                           message: str, site_id: str, model_version: str, conn):
        """Insert alert into database"""
        await conn.execute("""
            INSERT INTO pms_safety_alerts (alert_id, alert_type, severity, message,
                                         site_id, model_version, status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, 'active', $7)
        """, alert_id, alert_type, severity, message, site_id, model_version,
            datetime.utcnow())

    async def _update_performance_metrics(self, event: GroundTruthEvent, conn):
        """Update daily performance metrics"""
        try:
            date_str = event.timestamp.strftime('%Y-%m-%d')

            # Calculate daily metrics for this site
            daily_stats = await conn.fetchrow("""
                WITH predictions AS (
                    SELECT (payload_json->>'prediction_class') as pred_class,
                           (payload_json->>'prediction_score')::float as pred_score,
                           (payload_json->>'confidence_score')::float as confidence
                    FROM pms_events
                    WHERE event_type = 'prediction'
                      AND site_id = $1
                      AND DATE(created_at) = $2
                ),
                ground_truth AS (
                    SELECT (payload_json->>'true_class') as true_class
                    FROM pms_events
                    WHERE event_type = 'ground_truth'
                      AND site_id = $1
                      AND DATE(created_at) = $2
                )
                SELECT COUNT(*) as total_predictions,
                       AVG(pred_score) as avg_prediction_score,
                       AVG(confidence) as avg_confidence
                FROM predictions
            """, event.site_id, date_str)

            if daily_stats and daily_stats['total_predictions'] > 0:
                # Upsert daily metrics
                await conn.execute("""
                    INSERT INTO pms_daily_metrics (
                        metric_date, site_id, model_version, total_predictions,
                        avg_prediction_score, avg_confidence_score,
                        created_at, updated_at
                    ) VALUES ($1, $2, 'current', $3, $4, $5, $6, $6)
                    ON CONFLICT (metric_date, site_id, model_version)
                    DO UPDATE SET
                        total_predictions = EXCLUDED.total_predictions,
                        avg_prediction_score = EXCLUDED.avg_prediction_score,
                        avg_confidence_score = EXCLUDED.avg_confidence_score,
                        updated_at = EXCLUDED.updated_at
                """, date_str, event.site_id, daily_stats['total_predictions'],
                    float(daily_stats['avg_prediction_score'] or 0),
                    float(daily_stats['avg_confidence_score'] or 0),
                    datetime.utcnow())

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")


# Initialize PMS service
pms_service = None

@app.on_event("startup")
async def startup_event():
    global pms_service
    db_pool = await get_db_pool()
    pms_service = PMSService(db_pool)
    logger.info("PMS Service initialized")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'db_pool'):
        await app.state.db_pool.close()
    logger.info("PMS Service shutdown")

# API Endpoints

@app.post("/api/pms/event", response_model=dict)
async def submit_event(
    event_type: str,
    event_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Submit prediction, ground truth, or model update event"""
    try:
        if event_type == "prediction":
            event = PredictionEvent(**event_data)
            event_id = await pms_service.store_prediction_event(event)

        elif event_type == "ground_truth":
            event = GroundTruthEvent(**event_data)
            event_id = await pms_service.store_ground_truth_event(event)

        elif event_type == "model_update":
            event = ModelUpdateEvent(**event_data)
            event_id = await pms_service.store_model_update_event(event)

        else:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")

        return {"event_id": event_id, "status": "stored"}

    except Exception as e:
        logger.error(f"Error storing {event_type} event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pms/metrics", response_model=Dict[str, Any])
async def get_metrics(query: MetricsQuery = Depends()):
    """Get daily aggregated metrics"""
    try:
        metrics = await pms_service.get_daily_metrics(
            date=query.date,
            site_id=query.site_id,
            model_version=query.model_version
        )
        return metrics

    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pms/alerts/live")
async def get_live_alerts():
    """Server-Sent Events stream for live alerts"""
    async def alert_stream():
        try:
            while True:
                alerts = await pms_service.get_active_alerts()
                alert_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "alerts": [alert.dict() for alert in alerts]
                }
                yield f"data: {json.dumps(alert_data)}\n\n"
                await asyncio.sleep(5)  # Update every 5 seconds

        except asyncio.CancelledError:
            logger.info("Alert stream cancelled")
        except Exception as e:
            logger.error(f"Error in alert stream: {e}")

    return StreamingResponse(
        alert_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.post("/api/pms/close_alert", response_model=dict)
async def close_alert(alert_id: UUID, resolution_notes: str = None):
    """Close an active alert"""
    try:
        success = await pms_service.close_alert(alert_id, resolution_notes)
        if success:
            return {"status": "closed", "alert_id": str(alert_id)}
        else:
            raise HTTPException(status_code=404, detail="Alert not found or already closed")

    except Exception as e:
        logger.error(f"Error closing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pms/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Post-Market Surveillance API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)