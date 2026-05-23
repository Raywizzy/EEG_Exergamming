#!/usr/bin/env python3
"""
FastAPI Endpoints for Deep Learning Model Management and Prediction
Phase VII: RESTful API for model registry, training, and inference with GPU support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import uuid
import asyncio
import time
from datetime import datetime
from pathlib import Path
import tempfile
import torch
import logging
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "models"))

from models.model_registry import ModelRegistry, ModelMetadata, TrainingRun, ModelType, ModelStatus, Framework
from models.deep_learning.cnn_classifier import CNNConfig, CNNInferenceWrapper, CNNTrainer

# Initialize FastAPI app
app = FastAPI(
    title="EEG Deep Learning API",
    description="Phase VII: Deep learning model management and prediction API",
    version="1.0.0"
)

# Initialize model registry
model_registry = ModelRegistry()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ================================================================================

class ModelRegistrationRequest(BaseModel):
    model_name: str
    model_type: str  # 'cnn_spectrogram', 'rnn_temporal', 'transformer', 'ensemble'
    architecture: str
    version: str
    framework: str  # 'pytorch', 'tensorflow', 'sklearn'
    hyperparameters: Dict[str, Any]
    training_data_sources: List[str]
    performance_metrics: Dict[str, float]
    description: Optional[str] = None

class TrainingRequest(BaseModel):
    model_name: str
    experiment_name: str
    training_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    use_gpu: bool = True

class PredictionRequest(BaseModel):
    model_id: str
    input_type: str  # 'file_upload', 'raw_data'
    processing_options: Optional[Dict[str, Any]] = None

class ModelComparisonRequest(BaseModel):
    model_ids: List[str]
    experiment_name: str
    dataset_name: str
    validation_type: str = "loso"

class ModelDeploymentRequest(BaseModel):
    model_id: str
    site_ids: List[str]
    deployment_config: Optional[Dict[str, Any]] = None

# Response models
class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    version: str
    status: str
    is_deployed: bool
    performance_metrics: Dict[str, float]
    created_at: str

class PredictionResponse(BaseModel):
    prediction_id: str
    model_id: str
    prediction_class: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    processing_time_ms: int
    model_type: str
    metadata: Dict[str, Any]

class TrainingResponse(BaseModel):
    run_id: str
    status: str
    message: str

# ================================================================================
# SYSTEM AND HEALTH ENDPOINTS
# ================================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "EEG Deep Learning API - Phase VII",
        "version": "1.0.0",
        "endpoints": {
            "models": "/api/v1/models",
            "predictions": "/api/v1/predict",
            "training": "/api/v1/train",
            "health": "/health"
        },
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check model registry
        stats = model_registry.get_registry_stats()

        # Check GPU availability
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_registry": {
                "connected": True,
                "model_count": len(stats.get('model_statistics', [])),
                "training_runs": len(stats.get('training_statistics', []))
            },
            "gpu": gpu_info,
            "system": {
                "pytorch_version": torch.__version__,
                "python_version": sys.version
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"System unhealthy: {str(e)}")

# ================================================================================
# MODEL REGISTRY ENDPOINTS
# ================================================================================

@app.get("/api/v1/models", response_model=List[ModelResponse])
async def get_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    deployed_only: bool = False
):
    """Get all models with optional filtering."""
    try:
        # This would typically query the database
        # For now, return mock data since we need the full database integration
        models = []

        # Get registry stats
        stats = model_registry.get_registry_stats()

        return [
            ModelResponse(
                model_id="MODEL_CNN_DEMO_001",
                model_name="EEG_CNN_Spectrogram_v1",
                model_type="cnn_spectrogram",
                version="1.0.0",
                status="deployed",
                is_deployed=True,
                performance_metrics={
                    "balanced_accuracy": 0.924,
                    "sensitivity": 0.911,
                    "specificity": 0.937,
                    "auc": 0.962
                },
                created_at=datetime.now().isoformat()
            )
        ]

    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/register")
async def register_model(request: ModelRegistrationRequest):
    """Register a new model in the registry."""
    try:
        # Create model metadata
        metadata = ModelMetadata(
            model_id=f"MODEL_{uuid.uuid4().hex[:12].upper()}",
            model_name=request.model_name,
            model_type=ModelType(request.model_type),
            architecture=request.architecture,
            version=request.version,
            framework=Framework(request.framework),
            hyperparameters=request.hyperparameters,
            training_data_sources=request.training_data_sources,
            performance_metrics=request.performance_metrics,
            model_path=f"models/{request.model_name}/{request.version}",
            status=ModelStatus.VALIDATION
        )

        # Register model
        model_id = model_registry.register_model(metadata)

        logger.info(f"Registered model {model_id}")

        return {
            "model_id": model_id,
            "message": "Model registered successfully",
            "status": "validation",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model details."""
    try:
        # Load model metadata
        model, metadata = model_registry.load_model(model_id)

        return {
            "model_id": model_id,
            "metadata": metadata,
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/{model_id}/deploy")
async def deploy_model(model_id: str, request: ModelDeploymentRequest):
    """Deploy model to specified sites."""
    try:
        success = model_registry.deploy_model(model_id, request.site_ids)

        if success:
            return {
                "model_id": model_id,
                "deployment_sites": request.site_ids,
                "status": "deployed",
                "message": "Model deployed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Deployment failed")

    except Exception as e:
        logger.error(f"Failed to deploy model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/deployed")
async def get_deployed_models(site_id: Optional[str] = None):
    """Get all deployed models, optionally filtered by site."""
    try:
        deployed_models = model_registry.get_deployed_models(site_id)

        return {
            "deployed_models": deployed_models,
            "count": len(deployed_models),
            "site_filter": site_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get deployed models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# MODEL TRAINING ENDPOINTS
# ================================================================================

@app.post("/api/v1/train/cnn")
async def train_cnn_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Start CNN model training with GPU support."""
    try:
        # Create training run ID
        run_id = f"RUN_{uuid.uuid4().hex[:8].upper()}"

        # Create model metadata for training
        model_id = f"MODEL_{uuid.uuid4().hex[:12].upper()}"

        # Create CNN configuration from training request
        cnn_config = CNNConfig(
            **request.training_config.get('model_config', {}),
            **request.training_config.get('training_params', {})
        )

        # Create training run record
        training_run = TrainingRun(
            run_id=run_id,
            model_id=model_id,
            experiment_name=request.experiment_name,
            run_name=f"{request.model_name}_training",
            training_config=request.training_config,
            dataset_split=request.dataset_config,
            training_metrics={},
            validation_metrics={},
            final_performance={},
            gpu_used=request.use_gpu and torch.cuda.is_available()
        )

        # Start training run in registry
        model_registry.start_training_run(training_run)

        # Add training task to background
        background_tasks.add_task(
            _train_cnn_background,
            run_id, model_id, cnn_config, training_run
        )

        logger.info(f"Started CNN training run {run_id}")

        return TrainingResponse(
            run_id=run_id,
            status="started",
            message=f"CNN training started with GPU: {training_run.gpu_used}"
        )

    except Exception as e:
        logger.error(f"Failed to start CNN training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/train/{run_id}/status")
async def get_training_status(run_id: str):
    """Get training run status and metrics."""
    try:
        # Query training run status from database
        conn = model_registry.get_db_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM training_runs WHERE run_id = ?
            """, (run_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Training run not found")

            training_data = dict(row)

            return {
                "run_id": run_id,
                "status": training_data['status'],
                "training_metrics": json.loads(training_data['training_metrics'] or '{}'),
                "validation_metrics": json.loads(training_data['validation_metrics'] or '{}'),
                "final_performance": json.loads(training_data['final_performance'] or '{}'),
                "duration_seconds": training_data['training_duration_seconds'],
                "gpu_used": training_data['gpu_used'],
                "error_message": training_data['error_message'],
                "timestamp": datetime.now().isoformat()
            }

        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# PREDICTION ENDPOINTS
# ================================================================================

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_from_file(
    model_id: str,
    file: UploadFile = File(...),
    processing_options: Optional[str] = None
):
    """Make prediction from uploaded EEG file."""
    try:
        # Validate file type
        if not file.filename.endswith(('.fif', '.edf', '.bdf')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use .fif, .edf, or .bdf files."
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fif') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Load model
            model_obj, metadata = model_registry.load_model(model_id)

            # Determine model type and create appropriate inference wrapper
            model_type = metadata['model_type']

            if model_type == 'cnn_spectrogram':
                # Use CNN inference wrapper
                config = CNNConfig()  # Would load from metadata in production
                wrapper = CNNInferenceWrapper(metadata['model_path'], config)
                result = wrapper.predict_from_file(temp_file_path)
            else:
                # Add other model types here (RNN, Transformer, etc.)
                raise HTTPException(
                    status_code=400,
                    detail=f"Model type {model_type} not yet supported"
                )

            # Create prediction ID and save to database
            prediction_id = f"PRED_{uuid.uuid4().hex[:8].upper()}"

            # Store prediction in database
            conn = model_registry.get_db_connection()
            try:
                conn.execute("""
                    INSERT INTO model_predictions
                    (prediction_id, job_id, model_id, prediction_class,
                     confidence_score, class_probabilities, processing_time_ms,
                     model_version, prediction_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_id,
                    'API_PREDICTION',  # No job_id for direct API calls
                    model_id,
                    result['prediction_class'],
                    result['confidence_score'],
                    json.dumps(result.get('class_probabilities', {})),
                    result.get('processing_time_ms', 0),
                    metadata['version'],
                    json.dumps(result)
                ))
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Prediction {prediction_id} completed: {result['prediction_class']}")

            return PredictionResponse(
                prediction_id=prediction_id,
                model_id=model_id,
                prediction_class=result['prediction_class'],
                confidence_score=result['confidence_score'],
                class_probabilities=result.get('class_probabilities', {}),
                processing_time_ms=result.get('processing_time_ms', 0),
                model_type=model_type,
                metadata={
                    "filename": file.filename,
                    "file_size": len(content),
                    "segments_processed": result.get('segments_processed', 0),
                    "input_shape": result.get('input_shape')
                }
            )

        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/batch")
async def predict_batch(
    model_id: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks
):
    """Process multiple files for batch prediction."""
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size limited to 10 files"
            )

        batch_id = f"BATCH_{uuid.uuid4().hex[:8].upper()}"

        # Add batch processing task to background
        background_tasks.add_task(
            _process_batch_predictions,
            batch_id, model_id, files
        )

        return {
            "batch_id": batch_id,
            "status": "processing",
            "file_count": len(files),
            "message": "Batch prediction started",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# MODEL COMPARISON ENDPOINTS
# ================================================================================

@app.post("/api/v1/compare")
async def compare_models(request: ModelComparisonRequest):
    """Compare multiple models head-to-head."""
    try:
        comparison_id = model_registry.compare_models(
            model_ids=request.model_ids,
            experiment_name=request.experiment_name,
            dataset_name=request.dataset_name,
            validation_type=request.validation_type
        )

        return {
            "comparison_id": comparison_id,
            "experiment_name": request.experiment_name,
            "models_compared": len(request.model_ids),
            "validation_type": request.validation_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/compare/{comparison_id}")
async def get_comparison_results(comparison_id: str):
    """Get model comparison results."""
    try:
        conn = model_registry.get_db_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM model_comparisons WHERE comparison_id = ?
            """, (comparison_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Comparison not found")

            comparison_data = dict(row)

            return {
                "comparison_id": comparison_id,
                "experiment_name": comparison_data['experiment_name'],
                "model_ids": json.loads(comparison_data['model_ids']),
                "dataset_name": comparison_data['dataset_name'],
                "validation_type": comparison_data['validation_type'],
                "comparison_metrics": json.loads(comparison_data['comparison_metrics']),
                "statistical_tests": json.loads(comparison_data['statistical_tests'] or '{}'),
                "best_model_id": comparison_data['best_model_id'],
                "summary": comparison_data['comparison_summary'],
                "created_at": comparison_data['created_at']
            }

        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comparison results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# BACKGROUND TASKS
# ================================================================================

async def _train_cnn_background(run_id: str, model_id: str, config: CNNConfig, training_run: TrainingRun):
    """Background task for CNN training."""
    try:
        logger.info(f"Starting background CNN training for run {run_id}")

        # Mock training for demo (would implement real training here)
        await asyncio.sleep(2)  # Simulate training time

        # Mock training results
        final_performance = {
            "balanced_accuracy": 0.924,
            "sensitivity": 0.911,
            "specificity": 0.937,
            "auc": 0.962,
            "training_loss": 0.156,
            "validation_loss": 0.234
        }

        # Complete training run
        model_registry.complete_training_run(
            run_id=run_id,
            final_performance=final_performance,
            duration_seconds=120,  # 2 minutes
            status="completed"
        )

        logger.info(f"Completed CNN training run {run_id}")

    except Exception as e:
        logger.error(f"CNN training failed for run {run_id}: {e}")
        model_registry.complete_training_run(
            run_id=run_id,
            final_performance={},
            duration_seconds=0,
            status="failed",
            error_message=str(e)
        )

async def _process_batch_predictions(batch_id: str, model_id: str, files: List[UploadFile]):
    """Background task for batch predictions."""
    try:
        logger.info(f"Processing batch {batch_id} with {len(files)} files")

        # Mock batch processing
        await asyncio.sleep(1)

        logger.info(f"Completed batch processing {batch_id}")

    except Exception as e:
        logger.error(f"Batch processing failed for {batch_id}: {e}")

# ================================================================================
# STARTUP AND SHUTDOWN
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting EEG Deep Learning API...")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logger.info("API ready for requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down EEG Deep Learning API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)