"""
FastAPI Endpoints for Explainability Layer
Provides REST API for model explanation generation, retrieval, and export
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime
from pathlib import Path
import tempfile
import logging
import numpy as np
import json
import io
import zipfile

from ..explainability.grad_cam import CNNExplainer, GradCAMConfig
from ..explainability.attention_analysis import TransformerAttentionAnalyzer, AttentionAnalysisConfig
from ..explainability.tabular_attribution import Core15Explainer, TabularAttributionConfig
from ..explainability.artifact_manager import (
    ExplainabilityArtifactManager, ExplanationArtifact, ExplanationMetadata
)
from ..explainability.visualization import ExplanationVisualizer
from ..models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Pydantic models for API
class ExplanationRequest(BaseModel):
    model_id: str
    subject_id: str
    explanation_type: str = Field(..., regex="^(grad_cam|attention|feature_attribution|ensemble)$")
    config: Optional[Dict[str, Any]] = {}

class ExplanationResponse(BaseModel):
    explanation_id: str
    subject_id: str
    model_id: str
    explanation_type: str
    confidence: float
    predicted_class: int
    created_at: datetime
    visualization_url: Optional[str] = None
    download_url: Optional[str] = None

class ExplanationSearchRequest(BaseModel):
    subject_id: Optional[str] = None
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    explanation_type: Optional[str] = None
    confidence_min: Optional[float] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)

class BatchExplanationRequest(BaseModel):
    model_id: str
    explanation_type: str
    subject_ids: List[str]
    config: Optional[Dict[str, Any]] = {}

class ExportRequest(BaseModel):
    explanation_id: str
    export_format: str = Field(..., regex="^(pdf|json|zip)$")

# Initialize components
explainer_app = FastAPI(title="Explainability API", version="1.0.0")

# Global instances (would be injected in production)
artifact_manager: Optional[ExplainabilityArtifactManager] = None
model_registry: Optional[ModelRegistry] = None
visualizer: Optional[ExplanationVisualizer] = None

def get_artifact_manager() -> ExplainabilityArtifactManager:
    """Dependency injection for artifact manager"""
    global artifact_manager
    if artifact_manager is None:
        artifact_manager = ExplainabilityArtifactManager(
            db_path="data/platform.db",
            artifacts_dir="artifacts/explanations"
        )
    return artifact_manager

def get_model_registry() -> ModelRegistry:
    """Dependency injection for model registry"""
    global model_registry
    if model_registry is None:
        from ..models.model_registry import ModelRegistry
        model_registry = ModelRegistry("data/platform.db")
    return model_registry

def get_visualizer() -> ExplanationVisualizer:
    """Dependency injection for visualizer"""
    global visualizer
    if visualizer is None:
        visualizer = ExplanationVisualizer(Path("artifacts/visualizations"))
    return visualizer

@explainer_app.post("/api/v1/explain", response_model=ExplanationResponse)
async def generate_explanation(
    request: ExplanationRequest,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager),
    model_registry: ModelRegistry = Depends(get_model_registry),
    visualizer: ExplanationVisualizer = Depends(get_visualizer)
):
    """Generate model explanation for uploaded EEG data"""

    # Load model metadata
    model_metadata = model_registry.get_model(request.model_id)
    if not model_metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load EEG data
        eeg_data = np.load(tmp_file_path)

        # Generate explanation based on model type
        explanation = None

        if model_metadata.model_type == "cnn_spectrogram" and request.explanation_type == "grad_cam":
            config = GradCAMConfig(**request.config)
            explainer = CNNExplainer(model_metadata.model_path, config)
            explanation = explainer.explain_prediction(eeg_data, request.subject_id)

        elif model_metadata.model_type == "transformer" and request.explanation_type == "attention":
            config = AttentionAnalysisConfig(**request.config)
            explainer = TransformerAttentionAnalyzer(model_metadata.model_path, config)
            explanation = explainer.explain_prediction(eeg_data, request.subject_id)

        elif model_metadata.model_type == "core15+" and request.explanation_type == "feature_attribution":
            config = TabularAttributionConfig(**request.config)
            explainer = Core15Explainer(
                model_metadata.model_path,
                model_metadata.hyperparameters.get('feature_names', []),
                config
            )
            explanation = explainer.explain_prediction(eeg_data, request.subject_id)

        elif request.explanation_type == "ensemble":
            # Generate ensemble explanation combining multiple models
            explanation = await _generate_ensemble_explanation(
                request, eeg_data, model_registry, artifact_manager
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Explanation type {request.explanation_type} not supported for model type {model_metadata.model_type}"
            )

        if not explanation:
            raise HTTPException(status_code=500, detail="Failed to generate explanation")

        # Generate visualization
        fig, viz_bytes = visualizer.visualize_explanation(explanation)

        # Create artifact
        explanation_id = str(uuid.uuid4())
        metadata = ExplanationMetadata(
            explanation_id=explanation_id,
            subject_id=request.subject_id,
            model_id=request.model_id,
            model_type=model_metadata.model_type,
            explanation_type=request.explanation_type,
            created_at=datetime.now(),
            created_by="api_user",
            data_hash=explanation.get('metadata', {}).get('data_hash', ''),
            model_version=model_metadata.version,
            confidence_score=explanation['prediction']['confidence'],
            predicted_class=explanation['prediction']['class'],
            explanation_config=request.config,
            quality_metrics={},
            file_paths={}
        )

        artifact = ExplanationArtifact(
            metadata=metadata,
            explanation_data=explanation,
            visualizations={"main_plot": viz_bytes},
            summary_stats={},
            audit_trail=[]
        )

        # Store artifact
        stored_id = artifact_manager.store_explanation(artifact)

        # Clean up temporary file
        Path(tmp_file_path).unlink()

        return ExplanationResponse(
            explanation_id=stored_id,
            subject_id=request.subject_id,
            model_id=request.model_id,
            explanation_type=request.explanation_type,
            confidence=explanation['prediction']['confidence'],
            predicted_class=explanation['prediction']['class'],
            created_at=metadata.created_at,
            visualization_url=f"/api/v1/explanations/{stored_id}/visualization",
            download_url=f"/api/v1/explanations/{stored_id}/download"
        )

    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@explainer_app.post("/api/v1/explain/batch", response_model=List[ExplanationResponse])
async def batch_generate_explanations(
    request: BatchExplanationRequest,
    background_tasks: BackgroundTasks,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager),
    model_registry: ModelRegistry = Depends(get_model_registry)
):
    """Generate explanations for batch of subjects"""
    # This would be implemented to process multiple subjects
    # For now, return placeholder
    raise HTTPException(status_code=501, detail="Batch processing not yet implemented")

@explainer_app.get("/api/v1/explanations/{explanation_id}")
async def get_explanation(
    explanation_id: str,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager)
):
    """Retrieve explanation by ID"""
    artifact = artifact_manager.retrieve_explanation(explanation_id)

    if not artifact:
        raise HTTPException(status_code=404, detail="Explanation not found")

    return {
        "explanation_id": artifact.metadata.explanation_id,
        "subject_id": artifact.metadata.subject_id,
        "model_id": artifact.metadata.model_id,
        "model_type": artifact.metadata.model_type,
        "explanation_type": artifact.metadata.explanation_type,
        "confidence": artifact.metadata.confidence_score,
        "predicted_class": artifact.metadata.predicted_class,
        "created_at": artifact.metadata.created_at,
        "explanation_data": artifact.explanation_data,
        "summary_stats": artifact.summary_stats
    }

@explainer_app.get("/api/v1/explanations/{explanation_id}/visualization")
async def get_explanation_visualization(
    explanation_id: str,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager)
):
    """Get explanation visualization as PNG"""
    artifact = artifact_manager.retrieve_explanation(explanation_id)

    if not artifact:
        raise HTTPException(status_code=404, detail="Explanation not found")

    if "main_plot" not in artifact.visualizations:
        raise HTTPException(status_code=404, detail="Visualization not found")

    viz_bytes = artifact.visualizations["main_plot"]

    return StreamingResponse(
        io.BytesIO(viz_bytes),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=explanation_{explanation_id}.png"}
    )

@explainer_app.post("/api/v1/explanations/search")
async def search_explanations(
    request: ExplanationSearchRequest,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager)
):
    """Search explanations with filters"""
    filters = {k: v for k, v in request.dict().items() if v is not None and k != 'limit'}

    results = artifact_manager.search_explanations(filters, request.limit)

    return {
        "total_results": len(results),
        "explanations": [
            {
                "explanation_id": meta.explanation_id,
                "subject_id": meta.subject_id,
                "model_id": meta.model_id,
                "model_type": meta.model_type,
                "explanation_type": meta.explanation_type,
                "confidence": meta.confidence_score,
                "predicted_class": meta.predicted_class,
                "created_at": meta.created_at
            }
            for meta in results
        ]
    }

@explainer_app.post("/api/v1/explanations/{explanation_id}/export")
async def export_explanation(
    explanation_id: str,
    request: ExportRequest,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager)
):
    """Export explanation in specified format"""
    if request.export_format == "pdf":
        # Export as PDF
        output_path = Path(f"temp/explanation_{explanation_id}.pdf")
        output_path.parent.mkdir(exist_ok=True)

        success = artifact_manager.export_explanation_pdf(explanation_id, output_path)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to export PDF")

        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"explanation_{explanation_id}.pdf"
        )

    elif request.export_format == "zip":
        # Export as ZIP bundle
        output_path = Path(f"temp/explanation_{explanation_id}.zip")
        output_path.parent.mkdir(exist_ok=True)

        success = artifact_manager.export_explanation_bundle(explanation_id, output_path)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to export ZIP")

        return FileResponse(
            output_path,
            media_type="application/zip",
            filename=f"explanation_{explanation_id}.zip"
        )

    elif request.export_format == "json":
        # Export as JSON
        artifact = artifact_manager.retrieve_explanation(explanation_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Explanation not found")

        # Convert to JSON-serializable format
        export_data = {
            "metadata": {
                "explanation_id": artifact.metadata.explanation_id,
                "subject_id": artifact.metadata.subject_id,
                "model_id": artifact.metadata.model_id,
                "model_type": artifact.metadata.model_type,
                "explanation_type": artifact.metadata.explanation_type,
                "confidence": artifact.metadata.confidence_score,
                "predicted_class": artifact.metadata.predicted_class,
                "created_at": artifact.metadata.created_at.isoformat()
            },
            "explanation_data": _serialize_for_json(artifact.explanation_data),
            "summary_stats": artifact.summary_stats
        }

        return StreamingResponse(
            io.StringIO(json.dumps(export_data, indent=2)),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=explanation_{explanation_id}.json"}
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@explainer_app.get("/api/v1/explanations/statistics")
async def get_explanation_statistics(
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager)
):
    """Get overall explanation statistics"""
    stats = artifact_manager.get_explanation_statistics()
    return stats

@explainer_app.post("/api/v1/explanations/compare")
async def compare_explanations(
    explanation_ids: List[str],
    visualizer: ExplanationVisualizer = Depends(get_visualizer),
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager)
):
    """Compare multiple explanations"""
    explanations = []

    for exp_id in explanation_ids:
        artifact = artifact_manager.retrieve_explanation(exp_id)
        if artifact:
            explanations.append(artifact.explanation_data)

    if not explanations:
        raise HTTPException(status_code=404, detail="No valid explanations found")

    # Generate comparison visualization
    fig, viz_bytes = visualizer.create_comparison_visualization(explanations)

    return StreamingResponse(
        io.BytesIO(viz_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=explanation_comparison.png"}
    )

@explainer_app.get("/api/v1/explanations/{explanation_id}/dashboard")
async def get_interactive_dashboard(
    explanation_id: str,
    artifact_manager: ExplainabilityArtifactManager = Depends(get_artifact_manager),
    visualizer: ExplanationVisualizer = Depends(get_visualizer)
):
    """Get interactive dashboard for explanation"""
    artifact = artifact_manager.retrieve_explanation(explanation_id)

    if not artifact:
        raise HTTPException(status_code=404, detail="Explanation not found")

    dashboard_html = visualizer.create_interactive_dashboard(artifact.explanation_data)

    return StreamingResponse(
        io.StringIO(dashboard_html),
        media_type="text/html"
    )

# Helper functions
async def _generate_ensemble_explanation(
    request: ExplanationRequest,
    eeg_data: np.ndarray,
    model_registry: ModelRegistry,
    artifact_manager: ExplainabilityArtifactManager
) -> Dict[str, Any]:
    """Generate ensemble explanation combining multiple model types"""
    # This would combine explanations from multiple models
    # For now, return placeholder
    return {
        "subject_id": request.subject_id,
        "prediction": {
            "class": 1,
            "confidence": 0.85,
            "probabilities": [0.15, 0.85]
        },
        "ensemble_components": {
            "core15+": {"weight": 0.4, "prediction": 1, "confidence": 0.75},
            "cnn": {"weight": 0.3, "prediction": 1, "confidence": 0.90},
            "transformer": {"weight": 0.3, "prediction": 1, "confidence": 0.80}
        },
        "metadata": {
            "data_hash": "placeholder",
            "explanation_type": "ensemble"
        }
    }

def _serialize_for_json(data: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects for JSON export"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: _serialize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_serialize_for_json(item) for item in data]
    elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):  # PyTorch tensor
        return data.cpu().numpy().tolist()
    else:
        return data

# Include the explainer endpoints in the main app
def include_explainability_endpoints(main_app: FastAPI):
    """Include explainability endpoints in main FastAPI app"""
    main_app.mount("/explainability", explainer_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(explainer_app, host="0.0.0.0", port=8001)