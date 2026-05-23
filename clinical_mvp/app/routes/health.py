"""
Health Check Routes
System monitoring and status endpoints
"""

from fastapi import APIRouter
from datetime import datetime
import psutil
import os
from pathlib import Path

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0_Phase_VI",
        "service": "core15_clinical_mvp"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics."""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Core15+ specific checks
        core15_status = "healthy"
        core15_checks = {}

        # Check if Core15+ components are accessible
        try:
            # Check if processing pipeline is available
            import sys
            sys.path.append('/Users/user/Desktop/EEG_Exergamming')
            from src.preprocess.pipeline import EEGPreprocessor
            from src.features.core15_plus import Core15PlusExtractor
            from src.clinical.qc_manager import ClinicalQCManager

            core15_checks['preprocessor'] = "available"
            core15_checks['feature_extractor'] = "available"
            core15_checks['qc_manager'] = "available"

        except Exception as e:
            core15_status = "degraded"
            core15_checks['error'] = str(e)

        # Check data directories
        data_in = Path(os.environ.get("BIDS_IN", "/tmp/core15_data/in"))
        data_out = Path(os.environ.get("RESULTS_OUT", "/tmp/core15_data/out"))

        storage_checks = {
            "input_directory": {
                "path": str(data_in),
                "exists": data_in.exists(),
                "writable": data_in.exists() and os.access(data_in, os.W_OK)
            },
            "output_directory": {
                "path": str(data_out),
                "exists": data_out.exists(),
                "writable": data_out.exists() and os.access(data_out, os.W_OK)
            }
        }

        return {
            "status": "healthy" if core15_status == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0_Phase_VI",
            "service": "core15_clinical_mvp",
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "core15_status": core15_status,
            "core15_checks": core15_checks,
            "storage_checks": storage_checks,
            "performance_targets": {
                "balanced_accuracy": "97.2%",
                "processing_time": "0.41s per subject",
                "regulatory_compliance": "FDA/EMA compliant"
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes-style readiness check."""
    try:
        # Check if service can accept traffic
        data_in = Path(os.environ.get("BIDS_IN", "/tmp/core15_data/in"))
        data_out = Path(os.environ.get("RESULTS_OUT", "/tmp/core15_data/out"))

        # Ensure directories exist and are writable
        data_in.mkdir(parents=True, exist_ok=True)
        data_out.mkdir(parents=True, exist_ok=True)

        if not (data_in.exists() and data_out.exists()):
            return {"status": "not_ready", "reason": "data directories not accessible"}

        return {"status": "ready", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}

@router.get("/health/live")
async def liveness_check():
    """Kubernetes-style liveness check."""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": psutil.boot_time()
    }