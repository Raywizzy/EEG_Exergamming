"""
UI Routes for Clinical Interface
Web interface for clinicians to upload EEG data and view results
"""

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uuid
import os
import json
from datetime import datetime
from typing import Optional

from ..services.jobs import JobManager
from ..services.report_pdf import generate_clinical_pdf

router = APIRouter()

# Configuration
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
TEMPLATES = Jinja2Templates(directory=str(TEMPLATES_DIR))

DATA_IN = Path(os.environ.get("BIDS_IN", "/tmp/core15_data/in"))
DATA_OUT = Path(os.environ.get("RESULTS_OUT", "/tmp/core15_data/out"))

# Ensure directories exist
DATA_IN.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Job manager
job_manager = JobManager()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing recent jobs and system status."""
    recent_jobs = job_manager.list_jobs(limit=10)

    # Calculate summary statistics
    total_jobs = len(job_manager.list_jobs())
    successful_jobs = len([j for j in job_manager.list_jobs() if j.get('status') == 'succeeded'])
    success_rate = (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0

    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Core15+ Clinical Dashboard",
            "recent_jobs": recent_jobs,
            "total_jobs": total_jobs,
            "success_rate": success_rate,
            "performance_ba": 97.2,
            "processing_time": "0.41s",
            "now": datetime.now()
        }
    )

@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page for new validation jobs."""
    return TEMPLATES.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "title": "Upload EEG Data for Validation",
            "now": datetime.now()
        }
    )

@router.post("/upload")
async def handle_upload(
    request: Request,
    site_id: str = Form(..., description="Site identifier"),
    trial_id: str = Form(..., description="Trial identifier"),
    subjects: str = Form("", description="Comma-separated subject IDs"),
    bids_zip: Optional[UploadFile] = File(None, description="BIDS dataset ZIP file")
):
    """Handle file upload and create new validation job."""
    try:
        # Generate unique job ID
        job_uuid = uuid.uuid4()
        job_id = str(job_uuid)

        # Create job directories
        job_in_dir = DATA_IN / f"{trial_id}" / f"job-{job_uuid}"
        job_out_dir = DATA_OUT / f"{trial_id}" / f"job-{job_uuid}"
        job_in_dir.mkdir(parents=True, exist_ok=True)
        job_out_dir.mkdir(parents=True, exist_ok=True)

        # Handle file upload
        if bids_zip and bids_zip.filename:
            if not bids_zip.filename.endswith('.zip'):
                raise HTTPException(status_code=400, detail="Only ZIP files are supported")

            zip_path = job_in_dir / bids_zip.filename
            with zip_path.open("wb") as f:
                shutil.copyfileobj(bids_zip.file, f)

            # Extract ZIP file
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(job_in_dir)

            # Remove ZIP file after extraction
            zip_path.unlink()

        # Parse subjects list
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

        # Create validation job
        job_id = job_manager.create_job(
            site_id=site_id,
            trial_id=trial_id,
            bids_dir=str(job_in_dir),
            out_dir=str(job_out_dir),
            subjects=subject_list or None
        )

        # Redirect to job status page
        return RedirectResponse(url=f"/ui/jobs/{job_id}", status_code=303)

    except Exception as e:
        # Handle upload errors
        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Upload Error",
                "message": f"Failed to process upload: {str(e)}",
                "status_code": 400
            },
            status_code=400
        )

@router.get("/jobs", response_class=HTMLResponse)
async def jobs_table(request: Request):
    """Jobs table showing all validation jobs."""
    jobs = job_manager.list_jobs()

    return TEMPLATES.TemplateResponse(
        "jobs.html",
        {
            "request": request,
            "title": "Validation Jobs",
            "jobs": jobs,
            "now": datetime.now()
        }
    )

@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    """Detailed view of a specific job."""
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Load additional job details if available
    job_details = {}
    if job.get('out_dir'):
        out_dir = Path(job['out_dir'])

        # Load clinical report if available
        report_file = out_dir / "clinical_report.json"
        if report_file.exists():
            try:
                with report_file.open() as f:
                    job_details['clinical_report'] = json.load(f)
            except:
                pass

        # Load QC metrics if available
        qc_file = out_dir / "qc_summary.json"
        if qc_file.exists():
            try:
                with qc_file.open() as f:
                    job_details['qc_metrics'] = json.load(f)
            except:
                pass

    return TEMPLATES.TemplateResponse(
        "job_detail.html",
        {
            "request": request,
            "title": f"Job {job_id[:8]}...",
            "job": job,
            "job_details": job_details,
            "now": datetime.now()
        }
    )

@router.get("/jobs/{job_id}/download/{file_type}")
async def download_job_file(job_id: str, file_type: str):
    """Download job output files."""
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    out_dir = Path(job.get('out_dir', ''))

    if file_type == "pdf":
        # Generate or serve PDF report
        pdf_file = out_dir / "clinical_report.pdf"
        if not pdf_file.exists():
            # Generate PDF from available data
            job_data = {}
            if out_dir.exists():
                # Load available data for PDF generation
                report_file = out_dir / "clinical_report.json"
                if report_file.exists():
                    with report_file.open() as f:
                        job_data = json.load(f)

                # Generate PDF
                generate_clinical_pdf(
                    str(pdf_file),
                    site_id=job.get('site_id', 'Unknown'),
                    trial_id=job.get('trial_id', 'Unknown'),
                    job_data=job_data,
                    job_info=job
                )

        if pdf_file.exists():
            return FileResponse(
                pdf_file,
                media_type="application/pdf",
                filename=f"core15_report_{job_id[:8]}.pdf"
            )

    elif file_type == "csv":
        # Serve features CSV
        csv_file = out_dir / "core15_features.csv"
        if csv_file.exists():
            return FileResponse(
                csv_file,
                media_type="text/csv",
                filename=f"core15_features_{job_id[:8]}.csv"
            )

    elif file_type == "json":
        # Serve clinical report JSON
        json_file = out_dir / "clinical_report.json"
        if json_file.exists():
            return FileResponse(
                json_file,
                media_type="application/json",
                filename=f"core15_report_{job_id[:8]}.json"
            )

    raise HTTPException(status_code=404, detail="File not found")

@router.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """Help and documentation page."""
    return TEMPLATES.TemplateResponse(
        "help.html",
        {
            "request": request,
            "title": "Help & Documentation",
            "now": datetime.now()
        }
    )