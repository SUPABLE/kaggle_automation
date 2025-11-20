from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
import pandas as pd

from app.models.schemas import ReportRequest, ReportResponse
from app.core.config import settings
from app.services.report import ReportService
from app.api.routes.analysis import active_tasks

router = APIRouter()

@router.post("/{task_id}", response_model=ReportResponse)
async def generate_report(task_id: str, request: ReportRequest):
    """
    Generate a research-style article for the completed analysis
    """
    try:
        # Check if task exists and is completed
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task_data = active_tasks[task_id]
        if task_data["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed. Current status: {task_data['status']}"
            )

        if not task_data.get("results"):
            raise HTTPException(status_code=400, detail="No results available for this task")

        # Generate report
        report_service = ReportService()
        report_content = await report_service.generate_article(
            task_data["results"],
            format=request.format
        )

        # Save report to file
        report_dir = os.path.join(settings.reports_dir, task_id)
        os.makedirs(report_dir, exist_ok=True)

        report_filename = f"report.{request.format}"
        report_path = os.path.join(report_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return ReportResponse(
            task_id=task_id,
            report_content=report_content,
            report_format=request.format,
            generated_at=str(pd.Timestamp.now())
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/{task_id}/view", response_class=HTMLResponse)
async def view_report(task_id: str, format: str = "html"):
    """
    View the generated report as HTML
    """
    try:
        report_path = os.path.join(settings.reports_dir, task_id, f"report.{format}")

        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")

        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()

        return HTMLResponse(content=report_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to view report: {str(e)}")

@router.get("/{task_id}/download")
async def download_report(task_id: str, format: str = "html"):
    """
    Download the generated report
    """
    try:
        report_path = os.path.join(settings.reports_dir, task_id, f"report.{format}")

        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")

        # Return file for download
        from fastapi.responses import FileResponse
        return FileResponse(
            path=report_path,
            filename=f"kaggle_analysis_report_{task_id}.{format}",
            media_type="text/html" if format == "html" else "text/markdown"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")