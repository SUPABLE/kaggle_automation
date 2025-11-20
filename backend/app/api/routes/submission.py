from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import pandas as pd

from app.models.schemas import SubmissionDownloadRequest
from app.core.config import settings

router = APIRouter()

@router.get("/{task_id}/download")
async def download_submission(task_id: str):
    """
    Download the generated submission file
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

        if not task_data.get("results") or not task_data["results"].get("submission_data"):
            raise HTTPException(status_code=400, detail="No submission data available")

        # Get submission data
        submission_data = task_data["results"]["submission_data"]

        # Create submission DataFrame
        submission_df = pd.DataFrame(submission_data)

        # Save submission file
        submission_dir = os.path.join(settings.reports_dir, task_id)
        os.makedirs(submission_dir, exist_ok=True)

        submission_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)

        # Return file for download
        return FileResponse(
            path=submission_path,
            filename=f"submission_{task_id}.csv",
            media_type="text/csv"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download submission: {str(e)}")

@router.get("/{task_id}/info")
async def get_submission_info(task_id: str):
    """
    Get information about the generated submission
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

        if not task_data.get("results") or not task_data["results"].get("submission_data"):
            raise HTTPException(status_code=400, detail="No submission data available")

        submission_data = task_data["results"]["submission_data"]
        submission_df = pd.DataFrame(submission_data)

        return {
            "task_id": task_id,
            "submission_shape": submission_df.shape,
            "columns": submission_df.columns.tolist(),
            "sample_predictions": submission_df.head().to_dict('records'),
            "prediction_stats": {
                "min": float(submission_df.iloc[:, 1].min()) if len(submission_df.columns) > 1 else None,
                "max": float(submission_df.iloc[:, 1].max()) if len(submission_df.columns) > 1 else None,
                "mean": float(submission_df.iloc[:, 1].mean()) if len(submission_df.columns) > 1 else None,
                "null_count": int(submission_df.iloc[:, 1].isnull().sum()) if len(submission_df.columns) > 1 else 0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get submission info: {str(e)}")