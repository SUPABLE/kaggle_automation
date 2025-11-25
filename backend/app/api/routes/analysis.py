from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import uuid
import json
import os
from typing import Dict, Any

from app.models.schemas import (
    AnalysisRequest, AnalysisStartResponse, AnalysisStatus, TaskProgress
)
from app.core.config import settings
from app.services.eda import EDAService
from app.services.preprocessing import PreprocessingService
from app.services.cross_validation import CrossValidationService
from app.services.modeling import ModelingService
from app.services.ensemble import EnsembleService
from app.services.report import ReportService

router = APIRouter()

# Global dictionary to store task status and progress
active_tasks: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}

@router.post("/start", response_model=AnalysisStartResponse)
async def start_analysis(request: AnalysisRequest):
    """
    Start the full ML analysis pipeline
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Validate session exists
        session_dir = os.path.join(settings.uploads_dir, request.file_session_id)
        if not os.path.exists(session_dir):
            raise HTTPException(status_code=404, detail="Upload session not found")

        # Initialize task status
        active_tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_stage": "initialization",
            "message": "Task initialized",
            "error": None,
            "results": None,
            "config": request.config.dict(),
            "session_id": request.file_session_id
        }

        # Start analysis in background
        asyncio.create_task(run_analysis_pipeline(task_id, request))

        return AnalysisStartResponse(
            success=True,
            task_id=task_id,
            message="Analysis started successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@router.get("/{task_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(task_id: str):
    """
    Get the current status of an analysis task
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = active_tasks[task_id]

    return AnalysisStatus(
        task_id=task_id,
        status=task_data["status"],
        current_stage=task_data.get("current_stage"),
        progress_percentage=task_data["progress"],
        message=task_data.get("message"),
        error=task_data.get("error"),
        results=task_data.get("results")
    )

@router.websocket("/{task_id}/progress")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await websocket.accept()

    if task_id not in active_tasks:
        await websocket.send_json({"error": "Task not found"})
        await websocket.close()
        return

    websocket_connections[task_id] = websocket

    try:
        # Send initial status
        task_data = active_tasks[task_id]
        await websocket.send_json({
            "status": task_data["status"],
            "progress": task_data["progress"],
            "stage": task_data.get("current_stage"),
            "message": task_data.get("message")
        })

        # Keep connection alive and send updates
        while task_id in active_tasks and task_data["status"] in ["pending", "running"]:
            await asyncio.sleep(1)
            task_data = active_tasks[task_id]

            await websocket.send_json({
                "status": task_data["status"],
                "progress": task_data["progress"],
                "stage": task_data.get("current_stage"),
                "message": task_data.get("message"),
                "error": task_data.get("error")
            })

            if task_data["status"] in ["completed", "failed"]:
                break

    except WebSocketDisconnect:
        pass
    finally:
        if task_id in websocket_connections:
            del websocket_connections[task_id]

async def update_task_progress(task_id: str, progress: float, stage: str, message: str, error: str = None):
    """
    Update task progress and notify WebSocket connections
    """
    if task_id not in active_tasks:
        return

    active_tasks[task_id].update({
        "progress": progress,
        "current_stage": stage,
        "message": message,
        "error": error
    })

    # Send WebSocket update if connection exists
    if task_id in websocket_connections:
        try:
            await websocket_connections[task_id].send_json({
                "status": active_tasks[task_id]["status"],
                "progress": progress,
                "stage": stage,
                "message": message,
                "error": error
            })
        except:
            # WebSocket connection might be closed
            pass

async def run_analysis_pipeline(task_id: str, request: AnalysisRequest):
    """
    Run the complete ML analysis pipeline
    """
    try:
        # Update status to running
        active_tasks[task_id]["status"] = "running"

        # Initialize services
        eda_service = EDAService()
        preprocessing_service = PreprocessingService()
        cv_service = CrossValidationService()
        modeling_service = ModelingService()
        ensemble_service = EnsembleService()
        report_service = ReportService()

        # Load data paths
        session_dir = os.path.join(settings.uploads_dir, request.file_session_id)
        train_path = os.path.join(session_dir, "train.csv")
        test_path = os.path.join(session_dir, "test.csv")

        # Stage 1: Exploratory Data Analysis (10%)
        await update_task_progress(task_id, 10, "eda", "Running exploratory data analysis")
        eda_results = await eda_service.analyze(train_path, test_path, request.config)

        # Stage 2: Data Preprocessing (25%)
        await update_task_progress(task_id, 25, "preprocessing", "Preprocessing data")
        processed_data = await preprocessing_service.process(
            train_path, test_path, request.config, eda_results
        )

        # Stage 3: Cross-Validation Strategy (35%)
        await update_task_progress(task_id, 35, "cross_validation", "Setting up cross-validation strategy")
        cv_strategy = await cv_service.setup_cv(processed_data, request.config)

        # Stage 4: Model Training (70%)
        await update_task_progress(task_id, 40, "modeling", "Training models")
        model_results = await modeling_service.train_models(
            processed_data, cv_strategy, request.config
        )
        await update_task_progress(task_id, 70, "modeling", "Model training completed")

        # Stage 5: Ensemble Creation (85%)
        await update_task_progress(task_id, 75, "ensemble", "Creating ensemble models")
        ensemble_results = await ensemble_service.create_ensemble(
            model_results, processed_data, request.config
        )

        # Stage 6: Submission Generation (95%)
        await update_task_progress(task_id, 90, "submission", "Generating submission file")
        submission_data = await ensemble_service.generate_submission(
            ensemble_results, processed_data, request.config
        )

        # Store results
        results = {
            "eda_results": eda_results,
            "model_results": model_results,
            "ensemble_results": ensemble_results,
            "submission_data": submission_data,
            "config": request.config.dict()
        }

        active_tasks[task_id]["results"] = results
        active_tasks[task_id]["status"] = "completed"
        await update_task_progress(task_id, 100, "completed", "Analysis completed successfully")

    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        await update_task_progress(task_id, active_tasks[task_id]["progress"], "failed", f"Analysis failed: {str(e)}")