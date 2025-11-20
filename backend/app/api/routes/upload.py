from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import os
import uuid
import aiofiles
from typing import List

from app.models.schemas import FileUploadResponse
from app.core.config import settings
from app.utils.file_handler import validate_csv, save_uploaded_file, get_file_info

router = APIRouter()

@router.post("/", response_model=FileUploadResponse)
async def upload_dataset(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    session_id: str = Form(default=None)
):
    """
    Upload and validate train and test CSV files
    """
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create session directory
        session_dir = os.path.join(settings.uploads_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Validate file extensions
        for file in [train_file, test_file]:
            if not file.filename.endswith('.csv'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a CSV file"
                )

            if file.size > settings.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size // (1024*1024)}MB"
                )

        # Save files
        train_path = await save_uploaded_file(train_file, session_dir, "train.csv")
        test_path = await save_uploaded_file(test_file, session_dir, "test.csv")

        # Validate and analyze files
        train_info = await validate_csv(train_path)
        test_info = await validate_csv(test_path)

        # Check column consistency
        train_columns = set(train_info['columns'])
        test_columns = set(test_info['columns'])

        # Test file should have all columns except target (we'll determine target later)
        missing_in_test = train_columns - test_columns
        extra_in_test = test_columns - train_columns

        warnings = []
        if missing_in_test:
            warnings.append(f"Columns missing in test file: {', '.join(missing_in_test)}")
        if extra_in_test:
            warnings.append(f"Extra columns in test file: {', '.join(extra_in_test)}")

        return FileUploadResponse(
            success=True,
            message="Files uploaded and validated successfully",
            file_info={
                "session_id": session_id,
                "train_file": {
                    "name": train_file.filename,
                    "size": train_file.size,
                    "rows": train_info['shape'][0],
                    "columns": train_info['shape'][1]
                },
                "test_file": {
                    "name": test_file.filename,
                    "size": test_file.size,
                    "rows": test_info['shape'][0],
                    "columns": test_info['shape'][1]
                },
                "warnings": warnings
            },
            train_columns=train_info['columns'],
            test_columns=test_info['columns']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """
    Get information about uploaded files for a session
    """
    try:
        session_dir = os.path.join(settings.uploads_dir, session_id)
        if not os.path.exists(session_dir):
            raise HTTPException(status_code=404, detail="Session not found")

        train_path = os.path.join(session_dir, "train.csv")
        test_path = os.path.join(session_dir, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise HTTPException(status_code=404, detail="Files not found for this session")

        train_info = await get_file_info(train_path)
        test_info = await get_file_info(test_path)

        return {
            "session_id": session_id,
            "train_info": train_info,
            "test_info": test_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")