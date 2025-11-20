import pandas as pd
import os
import aiofiles
from typing import Dict, Any
from fastapi import UploadFile

async def save_uploaded_file(file: UploadFile, directory: str, filename: str) -> str:
    """
    Save uploaded file to specified directory
    """
    file_path = os.path.join(directory, filename)

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return file_path

async def validate_csv(file_path: str) -> Dict[str, Any]:
    """
    Validate CSV file and return basic information
    """
    try:
        # Read CSV with pandas
        df = pd.read_csv(file_path)

        # Basic validation
        if df.empty:
            raise ValueError("CSV file is empty")

        if df.columns.duplicated().any():
            raise ValueError("CSV file has duplicate column names")

        # Get file info
        file_size = os.path.getsize(file_path)

        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "file_size": file_size,
            "memory_usage": df.memory_usage(deep=True).sum()
        }

    except Exception as e:
        raise ValueError(f"CSV validation failed: {str(e)}")

async def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic file information
    """
    try:
        df = pd.read_csv(file_path)
        file_size = os.path.getsize(file_path)

        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "file_size": file_size,
            "sample_data": df.head().to_dict('records')
        }

    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

def ensure_directory(directory: str) -> None:
    """
    Ensure directory exists
    """
    os.makedirs(directory, exist_ok=True)

def clean_filename(filename: str) -> str:
    """
    Clean filename for safe storage
    """
    # Remove path components and unsafe characters
    filename = os.path.basename(filename)
    filename = ''.join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-'))
    return filename.strip()