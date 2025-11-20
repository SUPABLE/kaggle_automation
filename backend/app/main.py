from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

from app.api.routes import upload, analysis, report, submission
from app.core.config import settings

app = FastAPI(
    title="Kaggle Automation API",
    description="API for automated Kaggle competition workflows",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/api/analyze", tags=["analysis"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(submission.router, prefix="/api/submit", tags=["submission"])

# Mount static files
if os.path.exists("../shared/static"):
    app.mount("/static", StaticFiles(directory="../shared/static"), name="static")

@app.get("/")
async def root():
    return {"message": "Kaggle Automation API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)