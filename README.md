# Kaggle Automation Web Application

A complete end-to-end web application that automates Kaggle competition workflows from dataset upload to publishable research article generation. The system includes a Next.js frontend for user interaction, FastAPI backend for ML processing, and Docker configuration for local deployment.

## Features

### Frontend (Next.js + TailwindCSS)
- **Dataset Upload**: Drag-and-drop interface for train/test CSV files
- **Competition Configuration**: Forms for target column, task type, metrics, and rules
- **Analysis Dashboard**: Real-time monitoring of ML pipeline progress
- **Article Viewer**: Interactive display of generated research reports
- **Download Center**: Access submission files and detailed reports

### Backend (FastAPI + Python)
- **Automated EDA**: Comprehensive exploratory data analysis with visualizations
- **Intelligent Preprocessing**: Missing value handling, encoding, and feature engineering
- **Multi-Model Training**: LightGBM, XGBoost, CatBoost, Linear models with hyperparameter optimization
- **Ensemble Methods**: Weighted averaging, stacking, and blending strategies
- **Report Generation**: Research-style articles with embedded charts and insights
- **Cross-Validation**: Automatic strategy selection (stratified, time-series, group-aware)

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
```bash
git clone <repository_url>
cd kaggle_automation
```

2. **Start the application**:
```bash
docker-compose up
```

3. **Access the application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup

#### Frontend Development
```bash
cd kaggle_automation
npm install
npm run dev
```
Access at http://localhost:3000

#### Backend Development
```bash
cd kaggle_automation/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Access at http://localhost:8000

## Usage Guide

### 1. Upload Your Dataset
- Prepare your training and test CSV files
- Drag and drop both files onto the upload area
- Wait for validation to complete

### 2. Configure Competition Settings
- Select your target column from the dropdown
- Choose the appropriate task type (classification/regression)
- Set evaluation metrics and submission format
- Configure advanced options (feature engineering, optimization)

### 3. Start Automated Analysis
- Click "Start Analysis" to begin the pipeline
- Monitor progress on the analysis dashboard
- Pipeline includes:
  - Exploratory Data Analysis
  - Data Preprocessing & Feature Engineering
  - Model Training & Selection
  - Ensemble Creation
  - Submission Generation

### 4. Review Results
- View the generated research article with insights
- Download competition-ready submission file
- Explore detailed model performance metrics

## Project Structure

```
kaggle_automation/
├── frontend/                 # Next.js application (root level)
│   ├── src/
│   │   ├── app/              # App Router pages
│   │   │   ├── page.tsx      # Main upload/config page
│   │   │   ├── analysis/     # Analysis dashboard
│   │   │   └── article/      # Report viewer
│   │   ├── components/       # React components
│   │   │   ├── ui/           # Reusable UI components
│   │   │   ├── forms/        # Form components
│   │   │   └── charts/       # Data visualization
│   │   └── lib/              # Utilities and API client
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py           # Application entry point
│   │   ├── api/routes/       # API endpoints
│   │   ├── services/         # ML pipeline services
│   │   ├── models/           # Pydantic models
│   │   ├── core/             # Configuration
│   │   └── utils/            # Helper functions
│   └── requirements.txt      # Python dependencies
├── shared/                   # Shared resources
│   ├── templates/            # Report templates
│   ├── configs/              # Configuration files
│   └── static/               # Static assets
├── data/                     # Local data storage
│   ├── uploads/              # User uploaded datasets
│   ├── artifacts/            # Generated models/charts
│   └── reports/              # Generated articles
├── docker-compose.yml        # Multi-service orchestration
├── Dockerfile.frontend       # Frontend container
├── Dockerfile.backend        # Backend container
└── README.md                 # This file
```

## Supported File Formats

- **Input**: CSV files (train.csv and test.csv)
- **Output**:
  - submission.csv (competition-ready submission)
  - HTML report (research-style article)
  - Model artifacts (trained models and configurations)

## ML Pipeline Capabilities

### Data Analysis
- Statistical summaries and data quality assessment
- Missing value pattern analysis
- Feature correlation and importance analysis
- Target variable distribution analysis
- Automatic detection of data leakage and quality issues

### Model Training
- **Classification**: LightGBM, XGBoost, CatBoost, Logistic Regression
- **Regression**: LightGBM, XGBoost, CatBoost, Ridge, Linear Regression
- **Automatic CV Strategy**: Stratified K-Fold, Time Series, Group-aware
- **Hyperparameter Optimization**: Optuna-based optimization
- **Feature Selection**: Automated feature selection based on importance

### Ensemble Methods
- **Weighted Averaging**: Optimized weight allocation
- **Stacking**: Meta-learner integration
- **Blending**: Simple and advanced blending strategies

## Environment Variables

### Frontend
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend
```bash
PYTHONPATH=/app
DATA_DIR=/app/data
UPLOADS_DIR=/app/data/uploads
ARTIFACTS_DIR=/app/data/artifacts
REPORTS_DIR=/app/data/reports
```

## Example: Titanic Dataset

Here's how to use the application with the classic Titanic dataset:

1. **Download Data**:
   - Get train.csv and test.csv from Kaggle
   - Ensure they have matching columns (except target)

2. **Configure Competition**:
   - Target Column: `Survived`
   - Task Type: `Binary Classification`
   - Evaluation Metric: `Accuracy`
   - Submission Format: ID Column `PassengerId`, Prediction Column `Survived`

3. **Expected Results**:
   - Automated EDA revealing survival patterns
   - Multiple models with 80-85% accuracy
   - Ensemble model with improved performance
   - Detailed research article with insights

## API Endpoints

### Upload
- `POST /api/upload/` - Upload and validate dataset files
- `GET /api/upload/sessions/{session_id}/info` - Get session file information

### Analysis
- `POST /api/analyze/start` - Start ML analysis pipeline
- `GET /api/analyze/{task_id}/status` - Get analysis progress
- `WebSocket /api/analyze/{task_id}/progress` - Real-time progress updates

### Reports & Submissions
- `POST /api/report/{task_id}` - Generate research report
- `GET /api/report/{task_id}/view` - View report (HTML)
- `GET /api/report/{task_id}/download` - Download report
- `GET /api/submit/{task_id}/download` - Download submission file
- `GET /api/submit/{task_id}/info` - Get submission information

## Development

### Running Tests
```bash
# Frontend tests
npm run test

# Backend tests
cd backend
pytest
```

### Code Style
- **Frontend**: ESLint + Prettier
- **Backend**: Black + isort
- **TypeScript**: Strict typing throughout

### Adding New Models
1. Implement model in `backend/app/services/modeling.py`
2. Update model configuration schemas
3. Add model-specific hyperparameters
4. Update evaluation metrics if needed

## Troubleshooting

### Common Issues

1. **Docker Build Fails**:
   - Ensure Docker has enough memory (4GB+ recommended)
   - Check that all files are properly copied

2. **Backend Connection Error**:
   - Verify backend is running on port 8000
   - Check CORS configuration in `backend/app/main.py`

3. **File Upload Fails**:
   - Ensure CSV files are properly formatted
   - Check file size limits (100MB per file)
   - Verify columns match between train and test sets

4. **Analysis Fails**:
   - Check backend logs for detailed error messages
   - Ensure target column exists and has valid values
   - Verify sufficient data for cross-validation

### Performance Tips

- For large datasets (>100MB), consider preprocessing before upload
- Use appropriate CV strategy for your data type
- Enable hyperparameter optimization for better performance
- Consider feature engineering for domain-specific insights

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:8000/docs
3. Open an issue on the GitHub repository

## Changelog

### v1.0.0
- Initial release with complete ML pipeline
- Support for classification and regression tasks
- Automated EDA and report generation
- Docker deployment configuration
- Real-time progress monitoring