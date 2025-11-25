import pandas as pd
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader
import os
from datetime import datetime
import json
import base64
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px

from app.core.config import settings
from app.utils.plotting import create_feature_importance_plot, create_model_comparison_plot

class ReportService:
    def __init__(self):
        self.templates_dir = os.path.join(settings.shared_dir, "templates")
        self.reports_dir = settings.reports_dir

    async def generate_article(self, results: Dict[str, Any], format: str = "html") -> str:
        """
        Generate a research-style article from analysis results
        """
        try:
            # Extract results
            eda_results = results["eda_results"]
            model_results = results["model_results"]
            ensemble_results = results["ensemble_results"]
            config = results["config"]

            # Prepare article data
            article_data = self._prepare_article_data(results)

            # Generate article content
            if format == "html":
                content = await self._generate_html_article(article_data)
            else:  # markdown
                content = await self._generate_markdown_article(article_data)

            return content

        except Exception as e:
            raise Exception(f"Report generation failed: {str(e)}")

    def _prepare_article_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for article generation
        """
        eda_results = results["eda_results"]
        model_results = results["model_results"]
        ensemble_results = results["ensemble_results"]
        config = results["config"]

        # Dataset overview
        dataset_info = eda_results["dataset_info"]
        quality_assessment = eda_results["quality_assessment"]

        # Model performance
        best_model = model_results["best_model"]
        model_comparison = self._prepare_model_comparison(model_results["model_results"])

        # Feature importance
        feature_importance = model_results["feature_importance"]

        # Target analysis
        target_analysis = eda_results["target_analysis"]

        # Ensemble performance
        ensemble_performance = ensemble_results["ensemble_performance"]

        # EDA insights
        eda_insights = self._generate_eda_insights(eda_results)

        return {
            "title": f"Automated Analysis Report - {config.get('competition_name', 'Kaggle Competition')}",
            "date": datetime.now().strftime("%B %d, %Y"),
            "config": config,
            "dataset_info": dataset_info,
            "target_analysis": target_analysis,
            "quality_assessment": quality_assessment,
            "eda_insights": eda_insights,
            "model_comparison": model_comparison,
            "best_model": best_model,
            "feature_importance": feature_importance,
            "ensemble_performance": ensemble_performance,
            "visualizations": eda_results.get("visualizations", {})
        }

    def _prepare_model_comparison(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare model comparison data
        """
        comparison_data = []

        for result in model_results:
            comparison_data.append({
                "model_name": result["model_name"],
                "cv_score": f"{result['cv_score']:.4f} ± {result['cv_std']:.4f}",
                "training_time": f"{result['training_time']:.2f}s",
                "additional_metrics": result["additional_metrics"]
            })

        # Sort by CV score
        comparison_data.sort(key=lambda x: float(x["cv_score"].split(" ± ")[0]), reverse=True)

        return {
            "models": comparison_data,
            "best_score": comparison_data[0]["cv_score"] if comparison_data else "N/A"
        }

    def _generate_eda_insights(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights from EDA results
        """
        insights = []

        # Dataset size insights
        dataset_info = eda_results["dataset_info"]
        train_shape = dataset_info["train_shape"]
        test_shape = dataset_info["test_shape"]

        insights.append(f"Dataset contains {train_shape[0]:,} training samples and {test_shape[0]:,} test samples with {train_shape[1]} features.")

        # Target variable insights
        target_analysis = eda_results["target_analysis"]
        if target_analysis["type"] == "classification":
            class_balance = target_analysis["class_balance"]
            insights.append(f"Target variable has {target_analysis['unique_classes']} classes with a class imbalance ratio of {class_balance['majority_ratio']:.2f}.")
        else:
            stats = target_analysis["statistics"]
            insights.append(f"Target variable ranges from {stats['min']:.2f} to {stats['max']:.2f} with a mean of {stats['mean']:.2f}.")

        # Missing values insights
        missing_values = eda_results["missing_values"]
        if missing_values["train_missing"]:
            missing_features = list(missing_values["train_missing"].keys())
            insights.append(f"Found missing values in {len(missing_features)} features: {', '.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}")
        else:
            insights.append("No missing values found in the dataset.")

        # Feature type insights
        dtypes_info = eda_results["dtypes_info"]
        insights.append(f"Dataset contains {len(dtypes_info['categorical_columns'])} categorical and {len(dtypes_info['numerical_columns'])} numerical features.")

        # Correlation insights
        correlation_analysis = eda_results["correlation_analysis"]
        if correlation_analysis["high_correlation_pairs"]:
            high_corr_count = len(correlation_analysis["high_correlation_pairs"])
            insights.append(f"Found {high_corr_count} highly correlated feature pairs that may need attention.")

        # Quality assessment insights
        quality_assessment = eda_results["quality_assessment"]
        if quality_assessment["issues"]:
            insights.append(f"Data quality score: {quality_assessment['quality_score']}/100. Issues found: {', '.join(quality_assessment['issues'][:2])}")
        else:
            insights.append(f"High data quality score: {quality_assessment['quality_score']}/100 with no major issues detected.")

        return {
            "insights": insights,
            "summary": " ".join(insights[:3])  # Summary with first 3 insights
        }

    async def _generate_html_article(self, article_data: Dict[str, Any]) -> str:
        """
        Generate HTML article
        """
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 20px;
        }
        h3 {
            color: #7f8c8d;
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .insight-box {
            background-color: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .visualization {
            margin: 30px 0;
            text-align: center;
        }
        .best-model {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>

        <div class="metadata">
            <p><strong>Generated:</strong> {{ date }}</p>
            <p><strong>Task Type:</strong> {{ config.task_type|title }}</p>
            <p><strong>Evaluation Metric:</strong> {{ config.evaluation_metric|upper }}</p>
            <p><strong>Target Column:</strong> {{ config.target_column }}</p>
        </div>

        <h2>Abstract</h2>
        <p>This report presents an automated analysis of a Kaggle competition dataset. Our analysis explores {{ dataset_info.train_shape[1] }} features across {{ dataset_info.train_shape[0] }} training samples, identifying key patterns and relationships. We trained and evaluated multiple machine learning models, achieving the best performance with {{ best_model.model_name|title }} ({{ best_model.cv_score }}). The ensemble approach further improved performance to {{ ensemble_performance.ensemble_best }}, demonstrating {{ ensemble_performance.improvement > 0 and 'an improvement of' or 'similar performance to' }} the best individual model.</p>

        <h2>1. Data Overview</h2>

        <h3>Dataset Summary</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "{:,}".format(dataset_info.train_shape[0]) }}</div>
                <div>Training Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ dataset_info.train_shape[1] }}</div>
                <div>Features</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:,}".format(dataset_info.test_shape[0]) }}</div>
                <div>Test Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ quality_assessment.quality_score }}/100</div>
                <div>Data Quality Score</div>
            </div>
        </div>

        <h3>Target Variable Analysis</h3>
        {% if target_analysis.type == "classification" %}
        <p>The target variable is {{ target_analysis.type }} with {{ target_analysis.unique_classes }} classes. The majority class represents {{ "%.1f"|format(target_analysis.class_balance.majority_ratio * 100) }}% of the samples.</p>
        {% else %}
        <p>The target variable is continuous with a mean of {{ "%.2f"|format(target_analysis.statistics.mean) }} (±{{ "%.2f"|format(target_analysis.statistics.std) }}) and ranges from {{ "%.2f"|format(target_analysis.statistics.min) }} to {{ "%.2f"|format(target_analysis.statistics.max) }}.</p>
        {% endif %}

        <h3>Key EDA Insights</h3>
        <div class="insight-box">
            <ul>
            {% for insight in eda_insights.insights %}
                <li>{{ insight }}</li>
            {% endfor %}
            </ul>
        </div>

        {% if visualizations.target_distribution %}
        <div class="visualization">
            <h3>Target Distribution</h3>
            {{ visualizations.target_distribution|safe }}
        </div>
        {% endif %}

        <h2>2. Methodology</h2>

        <h3>Data Preprocessing</h3>
        <p>Our preprocessing pipeline included missing value imputation, categorical encoding, and feature scaling. We carefully handled different data types and ensured consistency between training and test datasets.</p>

        <h3>Cross-Validation Strategy</h3>
        <p>We employed {{ config.cv_strategy|replace('_', ' ')|title }} with {{ config.cv_folds }} folds to ensure robust model evaluation and prevent overfitting.</p>

        <h3>Model Selection</h3>
        <p>We trained and evaluated multiple models: Logistic Regression, Random Forest, LightGBM, XGBoost, and CatBoost, selecting the best performing model based on {{ config.evaluation_metric|upper }}.</p>

        <h2>3. Results</h2>

        <h3>Model Performance Comparison</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>CV Score</th>
                    <th>Training Time</th>
                </tr>
            </thead>
            <tbody>
            {% for model in model_comparison.models %}
                <tr {% if model.model_name == best_model.model_name %}class="best-model"{% endif %}>
                    <td>{{ model.model_name|title }}</td>
                    <td>{{ model.cv_score }}</td>
                    <td>{{ model.training_time }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ best_model.cv_score }}</div>
                <div>Best Model Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ ensemble_performance.ensemble_best }}</div>
                <div>Ensemble Score</div>
            </div>
            {% if ensemble_performance.improvement > 0 %}
            <div class="metric-card">
                <div class="metric-value">+{{ "%.4f"|format(ensemble_performance.improvement) }}</div>
                <div>Ensemble Improvement</div>
            </div>
            {% endif %}
        </div>

        <h3>Feature Importance</h3>
        <p>Top 10 most important features based on model aggregations:</p>
        <ul>
        {% for feature, importance in feature_importance.items()[:10] %}
            <li><strong>{{ feature }}</strong>: {{ "%.4f"|format(importance) }}</li>
        {% endfor %}
        </ul>

        <h2>4. Conclusions</h2>
        <p>Our automated analysis successfully identified key patterns in the dataset and built robust predictive models. The {{ best_model.model_name }} model achieved strong performance with a CV score of {{ best_model.cv_score }}, while the ensemble approach {{ ensemble_performance.improvement > 0 and 'further improved' or 'maintained' }} this performance at {{ ensemble_performance.ensemble_best }}.</p>

        <h3>Key Findings</h3>
        <ul>
            <li>Dataset contains {{ dataset_info.train_shape[1] }} features with {{ quality_assessment.quality_score }}/100 data quality</li>
            <li>{{ best_model.model_name|title }} emerged as the best individual model</li>
            <li>Ensemble methods {{ ensemble_performance.improvement > 0 and 'provided additional improvement' or 'showed similar performance' }}</li>
            <li>Most important features include domain-relevant variables that drive prediction accuracy</li>
        </ul>

        <h3>Limitations</h3>
        <p>This analysis uses automated methodologies that may miss domain-specific insights. Further improvements could be achieved through feature engineering and hyperparameter tuning.</p>

        <h3>Future Work</h3>
        <p>Future analyses could explore more sophisticated feature engineering, advanced ensemble techniques, and domain-specific preprocessing methods.</p>
    </div>
</body>
</html>
        """

        template = Template(html_template)
        return template.render(**article_data)

    async def _generate_markdown_article(self, article_data: Dict[str, Any]) -> str:
        """
        Generate Markdown article
        """
        markdown_template = """# {{ title }}

**Generated:** {{ date }}
**Task Type:** {{ config.task_type|title }}
**Evaluation Metric:** {{ config.evaluation_metric|upper }}
**Target Column:** {{ config.target_column }}

## Abstract

This report presents an automated analysis of a Kaggle competition dataset. Our analysis explores {{ dataset_info.train_shape[1] }} features across {{ dataset_info.train_shape[0] }} training samples, identifying key patterns and relationships. We trained and evaluated multiple machine learning models, achieving the best performance with {{ best_model.model_name|title }} ({{ best_model.cv_score }}). The ensemble approach further improved performance to {{ ensemble_performance.ensemble_best }}, demonstrating {{ ensemble_performance.improvement > 0 and 'an improvement of' or 'similar performance to' }} the best individual model.

## 1. Data Overview

### Dataset Summary

| Metric | Value |
|--------|-------|
| Training Samples | {{ "{:,}".format(dataset_info.train_shape[0]) }} |
| Features | {{ dataset_info.train_shape[1] }} |
| Test Samples | {{ "{:,}".format(dataset_info.test_shape[0]) }} |
| Data Quality Score | {{ quality_assessment.quality_score }}/100 |

### Target Variable Analysis

{% if target_analysis.type == "classification" %}
The target variable is {{ target_analysis.type }} with {{ target_analysis.unique_classes }} classes. The majority class represents {{ "%.1f"|format(target_analysis.class_balance.majority_ratio * 100) }}% of the samples.
{% else %}
The target variable is continuous with a mean of {{ "%.2f"|format(target_analysis.statistics.mean) }} (±{{ "%.2f"|format(target_analysis.statistics.std) }}) and ranges from {{ "%.2f"|format(target_analysis.statistics.min) }} to {{ "%.2f"|format(target_analysis.statistics.max) }}.
{% endif %}

### Key EDA Insights

{% for insight in eda_insights.insights %}
- {{ insight }}
{% endfor %}

## 2. Methodology

### Data Preprocessing
Our preprocessing pipeline included missing value imputation, categorical encoding, and feature scaling. We carefully handled different data types and ensured consistency between training and test datasets.

### Cross-Validation Strategy
We employed {{ config.cv_strategy|replace('_', ' ')|title }} with {{ config.cv_folds }} folds to ensure robust model evaluation and prevent overfitting.

### Model Selection
We trained and evaluated multiple models: Logistic Regression, Random Forest, LightGBM, XGBoost, and CatBoost, selecting the best performing model based on {{ config.evaluation_metric|upper }}.

## 3. Results

### Model Performance Comparison

| Model | CV Score | Training Time |
|-------|----------|---------------|
{% for model in model_comparison.models %}| {{ model.model_name|title }}{% if model.model_name == best_model.model_name %} ⭐{% endif %} | {{ model.cv_score }} | {{ model.training_time }} |
{% endfor %}

**Best Model Score:** {{ best_model.cv_score }}
**Ensemble Score:** {{ ensemble_performance.ensemble_best }}
{% if ensemble_performance.improvement > 0 %}
**Ensemble Improvement:** +{{ "%.4f"|format(ensemble_performance.improvement) }}
{% endif %}

### Feature Importance

Top 10 most important features:
{% for feature, importance in feature_importance.items()[:10] %}
1. **{{ feature }}**: {{ "%.4f"|format(importance) }}
{% endfor %}

## 4. Conclusions

Our automated analysis successfully identified key patterns in the dataset and built robust predictive models. The {{ best_model.model_name }} model achieved strong performance with a CV score of {{ best_model.cv_score }}, while the ensemble approach {{ ensemble_performance.improvement > 0 and 'further improved' or 'maintained' }} this performance at {{ ensemble_performance.ensemble_best }}.

### Key Findings

- Dataset contains {{ dataset_info.train_shape[1] }} features with {{ quality_assessment.quality_score }}/100 data quality
- {{ best_model.model_name|title }} emerged as the best individual model
- Ensemble methods {{ ensemble_performance.improvement > 0 and 'provided additional improvement' or 'showed similar performance' }}
- Most important features include domain-relevant variables that drive prediction accuracy

### Limitations

This analysis uses automated methodologies that may miss domain-specific insights. Further improvements could be achieved through feature engineering and hyperparameter tuning.

### Future Work

Future analyses could explore more sophisticated feature engineering, advanced ensemble techniques, and domain-specific preprocessing methods.

---
*This report was generated automatically using the Kaggle Automation platform.*
        """

        template = Template(markdown_template)
        return template.render(**article_data)