import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any

async def create_distribution_plots(df: pd.DataFrame, target_column: str, task_type: str) -> str:
    """Create distribution plots for target variable"""

    if task_type in ["binary_classification", "multiclass_classification"]:
        # Classification - create count plot
        value_counts = df[target_column].value_counts()

        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                text=value_counts.values,
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(value_counts)]
            )
        ])

        fig.update_layout(
            title=f'Distribution of {target_column}',
            xaxis_title=target_column,
            yaxis_title='Count',
            showlegend=False
        )

    else:  # regression
        # Regression - create histogram and box plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Histogram', 'Box Plot'],
            vertical_spacing=0.1
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=df[target_column].dropna(),
                nbinsx=30,
                name='Distribution',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=df[target_column].dropna(),
                name='Box Plot',
                marker_color='#ff7f0e'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'Distribution of {target_column}',
            height=600
        )

    return fig.to_html(include_plotlyjs='cdn')

async def create_correlation_heatmap(correlation_matrix: Dict[str, Dict[str, float]]) -> str:
    """Create correlation heatmap"""

    # Convert dict back to DataFrame
    corr_df = pd.DataFrame(correlation_matrix)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 8},
        hoverongaps=False
    ))

    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        width=800,
        height=800
    )

    return fig.to_html(include_plotlyjs='cdn')

def create_feature_importance_plot(feature_importance: Dict[str, float], top_n: int = 20) -> str:
    """Create feature importance bar plot"""

    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_features:
        return "No feature importance data available"

    features, importances = zip(*sorted_features)

    fig = go.Figure(data=[
        go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker_color='#1f77b4'
        )
    ])

    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=max(400, len(features) * 25)
    )

    return fig.to_html(include_plotlyjs='cdn')

def create_model_comparison_plot(model_results: list) -> str:
    """Create model performance comparison plot"""

    if not model_results:
        return "No model results available"

    model_names = [result['model_name'] for result in model_results]
    cv_scores = [result['cv_score'] for result in model_results]
    cv_stds = [result['cv_std'] for result in model_results]

    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=cv_scores,
            error_y=dict(type='data', array=cv_stds, visible=True),
            marker_color='#1f77b4',
            text=[f'{score:.4f}' for score in cv_scores],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='CV Score',
        xaxis_tickangle=-45
    )

    return fig.to_html(include_plotlyjs='cdn')

def create_learning_curves_plot(train_scores: list, val_scores: list, model_name: str) -> str:
    """Create learning curves plot"""

    epochs = list(range(1, len(train_scores) + 1))

    fig = go.Figure()

    # Training scores
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_scores,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    ))

    # Validation scores
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_scores,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'Learning Curves - {model_name}',
        xaxis_title='Epoch',
        yaxis_title='Score',
        hovermode='x unified'
    )

    return fig.to_html(include_plotlyjs='cdn')

def create_confusion_matrix_plot(cm_matrix: list, class_labels: list) -> str:
    """Create confusion matrix heatmap"""

    fig = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        text=cm_matrix,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )

    return fig.to_html(include_plotlyjs='cdn')

def create_residual_plot(y_true: list, y_pred: list) -> str:
    """Create residual analysis plot for regression"""

    residuals = np.array(y_true) - np.array(y_pred)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Residuals vs Predicted', 'Q-Q Plot', 'Histogram of Residuals', 'Scale-Location'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Q-Q Plot (simplified)
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.sort(np.random.normal(0, np.std(residuals), len(residuals)))

    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Q-Q'),
        row=1, col=2
    )

    # Histogram of residuals
    fig.add_trace(
        go.Histogram(x=residuals, name='Residual Distribution'),
        row=2, col=1
    )

    # Scale-Location plot
    fig.add_trace(
        go.Scatter(x=y_pred, y=np.abs(residuals), mode='markers', name='Scale-Location'),
        row=2, col=2
    )

    fig.update_layout(
        title='Residual Analysis',
        height=600,
        showlegend=False
    )

    return fig.to_html(include_plotlyjs='cdn')