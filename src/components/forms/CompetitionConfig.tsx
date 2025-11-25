'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button';

interface ConfigData {
  target_column: string;
  task_type: string;
  evaluation_metric: string;
  cv_strategy: string;
  cv_folds: number;
  submission_format: {
    id_column: string;
    prediction_column: string;
  };
  competition_rules?: any;
  feature_engineering: boolean;
  hyperparameter_optimization: boolean;
}

interface CompetitionConfigProps {
  onConfigSubmitted: (config: ConfigData) => void;
  availableColumns: string[];
  disabled?: boolean;
}

export const CompetitionConfig: React.FC<CompetitionConfigProps> = ({
  onConfigSubmitted,
  availableColumns,
  disabled = false,
}) => {
  const [config, setConfig] = useState<ConfigData>({
    target_column: '',
    task_type: 'binary_classification',
    evaluation_metric: 'accuracy',
    cv_strategy: 'stratified_kfold',
    cv_folds: 5,
    submission_format: {
      id_column: 'id',
      prediction_column: 'target',
    },
    feature_engineering: true,
    hyperparameter_optimization: true,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const taskTypes = [
    { value: 'binary_classification', label: 'Binary Classification' },
    { value: 'multiclass_classification', label: 'Multiclass Classification' },
    { value: 'regression', label: 'Regression' },
  ];

  const cvStrategies = [
    { value: 'stratified_kfold', label: 'Stratified K-Fold' },
    { value: 'kfold', label: 'K-Fold' },
    { value: 'time_series_split', label: 'Time Series Split' },
    { value: 'group_kfold', label: 'Group K-Fold' },
  ];

  const metricsByTaskType = {
    binary_classification: [
      { value: 'accuracy', label: 'Accuracy' },
      { value: 'f1', label: 'F1 Score' },
      { value: 'auc', label: 'AUC' },
      { value: 'log_loss', label: 'Log Loss' },
    ],
    multiclass_classification: [
      { value: 'accuracy', label: 'Accuracy' },
      { value: 'f1', label: 'F1 Score (Macro)' },
      { value: 'log_loss', label: 'Log Loss' },
    ],
    regression: [
      { value: 'rmse', label: 'Root Mean Squared Error' },
      { value: 'mae', label: 'Mean Absolute Error' },
      { value: 'r2', label: 'R² Score' },
    ],
  };

  const validateConfig = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!config.target_column) {
      newErrors.target_column = 'Target column is required';
    } else if (!availableColumns.includes(config.target_column)) {
      newErrors.target_column = 'Target column must exist in training data';
    }

    if (config.cv_folds < 3 || config.cv_folds > 10) {
      newErrors.cv_folds = 'CV folds must be between 3 and 10';
    }

    if (!config.submission_format.id_column) {
      newErrors.id_column = 'ID column is required';
    }

    if (!config.submission_format.prediction_column) {
      newErrors.prediction_column = 'Prediction column is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (validateConfig()) {
      onConfigSubmitted(config);
    }
  };

  const handleInputChange = (field: string, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));

    // Clear error for this field
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }

    // Auto-update evaluation metric when task type changes
    if (field === 'task_type') {
      const defaultMetrics = metricsByTaskType[value as keyof typeof metricsByTaskType];
      if (defaultMetrics && defaultMetrics.length > 0) {
        setConfig(prev => ({
          ...prev,
          task_type: value,
          evaluation_metric: defaultMetrics[0].value,
        }));
      }
    }
  };

  const handleSubmissionFormatChange = (field: string, value: string) => {
    setConfig(prev => ({
      ...prev,
      submission_format: {
        ...prev.submission_format,
        [field]: value,
      },
    }));

    // Clear error for this field
    const errorField = field === 'id_column' ? 'id_column' : 'prediction_column';
    if (errors[errorField]) {
      setErrors(prev => ({ ...prev, [errorField]: '' }));
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Target Column */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Target Column *
        </label>
        <select
          value={config.target_column}
          onChange={(e) => handleInputChange('target_column', e.target.value)}
          disabled={disabled || availableColumns.length === 0}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.target_column ? 'border-red-300' : 'border-gray-300'
          } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
        >
          <option value="">Select target column</option>
          {availableColumns.map((column) => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        {errors.target_column && (
          <p className="mt-1 text-sm text-red-600">{errors.target_column}</p>
        )}
      </div>

      {/* Task Type */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Task Type *
        </label>
        <select
          value={config.task_type}
          onChange={(e) => handleInputChange('task_type', e.target.value)}
          disabled={disabled}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            disabled ? 'bg-gray-100 cursor-not-allowed' : 'border-gray-300'
          }`}
        >
          {taskTypes.map((type) => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>
      </div>

      {/* Evaluation Metric */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Evaluation Metric *
        </label>
        <select
          value={config.evaluation_metric}
          onChange={(e) => handleInputChange('evaluation_metric', e.target.value)}
          disabled={disabled}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            disabled ? 'bg-gray-100 cursor-not-allowed' : 'border-gray-300'
          }`}
        >
          {metricsByTaskType[config.task_type as keyof typeof metricsByTaskType]?.map((metric) => (
            <option key={metric.value} value={metric.value}>
              {metric.label}
            </option>
          ))}
        </select>
      </div>

      {/* Cross-Validation Strategy */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Cross-Validation Strategy
        </label>
        <select
          value={config.cv_strategy}
          onChange={(e) => handleInputChange('cv_strategy', e.target.value)}
          disabled={disabled}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            disabled ? 'bg-gray-100 cursor-not-allowed' : 'border-gray-300'
          }`}
        >
          {cvStrategies.map((strategy) => (
            <option key={strategy.value} value={strategy.value}>
              {strategy.label}
            </option>
          ))}
        </select>
      </div>

      {/* CV Folds */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          CV Folds *
        </label>
        <input
          type="number"
          min="3"
          max="10"
          value={config.cv_folds}
          onChange={(e) => handleInputChange('cv_folds', parseInt(e.target.value))}
          disabled={disabled}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.cv_folds ? 'border-red-300' : 'border-gray-300'
          } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
        />
        {errors.cv_folds && (
          <p className="mt-1 text-sm text-red-600">{errors.cv_folds}</p>
        )}
      </div>

      {/* Submission Format */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-3">Submission Format</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              ID Column *
            </label>
            <input
              type="text"
              value={config.submission_format.id_column}
              onChange={(e) => handleSubmissionFormatChange('id_column', e.target.value)}
              disabled={disabled}
              placeholder="e.g., id, passenger_id"
              className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.id_column ? 'border-red-300' : 'border-gray-300'
              } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
            />
            {errors.id_column && (
              <p className="mt-1 text-sm text-red-600">{errors.id_column}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prediction Column *
            </label>
            <input
              type="text"
              value={config.submission_format.prediction_column}
              onChange={(e) => handleSubmissionFormatChange('prediction_column', e.target.value)}
              disabled={disabled}
              placeholder="e.g., target, survived"
              className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.prediction_column ? 'border-red-300' : 'border-gray-300'
              } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
            />
            {errors.prediction_column && (
              <p className="mt-1 text-sm text-red-600">{errors.prediction_column}</p>
            )}
          </div>
        </div>
      </div>

      {/* Advanced Options */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-3">Advanced Options</h3>
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.feature_engineering}
              onChange={(e) => handleInputChange('feature_engineering', e.target.checked)}
              disabled={disabled}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Enable automatic feature engineering
            </span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.hyperparameter_optimization}
              onChange={(e) => handleInputChange('hyperparameter_optimization', e.target.checked)}
              disabled={disabled}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Enable hyperparameter optimization
            </span>
          </label>
        </div>
      </div>

      {/* Submit Button */}
      <div>
        <Button
          type="submit"
          disabled={disabled}
          className="w-full"
        >
          Save Configuration
        </Button>
      </div>
    </form>
  );
};