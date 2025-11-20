'use client';

import { useState } from 'react';
import { FileUpload } from '@/components/ui/FileUpload';
import { CompetitionConfig } from '@/components/forms/CompetitionConfig';
import { Button } from '@/components/ui/Button';
import { Alert } from '@/components/ui/Alert';

interface FileData {
  train_file: File | null;
  test_file: File | null;
  session_id: string;
  file_info: any;
}

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

export default function Home() {
  const [fileData, setFileData] = useState<FileData | null>(null);
  const [configData, setConfigData] = useState<ConfigData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFilesUploaded = (data: FileData) => {
    setFileData(data);
    setError(null);
    setSuccess('Files uploaded successfully!');
  };

  const handleConfigSubmitted = (config: ConfigData) => {
    setConfigData(config);
    setError(null);
  };

  const handleStartAnalysis = async () => {
    if (!fileData || !configData) {
      setError('Please upload files and configure competition settings first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Start analysis
      const response = await fetch('/api/analyze/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config: configData,
          file_session_id: fileData.session_id,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start analysis');
      }

      const result = await response.json();

      if (result.success) {
        setSuccess('Analysis started! Redirecting to analysis dashboard...');
        setTimeout(() => {
          window.location.href = `/analysis?task_id=${result.task_id}`;
        }, 2000);
      } else {
        throw new Error(result.message || 'Failed to start analysis');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const isReadyToStart = fileData && configData;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Kaggle Automation</h1>
              <p className="text-gray-600 mt-1">Automated ML workflow for competition success</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Alert Messages */}
        {error && (
          <Alert type="error" message={error} className="mb-6" />
        )}
        {success && (
          <Alert type="success" message={success} className="mb-6" />
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - File Upload */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Dataset Upload</h2>
            <FileUpload onFilesUploaded={handleFilesUploaded} />

            {fileData?.file_info && (
              <div className="mt-6 p-4 bg-green-50 rounded-md">
                <h3 className="font-medium text-green-800 mb-2">Files Uploaded Successfully</h3>
                <div className="text-sm text-green-700">
                  <p><strong>Training:</strong> {fileData.file_info.train_file.name} ({fileData.file_info.train_file.rows.toLocaleString()} rows, {fileData.file_info.train_file.columns} columns)</p>
                  <p><strong>Test:</strong> {fileData.file_info.test_file.name} ({fileData.file_info.test_file.rows.toLocaleString()} rows, {fileData.file_info.test_file.columns} columns)</p>
                </div>
                {fileData.file_info.warnings && fileData.file_info.warnings.length > 0 && (
                  <div className="mt-2 text-yellow-700 text-sm">
                    <p className="font-medium">Warnings:</p>
                    <ul className="list-disc list-inside">
                      {fileData.file_info.warnings.map((warning: string, index: number) => (
                        <li key={index}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Panel - Competition Configuration */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Competition Configuration</h2>
            <CompetitionConfig
              onConfigSubmitted={handleConfigSubmitted}
              availableColumns={fileData?.file_info?.train_columns || []}
              disabled={!fileData}
            />
          </div>
        </div>

        {/* Bottom Panel - Start Analysis */}
        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">Ready to Start Analysis</h2>
              <p className="text-gray-600 mt-1">
                {isReadyToStart
                  ? 'All set! Click below to start the automated ML pipeline.'
                  : 'Please upload your dataset files and configure competition settings above.'
                }
              </p>
            </div>
            <Button
              onClick={handleStartAnalysis}
              disabled={!isReadyToStart || isLoading}
              loading={isLoading}
              className="px-8 py-3"
            >
              {isLoading ? 'Starting Analysis...' : 'Start Analysis'}
            </Button>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-blue-900 mb-3">How it works</h2>
          <ol className="list-decimal list-inside space-y-2 text-blue-800">
            <li><strong>Upload Dataset:</strong> Upload your training and test CSV files</li>
            <li><strong>Configure Competition:</strong> Set target column, task type, and evaluation metrics</li>
            <li><strong>Start Analysis:</strong> Our automated pipeline will perform EDA, model training, and ensembling</li>
            <li><strong>Review Results:</strong> Monitor progress and view detailed analysis dashboard</li>
            <li><strong>Download Submission:</strong> Get your competition-ready submission file and research report</li>
          </ol>
        </div>
      </main>
    </div>
  );
}