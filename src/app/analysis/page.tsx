'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Alert } from '@/components/ui/Alert';
import { Button } from '@/components/ui/Button';

interface AnalysisStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_stage?: string;
  progress_percentage: number;
  message?: string;
  error?: string;
  results?: any;
}

export default function AnalysisPage() {
  const searchParams = useSearchParams();
  const taskId = searchParams.get('task_id');

  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!taskId) {
      setError('No task ID provided');
      setIsLoading(false);
      return;
    }

    const fetchStatus = async () => {
      try {
        const response = await fetch(`/api/analyze/${taskId}/status`);
        if (!response.ok) {
          throw new Error('Failed to fetch status');
        }
        const data = await response.json();
        setStatus(data);

        if (data.status === 'completed') {
          setIsLoading(false);
        } else if (data.status === 'failed') {
          setError(data.error || 'Analysis failed');
          setIsLoading(false);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch status');
        setIsLoading(false);
      }
    };

    fetchStatus();

    // Poll for updates every 2 seconds if still running
    const interval = setInterval(() => {
      if (isLoading) {
        fetchStatus();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId, isLoading]);

  const getStageDescription = (stage?: string): string => {
    if (!stage) return 'Initializing...';

    const stageDescriptions: Record<string, string> = {
      'initialization': 'Initializing analysis pipeline...',
      'eda': 'Running exploratory data analysis...',
      'preprocessing': 'Preprocessing data and engineering features...',
      'cross_validation': 'Setting up cross-validation strategy...',
      'modeling': 'Training and evaluating machine learning models...',
      'ensemble': 'Creating ensemble models for better performance...',
      'submission': 'Generating competition submission file...',
      'completed': 'Analysis completed successfully!',
    };

    return stageDescriptions[stage] || stage;
  };

  const getProgressColor = (progress: number): string => {
    if (progress < 25) return 'bg-red-500';
    if (progress < 50) return 'bg-yellow-500';
    if (progress < 75) return 'bg-blue-500';
    return 'bg-green-500';
  };

  if (error && !status) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow p-6">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
            <p className="text-gray-600 mb-6">{error}</p>
            <Button onClick={() => window.history.back()}>
              Go Back
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analysis Dashboard</h1>
              <p className="text-gray-600 mt-1">
                Task ID: {taskId}
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() => window.history.back()}
            >
              Back to Home
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && status?.status !== 'failed' && (
          <Alert type="error" message={error} className="mb-6" />
        )}

        {status?.status === 'failed' && (
          <Alert type="error" message={status.error || 'Analysis failed'} className="mb-6" />
        )}

        {status?.status === 'completed' && (
          <Alert
            type="success"
            message="Analysis completed successfully! You can now view your results and download submissions."
            className="mb-6"
          />
        )}

        {/* Progress Card */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">
              {status?.status === 'completed' ? 'Analysis Complete' : 'Analysis Progress'}
            </h2>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">
                Status: <span className="font-medium capitalize">{status?.status}</span>
              </span>
              {status && (
                <span className="text-sm font-medium text-gray-900">
                  {Math.round(status.progress_percentage)}%
                </span>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          {status && (
            <div className="mb-6">
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${getProgressColor(status.progress_percentage)}`}
                  style={{ width: `${status.progress_percentage}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Current Stage */}
          {status && status.status !== 'completed' && (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-lg font-medium text-gray-900">
                {getStageDescription(status.current_stage)}
              </p>
              {status.message && (
                <p className="text-gray-600 mt-2">{status.message}</p>
              )}
            </div>
          )}

          {status?.status === 'completed' && (
            <div className="text-center py-8">
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
                <svg
                  className="h-6 w-6 text-green-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
              <p className="text-lg font-medium text-gray-900 mb-4">
                Analysis completed successfully!
              </p>
              <div className="flex justify-center space-x-4">
                <Button onClick={() => window.location.href = `/article?task_id=${taskId}`}>
                  View Report
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => window.location.href = `/api/submit/${taskId}/download`}
                >
                  Download Submission
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Analysis Stages */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            { stage: 'eda', title: 'Exploratory Data Analysis', description: 'Analyze data distributions and patterns' },
            { stage: 'preprocessing', title: 'Data Preprocessing', description: 'Clean and prepare data for modeling' },
            { stage: 'modeling', title: 'Model Training', description: 'Train multiple machine learning models' },
            { stage: 'ensemble', title: 'Ensemble Creation', description: 'Combine models for better performance' },
            { stage: 'submission', title: 'Submission Generation', description: 'Create competition-ready submission file' },
            { stage: 'completed', title: 'Report Generation', description: 'Generate detailed analysis report' },
          ].map((item) => {
            const currentStage = status?.current_stage;
            const isCompleted = status?.status === 'completed';
            const isCurrentStage = currentStage === item.stage;
            const stageIndex = ['eda', 'preprocessing', 'modeling', 'ensemble', 'submission', 'completed'].indexOf(item.stage);
            const currentIndex = currentStage ? ['eda', 'preprocessing', 'modeling', 'ensemble', 'submission', 'completed'].indexOf(currentStage) : -1;

            const stageStatus = isCompleted || stageIndex < currentIndex ? 'completed' : isCurrentStage ? 'current' : 'pending';

            return (
              <div
                key={item.stage}
                className={`bg-white rounded-lg shadow p-6 border-2 ${
                  stageStatus === 'completed' ? 'border-green-200' :
                  stageStatus === 'current' ? 'border-blue-200' :
                  'border-gray-200'
                }`}
              >
                <div className="flex items-center mb-3">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      stageStatus === 'completed' ? 'bg-green-500' :
                      stageStatus === 'current' ? 'bg-blue-500' :
                      'bg-gray-300'
                    }`}
                  >
                    {stageStatus === 'completed' ? (
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : stageStatus === 'current' ? (
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                    ) : (
                      <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                    )}
                  </div>
                  <h3 className="ml-3 font-medium text-gray-900">{item.title}</h3>
                </div>
                <p className="text-sm text-gray-600">{item.description}</p>
              </div>
            );
          })}
        </div>
      </main>
    </div>
  );
}