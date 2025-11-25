'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Alert } from '@/components/ui/Alert';
import { Button } from '@/components/ui/Button';

interface ReportData {
  task_id: string;
  report_content: string;
  report_format: string;
  generated_at: string;
}

export default function ArticlePage() {
  const searchParams = useSearchParams();
  const taskId = searchParams.get('task_id');

  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!taskId) {
      setError('No task ID provided');
      setLoading(false);
      return;
    }

    const fetchReport = async () => {
      try {
        const response = await fetch(`/api/report/${taskId}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            task_id: taskId,
            format: 'html',
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch report');
        }

        const data = await response.json();
        setReport(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch report');
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [taskId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading report...</p>
        </div>
      </div>
    );
  }

  if (error) {
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
              <h1 className="text-3xl font-bold text-gray-900">Analysis Report</h1>
              <p className="text-gray-600 mt-1">
                Task ID: {taskId}
              </p>
            </div>
            <div className="flex space-x-4">
              <Button
                variant="secondary"
                onClick={() => window.history.back()}
              >
                Back to Analysis
              </Button>
              <Button
                onClick={() => window.location.href = `/api/submit/${taskId}/download`}
              >
                Download Submission
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Report Actions */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Report Actions</h2>
              <p className="text-gray-600 mt-1">
                Generated on {report ? new Date(report.generated_at).toLocaleString() : 'Unknown'}
              </p>
            </div>
            <div className="flex space-x-4">
              <Button
                variant="secondary"
                onClick={() => {
                  if (report) {
                    const blob = new Blob([report.report_content], { type: 'text/html' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `kaggle_report_${taskId}.html`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }
                }}
              >
                Download Report (HTML)
              </Button>
              <Button
                onClick={() => {
                  if (report) {
                    window.print();
                  }
                }}
              >
                Print Report
              </Button>
            </div>
          </div>
        </div>

        {/* Report Content */}
        {report && (
          <div className="bg-white rounded-lg shadow">
            <div
              dangerouslySetInnerHTML={{ __html: report.report_content }}
              className="prose max-w-none"
            />
          </div>
        )}

        {/* Additional Actions */}
        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Additional Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Model Performance</h3>
              <p className="text-gray-600 mb-4">
                View detailed performance metrics and comparisons for all trained models.
              </p>
              <Button variant="secondary" disabled>
                View Detailed Metrics (Coming Soon)
              </Button>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Feature Analysis</h3>
              <p className="text-gray-600 mb-4">
                Explore feature importance and detailed analysis of variable relationships.
              </p>
              <Button variant="secondary" disabled>
                View Feature Analysis (Coming Soon)
              </Button>
            </div>
          </div>
        </div>

        {/* Back to Home */}
        <div className="mt-8 text-center">
          <Button
            variant="secondary"
            onClick={() => window.location.href = '/'}
          >
            Start New Analysis
          </Button>
        </div>
      </main>
    </div>
  );
}