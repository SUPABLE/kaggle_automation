// API client for Kaggle Automation backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface FileUploadResponse {
  success: boolean;
  message: string;
  file_info?: {
    session_id: string;
    train_file: {
      name: string;
      size: number;
      rows: number;
      columns: number;
    };
    test_file: {
      name: string;
      size: number;
      rows: number;
      columns: number;
    };
    warnings?: string[];
  };
  train_columns?: string[];
  test_columns?: string[];
}

export interface CompetitionConfig {
  target_column: string;
  task_type: 'binary_classification' | 'multiclass_classification' | 'regression';
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

export interface AnalysisStartResponse {
  success: boolean;
  task_id: string;
  message: string;
}

export interface AnalysisStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_stage?: string;
  progress_percentage: number;
  message?: string;
  error?: string;
  results?: any;
}

export interface ReportResponse {
  task_id: string;
  report_content: string;
  report_format: string;
  generated_at: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error occurred');
    }
  }

  // File upload endpoint
  async uploadFiles(
    trainFile: File,
    testFile: File,
    sessionId?: string
  ): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('train_file', trainFile);
    formData.append('test_file', testFile);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const response = await fetch(`${this.baseUrl}/api/upload/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Upload failed');
    }

    return await response.json();
  }

  // Start analysis
  async startAnalysis(
    config: CompetitionConfig,
    fileSessionId: string
  ): Promise<AnalysisStartResponse> {
    return this.request<AnalysisStartResponse>('/api/analyze/start', {
      method: 'POST',
      body: JSON.stringify({
        config,
        file_session_id: fileSessionId,
      }),
    });
  }

  // Get analysis status
  async getAnalysisStatus(taskId: string): Promise<AnalysisStatus> {
    return this.request<AnalysisStatus>(`/api/analyze/${taskId}/status`);
  }

  // Generate report
  async generateReport(
    taskId: string,
    format: 'html' | 'markdown' = 'html'
  ): Promise<ReportResponse> {
    return this.request<ReportResponse>(`/api/report/${taskId}`, {
      method: 'POST',
      body: JSON.stringify({
        task_id: taskId,
        format,
      }),
    });
  }

  // Get submission info
  async getSubmissionInfo(taskId: string): Promise<any> {
    return this.request<any>(`/api/submit/${taskId}/info`);
  }

  // Download submission file (returns blob for download)
  async downloadSubmission(taskId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/submit/${taskId}/download`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Download failed');
    }

    return response.blob();
  }

  // Download report file (returns blob for download)
  async downloadReport(taskId: string, format: 'html' | 'markdown' = 'html'): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/report/${taskId}/download?format=${format}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Download failed');
    }

    return response.blob();
  }

  // Health check
  async healthCheck(): Promise<{ status: string }> {
    return this.request<{ status: string }>('/health');
  }

  // Get session info
  async getSessionInfo(sessionId: string): Promise<any> {
    return this.request<any>(`/api/upload/sessions/${sessionId}/info`);
  }
}

// Create singleton instance
export const apiClient = new ApiClient();

// Export individual functions for convenience
export const uploadFiles = apiClient.uploadFiles.bind(apiClient);
export const startAnalysis = apiClient.startAnalysis.bind(apiClient);
export const getAnalysisStatus = apiClient.getAnalysisStatus.bind(apiClient);
export const generateReport = apiClient.generateReport.bind(apiClient);
export const getSubmissionInfo = apiClient.getSubmissionInfo.bind(apiClient);
export const downloadSubmission = apiClient.downloadSubmission.bind(apiClient);
export const downloadReport = apiClient.downloadReport.bind(apiClient);
export const healthCheck = apiClient.healthCheck.bind(apiClient);
export const getSessionInfo = apiClient.getSessionInfo.bind(apiClient);