'use client';

import React, { useState, useCallback } from 'react';

interface FileData {
  train_file: File | null;
  test_file: File | null;
  session_id: string;
  file_info: any;
}

interface FileUploadProps {
  onFilesUploaded: (data: FileData) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFilesUploaded }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<{
    train: File | null;
    test: File | null;
  }>({ train: null, test: null });

  const validateFile = (file: File): boolean => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Only CSV files are allowed');
      return false;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB
      setError('File size must be less than 100MB');
      return false;
    }

    return true;
  };

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length !== 2) {
      setError('Please upload exactly 2 files: train.csv and test.csv');
      return;
    }

    const fileArray = Array.from(files);
    const trainFile = fileArray.find(f => f.name.toLowerCase().includes('train')) || fileArray[0];
    const testFile = fileArray.find(f => f.name.toLowerCase().includes('test')) || fileArray[1];

    if (!validateFile(trainFile) || !validateFile(testFile)) {
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('train_file', trainFile);
      formData.append('test_file', testFile);

      const response = await fetch('/api/upload/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();

      if (result.success) {
        setUploadedFiles({ train: trainFile, test: testFile });
        onFilesUploaded({
          train_file: trainFile,
          test_file: testFile,
          session_id: result.file_info.session_id,
          file_info: result.file_info,
        });
      } else {
        throw new Error(result.message || 'Upload failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [onFilesUploaded]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length === 2) {
      handleFiles(e.dataTransfer.files);
    } else {
      setError('Please upload exactly 2 files');
    }
  }, [handleFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  }, [handleFiles]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        } ${uploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !uploading && document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".csv"
          onChange={handleFileInput}
          disabled={uploading}
          className="hidden"
        />

        {uploading ? (
          <div className="space-y-2">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="text-gray-600">Uploading and validating files...</p>
          </div>
        ) : (
          <div className="space-y-2">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              stroke="currentColor"
              fill="none"
              viewBox="0 0 48 48"
            >
              <path
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div>
              <p className="text-gray-600">
                Drag and drop your CSV files here, or click to select files
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Upload exactly 2 files: train.csv and test.csv (max 100MB each)
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {uploadedFiles.train && uploadedFiles.test && (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Uploaded Files:</h4>
          <div className="bg-gray-50 rounded-md p-4 space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium">Training:</span>
              <span>{uploadedFiles.train.name} ({formatFileSize(uploadedFiles.train.size)})</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium">Test:</span>
              <span>{uploadedFiles.test.name} ({formatFileSize(uploadedFiles.test.size)})</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};