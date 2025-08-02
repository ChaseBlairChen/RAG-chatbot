// components/upload/UploadQueue.tsx
import React from 'react';

interface UploadQueueProps {
  uploadQueue: File[];
  currentlyUploading: File | null;
  uploadProgress: number;
  isAnalyzing: boolean;
  onRemoveFromQueue: (index: number) => void;
  onClearQueue: () => void;
  onUploadOnly: () => void;
  onUploadAndAnalyze: () => void;
}

export const UploadQueue: React.FC<UploadQueueProps> = ({
  uploadQueue,
  currentlyUploading,
  uploadProgress,
  isAnalyzing,
  onRemoveFromQueue,
  onClearQueue,
  onUploadOnly,
  onUploadAndAnalyze
}) => {
  if (uploadQueue.length === 0) return null;

  return (
    <div className="mt-6 p-4 bg-stone-100 rounded-lg border border-stone-200">
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-medium text-stone-900">Upload Queue ({uploadQueue.length} files)</h4>
        <div className="flex gap-2">
          <button
            onClick={onClearQueue}
            className="text-stone-700 hover:text-stone-900 text-sm font-medium"
          >
            Clear All
          </button>
          <button
            onClick={onUploadOnly}
            disabled={isAnalyzing}
            className="bg-stone-700 text-white px-4 py-2 rounded-lg hover:bg-stone-800 disabled:bg-gray-300 transition-all font-medium text-sm"
          >
            {isAnalyzing ? 'Uploading...' : 'Upload Only'}
          </button>
          <button
            onClick={onUploadAndAnalyze}
            disabled={isAnalyzing}
            className="bg-green-700 text-white px-4 py-2 rounded-lg hover:bg-green-800 disabled:bg-gray-300 transition-all font-medium text-sm"
          >
            {isAnalyzing ? 'Processing...' : 'Upload & Analyze All'}
          </button>
        </div>
      </div>
      
      {/* File List */}
      <div className="space-y-2 max-h-40 overflow-y-auto">
        {uploadQueue.map((file, index) => (
          <div key={`${file.name}-${index}`} className="flex items-center justify-between bg-white rounded-lg p-3 border">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-stone-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <div>
                <p className="font-medium text-gray-900 text-sm">{file.name}</p>
                <p className="text-xs text-gray-600">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              {currentlyUploading === file && (
                <span className="text-xs bg-stone-200 text-stone-700 px-2 py-1 rounded-full">
                  Uploading...
                </span>
              )}
            </div>
            <button
              onClick={() => onRemoveFromQueue(index)}
              disabled={isAnalyzing}
              className="text-red-600 hover:text-red-700 p-1 disabled:opacity-50"
              title="Remove from queue"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ))}
      </div>
      
      {/* Progress Bar */}
      {isAnalyzing && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-sm text-blue-700 mb-2">
            <span>Processing documents...</span>
            <span>{Math.round(uploadProgress)}%</span>
          </div>
          <div className="bg-blue-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          {currentlyUploading && (
            <p className="text-xs text-blue-600 mt-2">
              Currently uploading: {currentlyUploading.name}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
