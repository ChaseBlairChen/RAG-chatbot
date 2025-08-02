// components/upload/UploadStatus.tsx
import React from 'react';
import { UploadStatus as UploadStatusType } from '../../types';
import { getStatusColor } from '../../utils/helpers';

interface UploadStatusProps {
  uploadStatuses: Map<string, UploadStatusType>;
}

export const UploadStatusComponent: React.FC<UploadStatusProps> = ({ uploadStatuses }) => {
  if (uploadStatuses.size === 0) return null;

  return (
    <div className="mt-6 space-y-3">
      <h4 className="font-medium text-gray-900">Processing Status</h4>
      {Array.from(uploadStatuses.values()).map(status => (
        <div key={status.fileId} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-gray-900">{status.filename}</span>
            <span className={`text-xs px-2 py-1 rounded-full ${getStatusColor(status.status)}`}>
              {status.status}
            </span>
          </div>
          <p className="text-sm text-gray-600 mb-2">{status.message}</p>
          <div className="bg-gray-200 rounded-full h-2">
            <div 
              className="bg-stone-700 h-2 rounded-full transition-all" 
              style={{ width: `${status.progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-xs text-gray-500 mt-1">
            <span>{status.progress}% complete</span>
            {status.pagesProcessed !== undefined && (
              <span>{status.pagesProcessed} pages • {status.chunksCreated} chunks</span>
            )}
          </div>
          {status.status === 'failed' && status.errors && status.errors.length > 0 && (
            <div className="mt-2 text-xs text-red-600">
              {status.errors.join(', ')}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};
