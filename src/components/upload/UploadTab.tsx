// components/upload/UploadTab.tsx
import React from 'react';
import { UploadZone } from './UploadZone';
import { UploadQueue } from './UploadQueue';
import { UploadStatusComponent } from './UploadStatus';
import { UploadResults } from './UploadResults';
import { UploadStatus } from '../../types';

interface UploadTabProps {
  uploadQueue: File[];
  currentlyUploading: File | null;
  uploadProgress: number;
  uploadResults: any[];
  uploadStatuses: Map<string, UploadStatus>;
  isAnalyzing: boolean;
  onFileSelect: (files: FileList | null) => void;
  onRemoveFromQueue: (index: number) => void;
  onClearQueue: () => void;
  onUploadAll: (runAnalysis: boolean) => void;
  onSetActiveTab: (tab: string) => void;
}

export const UploadTab: React.FC<UploadTabProps> = ({
  uploadQueue,
  currentlyUploading,
  uploadProgress,
  uploadResults,
  uploadStatuses,
  isAnalyzing,
  onFileSelect,
  onRemoveFromQueue,
  onClearQueue,
  onUploadAll,
  onSetActiveTab
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Upload & Analyze Documents</h2>
        <div className="text-sm text-gray-500">
          {uploadQueue.length > 0 && `${uploadQueue.length} file${uploadQueue.length !== 1 ? 's' : ''} queued`}
        </div>
      </div>
      
      <div className="max-w-4xl">
        <UploadZone onFileSelect={onFileSelect} />

        <UploadQueue
          uploadQueue={uploadQueue}
          currentlyUploading={currentlyUploading}
          uploadProgress={uploadProgress}
          isAnalyzing={isAnalyzing}
          onRemoveFromQueue={onRemoveFromQueue}
          onClearQueue={onClearQueue}
          onUploadOnly={() => onUploadAll(false)}
          onUploadAndAnalyze={() => onUploadAll(true)}
        />

        <UploadStatusComponent uploadStatuses={uploadStatuses} />
        
        <UploadResults uploadResults={uploadResults} />
        
        {/* Instructions */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Supported File Types:</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• <strong>PDF:</strong> Portable Document Format</li>
              <li>• <strong>DOC/DOCX:</strong> Microsoft Word documents</li>
              <li>• <strong>TXT:</strong> Plain text files</li>
              <li>• <strong>RTF:</strong> Rich Text Format</li>
            </ul>
          </div>
          
          <div className="p-4 bg-green-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">🚀 New Features:</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• <strong>Real-time Progress:</strong> Track document processing status</li>
              <li>• <strong>Upload & Analyze:</strong> Automatically runs comprehensive analysis</li>
              <li>• <strong>Batch Upload:</strong> Process multiple documents at once</li>
              <li>• <strong>Background Processing:</strong> Large documents process asynchronously</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
