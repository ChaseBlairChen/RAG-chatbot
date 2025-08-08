// components/upload/UploadTab.tsx
import React from 'react';
import { UploadZone } from './UploadZone';
import { UploadQueue } from './UploadQueue';
import { UploadResults } from './UploadResults';
import { UploadStatusComponent } from './UploadStatus';

interface UploadTabProps {
  uploadQueue: File[];
  currentlyUploading: File | null;
  uploadProgress: number;
  uploadResults: any[];
  uploadStatuses: Map<string, any>;
  isAnalyzing: boolean;
  onFileSelect: (files: FileList | null) => void;
  onRemoveFromQueue: (index: number) => void;
  onClearQueue: () => void;
  onUploadAll: (runAnalysisAfter: boolean) => Promise<string[]>;
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
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-bold text-gray-900">Document Upload & Analysis</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload your legal documents for AI-powered analysis, risk assessment, and comprehensive legal insights.
        </p>
      </div>

      {/* Upload Zone */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
        <UploadZone onFileSelect={onFileSelect} />
      </div>

      {/* Upload Queue */}
      {uploadQueue.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Upload Queue</h2>
            <button
              onClick={onClearQueue}
              className="text-sm text-red-600 hover:text-red-700 font-medium transition-colors"
            >
              Clear All
            </button>
          </div>
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
          
          {/* Upload Actions */}
          <div className="mt-6 flex gap-4">
            <button
              onClick={() => onUploadAll(false)}
              disabled={currentlyUploading !== null}
              className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
            >
              Upload Documents
            </button>
            <button
              onClick={() => onUploadAll(true)}
              disabled={currentlyUploading !== null}
              className="flex-1 bg-green-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
            >
              Upload & Analyze
            </button>
          </div>
        </div>
      )}

      {/* Upload Progress */}
      {currentlyUploading && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <UploadStatusComponent
            uploadStatuses={uploadStatuses}
          />
        </div>
      )}

      {/* Upload Results */}
      {uploadResults.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Upload Results</h2>
            <button
              onClick={() => onSetActiveTab('documents')}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors"
            >
              View All Documents →
            </button>
          </div>
          <UploadResults uploadResults={uploadResults} />
        </div>
      )}

      {/* Analysis Status */}
      {isAnalyzing && (
        <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
            <div>
              <h3 className="font-semibold text-blue-900">Analyzing Documents</h3>
              <p className="text-sm text-blue-700">Running comprehensive legal analysis...</p>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-6 border border-blue-200">
          <div className="text-3xl mb-3"></div>
          <h3 className="font-semibold text-blue-900 mb-2">Document Types</h3>
          <p className="text-sm text-blue-700">
            Contracts, agreements, legal briefs, court documents, and more
          </p>
        </div>
        
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-6 border border-green-200">
          <div className="text-3xl mb-3"></div>
          <h3 className="font-semibold text-green-900 mb-2">Analysis Features</h3>
          <p className="text-sm text-green-700">
            Risk assessment, clause extraction, timeline analysis, and missing clauses detection
          </p>
        </div>
        
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-6 border border-purple-200">
          <div className="text-3xl mb-3">⚡</div>
          <h3 className="font-semibold text-purple-900 mb-2">Fast Processing</h3>
          <p className="text-sm text-purple-700">
            AI-powered analysis with results in minutes, not hours
          </p>
        </div>
      </div>
    </div>
  );
};