// components/documents/DocumentItem.tsx
import React from 'react';
import type { DocumentAnalysis } from '../../types';

interface DocumentItemProps {
  document: any;
  analysis?: DocumentAnalysis;
  isAnalyzing: boolean;
  onAnalyze: () => void;
  onDelete: () => void;
}

export const DocumentItem: React.FC<DocumentItemProps> = ({
  document,
  analysis,
  isAnalyzing,
  onAnalyze,
  onDelete
}) => {
  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf': return '📄';
      case 'doc': case 'docx': return '📝';
      case 'txt': return '📃';
      case 'rtf': return '📄';
      default: return '📁';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-yellow-100 text-yellow-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-all">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="text-3xl">{getFileIcon(document.filename)}</div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-gray-900 truncate" title={document.filename}>
              {document.filename}
            </h3>
            <p className="text-sm text-gray-500">
              {formatFileSize(document.file_size || 0)} • {document.pages_processed || 0} pages
            </p>
          </div>
        </div>
        
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(document.status)}`}>
          {document.status}
        </div>
      </div>

      {/* Document Info */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Uploaded:</span>
          <span className="text-gray-900">
            {new Date(document.uploaded_at).toLocaleDateString()}
          </span>
        </div>
        
        {document.processing_time && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Processing Time:</span>
            <span className="text-gray-900">{document.processing_time.toFixed(2)}s</span>
          </div>
        )}
      </div>

      {/* Analysis Status */}
      {analysis && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-blue-700 font-medium">Analysis Status</span>
            {analysis.confidence && (
              <span className="text-blue-600">
                {Math.round(analysis.confidence * 100)}% confidence
              </span>
            )}
          </div>
          {analysis.lastAnalyzed && (
            <p className="text-xs text-blue-600 mt-1">
              Last analyzed: {new Date(analysis.lastAnalyzed).toLocaleString()}
            </p>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={onAnalyze}
          disabled={isAnalyzing || document.status !== 'completed'}
          className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all text-sm"
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze'}
        </button>
        
        <button
          onClick={onDelete}
          className="px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-all"
          title="Delete document"
        >
          🗑️
        </button>
      </div>

      {/* Warnings */}
      {document.warnings && document.warnings.length > 0 && (
        <div className="mt-3 p-2 bg-yellow-50 rounded-lg border border-yellow-200">
          <p className="text-xs text-yellow-800">
            ⚠️ {document.warnings.length} warning{document.warnings.length !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </div>
  );
};