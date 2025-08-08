// components/documents/DocumentsTab.tsx
import React from 'react';
import { DocumentItem } from './DocumentItem';
import { EmptyState } from '../common/EmptyState';
import type { DocumentAnalysis } from '../../types';

interface DocumentsTabProps {
  documentAnalyses: DocumentAnalysis[];
  userDocuments: any[];
  isAnalyzing: boolean;
  onAnalyze: (documentId: string) => void;
  onDelete: (documentId: string) => void;
  onSetActiveTab: (tab: string) => void;
}

export const DocumentsTab: React.FC<DocumentsTabProps> = ({
  documentAnalyses,
  userDocuments,
  isAnalyzing,
  onAnalyze,
  onDelete,
  onSetActiveTab
}) => {
  if (userDocuments.length === 0) {
    return (
      <EmptyState
        icon=""
        title="No Documents Yet"
        description="Upload your first legal document to get started with AI-powered analysis."
        actionText="Upload Documents"
        onAction={() => onSetActiveTab('upload')}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">My Documents</h1>
          <p className="text-gray-600 mt-1">
            {userDocuments.length} document{userDocuments.length !== 1 ? 's' : ''} uploaded
          </p>
        </div>
        
        <button
          onClick={() => onSetActiveTab('upload')}
          className="bg-blue-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-blue-700 transition-all"
        >
          Upload More
        </button>
      </div>

      {/* Documents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {userDocuments.map((document) => {
          const analysis = documentAnalyses.find(d => d.id === document.file_id);
          
          return (
            <DocumentItem
              key={document.file_id}
              document={document}
              analysis={analysis}
              isAnalyzing={isAnalyzing}
              onAnalyze={() => onAnalyze(document.file_id)}
              onDelete={() => onDelete(document.file_id)}
            />
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-50 rounded-2xl p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => onSetActiveTab('analysis')}
            className="bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 transition-all"
          >
            🔍 Run Analysis
          </button>
          <button
            onClick={() => onSetActiveTab('results')}
            className="bg-purple-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-purple-700 transition-all"
          >
            📊 View Results
          </button>
          <button
            onClick={() => onSetActiveTab('upload')}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-all"
          >
            📤 Upload More
          </button>
        </div>
      </div>
    </div>
  );
};