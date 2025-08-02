// components/documents/DocumentsTab.tsx
import React from 'react';
import type { DocumentAnalysis } from '../../types';
import { DocumentItem } from './DocumentItem';
import { EmptyState } from '../common/EmptyState';

interface DocumentsTabProps {
  documentAnalyses: DocumentAnalysis[];
  userDocuments: any[];
  isAnalyzing: boolean;
  onAnalyze: (docId: string) => void;
  onDelete: (docId: string) => void;
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
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">My Documents</h2>
        <div className="text-sm text-gray-500">
          {userDocuments.length} document{userDocuments.length !== 1 ? 's' : ''}
        </div>
      </div>
      
      {userDocuments.length === 0 ? (
        <EmptyState
          icon={
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          }
          title="No documents uploaded"
          description="Upload your first legal document to get started"
          action={{
            label: "Upload Document",
            onClick: () => onSetActiveTab('upload')
          }}
        />
      ) : (
        <div className="space-y-4">
          {documentAnalyses.map((doc) => (
            <DocumentItem
              key={doc.id}
              document={doc}
              isAnalyzing={isAnalyzing}
              onAnalyze={onAnalyze}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
};