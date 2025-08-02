// components/analysis/AnalysisTab.tsx
import React from 'react';
import type { DocumentAnalysis } from '../../types';
import { ANALYSIS_TOOLS } from '../../utils/constants';
import { AnalysisToolCard } from './AnalysisToolCard';
import { DocumentSelector } from './DocumentSelector';
import { EmptyState } from '../common/EmptyState';

interface AnalysisTabProps {
  userDocuments: any[];
  documentAnalyses: DocumentAnalysis[];
  isAnalyzing: boolean;
  selectedDocument: string | null;
  setSelectedDocument: (value: string | null) => void;
  onRunAnalysis: (toolId: string, documentId?: string) => void;
  onSetActiveTab: (tab: string) => void;
}

export const AnalysisTab: React.FC<AnalysisTabProps> = ({
  userDocuments,
  documentAnalyses,
  isAnalyzing,
  selectedDocument,
  setSelectedDocument,
  onRunAnalysis,
  onSetActiveTab
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Analysis Tools</h2>
        <div className="text-sm text-gray-500">
          {userDocuments.length > 0 && `${userDocuments.length} document${userDocuments.length !== 1 ? 's' : ''} available`}
        </div>
      </div>
      
      {userDocuments.length === 0 ? (
        <EmptyState
          icon={
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          title="No documents to analyze"
          description="Upload documents first to run analysis tools"
          action={{
            label: "Upload Documents",
            onClick: () => onSetActiveTab('upload')
          }}
        />
      ) : (
        <>
          <DocumentSelector
            selectedDocument={selectedDocument}
            setSelectedDocument={setSelectedDocument}
            documentAnalyses={documentAnalyses}
            userDocumentsCount={userDocuments.length}
          />

          {/* Analysis Tools Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {ANALYSIS_TOOLS.map((tool) => (
              <AnalysisToolCard
                key={tool.id}
                tool={tool}
                isAnalyzing={isAnalyzing}
                selectedDocument={selectedDocument}
                onRunAnalysis={onRunAnalysis}
              />
            ))}
          </div>

          {/* Analysis Tips */}
          <div className="mt-8 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
            <h4 className="font-medium text-yellow-900 mb-2">💡 Analysis Tips:</h4>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• <strong>Complete Document Analysis</strong> is recommended - it runs all tools at once for comprehensive insights</li>
              <li>• Select "All Documents" to analyze your entire document collection</li>
              <li>• Choose specific documents for targeted analysis</li>
              <li>• Results include confidence scores and source citations</li>
              <li>• All analyses can be downloaded as text reports</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};