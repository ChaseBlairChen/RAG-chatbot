// components/analysis/AnalysisTab.tsx
import React from 'react';
import { DocumentSelector } from './DocumentSelector';
import { AnalysisToolCard } from './AnalysisToolCard';
import { ANALYSIS_TOOLS } from '../../utils/constants';
import { EmptyState } from '../common/EmptyState';

interface AnalysisTabProps {
  userDocuments: any[];
  documentAnalyses: any[];
  isAnalyzing: boolean;
  selectedDocument: string | null;
  setSelectedDocument: (documentId: string | null) => void;
  onRunAnalysis: (toolId: string, documentId?: string, useEnhancedRag?: boolean) => void;
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
  if (userDocuments.length === 0) {
    return (
      <EmptyState
        icon=""
        title="No Documents to Analyze"
        description="Upload documents first to use the analysis tools."
        actionText="Upload Documents"
        onAction={() => onSetActiveTab('upload')}
      />
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-bold text-gray-900">Document Analysis Tools</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Choose from our comprehensive suite of AI-powered legal analysis tools to extract insights from your documents.
        </p>
      </div>

      {/* Document Selector */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <DocumentSelector
          userDocuments={userDocuments}
          selectedDocument={selectedDocument}
          onSelectDocument={setSelectedDocument}
        />
      </div>

      {/* Analysis Tools Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {ANALYSIS_TOOLS.map((tool) => (
          <AnalysisToolCard
            key={tool.id}
            tool={tool}
            isAnalyzing={isAnalyzing}
            onRunAnalysis={() => onRunAnalysis(tool.id, selectedDocument || undefined)}
          />
        ))}
      </div>

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Analysis</h3>
        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => onRunAnalysis('comprehensive', selectedDocument || undefined)}
            disabled={isAnalyzing}
            className="bg-blue-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
          >
            🚀 Run Complete Analysis
          </button>
          <button
            onClick={() => onSetActiveTab('results')}
            className="bg-purple-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-purple-700 transition-all"
          >
            📊 View Previous Results
          </button>
        </div>
      </div>

      {/* Analysis Tips */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6">
        <h3 className="text-lg font-semibold text-yellow-900 mb-3">💡 Analysis Tips</h3>
        <ul className="space-y-2 text-sm text-yellow-800">
          <li>• <strong>Complete Analysis:</strong> Best for comprehensive document review</li>
          <li>• <strong>Risk Assessment:</strong> Identifies potential legal risks and issues</li>
          <li>• <strong>Clause Extraction:</strong> Finds key contractual terms automatically</li>
          <li>• <strong>Timeline Analysis:</strong> Extracts deadlines and important dates</li>
        </ul>
      </div>
    </div>
  );
};