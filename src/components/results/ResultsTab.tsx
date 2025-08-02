// components/results/ResultsTab.tsx
import React from 'react';
import { AnalysisResult } from '../../types';
import { AnalysisResultComponent } from './AnalysisResult';
import { EmptyState } from '../common/EmptyState';

interface ResultsTabProps {
  analysisResults: AnalysisResult[];
  isAnalyzing: boolean;
  onRerunAnalysis: (documentId: string) => void;
  onDownloadResult: (resultId: number) => void;
  onClearResults: () => void;
  onSetActiveTab: (tab: string) => void;
}

export const ResultsTab: React.FC<ResultsTabProps> = ({
  analysisResults,
  isAnalyzing,
  onRerunAnalysis,
  onDownloadResult,
  onClearResults,
  onSetActiveTab
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Analysis Results</h2>
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span>{analysisResults.length} result{analysisResults.length !== 1 ? 's' : ''}</span>
          {analysisResults.length > 0 && (
            <button
              onClick={onClearResults}
              className="text-red-600 hover:text-red-700 font-medium"
            >
              Clear All Results
            </button>
          )}
        </div>
      </div>
      
      {analysisResults.length === 0 ? (
        <EmptyState
          icon={
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          title="No analysis results"
          description="Run analysis tools to see comprehensive results here"
          action={{
            label: "Go to Analysis Tools",
            onClick: () => onSetActiveTab('analysis')
          }}
        />
      ) : (
        <div className="space-y-6">
          {analysisResults.map((result) => (
            <AnalysisResultComponent
              key={result.id}
              result={result}
              isAnalyzing={isAnalyzing}
              onRerun={onRerunAnalysis}
              onDownload={onDownloadResult}
            />
          ))}
        </div>
      )}
    </div>
  );
};
