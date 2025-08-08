// components/results/ResultsTab.tsx
import React from 'react';
import { AnalysisResultComponent } from './AnalysisResult';
import { EmptyState } from '../common/EmptyState';

interface ResultsTabProps {
  analysisResults: any[];
  isAnalyzing: boolean;
  onRerunAnalysis: (documentId: string) => void;
  onDownloadResult: (id: number) => void;
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
  if (analysisResults.length === 0) {
    return (
      <EmptyState
        icon="📊"
        title="No Analysis Results"
        description="Run document analysis to see results here."
        action={{
          label: "Run Analysis",
          onClick: () => onSetActiveTab('analysis')
        }}
      />
    );
  }

  const completedResults = analysisResults.filter(r => r.status === 'completed');
  const processingResults = analysisResults.filter(r => r.status === 'processing');
  const failedResults = analysisResults.filter(r => r.status === 'failed');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analysis Results</h1>
          <p className="text-gray-600 mt-1">
            {completedResults.length} completed, {processingResults.length} processing, {failedResults.length} failed
          </p>
        </div>
        
        <div className="flex gap-3">
          <button
            onClick={() => onSetActiveTab('analysis')}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-all"
          >
            Run New Analysis
          </button>
          <button
            onClick={onClearResults}
            className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg font-medium hover:bg-gray-200 transition-all"
          >
            Clear All
          </button>
        </div>
      </div>

      {/* Processing Results */}
      {processingResults.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
          <h2 className="text-lg font-semibold text-blue-900 mb-4">Processing...</h2>
          <div className="space-y-3">
            {processingResults.map((result) => (
              <div key={result.id} className="flex items-center gap-3">
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                <span className="text-blue-800">{result.toolTitle} on {result.document}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Failed Results */}
      {failedResults.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-2xl p-6">
          <h2 className="text-lg font-semibold text-red-900 mb-4">Failed Analyses</h2>
          <div className="space-y-3">
            {failedResults.map((result) => (
              <div key={result.id} className="flex items-center justify-between">
                <span className="text-red-800">{result.toolTitle} on {result.document}</span>
                <button
                  onClick={() => onRerunAnalysis(result.documentId)}
                  className="text-sm text-red-600 hover:text-red-700 font-medium"
                >
                  Retry
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Completed Results */}
      {completedResults.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">Completed Analyses</h2>
          {completedResults.map((result) => (
            <AnalysisResultComponent
              key={result.id}
              result={result}
              isAnalyzing={isAnalyzing}
              onDownload={onDownloadResult}
              onRerun={onRerunAnalysis}
            />
          ))}
        </div>
      )}
    </div>
  );
};