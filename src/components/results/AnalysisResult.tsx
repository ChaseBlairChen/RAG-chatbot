// components/results/AnalysisResult.tsx
import React from 'react';
import { AnalysisResult as AnalysisResultType } from '../../types';
import { getStatusColor } from '../../utils/helpers';
import { renderMarkdown } from '../../utils/markdown';

interface AnalysisResultProps {
  result: AnalysisResultType;
  isAnalyzing: boolean;
  onRerun: (documentId: string) => void;
  onDownload: (resultId: number) => void;
}

export const AnalysisResultComponent: React.FC<AnalysisResultProps> = ({
  result,
  isAnalyzing,
  onRerun,
  onDownload
}) => {
  return (
    <div className="border border-gray-200 rounded-xl p-6">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="font-semibold text-gray-900">{result.toolTitle}</h3>
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${getStatusColor(result.status)}`}>
              {result.status}
            </span>
            {result.confidence && (
              <span className="text-xs bg-stone-100 text-stone-700 px-2 py-1 rounded-full">
                {Math.round(result.confidence * 100)}% confidence
              </span>
            )}
            {result.analysisType === 'comprehensive' && (
              <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
                COMPLETE ANALYSIS
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600">
            {result.timestamp} • {result.document}
            {result.documentId && result.documentId !== 'all' && (
              <span className="ml-2 text-stone-600">• Specific Document</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {result.documentId && result.documentId !== 'all' && (
            <button
              onClick={() => onRerun(result.documentId)}
              disabled={isAnalyzing}
              className="text-stone-700 hover:text-stone-900 p-2 hover:bg-stone-100 rounded-lg transition-all"
              title="Rerun analysis"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            <div>
              <p className="text-sm font-medium text-amber-800 mb-1">Warnings:</p>
              <ul className="text-sm text-amber-700 space-y-1">
                {result.warnings.map((warning, i) => (
                  <li key={i}>• {warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
      
      {result.sources && result.sources.length > 0 && (
        <div className="border-t border-gray-200 pt-4">
          <p className="text-sm font-medium text-gray-900 mb-2">Sources:</p>
          <div className="space-y-1">
            {result.sources.slice(0, 5).map((source: any, i: number) => (
              <p key={i} className="text-xs text-gray-600">
                • {source.file_name} {source.page && `(Page ${source.page})`}
                {source.relevance && ` - ${Math.round(source.relevance * 100)}% relevant`}
              </p>
            ))}
            {result.sources.length > 5 && (
              <p className="text-xs text-gray-500">
                +{result.sources.length - 5} more sources
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
            </button>
          )}
          
          <button
            onClick={() => onDownload(result.id)}
            className="text-gray-600 hover:text-gray-700 p-2 hover:bg-gray-50 rounded-lg transition-all"
            title="Download result"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </button>
        </div>
      </div>
      
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div 
          className="prose prose-sm max-w-none text-gray-800"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(result.analysis) }}
        />
      </div>
      
      {result.warnings && result.warnings.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
