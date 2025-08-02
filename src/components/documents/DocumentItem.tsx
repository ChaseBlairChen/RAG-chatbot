// components/documents/DocumentItem.tsx
import React from 'react';
import { DocumentAnalysis } from '../../types';

interface DocumentItemProps {
  document: DocumentAnalysis;
  isAnalyzing: boolean;
  onAnalyze: (docId: string) => void;
  onDelete: (docId: string) => void;
}

export const DocumentItem: React.FC<DocumentItemProps> = ({ 
  document, 
  isAnalyzing, 
  onAnalyze, 
  onDelete 
}) => {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-stone-100 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-stone-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h4 className="font-medium text-gray-900">{document.filename}</h4>
            <p className="text-sm text-gray-600">
              Uploaded {new Date(document.uploadedAt).toLocaleDateString()} • {document.pagesProcessed} pages
              {document.lastAnalyzed && (
                <span className="ml-2 text-green-600">
                  • Analyzed {new Date(document.lastAnalyzed).toLocaleDateString()}
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onAnalyze(document.id)}
            disabled={isAnalyzing}
            className="bg-green-600 text-white px-3 py-2 rounded-lg hover:bg-green-700 disabled:bg-gray-300 transition-all font-medium text-sm"
            title="Run comprehensive analysis"
          >
            {isAnalyzing ? 'Analyzing...' : '🔍 Analyze'}
          </button>
          
          <button
            onClick={() => onDelete(document.id)}
            className="text-red-600 hover:text-red-700 p-2 hover:bg-red-50 rounded-lg transition-all"
            title="Delete document"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>
      
      {document.confidence && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>Last Analysis Confidence:</span>
            <div className="flex items-center gap-2">
              <div className="w-16 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full transition-all" 
                  style={{ width: `${(document.confidence * 100)}%` }}
                />
              </div>
              <span className="text-xs font-medium">{Math.round(document.confidence * 100)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
