// components/analysis/AnalysisToolCard.tsx
import React from 'react';
import { AnalysisTool } from '../../types';

interface AnalysisToolCardProps {
  tool: AnalysisTool;
  isAnalyzing: boolean;
  selectedDocument: string | null;
  onRunAnalysis: (toolId: string, documentId?: string) => void;
}

export const AnalysisToolCard: React.FC<AnalysisToolCardProps> = ({
  tool,
  isAnalyzing,
  selectedDocument,
  onRunAnalysis
}) => {
  return (
    <div className={`border border-gray-200 rounded-xl p-6 hover:shadow-sm transition-all ${
      tool.isComprehensive ? 'ring-2 ring-green-200 bg-green-50' : ''
    }`}>
      <div className="flex items-start gap-4 mb-4">
        <div className="text-2xl">{tool.icon}</div>
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 mb-1">
            {tool.title}
            {tool.isComprehensive && (
              <span className="ml-2 text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
                RECOMMENDED
              </span>
            )}
          </h3>
          <p className="text-sm text-gray-600 mb-3">{tool.description}</p>
          
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
              {tool.category}
            </span>
            <span className={`text-xs px-2 py-1 rounded-full ${
              tool.riskLevel === 'low' ? 'bg-green-100 text-green-700' :
              tool.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}>
              {tool.riskLevel} risk
            </span>
          </div>
          
          <div className="text-xs text-gray-500 mb-4">
            <strong>Ideal for:</strong> {tool.idealFor.join(', ')}
          </div>
        </div>
      </div>
      
      <button
        onClick={() => onRunAnalysis(tool.id, selectedDocument || undefined)}
        disabled={isAnalyzing}
        className={`w-full py-2 px-4 rounded-lg transition-all font-medium text-sm ${
          tool.isComprehensive 
            ? 'bg-green-600 text-white hover:bg-green-700 disabled:bg-gray-300'
            : 'bg-stone-800 text-white hover:bg-stone-900 disabled:bg-gray-300'
        } disabled:cursor-not-allowed`}
      >
        {isAnalyzing ? 'Running Analysis...' : 
         tool.isComprehensive ? 'Run Complete Analysis' : 'Run Analysis'}
      </button>
      
      {tool.isComprehensive && (
        <p className="text-xs text-green-700 mt-2 text-center">
          🚀 Runs all analyses at once - most efficient option!
        </p>
      )}
    </div>
  );
};
