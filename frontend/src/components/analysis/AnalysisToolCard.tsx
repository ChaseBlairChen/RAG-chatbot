// components/analysis/AnalysisToolCard.tsx
import React from 'react';
import type { AnalysisTool } from '../../types';

interface AnalysisToolCardProps {
  tool: AnalysisTool;
  isAnalyzing: boolean;
  onRunAnalysis: () => void;
}

export const AnalysisToolCard: React.FC<AnalysisToolCardProps> = ({
  tool,
  isAnalyzing,
  onRunAnalysis
}) => {
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-all">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="text-4xl">{tool.icon}</div>
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(tool.riskLevel)}`}>
          {tool.riskLevel} risk
        </div>
      </div>

      {/* Content */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-gray-900">{tool.title}</h3>
        <p className="text-sm text-gray-600">{tool.description}</p>

        {/* Ideal For */}
        <div>
          <p className="text-xs font-medium text-gray-700 mb-1">Ideal for:</p>
          <div className="flex flex-wrap gap-1">
            {tool.idealFor.map((item, index) => (
              <span
                key={index}
                className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full"
              >
                {item}
              </span>
            ))}
          </div>
        </div>

        {/* Category */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Category:</span>
          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
            {tool.category}
          </span>
        </div>
      </div>

      {/* Action Button */}
      <button
        onClick={onRunAnalysis}
        disabled={isAnalyzing}
        className={`w-full mt-4 px-4 py-3 rounded-xl font-medium transition-all ${
          tool.isComprehensive
            ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700'
            : 'bg-gray-900 text-white hover:bg-gray-800'
        } disabled:bg-gray-300 disabled:cursor-not-allowed`}
      >
        {isAnalyzing ? (
          <div className="flex items-center justify-center gap-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Analyzing...
          </div>
        ) : (
          `Run ${tool.title}`
        )}
      </button>
    </div>
  );
};