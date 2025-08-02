// components/analysis/DocumentSelector.tsx
import React from 'react';
import { DocumentAnalysis } from '../../types';

interface DocumentSelectorProps {
  selectedDocument: string | null;
  setSelectedDocument: (value: string | null) => void;
  documentAnalyses: DocumentAnalysis[];
  userDocumentsCount: number;
}

export const DocumentSelector: React.FC<DocumentSelectorProps> = ({
  selectedDocument,
  setSelectedDocument,
  documentAnalyses,
  userDocumentsCount
}) => {
  return (
    <div className="mb-6 p-4 bg-stone-100 rounded-lg border border-stone-200">
      <h4 className="font-medium text-stone-900 mb-3">Select Target Documents:</h4>
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2">
          <input
            type="radio"
            name="analysisScope"
            value="all"
            checked={selectedDocument === null}
            onChange={() => setSelectedDocument(null)}
            className="w-4 h-4"
          />
          <span className="text-sm text-stone-800">All Documents ({userDocumentsCount})</span>
        </label>
        
        <select
          value={selectedDocument || ''}
          onChange={(e) => setSelectedDocument(e.target.value || null)}
          className="bg-white border border-stone-300 rounded px-3 py-1 text-sm"
        >
          <option value="">Select Specific Document...</option>
          {documentAnalyses.map(doc => (
            <option key={doc.id} value={doc.id}>{doc.filename}</option>
          ))}
        </select>
      </div>
    </div>
  );
};
