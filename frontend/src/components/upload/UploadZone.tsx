// components/upload/UploadZone.tsx
import React from 'react';

interface UploadZoneProps {
  onFileSelect: (files: FileList | null) => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onFileSelect }) => {
  return (
    <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-gray-400 transition-all">
      <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      </div>
      
      <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Documents</h3>
      <p className="text-gray-600 mb-4">
        Support for PDF, DOC, DOCX, RTF, and TXT files (max 50MB each)
        <br />
        <strong>Select multiple files and choose to auto-analyze after upload</strong>
      </p>
      
      <input
        type="file"
        onChange={(e) => onFileSelect(e.target.files)}
        accept=".pdf,.doc,.docx,.txt,.rtf"
        className="hidden"
        id="file-upload"
        multiple
      />
      <label
        htmlFor="file-upload"
        className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-stone-800 hover:bg-stone-900 cursor-pointer transition-all"
      >
        Choose Files
      </label>
    </div>
  );
};
