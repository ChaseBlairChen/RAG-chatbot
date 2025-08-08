// components/upload/UploadZone.tsx
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadZoneProps {
  onFileSelect: (files: FileList | null) => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onFileSelect }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Convert File[] to FileList-like object
    const dataTransfer = new DataTransfer();
    acceptedFiles.forEach(file => dataTransfer.items.add(file));
    onFileSelect(dataTransfer.files);
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/rtf': ['.rtf']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: true
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-200 cursor-pointer ${
        isDragActive && !isDragReject
          ? 'border-blue-400 bg-blue-50'
          : isDragReject
          ? 'border-red-400 bg-red-50'
          : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
      }`}
    >
      <input {...getInputProps()} />
      
      <div className="space-y-4">
        <div className="text-6xl">
          {isDragActive && !isDragReject ? '📁' : isDragReject ? '❌' : '📤'}
        </div>
        
        <div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            {isDragActive && !isDragReject
              ? 'Drop your documents here'
              : isDragReject
              ? 'Invalid file type'
              : 'Upload Legal Documents'
            }
          </h3>
          
          <p className="text-gray-600">
            {isDragActive && !isDragReject
              ? 'Release to upload'
              : isDragReject
              ? 'Please upload valid document files'
              : 'Drag & drop files here, or click to browse'
            }
          </p>
        </div>

        {!isDragActive && !isDragReject && (
          <div className="space-y-2">
            <p className="text-sm text-gray-500">
              Supported formats: PDF, DOC, DOCX, TXT, RTF
            </p>
            <p className="text-sm text-gray-500">
              Maximum file size: 50MB per file
            </p>
          </div>
        )}

        {!isDragActive && (
          <button className="bg-blue-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-blue-700 transition-colors">
            Choose Files
          </button>
        )}
      </div>
    </div>
  );
};