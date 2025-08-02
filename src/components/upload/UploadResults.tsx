// components/upload/UploadResults.tsx
import React from 'react';

interface UploadResult {
  filename: string;
  success: boolean;
  pages_processed?: number;
  processing_time?: number;
  status?: string;
  error?: string;
  warnings?: string[];
}

interface UploadResultsProps {
  uploadResults: UploadResult[];
}

export const UploadResults: React.FC<UploadResultsProps> = ({ uploadResults }) => {
  if (uploadResults.length === 0) return null;

  return (
    <div className="mt-6 p-4 bg-gray-50 rounded-lg">
      <h4 className="font-medium text-gray-900 mb-3">Upload Results</h4>
      <div className="space-y-2">
        {uploadResults.map((result, index) => (
          <div key={index} className={`p-3 rounded-lg border ${
            result.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center gap-2">
              {result.success ? (
                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
              <div className="flex-1">
                <p className={`font-medium text-sm ${result.success ? 'text-green-800' : 'text-red-800'}`}>
                  {result.filename}
                </p>
                {result.success ? (
                  <p className="text-xs text-green-600">
                    ✅ {result.pages_processed} pages processed in {result.processing_time?.toFixed(2)}s
                    {result.status === 'processing' && ' (Processing in background...)'}
                  </p>
                ) : (
                  <p className="text-xs text-red-600">
                    ❌ {result.error}
                  </p>
                )}
              </div>
            </div>
            {result.warnings && result.warnings.length > 0 && (
              <div className="mt-2 text-xs text-yellow-700">
                <strong>Warnings:</strong> {result.warnings.join(', ')}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
