// components/layout/BackendWarning.tsx
import React from 'react';
import { useBackend } from '../../contexts/BackendContext';

export const BackendWarning: React.FC = () => {
  const { isBackendConfigured, connectionError } = useBackend();

  if (isBackendConfigured) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-100 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center gap-3">
        <svg className="w-5 h-5 text-amber-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
        <div className="flex-1">
          <p className="text-sm text-amber-800">
            {connectionError || "Connecting to backend server..."}
          </p>
        </div>
      </div>
    </div>
  );
};