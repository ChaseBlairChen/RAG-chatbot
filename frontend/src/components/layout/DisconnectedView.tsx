// components/layout/DisconnectedView.tsx
import React from 'react';
import { useBackend } from '../../contexts/BackendContext';

export const DisconnectedView: React.FC = () => {
  const { connectionError, testConnection } = useBackend();

  return (
    <div className="min-h-[calc(100vh-250px)] flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-12 text-center max-w-2xl">
        <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <h3 className="text-2xl font-semibold text-gray-900 mb-3">Connecting to Legally Backend</h3>
        <p className="text-gray-600 mb-6">
          Please wait while we connect to the Legally backend server...
        </p>
        {connectionError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800 text-sm">{connectionError}</p>
          </div>
        )}
        <div className="flex gap-3 justify-center">
          <button
            onClick={() => testConnection()}
            className="bg-stone-800 text-white px-6 py-3 rounded-lg hover:bg-stone-900 transition-all font-medium"
          >
            Retry Connection
          </button>
        </div>
      </div>
    </div>
  );
};
