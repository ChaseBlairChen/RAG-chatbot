import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const AdminPanel: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [billSearchQuery, setBillSearchQuery] = useState('HB 1001');
  const [userId, setUserId] = useState('user_demo');

  const apiService = new ApiService(backendUrl, apiToken);

  const runAdminCommand = async (endpoint: string, data?: any) => {
    setLoading(true);
    try {
      let response;
      if (data) {
        const formData = new FormData();
        Object.keys(data).forEach(key => {
          formData.append(key, data[key]);
        });
        
        const fetchResponse = await fetch(`${backendUrl}/admin${endpoint}`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${apiToken}`,
          },
          body: formData,
        });
        response = await fetchResponse.json();
      } else {
        response = await apiService.post(`/admin${endpoint}`);
      }
      
      setResults({ endpoint, data: response, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ endpoint, error: error instanceof Error ? error.message : 'Unknown error', timestamp: new Date().toLocaleString() });
    }
    setLoading(false);
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center">
          <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <h2 className="text-2xl font-semibold text-gray-900">Admin Panel</h2>
        <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">ADMIN ONLY</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {/* Document Health */}
        <button
          onClick={() => runAdminCommand('/document-health')}
          disabled={loading}
          className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200 text-left transition-all"
        >
          <div className="font-medium text-blue-900">📊 Document Health</div>
          <div className="text-sm text-blue-700">Check document tracking status</div>
        </button>

        {/* Cleanup Containers */}
        <button
          onClick={() => runAdminCommand('/cleanup-containers')}
          disabled={loading}
          className="p-4 bg-yellow-50 hover:bg-yellow-100 rounded-lg border border-yellow-200 text-left transition-all"
        >
          <div className="font-medium text-yellow-900">🧹 Cleanup Containers</div>
          <div className="text-sm text-yellow-700">Clean orphaned document containers</div>
        </button>

        {/* Sync Document Tracking */}
        <button
          onClick={() => runAdminCommand('/sync-document-tracking')}
          disabled={loading}
          className="p-4 bg-green-50 hover:bg-green-100 rounded-lg border border-green-200 text-left transition-all"
        >
          <div className="font-medium text-green-900">🔄 Sync Tracking</div>
          <div className="text-sm text-green-700">Sync document tracking data</div>
        </button>

        {/* Emergency Clear */}
        <button
          onClick={() => {
            if (confirm('⚠️ This will clear ALL document tracking. Continue?')) {
              runAdminCommand('/emergency-clear-tracking');
            }
          }}
          disabled={loading}
          className="p-4 bg-red-50 hover:bg-red-100 rounded-lg border border-red-200 text-left transition-all"
        >
          <div className="font-medium text-red-900">🚨 Emergency Clear</div>
          <div className="text-sm text-red-700">Clear all document tracking</div>
        </button>
      </div>

      {/* Debug Functions */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium text-gray-900 mb-3">🔧 Debug Functions</h3>
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={billSearchQuery}
              onChange={(e) => setBillSearchQuery(e.target.value)}
              placeholder="Bill number (e.g., HB 1001)"
              className="flex-1 px-3 py-2 border border-gray-200 rounded text-sm"
            />
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="User ID"
              className="flex-1 px-3 py-2 border border-gray-200 rounded text-sm"
            />
            <button
              onClick={() => runAdminCommand('/debug/test-bill-search', {
                bill_number: billSearchQuery,
                user_id: userId
              })}
              disabled={loading}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition-all text-sm"
            >
              Test Bill Search
            </button>
          </div>
        </div>
      </div>

      {/* Results Display */}
      {results && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900">Results</h3>
            <span className="text-xs text-gray-500">{results.timestamp}</span>
          </div>
          <div className="text-sm">
            <div className="mb-2"><strong>Endpoint:</strong> {results.endpoint}</div>
            {results.error ? (
              <div className="text-red-600">
                <strong>Error:</strong> {results.error}
              </div>
            ) : (
              <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
                {JSON.stringify(results.data, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Running admin command...</p>
        </div>
      )}
    </div>
  );
};
