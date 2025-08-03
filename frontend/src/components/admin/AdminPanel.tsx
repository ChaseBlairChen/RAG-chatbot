// components/admin/AdminPanel.tsx
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

  // Enhanced result rendering function
  const renderAdminResult = () => {
    if (!results) return null;

    if (results.error) {
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-800">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <strong>Error:</strong> {results.error}
          </div>
        </div>
      );
    }

    const data = results.data;
    
    // Document Health Results
    if (results.endpoint === '/document-health') {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-900">{data.uploaded_files_count}</div>
              <div className="text-sm text-blue-700">Uploaded Files</div>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-900">{data.container_directories}</div>
              <div className="text-sm text-green-700">Container Directories</div>
            </div>
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-900">{data.users_with_containers}</div>
              <div className="text-sm text-purple-700">Users with Containers</div>
            </div>
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-orange-900">{data.orphaned_files?.length || 0}</div>
              <div className="text-sm text-orange-700">Orphaned Files</div>
            </div>
          </div>

          {data.recommendations && data.recommendations.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-medium text-yellow-900 mb-2">💡 Recommendations</h4>
              <ul className="space-y-1">
                {data.recommendations.map((rec: string, idx: number) => (
                  <li key={idx} className="text-sm text-yellow-800">• {rec}</li>
                ))}
              </ul>
            </div>
          )}

          {data.container_errors && data.container_errors.length > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="font-medium text-red-900 mb-2">❌ Container Errors</h4>
              <ul className="space-y-1">
                {data.container_errors.map((error: string, idx: number) => (
                  <li key={idx} className="text-sm text-red-800">• {error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    }

    // Cleanup Results
    if (results.endpoint === '/cleanup-containers') {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="text-xl font-bold text-blue-900">{data.containers_checked}</div>
              <div className="text-sm text-blue-700">Containers Checked</div>
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="text-xl font-bold text-yellow-900">{data.orphaned_documents_found}</div>
              <div className="text-sm text-yellow-700">Orphaned Documents</div>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-xl font-bold text-green-900">{data.cleanup_performed ? 'Yes' : 'No'}</div>
              <div className="text-sm text-green-700">Cleanup Performed</div>
            </div>
          </div>

          {data.errors && data.errors.length > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="font-medium text-red-900 mb-2">Errors During Cleanup</h4>
              <ul className="space-y-1">
                {data.errors.map((error: string, idx: number) => (
                  <li key={idx} className="text-sm text-red-800">• {error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    }

    // Bill Search Debug Results
    if (results.endpoint === '/debug/test-bill-search') {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="text-xl font-bold text-blue-900">{data.total_chunks}</div>
              <div className="text-sm text-blue-700">Total Chunks in Database</div>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-xl font-bold text-green-900">{data.chunks_with_bill_metadata?.length || 0}</div>
              <div className="text-sm text-green-700">Chunks with Bill Metadata</div>
            </div>
          </div>

          {data.chunks_with_bill_metadata && data.chunks_with_bill_metadata.length > 0 && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">📋 Found Bill Chunks</h4>
              <div className="space-y-2">
                {data.chunks_with_bill_metadata.map((chunk: any, idx: number) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded border">
                    <div className="font-medium text-sm">Chunk {chunk.chunk_index}</div>
                    <div className="text-xs text-gray-600 mb-2">Bills: {chunk.contains_bills}</div>
                    <div className="text-xs text-gray-700">{chunk.content_preview}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {data.text_search_preview && (
            <div className="bg-gray-50 border rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">🔍 Text Search Preview</h4>
              <div className="text-sm text-gray-700 bg-white p-3 rounded border">
                {data.text_search_preview}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Emergency Clear Results
    if (results.endpoint === '/emergency-clear-tracking') {
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <h4 className="font-medium text-red-900">Emergency Clear Results</h4>
          </div>
          <div className="text-sm text-red-800">
            <p><strong>Status:</strong> {data.status}</p>
            <p><strong>Files Cleared:</strong> {data.cleared_files}</p>
            <p><strong>Time:</strong> {data.timestamp}</p>
            {data.warning && <p className="mt-2 font-medium">⚠️ {data.warning}</p>}
          </div>
        </div>
      );
    }

    // Default JSON rendering for other types
    return (
      <div className="bg-white border rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-3">Raw Response</h4>
        <pre className="text-xs overflow-auto max-h-96 bg-gray-50 p-3 rounded">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    );
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
          className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200 text-left transition-all group"
        >
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <div className="font-medium text-blue-900">Document Health</div>
          </div>
          <div className="text-sm text-blue-700">Check document tracking status and container health</div>
        </button>

        {/* Cleanup Containers */}
        <button
          onClick={() => runAdminCommand('/cleanup-containers')}
          disabled={loading}
          className="p-4 bg-yellow-50 hover:bg-yellow-100 rounded-lg border border-yellow-200 text-left transition-all"
        >
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <div className="font-medium text-yellow-900">Cleanup Containers</div>
          </div>
          <div className="text-sm text-yellow-700">Clean orphaned document containers and files</div>
        </button>

        {/* Sync Document Tracking */}
        <button
          onClick={() => runAdminCommand('/sync-document-tracking')}
          disabled={loading}
          className="p-4 bg-green-50 hover:bg-green-100 rounded-lg border border-green-200 text-left transition-all"
        >
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <div className="font-medium text-green-900">Sync Tracking</div>
          </div>
          <div className="text-sm text-green-700">Synchronize document tracking data with containers</div>
        </button>

        {/* Emergency Clear */}
        <button
          onClick={() => {
            if (confirm('⚠️ This will clear ALL document tracking. This action cannot be undone. Continue?')) {
              runAdminCommand('/emergency-clear-tracking');
            }
          }}
          disabled={loading}
          className="p-4 bg-red-50 hover:bg-red-100 rounded-lg border border-red-200 text-left transition-all"
        >
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="font-medium text-red-900">Emergency Clear</div>
          </div>
          <div className="text-sm text-red-700">Clear all document tracking (DANGEROUS)</div>
        </button>
      </div>

      {/* Debug Functions */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          Debug Functions
        </h3>
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={billSearchQuery}
              onChange={(e) => setBillSearchQuery(e.target.value)}
              placeholder="Bill number (e.g., HB 1001)"
              className="flex-1 px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="User ID (e.g., user_demo)"
              className="flex-1 px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <button
              onClick={() => runAdminCommand('/debug/test-bill-search', {
                bill_number: billSearchQuery,
                user_id: userId
              })}
              disabled={loading}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition-all text-sm font-medium"
            >
              {loading ? 'Testing...' : 'Test Bill Search'}
            </button>
          </div>
          <p className="text-xs text-gray-600">
            Test if specific bills can be found in user documents. Useful for debugging search functionality.
          </p>
        </div>
      </div>

      {/* Results Display with Enhanced Formatting */}
      {results && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-900 flex items-center gap-2">
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Admin Command Results
            </h3>
            <div className="text-right">
              <div className="text-xs text-gray-500">{results.timestamp}</div>
              <div className="text-xs font-mono text-gray-600">{results.endpoint}</div>
            </div>
          </div>
          {renderAdminResult()}
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-red-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Running admin command...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few moments...</p>
        </div>
      )}
    </div>
  );
};
