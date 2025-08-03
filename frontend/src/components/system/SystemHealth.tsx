// components/system/SystemHealth.tsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const SystemHealth: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [healthData, setHealthData] = useState<any>(null);
  const [detailedHealth, setDetailedHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const apiService = new ApiService(backendUrl, apiToken);

  const loadHealthData = async () => {
    setLoading(true);
    try {
      const [basic, detailed] = await Promise.all([
        apiService.get('/health'),
        apiService.get('/health/detailed')
      ]);
      setHealthData(basic);
      setDetailedHealth(detailed);
    } catch (error) {
      console.error('Failed to load health data:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadHealthData();
    const interval = setInterval(loadHealthData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: boolean | string) => {
    if (typeof status === 'boolean') {
      return status ? 'text-green-600' : 'text-red-600';
    }
    return status === 'healthy' ? 'text-green-600' : 'text-red-600';
  };

  const getStatusIcon = (status: boolean | string) => {
    const isHealthy = typeof status === 'boolean' ? status : status === 'healthy';
    return isHealthy ? '✅' : '❌';
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">System Health</h2>
        <button
          onClick={loadHealthData}
          disabled={loading}
          className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 transition-all text-sm"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {healthData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Basic Health */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium text-gray-900 mb-3">🏥 Basic Health</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Status:</span>
                <span className={getStatusColor(healthData.status)}>
                  {getStatusIcon(healthData.status)} {healthData.status}
                </span>
              </div>
            </div>
          </div>

          {/* Features */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium text-gray-900 mb-3">🔧 Features</h3>
            <div className="space-y-2 text-sm">
              {healthData.features && Object.entries(healthData.features).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="capitalize">{key.replace('_', ' ')}:</span>
                  <span className={getStatusColor(value as boolean)}>
                    {getStatusIcon(value as boolean)} {value ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {detailedHealth && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Components */}
          {detailedHealth.components && (
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-medium text-blue-900 mb-3">🏗️ Components</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Container Manager:</span>
                  <span className={getStatusColor(detailedHealth.components.container_manager === 'healthy')}>
                    {getStatusIcon(detailedHealth.components.container_manager === 'healthy')} 
                    {detailedHealth.components.container_manager}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Uploaded Files:</span>
                  <span className="text-gray-600">{detailedHealth.components.uploaded_files_count}</span>
                </div>
                <div className="flex justify-between">
                  <span>Active Sessions:</span>
                  <span className="text-gray-600">{detailedHealth.components.active_sessions}</span>
                </div>
              </div>
            </div>
          )}

          {/* Document Processors */}
          {detailedHealth.features?.document_processors && (
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-medium text-green-900 mb-3">📄 Document Processors</h3>
              <div className="space-y-2 text-sm">
                {Object.entries(detailedHealth.features.document_processors).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="capitalize">{key}:</span>
                    <span className={getStatusColor(value as boolean)}>
                      {getStatusIcon(value as boolean)} {value ? 'Available' : 'Not Available'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {loading && !healthData && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading system health...</p>
        </div>
      )}
    </div>
  );
};
