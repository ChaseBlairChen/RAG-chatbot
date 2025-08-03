// ==================== ./src/components/admin/AdminPanel.tsx ====================
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

// ==================== ./src/components/immigration/ImmigrationTools.tsx ====================
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const ImmigrationTools: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [formData, setFormData] = useState({
    caseType: 'asylum',
    clientName: '',
    country: '',
    testimony: ''
  });

  const apiService = new ApiService(backendUrl, apiToken);

  const createCase = async () => {
    if (!formData.clientName) {
      alert('Please enter client name');
      return;
    }

    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('case_type', formData.caseType);
      postData.append('client_name', formData.clientName);
      postData.append('language', 'en');

      const response = await fetch(`${backendUrl}/immigration/cases/create`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
        },
        body: postData,
      });

      const data = await response.json();
      setResults({ type: 'case_creation', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'case_creation', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  const researchCountryConditions = async () => {
    if (!formData.country) {
      alert('Please enter country name');
      return;
    }

    setLoading(true);
    try {
      const requestData = {
        country: formData.country,
        topics: ['persecution', 'human_rights', 'government', 'violence'],
        date_range: 'last_2_years'
      };

      const data = await apiService.post('/immigration/country-conditions/research', requestData);
      setResults({ type: 'country_conditions', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'country_conditions', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  const analyzeTestimony = async () => {
    if (!formData.testimony || !formData.country) {
      alert('Please enter both testimony and country');
      return;
    }

    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('testimony', formData.testimony);
      postData.append('country', formData.country);

      const response = await fetch(`${backendUrl}/immigration/credible-fear/analyze`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
        },
        body: postData,
      });

      const data = await response.json();
      setResults({ type: 'testimony_analysis', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'testimony_analysis', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  const getUpcomingDeadlines = async () => {
    setLoading(true);
    try {
      const data = await apiService.get('/immigration/deadlines/upcoming?days_ahead=30');
      setResults({ type: 'deadlines', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'deadlines', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 className="text-2xl font-semibold text-gray-900">Immigration Law Tools</h2>
        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">SPECIALIZED</span>
      </div>

      {/* Case Management */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-medium text-blue-900 mb-3">📋 Case Management</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <select
            value={formData.caseType}
            onChange={(e) => setFormData(prev => ({...prev, caseType: e.target.value}))}
            className="px-3 py-2 border border-gray-200 rounded"
          >
            <option value="asylum">Asylum</option>
            <option value="family_based">Family Based</option>
            <option value="employment_based">Employment Based</option>
            <option value="removal_defense">Removal Defense</option>
            <option value="naturalization">Naturalization</option>
          </select>
          <input
            type="text"
            value={formData.clientName}
            onChange={(e) => setFormData(prev => ({...prev, clientName: e.target.value}))}
            placeholder="Client Name"
            className="px-3 py-2 border border-gray-200 rounded"
          />
        </div>
        <button
          onClick={createCase}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-all"
        >
          Create Immigration Case
        </button>
      </div>

      {/* Country Conditions Research */}
      <div className="mb-6 p-4 bg-green-50 rounded-lg">
        <h3 className="font-medium text-green-900 mb-3">🌍 Country Conditions Research</h3>
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={formData.country}
            onChange={(e) => setFormData(prev => ({...prev, country: e.target.value}))}
            placeholder="Country (e.g., Afghanistan, Myanmar)"
            className="flex-1 px-3 py-2 border border-gray-200 rounded"
          />
          <button
            onClick={researchCountryConditions}
            disabled={loading}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-all"
          >
            Research Conditions
          </button>
        </div>
      </div>

      {/* Credible Fear Analysis */}
      <div className="mb-6 p-4 bg-purple-50 rounded-lg">
        <h3 className="font-medium text-purple-900 mb-3">🗣️ Credible Fear Analysis</h3>
        <textarea
          value={formData.testimony}
          onChange={(e) => setFormData(prev => ({...prev, testimony: e.target.value}))}
          placeholder="Enter client testimony for analysis..."
          className="w-full px-3 py-2 border border-gray-200 rounded mb-4 h-24"
        />
        <button
          onClick={analyzeTestimony}
          disabled={loading || !formData.testimony || !formData.country}
          className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-all"
        >
          Analyze Testimony
        </button>
      </div>

      {/* Deadlines */}
      <div className="mb-6 p-4 bg-orange-50 rounded-lg">
        <h3 className="font-medium text-orange-900 mb-3">⏰ Deadline Management</h3>
        <button
          onClick={getUpcomingDeadlines}
          disabled={loading}
          className="bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700 transition-all"
        >
          Get Upcoming Deadlines
        </button>
      </div>

      {/* Results Display */}
      {results && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900">Results: {results.type}</h3>
            <span className="text-xs text-gray-500">{results.timestamp}</span>
          </div>
          {results.error ? (
            <div className="text-red-600 text-sm">
              <strong>Error:</strong> {results.error}
            </div>
          ) : (
            <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
              {JSON.stringify(results.data, null, 2)}
            </pre>
          )}
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Processing immigration request...</p>
        </div>
      )}
    </div>
  );
};

// ==================== ./src/components/system/SystemHealth.tsx ====================
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

// ==================== ./src/components/layout/TabNavigation.tsx ====================
import React from 'react';

interface Tab {
  id: string;
  label: string;
  icon: string;
  badge?: number | null;
  adminOnly?: boolean;
}

interface TabNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  userDocumentsCount: number;
  analysisResultsCount: number;
  isBackendConfigured: boolean;
}

export const TabNavigation: React.FC<TabNavigationProps> = ({
  activeTab,
  setActiveTab,
  userDocumentsCount,
  analysisResultsCount,
  isBackendConfigured
}) => {
  const tabs: Tab[] = [
    { id: 'chat', label: 'Smart Chat', icon: '💬' },
    { id: 'upload', label: 'Upload & Analyze', icon: '📤' },
    { id: 'documents', label: 'My Documents', icon: '📁', badge: userDocumentsCount > 0 ? userDocumentsCount : null },
    { id: 'analysis', label: 'Analysis Tools', icon: '🔍' },
    { id: 'results', label: 'Results', icon: '📊', badge: analysisResultsCount > 0 ? analysisResultsCount : null },
    { id: 'legal-search', label: 'Legal Search', icon: '⚖️' },
    { id: 'immigration', label: 'Immigration', icon: '🗽' },
    { id: 'system-health', label: 'System Health', icon: '🏥' },
    { id: 'admin', label: 'Admin', icon: '🔧', adminOnly: true },
  ];

  return (
    <nav className="bg-white border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex space-x-8 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              disabled={!isBackendConfigured}
              className={`relative py-4 px-1 text-sm font-medium transition-all border-b-2 whitespace-nowrap ${
                activeTab === tab.id
                  ? 'text-slate-900 border-slate-900'
                  : 'text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300'
              } ${!isBackendConfigured ? 'cursor-not-allowed opacity-50' : ''} ${
                tab.adminOnly ? 'text-red-600 hover:text-red-700' : ''
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-base">{tab.icon}</span>
                <span>{tab.label}</span>
                {tab.badge && (
                  <span className="ml-1 bg-slate-900 text-white text-xs font-medium px-2 py-0.5 rounded-full">
                    {tab.badge}
                  </span>
                )}
                {tab.adminOnly && (
                  <span className="ml-1 text-xs bg-red-100 text-red-700 px-1 py-0.5 rounded">
                    ADMIN
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
};

// ==================== ./src/App.tsx ====================
import React, { useState, useEffect, useMemo } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { BackendProvider, useBackend } from './contexts/BackendContext';
import { ApiService } from './services/api';
import { LoginScreen } from './components/auth/LoginScreen';
import { AppHeader } from './components/layout/AppHeader';
import { BackendWarning } from './components/layout/BackendWarning';
import { TabNavigation } from './components/layout/TabNavigation';
import { DisconnectedView } from './components/layout/DisconnectedView';
import { ChatTab } from './components/chat/ChatTab';
import { UploadTab } from './components/upload/UploadTab';
import { DocumentsTab } from './components/documents/DocumentsTab';
import { AnalysisTab } from './components/analysis/AnalysisTab';
import { ResultsTab } from './components/results/ResultsTab';
import { LegalDatabaseSearch } from './components/legal/LegalDatabaseSearch';
import { ImmigrationTools } from './components/immigration/ImmigrationTools';
import { SystemHealth } from './components/system/SystemHealth';
import { AdminPanel } from './components/admin/AdminPanel';
import { useChat } from './hooks/useChat';
import { useDocuments } from './hooks/useDocuments';
import { useAnalysis } from './hooks/useAnalysis';

function MainApp() {
  const { isLoggedIn, currentUser, apiToken } = useAuth();
  const { backendUrl, isBackendConfigured } = useBackend();
  const [activeTab, setActiveTab] = useState('chat');

  // Create API service instance
  const apiService = useMemo(() => {
    return new ApiService(backendUrl, apiToken);
  }, [backendUrl, apiToken]);

  // Use hooks
  const { messages, isLoading, sessionId, sendMessage, addWelcomeMessage } = useChat(apiService);
  const {
    userDocuments,
    documentAnalyses,
    uploadQueue,
    currentlyUploading,
    uploadProgress,
    uploadResults,
    uploadStatuses,
    setDocumentAnalyses,
    loadUserDocuments,
    handleFileUpload,
    uploadAllDocuments,
    deleteDocument,
    removeFromQueue,
    clearQueue,
    clearStatuses
  } = useDocuments(apiService, isBackendConfigured);
  
  const {
    analysisResults,
    isAnalyzing,
    selectedDocumentForAnalysis,
    setSelectedDocumentForAnalysis,
    runAnalysis,
    runComprehensiveDocumentAnalysis,
    downloadResult,
    clearResults
  } = useAnalysis(apiService, documentAnalyses, setDocumentAnalyses, sessionId, currentUser?.user_id);

  // Initialize welcome message
  useEffect(() => {
    if (isLoggedIn && currentUser && messages.length === 0) {
      addWelcomeMessage(currentUser.username);
    }
  }, [isLoggedIn, currentUser, messages.length, addWelcomeMessage]);

  // Load documents when backend is configured
  useEffect(() => {
    if (isBackendConfigured) {
      loadUserDocuments();
    }
  }, [isBackendConfigured, loadUserDocuments]);

  // Handle logout cleanup
  useEffect(() => {
    if (!isLoggedIn) {
      clearStatuses();
      clearResults();
    }
  }, [isLoggedIn, clearStatuses, clearResults]);

  // Handle upload and analysis
  const handleUploadAll = async (runAnalysisAfter: boolean) => {
    const uploadedDocIds = await uploadAllDocuments(runAnalysisAfter);
    
    if (runAnalysisAfter && uploadedDocIds.length > 0) {
      setActiveTab('results');
      
      // Wait for processing to complete before running analysis
      setTimeout(async () => {
        for (const docId of uploadedDocIds) {
          try {
            await runComprehensiveDocumentAnalysis(docId);
          } catch (analysisError) {
            console.error(`Analysis failed for document ${docId}:`, analysisError);
          }
        }
      }, 5000);
    } else if (uploadedDocIds.length > 0) {
      setActiveTab('documents');
    }
  };

  if (!isLoggedIn) {
    return <LoginScreen />;
  }

  return (
    <div className="flex flex-col h-screen bg-stone-50">
      <AppHeader sessionId={sessionId} />
      <BackendWarning />
      <TabNavigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        userDocumentsCount={userDocuments.length}
        analysisResultsCount={analysisResults.length}
        isBackendConfigured={isBackendConfigured}
      />

      <div className="flex-grow overflow-auto">
        <div className="w-full max-w-7xl mx-auto p-6">
          {!isBackendConfigured ? (
            <DisconnectedView />
          ) : (
            <>
              {activeTab === 'chat' && (
                <ChatTab
                  messages={messages}
                  isLoading={isLoading}
                  sessionId={sessionId}
                  sendMessage={sendMessage}
                />
              )}
              
              {activeTab === 'upload' && (
                <UploadTab
                  uploadQueue={uploadQueue}
                  currentlyUploading={currentlyUploading}
                  uploadProgress={uploadProgress}
                  uploadResults={uploadResults}
                  uploadStatuses={uploadStatuses}
                  isAnalyzing={isAnalyzing}
                  onFileSelect={handleFileUpload}
                  onRemoveFromQueue={removeFromQueue}
                  onClearQueue={clearQueue}
                  onUploadAll={handleUploadAll}
                  onSetActiveTab={setActiveTab}
                />
              )}
              
              {activeTab === 'documents' && (
                <DocumentsTab
                  documentAnalyses={documentAnalyses}
                  userDocuments={userDocuments}
                  isAnalyzing={isAnalyzing}
                  onAnalyze={runComprehensiveDocumentAnalysis}
                  onDelete={deleteDocument}
                  onSetActiveTab={setActiveTab}
                />
              )}
              
              {activeTab === 'analysis' && (
                <AnalysisTab
                  userDocuments={userDocuments}
                  documentAnalyses={documentAnalyses}
                  isAnalyzing={isAnalyzing}
                  selectedDocument={selectedDocumentForAnalysis}
                  setSelectedDocument={setSelectedDocumentForAnalysis}
                  onRunAnalysis={runAnalysis}
                  onSetActiveTab={setActiveTab}
                />
              )}
              
              {activeTab === 'results' && (
                <ResultsTab
                  analysisResults={analysisResults}
                  isAnalyzing={isAnalyzing}
                  onRerunAnalysis={runComprehensiveDocumentAnalysis}
                  onDownloadResult={(id) => downloadResult(id, currentUser)}
                  onClearResults={clearResults}
                  onSetActiveTab={setActiveTab}
                />
              )}
              
              {activeTab === 'legal-search' && (
                <LegalDatabaseSearch />
              )}
              
              {activeTab === 'immigration' && (
                <ImmigrationTools />
              )}
              
              {activeTab === 'system-health' && (
                <SystemHealth />
              )}
              
              {activeTab === 'admin' && (
                <AdminPanel />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BackendProvider>
        <MainApp />
      </BackendProvider>
    </AuthProvider>
  );
}
