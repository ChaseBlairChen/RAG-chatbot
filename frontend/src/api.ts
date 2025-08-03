// ==================== ./src/services/api.ts ====================
export class ApiService {
  private baseUrl: string;
  private apiToken: string;

  constructor(baseUrl: string, apiToken: string) {
    this.baseUrl = baseUrl;
    this.apiToken = apiToken;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  async uploadFile(endpoint: string, formData: FormData): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5min for uploads

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
        },
        body: formData,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (${response.status}): ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  // Fixed method for searching free legal databases
  async searchFreeLegalDatabases(query: string): Promise<any> {
    const formData = new FormData();
    formData.append('query', query);
    
    try {
      const response = await fetch(`${this.baseUrl}/external/search-free`, {
        method: 'POST',
        signal: AbortSignal.timeout(30000),
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed (${response.status}): ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      throw error;
    }
  }

  // Method for searching premium legal databases
  async searchLegalDatabases(query: string, databases: string[]): Promise<any> {
    const formData = new FormData();
    formData.append('query', query);
    databases.forEach(db => formData.append('databases', db));
    
    try {
      const response = await fetch(`${this.baseUrl}/external/search`, {
        method: 'POST',
        signal: AbortSignal.timeout(30000),
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed (${response.status}): ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      throw error;
    }
  }
}

// ==================== ./src/components/legal/LegalDatabaseSearch.tsx ====================
import React, { useState } from 'react';
import { useBackend } from '../../contexts/BackendContext';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';

export const LegalDatabaseSearch: React.FC = () => {
  const { backendUrl } = useBackend();
  const { apiToken, currentUser } = useAuth();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchType, setSearchType] = useState<'free' | 'premium'>('free');
  const [selectedDatabases, setSelectedDatabases] = useState<string[]>(['harvard_caselaw', 'courtlistener']);
  
  const apiService = new ApiService(backendUrl, apiToken);

  const availableDatabases = {
    free: [
      { id: 'harvard_caselaw', name: 'Harvard Caselaw Access Project', description: 'Comprehensive case law database' },
      { id: 'courtlistener', name: 'CourtListener', description: 'Federal and state court data' },
      { id: 'federal_register', name: 'Federal Register', description: 'Government regulations and notices' }
    ],
    premium: [
      { id: 'lexisnexis', name: 'LexisNexis', description: 'Premium legal research platform' },
      { id: 'westlaw', name: 'Westlaw', description: 'Premium legal database' }
    ]
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      let response;
      
      if (searchType === 'free') {
        response = await apiService.searchFreeLegalDatabases(query);
      } else {
        response = await apiService.searchLegalDatabases(query, selectedDatabases);
      }
      
      setResults(response.results || []);
    } catch (error) {
      console.error('Search failed:', error);
      alert(`Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setResults([]);
    }
    setLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const toggleDatabase = (dbId: string) => {
    setSelectedDatabases(prev => 
      prev.includes(dbId) 
        ? prev.filter(id => id !== dbId)
        : [...prev, dbId]
    );
  };

  const canUsePremium = currentUser?.subscription_tier === 'premium';

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Legal Database Search</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Search Mode:</span>
          <select 
            value={searchType} 
            onChange={(e) => setSearchType(e.target.value as 'free' | 'premium')}
            className="bg-white border border-gray-200 rounded px-3 py-1 text-sm"
            disabled={!canUsePremium}
          >
            <option value="free">Free Databases</option>
            <option value="premium" disabled={!canUsePremium}>
              Premium Databases {!canUsePremium && '(Premium Only)'}
            </option>
          </select>
        </div>
      </div>
      
      <p className="text-gray-600 mb-4">
        {searchType === 'free' 
          ? 'Search through millions of cases from Harvard Caselaw, CourtListener, and Federal Register'
          : 'Access premium legal databases including LexisNexis and Westlaw'
        }
      </p>

      {/* Database Selection */}
      {searchType === 'premium' && canUsePremium && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-2">Select Databases:</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {availableDatabases.premium.map(db => (
              <label key={db.id} className="flex items-center gap-2 p-2 hover:bg-gray-100 rounded">
                <input
                  type="checkbox"
                  checked={selectedDatabases.includes(db.id)}
                  onChange={() => toggleDatabase(db.id)}
                  className="w-4 h-4"
                />
                <div>
                  <div className="font-medium text-sm">{db.name}</div>
                  <div className="text-xs text-gray-600">{db.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      )}

      {searchType === 'free' && (
        <div className="mb-4 p-4 bg-green-50 rounded-lg border border-green-200">
          <h4 className="font-medium text-green-900 mb-2">Free Databases Included:</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {availableDatabases.free.map(db => (
              <div key={db.id} className="p-2">
                <div className="font-medium text-sm text-green-800">✓ {db.name}</div>
                <div className="text-xs text-green-700">{db.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="flex gap-3 mb-6">
        <input 
          type="text"
          value={query} 
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search cases... (e.g., 'miranda rights', 'contract breach', 'negligence')"
          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500"
          disabled={loading}
        />
        <button 
          onClick={handleSearch} 
          disabled={loading || !query.trim() || (searchType === 'premium' && selectedDatabases.length === 0)}
          className="bg-slate-900 text-white px-6 py-3 rounded-xl hover:bg-slate-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-medium"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">
              Found {results.length} results:
            </h3>
            <span className="text-sm text-gray-500">
              Searched: {searchType === 'free' ? 'Free databases' : selectedDatabases.join(', ')}
            </span>
          </div>
          
          {results.map((result, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-semibold text-gray-900 flex-1">{result.title}</h4>
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full ml-2">
                  {result.source_database?.replace('_', ' ').toUpperCase() || 'Unknown Source'}
                </span>
              </div>
              
              <div className="text-sm text-gray-600 mb-2 space-y-1">
                {result.court && <div><strong>Court:</strong> {result.court}</div>}
                {result.date && <div><strong>Date:</strong> {result.date}</div>}
                {result.citation && <div><strong>Citation:</strong> {result.citation}</div>}
                {result.docket_number && <div><strong>Docket:</strong> {result.docket_number}</div>}
                {result.agency && <div><strong>Agency:</strong> {result.agency}</div>}
                {result.type && <div><strong>Type:</strong> {result.type}</div>}
              </div>
              
              {(result.preview || result.snippet || result.description || result.summary) && (
                <p className="text-gray-700 mb-3">
                  {result.preview || result.snippet || result.description || result.summary}
                </p>
              )}
              
              <div className="flex items-center justify-between">
                {result.url && (
                  <a 
                    href={result.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-700 font-medium text-sm"
                  >
                    View Full Document →
                  </a>
                )}
                {result.pdf_url && (
                  <a 
                    href={result.pdf_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-green-600 hover:text-green-700 font-medium text-sm ml-4"
                  >
                    Download PDF →
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {results.length === 0 && query && !loading && (
        <div className="text-center py-8 text-gray-500">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <p>No results found for "{query}"</p>
          <p className="text-sm mt-1">Try different search terms or broader keywords</p>
        </div>
      )}

      {/* Search Tips */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">💡 Search Tips:</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use specific legal terms: "miranda rights", "contract breach"</li>
            <li>• Include case names: "Brown v. Board"</li>
            <li>• Try statute citations: "42 USC 1983"</li>
            <li>• Use quotation marks for exact phrases</li>
          </ul>
        </div>
        
        <div className="p-4 bg-green-50 rounded-lg">
          <h4 className="font-medium text-green-900 mb-2">📚 Available Sources:</h4>
          <ul className="text-sm text-green-800 space-y-1">
            <li>• <strong>Harvard Caselaw:</strong> Historical and recent cases</li>
            <li>• <strong>CourtListener:</strong> Federal and state courts</li>
            <li>• <strong>Federal Register:</strong> Government regulations</li>
            {canUsePremium && (
              <>
                <li>• <strong>LexisNexis:</strong> Premium legal research (Premium)</li>
                <li>• <strong>Westlaw:</strong> Comprehensive case law (Premium)</li>
              </>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

// ==================== ./src/utils/constants.ts ====================
import type { AnalysisTool } from '../types';

export const ANALYSIS_TOOLS: AnalysisTool[] = [
  {
    id: 'comprehensive',
    title: 'Complete Document Analysis',
    description: 'Run all analysis tools at once for comprehensive insights',
    prompt: 'Provide a comprehensive legal analysis including: summary, key clauses, risks, timeline, obligations, and missing clauses.',
    icon: '🔍',
    category: 'Complete',
    idealFor: ['Any legal document'],
    riskLevel: 'low',
    isComprehensive: true
  },
  {
    id: 'summarize',
    title: 'Legal Document Summarization',
    description: 'Get plain English summaries while keeping legal tone intact',
    prompt: 'Summarize this legal document in plain English, keeping the legal tone intact. Highlight purpose, parties involved, and key terms.',
    icon: '📄',
    category: 'Analysis',
    idealFor: ['Contracts', 'Case briefs', 'Discovery documents'],
    riskLevel: 'low'
  },
  {
    id: 'extract-clauses',
    title: 'Key Clause Extraction',
    description: 'Extract termination, indemnification, liability clauses automatically',
    prompt: 'Extract and list the clauses related to termination, indemnification, liability, governing law, and confidentiality.',
    icon: '📋',
    category: 'Extraction',
    idealFor: ['NDAs', 'Employment agreements', 'Service contracts'],
    riskLevel: 'low'
  },
  {
    id: 'missing-clauses',
    title: 'Missing Clause Detection',
    description: 'Flag commonly expected clauses that might be missing',
    prompt: 'Analyze this contract and flag any commonly expected legal clauses that are missing, such as limitation of liability or dispute resolution.',
    icon: '⚠️',
    category: 'Risk Assessment',
    idealFor: ['Startup contracts', 'Vendor agreements'],
    riskLevel: 'medium'
  },
  {
    id: 'risk-flagging',
    title: 'Legal Risk Flagging',
    description: 'Identify clauses that may pose legal risks to signing party',
    prompt: 'Identify any clauses that may pose legal risks to the signing party, such as unilateral termination, broad indemnity, or vague obligations.',
    icon: '🚩',
    category: 'Risk Assessment',
    idealFor: ['Lease agreements', 'IP transfer agreements'],
    riskLevel: 'high'
  },
  {
    id: 'timeline-extraction',
    title: 'Timeline & Deadline Extraction',
    description: 'Extract all dates, deadlines, and renewal periods',
    prompt: 'Extract and list all dates, deadlines, renewal periods, and notice periods mentioned in this document.',
    icon: '📅',
    category: 'Extraction',
    idealFor: ['Leases', 'Licensing deals'],
    riskLevel: 'low'
  },
  {
    id: 'obligations',
    title: 'Obligation Summary',
    description: 'List all required actions and obligations with deadlines',
    prompt: 'List all actions or obligations the signing party is required to perform, along with associated deadlines or conditions.',
    icon: '✅',
    category: 'Analysis',
    idealFor: ['Service contracts', 'Compliance agreements'],
    riskLevel: 'low'
  }
];

export const TEST_ACCOUNTS = [
  { username: 'demo', password: 'demo123', email: 'demo@legalassistant.ai', role: 'user', subscription_tier: 'free' },
  { username: 'tester1', password: 'test123', email: 'tester1@company.com', role: 'user', subscription_tier: 'premium' },
  { username: 'tester2', password: 'test456', email: 'tester2@company.com', role: 'user', subscription_tier: 'free' },
  { username: 'lawyer1', password: 'legal123', email: 'lawyer1@lawfirm.com', role: 'user', subscription_tier: 'premium' }
];

// UPDATE THIS to match your actual backend URL
export const DEFAULT_BACKEND_URL = "http://localhost:8000";

// ==================== ./src/components/layout/TabNavigation.tsx ====================
import React from 'react';

interface Tab {
  id: string;
  label: string;
  icon: string;
  badge?: number | null;
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
  ];

  return (
    <nav className="bg-white border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex space-x-8">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              disabled={!isBackendConfigured}
              className={`relative py-4 px-1 text-sm font-medium transition-all border-b-2 ${
                activeTab === tab.id
                  ? 'text-slate-900 border-slate-900'
                  : 'text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300'
              } ${!isBackendConfigured ? 'cursor-not-allowed opacity-50' : ''}`}
            >
              <div className="flex items-center gap-2">
                <span className="text-base">{tab.icon}</span>
                <span>{tab.label}</span>
                {tab.badge && (
                  <span className="ml-1 bg-slate-900 text-white text-xs font-medium px-2 py-0.5 rounded-full">
                    {tab.badge}
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

// ==================== ./src/contexts/BackendContext.tsx ====================
import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { BackendCapabilities } from '../types';
import { DEFAULT_BACKEND_URL } from '../utils/constants';
import { useAuth } from './AuthContext';

interface BackendContextType {
  backendUrl: string;
  isBackendConfigured: boolean;
  connectionError: string;
  backendCapabilities: BackendCapabilities;
  testConnection: () => Promise<void>;
}

const BackendContext = createContext<BackendContextType | undefined>(undefined);

export const useBackend = () => {
  const context = useContext(BackendContext);
  if (!context) {
    throw new Error('useBackend must be used within a BackendProvider');
  }
  return context;
};

interface BackendProviderProps {
  children: ReactNode;
}

export const BackendProvider: React.FC<BackendProviderProps> = ({ children }) => {
  const { isLoggedIn, apiToken, currentUser } = useAuth();
  const [backendUrl] = useState(DEFAULT_BACKEND_URL);
  const [isBackendConfigured, setIsBackendConfigured] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const [backendCapabilities, setBackendCapabilities] = useState<BackendCapabilities>({
    hasChat: false,
    hasDocumentAnalysis: false,
    enhancedRag: false,
    userContainers: false,
    version: '',
    subscriptionTier: 'free'
  });

  const testConnection = async () => {
    if (!backendUrl) {
      setConnectionError('No backend URL configured');
      return;
    }
    
    setConnectionError('Testing connection...');
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const healthResponse = await fetch(`${backendUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);

      if (!healthResponse.ok) {
        throw new Error(`Backend returned ${healthResponse.status}: ${healthResponse.statusText}`);
      }

      const healthData = await healthResponse.json();
      
      // Check if this is the legal assistant backend
      if (healthData.status === 'healthy' || healthData.features) {
        setBackendCapabilities({
          hasChat: true,
          hasDocumentAnalysis: true,
          enhancedRag: healthData.features?.ai_enabled || false,
          userContainers: true,
          version: 'Legal Assistant Backend',
          subscriptionTier: currentUser?.subscription_tier || 'free'
        });
        setIsBackendConfigured(true);
        setConnectionError('');
      } else {
        throw new Error("Backend doesn't appear to be the Legal Assistant API");
      }
    } catch (error: unknown) {
      console.error('Failed to check backend capabilities:', error);
      setIsBackendConfigured(false);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          setConnectionError('Connection timeout - backend may be down or slow');
        } else if (error.message.includes('fetch') || error.name === 'TypeError') {
          setConnectionError(`Cannot connect to backend - check if server is running at ${backendUrl}`);
        } else {
          setConnectionError(`Backend error: ${error.message}`);
        }
      } else {
        setConnectionError('Unknown error occurred while connecting to backend');
      }
    }
  };

  useEffect(() => {
    if (isLoggedIn && apiToken) {
      testConnection();
    }
  }, [isLoggedIn, apiToken]);

  return (
    <BackendContext.Provider
      value={{
        backendUrl,
        isBackendConfigured,
        connectionError,
        backendCapabilities,
        testConnection
      }}
    >
      {children}
    </BackendContext.Provider>
  );
};

// ==================== ./src/hooks/useAnalysis.ts ====================
import { useState, useCallback } from 'react';
import type { AnalysisResult, DocumentAnalysis } from '../types';
import { ApiService } from '../services/api';
import { ANALYSIS_TOOLS } from '../utils/constants';

export const useAnalysis = (
  apiService: ApiService,
  documentAnalyses: DocumentAnalysis[],
  setDocumentAnalyses: React.Dispatch<React.SetStateAction<DocumentAnalysis[]>>,
  sessionId: string,
  currentUserId?: string
) => {
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedDocumentForAnalysis, setSelectedDocumentForAnalysis] = useState<string | null>(null);

  const runComprehensiveDocumentAnalysis = useCallback(async (documentId: string) => {
    const document = documentAnalyses.find(d => d.id === documentId);
    if (!document) return;

    setIsAnalyzing(true);

    const processingResult: AnalysisResult = {
      id: Date.now() + Math.random(),
      toolId: 'comprehensive',
      toolTitle: 'Complete Document Analysis',
      document: document.filename,
      documentId: documentId,
      analysis: 'Running comprehensive analysis including summary, clauses, risks, timeline, obligations, and missing clauses...',
      timestamp: new Date().toLocaleString(),
      status: 'processing',
      sources: [],
      analysisType: 'comprehensive'
    };
    
    setAnalysisResults(prev => [processingResult, ...prev]);

    try {
      const requestBody = {
        document_id: documentId,
        analysis_types: ['comprehensive'],
        user_id: currentUserId || "default_user",
        session_id: sessionId,
        response_style: "detailed"
      };

      const data = await apiService.post<any>('/analysis/comprehensive-analysis', requestBody);
      
      const analysisText = `# Comprehensive Legal Document Analysis

## Document Summary
${data.document_summary || 'No summary available'}

## Key Clauses Analysis
${data.key_clauses || 'No clauses analysis available'}

## Risk Assessment
${data.risk_assessment || 'No risk assessment available'}

## Timeline & Deadlines
${data.timeline_deadlines || 'No timeline information available'}

## Party Obligations
${data.party_obligations || 'No obligations analysis available'}

## Missing Clauses Analysis
${data.missing_clauses || 'No missing clauses analysis available'}`;

      setAnalysisResults(prev => prev.map(r => 
        r.id === processingResult.id 
          ? {
              ...r,
              analysis: analysisText,
              confidence: data.overall_confidence || 0.8,
              status: 'completed',
              sources: data.sources_by_section?.summary || [],
              warnings: data.warnings || []
            }
          : r
      ));

      setDocumentAnalyses(prev => prev.map(d => 
        d.id === documentId 
          ? {
              ...d,
              analysisResults: {
                summary: data.document_summary,
                clauses: data.key_clauses,
                risks: data.risk_assessment,
                timeline: data.timeline_deadlines,
                obligations: data.party_obligations,
                missingClauses: data.missing_clauses
              },
              lastAnalyzed: new Date().toISOString(),
              confidence: data.overall_confidence || 0.8
            }
          : d
      ));

    } catch (error) {
      console.error('Comprehensive analysis failed:', error);
      
      let errorMessage = 'Comprehensive analysis failed';
      if (error instanceof Error) {
        errorMessage = `Analysis failed: ${error.message}`;
      }
      
      setAnalysisResults(prev => prev.map(r => 
        r.id === processingResult.id 
          ? {
              ...r,
              analysis: errorMessage,
              status: 'failed',
              warnings: ['Make sure the document was uploaded successfully and you have proper access.']
            }
          : r
      ));
    } finally {
      setIsAnalyzing(false);
    }
  }, [documentAnalyses, apiService, sessionId, currentUserId, setDocumentAnalyses]);

  const runAnalysis = useCallback(async (toolId: string, documentId?: string, useEnhancedRag: boolean = true) => {
    const tool = ANALYSIS_TOOLS.find(t => t.id === toolId);
    if (!tool) return;

    if (tool.isComprehensive) {
      if (documentId) {
        await runComprehensiveDocumentAnalysis(documentId);
      } else {
        for (const doc of documentAnalyses) {
          await runComprehensiveDocumentAnalysis(doc.id);
        }
      }
      return;
    }

    setIsAnalyzing(true);

    const targetDoc = documentId ? documentAnalyses.find(d => d.id === documentId) : null;
    const docName = targetDoc ? targetDoc.filename : "User Documents";

    const processingResult: AnalysisResult = {
      id: Date.now() + Math.random(),
      toolId: toolId,
      toolTitle: tool.title,
      document: docName,
      documentId: documentId || 'all',
      analysis: `Running ${tool.title.toLowerCase()} on ${docName}...`,
      timestamp: new Date().toLocaleString(),
      status: 'processing',
      sources: [],
      analysisType: toolId
    };
    
    setAnalysisResults(prev => [processingResult, ...prev]);

    try {
      const requestBody = {
        question: tool.prompt,
        session_id: sessionId || undefined,
        response_style: "detailed",
        search_scope: documentId ? "user_only" : "user_only",
        use_enhanced_rag: useEnhancedRag,
        document_id: documentId || undefined
      };

      const data = await apiService.post<any>('/ask', requestBody);
      
      const analysisText = data.response || 'Analysis could not be completed.';
      const status = data.error ? 'failed' : 'completed';
      
      setAnalysisResults(prev => prev.map(r => 
        r.id === processingResult.id 
          ? {
              ...r,
              analysis: analysisText,
              confidence: data.confidence_score || 0.7,
              status: status,
              sources: data.sources || [],
              warnings: data.error ? [data.error] : []
            }
          : r
      ));

    } catch (error) {
      console.error('Analysis failed:', error);
      
      let errorMessage = 'Analysis failed';
      if (error instanceof Error) {
        errorMessage = `Analysis failed: ${error.message}`;
      }
      
      setAnalysisResults(prev => prev.map(r => 
        r.id === processingResult.id 
          ? {
              ...r,
              analysis: errorMessage,
              status: 'failed',
              warnings: ['Make sure you have uploaded documents and have proper authentication.']
            }
          : r
      ));
    } finally {
      setIsAnalyzing(false);
    }
  }, [documentAnalyses, apiService, sessionId, runComprehensiveDocumentAnalysis]);

  const downloadResult = useCallback((resultId: number, currentUser: any) => {
    const result = analysisResults.find(r => r.id === resultId);
    if (!result) return;

    const content = `Legal Document Analysis Report
Generated: ${result.timestamp}
Analysis Type: ${result.toolTitle}
Document: ${result.document}
User: ${currentUser?.username}
Status: ${result.status}
Confidence Score: ${result.confidence ? Math.round(result.confidence * 100) + '%' : 'N/A'}

ANALYSIS RESULTS:
${result.analysis}

${result.extractedData ? '\nEXTRACTED DATA:\n' + JSON.stringify(result.extractedData, null, 2) : ''}
${result.warnings && result.warnings.length > 0 ? '\nWARNINGS:\n' + result.warnings.join('\n') : ''}

---
Generated by Legally — powered by AI
User: ${currentUser?.username} (${currentUser?.subscription_tier})
This analysis is for informational purposes only and does not constitute legal advice.`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `legal-analysis-${result.toolId}-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [analysisResults]);

  const clearResults = useCallback(() => {
    setAnalysisResults([]);
  }, []);

  return {
    analysisResults,
    isAnalyzing,
    selectedDocumentForAnalysis,
    setSelectedDocumentForAnalysis,
    runAnalysis,
    runComprehensiveDocumentAnalysis,
    downloadResult,
    clearResults
  };
};

// ==================== ./src/hooks/useDocuments.ts ====================
import { useState, useCallback, useEffect } from 'react';
import type { DocumentAnalysis, UploadStatus } from '../types';
import { ApiService } from '../services/api';
import { validateFileBeforeUpload } from '../utils/fileValidation';

export const useDocuments = (apiService: ApiService, isBackendConfigured: boolean) => {
  const [userDocuments, setUserDocuments] = useState<any[]>([]);
  const [documentAnalyses, setDocumentAnalyses] = useState<DocumentAnalysis[]>([]);
  const [uploadQueue, setUploadQueue] = useState<File[]>([]);
  const [currentlyUploading, setCurrentlyUploading] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResults, setUploadResults] = useState<any[]>([]);
  const [uploadStatuses, setUploadStatuses] = useState<Map<string, UploadStatus>>(new Map());
  const [statusCheckIntervals, setStatusCheckIntervals] = useState<Map<string, NodeJS.Timeout>>(new Map());

  // Cleanup intervals on unmount
  useEffect(() => {
    return () => {
      statusCheckIntervals.forEach(interval => clearInterval(interval));
    };
  }, [statusCheckIntervals]);

  const loadUserDocuments = useCallback(async () => {
    if (!isBackendConfigured) return;

    try {
      const data = await apiService.get<any>('/documents/user/documents');
      setUserDocuments(data.documents || []);
      
      const docAnalyses = (data.documents || []).map((doc: any) => ({
        id: doc.file_id,
        filename: doc.filename,
        uploadedAt: doc.uploaded_at,
        pagesProcessed: doc.pages_processed,
        analysisResults: {},
        lastAnalyzed: null,
        confidence: null
      }));
      setDocumentAnalyses(docAnalyses);
    } catch (error) {
      console.error('Failed to load user documents:', error);
    }
  }, [apiService, isBackendConfigured]);

  const checkDocumentStatus = useCallback(async (fileId: string): Promise<UploadStatus | null> => {
    try {
      const data = await apiService.get<any>(`/documents/user/documents/${fileId}/status`);
      return {
        fileId: data.file_id,
        filename: data.filename,
        status: data.status,
        progress: data.progress || 0,
        message: data.message || '',
        pagesProcessed: data.pages_processed,
        chunksCreated: data.chunks_created,
        processingTime: data.processing_time,
        errors: data.errors || []
      };
    } catch (error) {
      console.error('Status check error:', error);
      return null;
    }
  }, [apiService]);

  const startProgressTracking = useCallback((fileId: string, filename: string) => {
    const initialStatus: UploadStatus = {
      fileId,
      filename,
      status: 'uploading',
      progress: 0,
      message: 'Uploading document...'
    };
    
    setUploadStatuses(prev => new Map(prev).set(fileId, initialStatus));

    const interval = setInterval(async () => {
      const status = await checkDocumentStatus(fileId);
      
      if (status) {
        setUploadStatuses(prev => new Map(prev).set(fileId, status));
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          setStatusCheckIntervals(prev => {
            const newMap = new Map(prev);
            newMap.delete(fileId);
            return newMap;
          });
          
          if (status.status === 'completed') {
            setTimeout(() => {
              setUploadStatuses(prev => {
                const newMap = new Map(prev);
                newMap.delete(fileId);
                return newMap;
              });
            }, 5000);
          }
        }
      }
    }, 2000);

    setStatusCheckIntervals(prev => new Map(prev).set(fileId, interval));

    setTimeout(() => {
      if (statusCheckIntervals.has(fileId)) {
        clearInterval(interval);
        setStatusCheckIntervals(prev => {
          const newMap = new Map(prev);
          newMap.delete(fileId);
          return newMap;
        });
        
        setUploadStatuses(prev => new Map(prev).set(fileId, {
          ...initialStatus,
          status: 'failed',
          message: 'Document processing timed out',
          errors: ['Processing took too long']
        }));
      }
    }, 5 * 60 * 1000);
  }, [checkDocumentStatus, statusCheckIntervals]);

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return;

    const fileArray = Array.from(files);
    const validFiles: File[] = [];
    const errors: string[] = [];

    fileArray.forEach(file => {
      const validationError = validateFileBeforeUpload(file);
      if (validationError) {
        errors.push(validationError);
      } else {
        validFiles.push(file);
      }
    });

    if (errors.length > 0) {
      alert('Some files had issues:\n\n' + errors.join('\n\n'));
    }

    if (validFiles.length > 0) {
      setUploadQueue(prev => [...prev, ...validFiles]);
    }
  }, []);

  const uploadAllDocuments = useCallback(async (runAnalysisAfter = false): Promise<string[]> => {
    if (uploadQueue.length === 0) return [];

    setUploadResults([]);
    const results: any[] = [];
    const uploadedDocIds: string[] = [];

    for (let i = 0; i < uploadQueue.length; i++) {
      const file = uploadQueue[i];
      setCurrentlyUploading(file);
      setUploadProgress(((i) / uploadQueue.length) * 100);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const data = await apiService.uploadFile('/documents/user/upload', formData);
        
        if (data.status === 'processing' && data.file_id) {
          startProgressTracking(data.file_id, file.name);
        }
        
        results.push({
          filename: file.name,
          success: true,
          pages_processed: data.pages_processed,
          file_id: data.file_id,
          processing_time: data.processing_time,
          warnings: data.warnings || [],
          status: data.status
        });

        uploadedDocIds.push(data.file_id);
      } catch (error) {
        console.error(`Upload failed for ${file.name}:`, error);
        
        let errorMessage = 'Upload failed';
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        results.push({
          filename: file.name,
          success: false,
          error: errorMessage
        });
      }
    }

    setUploadProgress(100);
    setUploadResults(results);
    setCurrentlyUploading(null);
    setUploadQueue([]);
    
    setTimeout(async () => {
      await loadUserDocuments();
    }, 3000);

    return uploadedDocIds;
  }, [uploadQueue, apiService, startProgressTracking, loadUserDocuments]);

  const deleteDocument = useCallback(async (fileId: string) => {
    try {
      await apiService.delete(`/documents/user/documents/${fileId}`);
      await loadUserDocuments();
      alert('Document deleted successfully');
    } catch (error) {
      console.error('Delete failed:', error);
      alert(`Delete failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [apiService, loadUserDocuments]);

  const removeFromQueue = useCallback((index: number) => {
    setUploadQueue(prev => prev.filter((_, i) => i !== index));
  }, []);

  const clearQueue = useCallback(() => {
    setUploadQueue([]);
    setUploadResults([]);
  }, []);

  const clearStatuses = useCallback(() => {
    statusCheckIntervals.forEach(interval => clearInterval(interval));
    setStatusCheckIntervals(new Map());
    setUploadStatuses(new Map());
  }, [statusCheckIntervals]);

  return {
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
  };
};

// ==================== ./package.json ====================
{
  "name": "legal-assistant-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "eslint": "^8.55.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.3.6",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }
}

// ==================== ./vite.config.ts ====================
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})

// ==================== ./tsconfig.json ====================
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}

// ==================== ./tsconfig.node.json ====================
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}

// ==================== ./tailwind.config.js ====================
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        stone: {
          50: '#fafaf9',
          100: '#f5f5f4',
          200: '#e7e5e4',
          300: '#d6d3d1',
          400: '#a8a29e',
          500: '#78716c',
          600: '#57534e',
          700: '#44403c',
          800: '#292524',
          900: '#1c1917',
        }
      }
    },
  },
  plugins: [],
}

// ==================== ./postcss.config.js ====================
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}

// ==================== ./index.html ====================
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Legally - AI-Powered Legal Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>

// ==================== ./src/index.css ====================
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    font-family: 'Inter', system-ui, sans-serif;
  }
}

@layer components {
  .prose {
    color: #374151;
  }
  
  .prose h1 {
    color: #111827;
    font-weight: 700;
    font-size: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
  }
  
  .prose h2 {
    color: #111827;
    font-weight: 600;
    font-size: 1.25rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
  }
  
  .prose h3 {
    color: #111827;
    font-weight: 600;
    font-size: 1.125rem;
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
  }
  
  .prose p {
    margin-bottom: 0.75rem;
    line-height: 1.6;
  }
  
  .prose ul {
    list-style-type: disc;
    margin-left: 1.5rem;
    margin-bottom: 0.75rem;
  }
  
  .prose ol {
    list-style-type: decimal;
    margin-left: 1.5rem;
    margin-bottom: 0.75rem;
  }
  
  .prose li {
    margin-bottom: 0.25rem;
  }
  
  .prose strong {
    font-weight: 600;
    color: #111827;
  }
  
  .prose em {
    font-style: italic;
  }
  
  .prose code {
    background-color: #f3f4f6;
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
  }
  
  .prose a {
    color: #2563eb;
    text-decoration: underline;
  }
  
  .prose a:hover {
    color: #1d4ed8;
  }
}
