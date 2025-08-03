// ==================== src/components/layout/SidebarNavigation.tsx ====================
import React from 'react';

interface Tab {
  id: string;
  label: string;
  icon: string;
  badge?: number | null;
  adminOnly?: boolean;
}

interface SidebarNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  userDocumentsCount: number;
  analysisResultsCount: number;
  isBackendConfigured: boolean;
}

export const SidebarNavigation: React.FC<SidebarNavigationProps> = ({
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
    <nav className="w-64 bg-white border-r border-gray-200 shadow-sm flex flex-col h-full">
      {/* Sidebar Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-stone-200 rounded-xl flex items-center justify-center shadow-sm">
            <svg className="w-6 h-6 text-stone-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Legally</h1>
            <p className="text-xs text-gray-500">— powered by AI</p>
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            disabled={!isBackendConfigured}
            className={`w-full flex items-center gap-3 px-3 py-3 text-left rounded-xl transition-all duration-200 group ${
              activeTab === tab.id
                ? 'bg-slate-900 text-white shadow-lg transform scale-[0.98]'
                : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
            } ${!isBackendConfigured ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} ${
              tab.adminOnly && activeTab !== tab.id ? 'hover:bg-red-50 hover:text-red-700' : ''
            }`}
          >
            <span className={`text-xl transition-transform duration-200 ${
              activeTab === tab.id ? 'scale-110' : 'group-hover:scale-105'
            }`}>
              {tab.icon}
            </span>
            
            <div className="flex-1 min-w-0">
              <span className={`text-sm font-medium block truncate ${
                tab.adminOnly && activeTab !== tab.id ? 'text-red-600' : ''
              }`}>
                {tab.label}
              </span>
              {tab.adminOnly && (
                <span className="text-xs text-red-500 opacity-75">Admin Only</span>
              )}
            </div>

            {tab.badge && (
              <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                activeTab === tab.id
                  ? 'bg-white text-slate-900'
                  : 'bg-slate-900 text-white'
              }`}>
                {tab.badge}
              </span>
            )}

            {tab.adminOnly && (
              <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">
                🔒
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Backend Status Footer */}
      <div className="p-3 border-t border-gray-200">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${
          isBackendConfigured 
            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
            : 'bg-rose-50 text-rose-700 border border-rose-200'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            isBackendConfigured ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500 animate-pulse'
          }`} />
          <span className="font-medium">
            {isBackendConfigured ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
    </nav>
  );
};

// ==================== src/App.tsx ====================
import React, { useState, useEffect, useMemo } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { BackendProvider, useBackend } from './contexts/BackendContext';
import { ApiService } from './services/api';
import { LoginScreen } from './components/auth/LoginScreen';
import { AppHeader } from './components/layout/AppHeader';
import { BackendWarning } from './components/layout/BackendWarning';
import { SidebarNavigation } from './components/layout/SidebarNavigation';
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
    <div className="flex h-screen bg-stone-50">
      {/* Sidebar Navigation */}
      <SidebarNavigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        userDocumentsCount={userDocuments.length}
        analysisResultsCount={analysisResults.length}
        isBackendConfigured={isBackendConfigured}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <AppHeader sessionId={sessionId} />
        <BackendWarning />

        {/* Content */}
        <div className="flex-1 overflow-auto">
          <div className="p-6 max-w-full">
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

// ==================== src/components/layout/AppHeader.tsx ====================
import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useBackend } from '../../contexts/BackendContext';
import { getSubscriptionBadgeClass } from '../../utils/helpers';

interface AppHeaderProps {
  sessionId?: string;
}

export const AppHeader: React.FC<AppHeaderProps> = ({ sessionId }) => {
  const { currentUser, logout } = useAuth();
  const { backendCapabilities } = useBackend();

  return (
    <header className="bg-white shadow-sm border-b border-gray-100">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left side - Welcome message */}
          <div>
            <h2 className="text-lg font-medium text-gray-900">
              Welcome back, {currentUser?.username}
            </h2>
            <p className="text-sm text-gray-600">
              Legal document analysis and case management platform
            </p>
          </div>

          {/* Right side - User info and controls */}
          <div className="flex items-center gap-3">
            {/* User Info */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-full">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-sm text-gray-700">{currentUser?.username}</span>
            </div>

            {/* Subscription Badge */}
            <div className={`px-3 py-1.5 rounded-full text-xs font-medium ${getSubscriptionBadgeClass(currentUser?.subscription_tier || 'free')}`}>
              {currentUser?.subscription_tier.toUpperCase()}
              {currentUser?.subscription_tier === 'premium' && (
                <span className="ml-1" title="Has access to external legal databases">🔗</span>
              )}
            </div>

            {/* Enhanced RAG Status */}
            {backendCapabilities.enhancedRag && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-stone-100 text-stone-700 border border-stone-200">
                <span>🧠</span>
                <span>Smart RAG</span>
              </div>
            )}
            
            {/* Session Info */}
            {sessionId && (
              <div className="text-xs text-gray-500 font-mono bg-gray-50 px-3 py-1.5 rounded-lg">
                Session: {sessionId.substring(0, 8)}
              </div>
            )}

            {/* Logout Button */}
            <button
              onClick={logout}
              className="p-2 bg-gray-50 hover:bg-gray-100 rounded-lg transition-all hover:shadow-sm text-gray-600 hover:text-gray-700"
              title="Logout"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

// ==================== src/components/layout/BackendWarning.tsx ====================
import React from 'react';
import { useBackend } from '../../contexts/BackendContext';

export const BackendWarning: React.FC = () => {
  const { isBackendConfigured, connectionError } = useBackend();

  if (isBackendConfigured) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-100 px-6 py-3">
      <div className="flex items-center gap-3">
        <svg className="w-5 h-5 text-amber-600 flex-shrink-0 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
        <div className="flex-1">
          <p className="text-sm text-amber-800 font-medium">
            {connectionError || "Connecting to backend server..."}
          </p>
        </div>
      </div>
    </div>
  );
};

// ==================== src/utils/constants.ts ====================
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

// UPDATE THIS to your actual Cloudflare tunnel URL
export const DEFAULT_BACKEND_URL = "https://accurately-feb-distinguished-optical.trycloudflare.com";
