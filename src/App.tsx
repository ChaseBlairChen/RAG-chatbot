// App.tsx
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
