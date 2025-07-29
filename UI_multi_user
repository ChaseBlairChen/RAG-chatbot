import { useState, useRef, useEffect } from "react";

type Message = {
  from: "user" | "bot";
  text: string;
  confidence?: number;
  expandAvailable?: boolean;
  sources?: any[];
};

type AnalysisResult = {
  id: number;
  toolId: string;
  toolTitle: string;
  document: string;
  analysis: string;
  confidence?: number;
  timestamp: string;
  sources?: any[];
  status: 'completed' | 'failed' | 'processing';
  extractedData?: any;
  warnings?: string[];
};

const analysisTools = [
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

export default function UnifiedLegalAssistant() {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { from: "bot", text: "Hello! I'm your AI Legal Assistant with Smart RAG capabilities. I can help you understand legal documents, analyze your uploaded files, and answer questions about law. How can I assist you today?" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [responseStyle, setResponseStyle] = useState("balanced");
  const [sessionId, setSessionId] = useState("");
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Document analysis state
  const [activeTab, setActiveTab] = useState("chat");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Backend configuration state
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");
  const [apiToken, setApiToken] = useState("demo-user-token");
  const [isBackendConfigured, setIsBackendConfigured] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [backendCapabilities, setBackendCapabilities] = useState({
    hasChat: false,
    hasDocumentAnalysis: false,
    enhancedRag: false,
    userContainers: false,
    version: "",
    subscriptionTier: "free"
  });

  // New unified backend settings
  const [searchScope, setSearchScope] = useState("all");
  const [useEnhancedRag, setUseEnhancedRag] = useState(true);
  const [userDocuments, setUserDocuments] = useState([]);

  // Simple markdown-like renderer without external dependencies
  const renderMarkdown = (text: string) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br/>')
      .replace(/^/, '<p>')
      .replace(/$/, '</p>')
      // Handle lists
      .replace(/^\* (.*$)/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
      // Handle numbered lists
      .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>')
      // Handle code blocks
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Handle links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  };

  // Enhanced error handling for backend connection - UPDATED VERSION
  const checkBackendCapabilities = async () => {
    await testConnectionWithUrl(backendUrl);
  };

  // Better error handling for document loading
  const loadUserDocuments = async () => {
    // FIXED: Allow demo-user-token for loading documents
    if (!isBackendConfigured || !apiToken) return;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const response = await fetch(`${backendUrl}/user/documents`, {
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${apiToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        setUserDocuments(data.documents || []);
      } else {
        console.error("Failed to load user documents:", response.status, response.statusText);
      }
    } catch (error) {
      console.error("Failed to load user documents:", error);
      // Don't show error to user, just log it
    }
  };

  useEffect(() => {
    // Don't auto-check on component mount, wait for user configuration
  }, []);

  useEffect(() => {
    if (isBackendConfigured) {
      loadUserDocuments();
    }
  }, [isBackendConfigured, apiToken]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // FIXED: Enhanced error handling for chat with updated request body
  async function sendMessage(expandRequest = false) {
    if (!input.trim() || isLoading) return;

    if (!isBackendConfigured) {
      setMessages(msgs => [...msgs, { 
        from: "bot", 
        text: "Please configure the backend URL and API token in the settings first. Click the ⚙️ button in the header." 
      }]);
      return;
    }

    const userMessage: Message = { from: "user", text: input };
    setMessages((msgs) => [...msgs, userMessage]);
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    try {
      const requestBody = {
        question: expandRequest ? `Please provide more detailed information about: ${currentInput}` : currentInput, // ✅ FIXED: Changed from 'query' to 'question'
        session_id: sessionId || undefined,
        response_style: responseStyle,
        search_scope: searchScope,
        use_enhanced_rag: useEnhancedRag
      };

      const headers: any = {
        "Content-Type": "application/json"
      };

      // Always add auth header if we have a token
      if (apiToken) {
        headers["Authorization"] = `Bearer ${apiToken}`;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const res = await fetch(`${backendUrl}/ask`, {
        method: "POST",
        signal: controller.signal,
        headers: headers,
        body: JSON.stringify(requestBody),
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error ${res.status}: ${errorText || res.statusText}`);
      }

      const data = await res.json();
      
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      const botText = data.response || data.error || "No response received from server";
      const botMessage: Message = { 
        from: "bot", 
        text: botText,
        confidence: data.confidence_score,
        expandAvailable: data.expand_available,
        sources: data.sources || []
      };
      setMessages((msgs) => [...msgs, botMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      let errorMessage = "Failed to connect to server.";
      
      if (error.name === 'AbortError') {
        errorMessage = "Request timed out - the server may be busy.";
      } else if (error.message.includes('fetch')) {
        errorMessage = "Network error - check your connection and backend URL.";
      } else if (error.message.includes('401') || error.message.includes('403')) {
        errorMessage = "Authentication failed - check your API token.";
      } else {
        errorMessage = `Error: ${error.message}`;
      }
      
      const botMessage: Message = { 
        from: "bot", 
        text: errorMessage + ((!apiToken || apiToken === "demo-user-token") ? " (Try setting a proper API token in settings)" : "")
      };
      setMessages((msgs) => [...msgs, botMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  function requestExpansion(messageText: string) {
    setInput(`Please expand on: ${messageText.slice(0, 100)}...`);
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
      }

      const allowedTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'application/rtf'
      ];
      
      if (!allowedTypes.includes(file.type)) {
        alert('Please upload a PDF, DOC, DOCX, RTF, or TXT file');
        return;
      }

      setUploadedFile(file);
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(0), 3000);
    }
  };

  const uploadDocument = async () => {
    if (!uploadedFile) return;

    // FIXED: Allow demo-user-token for uploads
    if (!apiToken) {
      alert('Please set an API token in settings to upload documents.');
      return;
    }

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for file upload

      const response = await fetch(`${backendUrl}/user/upload`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${apiToken}`
        },
        body: formData
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (${response.status}): ${errorText || response.statusText}`);
      }

      const data = await response.json();
      
      alert(`Document uploaded successfully! ${data.pages_processed} pages processed.`);
      
      // Reload user documents
      loadUserDocuments();
      
      // Clear uploaded file
      setUploadedFile(null);
      setActiveTab('analysis');
      
    } catch (error) {
      console.error('Upload failed:', error);
      let errorMessage = 'Upload failed';
      
      if (error.name === 'AbortError') {
        errorMessage = 'Upload timed out - file may be too large or server is slow';
      } else {
        errorMessage = `Upload failed: ${error.message}`;
      }
      
      alert(errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // FIXED: Updated runAnalysis function with 'question' instead of 'query'
  const runAnalysis = async (toolId: string) => {
    // FIXED: Allow demo-user-token for analysis
    if (!apiToken) {
      alert('Please set an API token in settings to run analysis.');
      return;
    }

    const tool = analysisTools.find(t => t.id === toolId);
    if (!tool) return;

    setIsAnalyzing(true);

    const processingResult: AnalysisResult = {
      id: Date.now(),
      toolId: toolId,
      toolTitle: tool.title,
      document: "User Documents",
      analysis: 'Running analysis on your uploaded documents...',
      timestamp: new Date().toLocaleString(),
      status: 'processing',
      sources: []
    };
    setAnalysisResults(prev => [processingResult, ...prev]);
    setActiveTab('results');

    try {
      const requestBody = {
        question: tool.prompt, // ✅ FIXED: Changed from 'query' to 'question'
        session_id: sessionId || undefined,
        response_style: "detailed",
        search_scope: "user_only",
        use_enhanced_rag: useEnhancedRag
      };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout

      const response = await fetch(`${backendUrl}/ask`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${apiToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis failed (${response.status}): ${errorText || response.statusText}`);
      }

      const data = await response.json();
      
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

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
      if (error.name === 'AbortError') {
        errorMessage = 'Analysis timed out - the request took too long';
      } else {
        errorMessage = `Analysis failed: ${error.message}`;
      }
      
      setAnalysisResults(prev => prev.map(r => 
        r.id === processingResult.id 
          ? {
              ...r,
              analysis: errorMessage,
              status: 'failed',
              warnings: ['Make sure you have uploaded documents and have a valid API token.']
            }
          : r
      ));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadResult = (resultId: number) => {
    const result = analysisResults.find(r => r.id === resultId);
    if (!result) return;

    const content = `Legal Document Analysis Report
Generated: ${result.timestamp}
Analysis Type: ${result.toolTitle}
Document: ${result.document}
Status: ${result.status}
Confidence Score: ${result.confidence ? Math.round(result.confidence * 100) + '%' : 'N/A'}

ANALYSIS RESULTS:
${result.analysis}

${result.extractedData ? '\nEXTRACTED DATA:\n' + JSON.stringify(result.extractedData, null, 2) : ''}

${result.warnings && result.warnings.length > 0 ? '\nWARNINGS:\n' + result.warnings.join('\n') : ''}

---
Generated by Legal Document Analysis Assistant
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
  };

  const deleteUserDocument = async (fileId: string) => {
    // FIXED: Allow demo-user-token for document deletion
    if (!apiToken) {
      alert('Authentication required to delete documents.');
      return;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch(`${backendUrl}/user/documents/${fileId}`, {
        method: 'DELETE',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${apiToken}`
        }
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        loadUserDocuments(); // Refresh the list
        alert('Document deleted successfully');
      } else {
        const errorText = await response.text();
        throw new Error(`Delete failed (${response.status}): ${errorText || response.statusText}`);
      }
    } catch (error) {
      console.error('Delete failed:', error);
      alert(`Delete failed: ${error.message}`);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-emerald-100 text-emerald-800';
      case 'failed': return 'bg-rose-100 text-rose-800';
      case 'processing': return 'bg-sky-100 text-sky-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Enhanced settings dialog with better UX - FIXED VERSION
  const openSettings = () => {
    const newUrl = prompt('Enter backend URL (e.g., http://18.232.139.244:8000):', backendUrl || 'http://localhost:8000');
    if (newUrl !== null) {
      const trimmedUrl = newUrl.trim();
      setBackendUrl(trimmedUrl);
      setIsBackendConfigured(false);
      setConnectionError("");
      
      // Test connection with the new URL directly, don't wait for state update
      if (trimmedUrl) {
        testConnectionWithUrl(trimmedUrl);  // Use new function
      }
    }
    
    const newToken = prompt('Enter API token (or "demo-user-token" for demo):', apiToken || 'demo-user-token');
    if (newToken !== null) {
      setApiToken(newToken.trim());
    }
  };

  // New function to test connection with specific URL
  const testConnectionWithUrl = async (url) => {
    if (!url) {
      alert('Please set a backend URL first');
      return;
    }
    
    setConnectionError("Testing connection...");
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const healthResponse = await fetch(`${url}/health`, {
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
      
      // Check if it's the unified backend
      if (healthData.version && healthData.version.includes("SmartRAG")) {
        setBackendCapabilities({
          hasChat: true,
          hasDocumentAnalysis: true,
          enhancedRag: healthData.components?.enhanced_rag?.enabled || false,
          userContainers: healthData.components?.user_containers?.enabled || false,
          version: healthData.version,
          subscriptionTier: "free"
        });
        setIsBackendConfigured(true);
        setConnectionError("");
        
        // Get subscription status if token is available
        if (apiToken && apiToken !== "demo-user-token") {
          try {
            const subController = new AbortController();
            const subTimeoutId = setTimeout(() => subController.abort(), 5000);
            
            const subResponse = await fetch(`${url}/subscription/status`, {
              signal: subController.signal,
              headers: {
                'Authorization': `Bearer ${apiToken}`,
                'Content-Type': 'application/json'
              }
            });
            
            clearTimeout(subTimeoutId);
            
            if (subResponse.ok) {
              const subData = await subResponse.json();
              setBackendCapabilities(prev => ({
                ...prev,
                subscriptionTier: subData.subscription_tier
              }));
            }
          } catch (e) {
            console.log("Could not fetch subscription status:", e);
          }
        }
      } else {
        throw new Error("Backend doesn't support Smart RAG features");
      }
    } catch (error) {
      console.error("Failed to check backend capabilities:", error);
      setIsBackendConfigured(false);
      
      if (error.name === 'AbortError') {
        setConnectionError("Connection timeout - backend may be down or slow");
      } else if (error.message.includes('fetch') || error.name === 'TypeError') {
        setConnectionError("Cannot connect to backend - check if server is running at " + url);
      } else {
        setConnectionError(`Backend error: ${error.message}`);
      }
    }
  };

  // Updated manual connection test button
  const testConnection = async () => {
    await testConnectionWithUrl(backendUrl);
  };

  const getSubscriptionBadge = () => {
    const tier = backendCapabilities.subscriptionTier;
    const colors = {
      free: 'bg-gray-100 text-gray-800',
      premium: 'bg-blue-100 text-blue-800',
      enterprise: 'bg-purple-100 text-purple-800'
    };
    return colors[tier] || colors.free;
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-gray-50 to-slate-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-slate-900 to-slate-700 rounded-xl flex items-center justify-center shadow-md">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Smart Legal Assistant
                </h1>
                <p className="text-xs text-gray-500">Multi-User AI Platform with Enhanced RAG</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {/* Backend Status */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
                isBackendConfigured 
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                  : 'bg-rose-50 text-rose-700 border border-rose-200'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isBackendConfigured ? 'bg-emerald-500' : 'bg-rose-500'} animate-pulse`} />
                <span>
                  {isBackendConfigured 
                    ? `Connected ${backendCapabilities.version ? `(${backendCapabilities.version})` : ''}` 
                    : 'Disconnected'
                  }
                </span>
              </div>

              {/* Test Connection Button */}
              {!isBackendConfigured && backendUrl && (
                <button
                  onClick={testConnection}
                  className="px-3 py-1.5 bg-blue-50 text-blue-700 text-xs font-medium rounded-full hover:bg-blue-100 transition-all"
                >
                  Test Connection
                </button>
              )}

              {/* Subscription Tier */}
              {isBackendConfigured && (
                <div className={`px-3 py-1.5 rounded-full text-xs font-medium ${getSubscriptionBadge()}`}>
                  {backendCapabilities.subscriptionTier.toUpperCase()}
                </div>
              )}

              {/* Enhanced RAG Status */}
              {isBackendConfigured && backendCapabilities.enhancedRag && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700 border border-blue-200">
                  <span>🧠</span>
                  <span>Smart RAG</span>
                </div>
              )}
              
              {/* Settings Button */}
              <button
                onClick={openSettings}
                className="p-2 bg-gray-50 hover:bg-gray-100 rounded-lg transition-all hover:shadow-sm"
                title="Configure Backend & Token"
              >
                <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
              
              {sessionId && (
                <div className="text-xs text-gray-500 font-mono bg-gray-50 px-3 py-1.5 rounded-lg">
                  Session: {sessionId.substring(0, 8)}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Backend Warning */}
      {!isBackendConfigured && (
        <div className="bg-amber-50 border-b border-amber-100 px-6 py-3">
          <div className="max-w-7xl mx-auto flex items-center gap-3">
            <svg className="w-5 h-5 text-amber-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div className="flex-1">
              <p className="text-sm text-amber-800">
                {connectionError || "Configure your backend URL and API token to enable AI features. Click the settings icon to get started."}
              </p>
            </div>
            {backendUrl && !isBackendConfigured && (
              <button
                onClick={testConnection}
                className="bg-amber-100 text-amber-800 px-3 py-1 rounded text-sm hover:bg-amber-200 transition-all"
              >
                Test Connection
              </button>
            )}
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <nav className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-8">
            {[
              { id: 'chat', label: 'Smart Chat', icon: '💬', available: true },
              { id: 'upload', label: 'Upload Documents', icon: '📤', available: true },
              { id: 'documents', label: 'My Documents', icon: '📁', badge: userDocuments.length > 0 ? userDocuments.length : null, available: true },
              { id: 'analysis', label: 'Analysis Tools', icon: '🔍', available: true },
              { id: 'results', label: 'Results', icon: '📊', badge: analysisResults.length > 0 ? analysisResults.length : null, available: true }
            ].map(tab => (
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

      {/* Main Content */}
      <div className="flex-grow overflow-auto">
        <div className="w-full max-w-7xl mx-auto p-6">
          
          {/* Backend Not Configured */}
          {!isBackendConfigured && (
            <div className="min-h-[calc(100vh-250px)] flex items-center justify-center">
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-12 text-center max-w-2xl">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 mb-3">Backend Connection Required</h3>
                <p className="text-gray-600 mb-6">
                  Connect to the Smart Legal Assistant backend to unlock AI-powered legal analysis and enhanced RAG capabilities.
                </p>
                {connectionError && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                    <p className="text-red-800 text-sm">{connectionError}</p>
                  </div>
                )}
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={openSettings}
                    className="bg-slate-900 text-white px-6 py-3 rounded-lg hover:bg-slate-800 transition-all font-medium"
                  >
                    Configure Backend
                  </button>
                  {backendUrl && (
                    <button
                      onClick={testConnection}
                      className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-200 transition-all font-medium"
                    >
                      Test Connection
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
          
          {/* Chat Tab */}
          {activeTab === 'chat' && isBackendConfigured && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 h-[calc(100vh-200px)] flex flex-col overflow-hidden">
              <div className="flex-grow overflow-y-auto p-6 space-y-4">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.from === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-3xl rounded-2xl px-4 py-3 ${
                      msg.from === "user" 
                        ? "bg-slate-900 text-white" 
                        : "bg-gray-50 text-gray-900 border border-gray-100"
                    }`}>
                      <div 
                        className="prose prose-sm max-w-none"
                        dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.text) }}
                      />
                      
                      {/* Show confidence and sources for bot messages */}
                      {msg.from === "bot" && (msg.confidence || msg.sources?.length > 0) && (
                        <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                          {msg.confidence && (
                            <div className="flex items-center gap-2 mb-2">
                              <span>Confidence:</span>
                              <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-20">
                                <div 
                                  className="bg-blue-500 h-2 rounded-full transition-all" 
                                  style={{ width: `${(msg.confidence * 100)}%` }}
                                />
                              </div>
                              <span>{Math.round((msg.confidence || 0) * 100)}%</span>
                            </div>
                          )}
                          {msg.sources && msg.sources.length > 0 && (
                            <div>
                              <span className="font-medium">Sources: </span>
                              {msg.sources.slice(0, 3).map((source: any, i: number) => (
                                <span key={i} className="mr-2">
                                  {source.file_name}
                                  {i < Math.min(msg.sources.length - 1, 2) ? ',' : ''}
                                </span>
                              ))}
                              {msg.sources.length > 3 && <span>+{msg.sources.length - 3} more</span>}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Expand button for bot messages */}
                      {msg.from === "bot" && msg.expandAvailable && (
                        <button
                          onClick={() => requestExpansion(msg.text)}
                          className="mt-2 text-xs bg-blue-50 text-blue-600 px-3 py-1 rounded-full hover:bg-blue-100 transition-all"
                        >
                          Expand Answer
                        </button>
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-50 rounded-2xl px-4 py-3 border border-gray-100">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                        <span className="text-sm text-gray-500 ml-2">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
              
              {/* Chat Input */}
              <div className="p-6 border-t border-gray-100 bg-gray-50">
                {/* Settings Row */}
                <div className="flex items-center gap-4 mb-4 text-sm">
                  <div className="flex items-center gap-2">
                    <label className="text-gray-600">Style:</label>
                    <select 
                      value={responseStyle} 
                      onChange={(e) => setResponseStyle(e.target.value)}
                      className="bg-white border border-gray-200 rounded px-2 py-1 text-sm"
                    >
                      <option value="concise">Concise</option>
                      <option value="balanced">Balanced</option>
                      <option value="detailed">Detailed</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <label className="text-gray-600">Search:</label>
                    <select 
                      value={searchScope} 
                      onChange={(e) => setSearchScope(e.target.value)}
                      className="bg-white border border-gray-200 rounded px-2 py-1 text-sm"
                    >
                      <option value="all">All Sources</option>
                      <option value="user_only">My Documents Only</option>
                      <option value="default_only">Default Database Only</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-1 text-gray-600">
                      <input 
                        type="checkbox" 
                        checked={useEnhancedRag} 
                        onChange={(e) => setUseEnhancedRag(e.target.checked)}
                        className="w-4 h-4"
                      />
                      Enhanced RAG
                    </label>
                  </div>
                </div>
                
                {/* Input Row */}
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask a legal question..."
                    className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent"
                    disabled={isLoading}
                  />
                  <button
                    onClick={() => sendMessage()}
                    disabled={isLoading || !input.trim()}
                    className="bg-slate-900 text-white px-6 py-3 rounded-xl hover:bg-slate-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-medium"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {/* Upload Tab */}
          {activeTab === 'upload' && isBackendConfigured && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Upload Legal Documents</h2>
              
              <div className="max-w-2xl">
                <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-gray-400 transition-all">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Document</h3>
                  <p className="text-gray-600 mb-4">
                    Support for PDF, DOC, DOCX, RTF, and TXT files (max 10MB)
                  </p>
                  
                  <input
                    type="file"
                    onChange={handleFileUpload}
                    accept=".pdf,.doc,.docx,.txt,.rtf"
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-slate-900 hover:bg-slate-800 cursor-pointer transition-all"
                  >
                    Choose File
                  </label>
                </div>
                
                {uploadedFile && (
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center gap-3">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <div className="flex-1">
                        <p className="font-medium text-blue-900">{uploadedFile.name}</p>
                        <p className="text-sm text-blue-700">
                          {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <button
                        onClick={uploadDocument}
                        disabled={isAnalyzing}
                        className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-300 transition-all font-medium"
                      >
                        {isAnalyzing ? 'Uploading...' : 'Upload'}
                      </button>
                    </div>
                    
                    {uploadProgress > 0 && (
                      <div className="mt-3">
                        <div className="bg-blue-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                            style={{ width: `${uploadProgress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-900 mb-2">Supported File Types:</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• <strong>PDF:</strong> Portable Document Format</li>
                    <li>• <strong>DOC/DOCX:</strong> Microsoft Word documents</li>
                    <li>• <strong>TXT:</strong> Plain text files</li>
                    <li>• <strong>RTF:</strong> Rich Text Format</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
          
          {/* Documents Tab */}
          {activeTab === 'documents' && isBackendConfigured && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold text-gray-900">My Documents</h2>
                <div className="text-sm text-gray-500">
                  {userDocuments.length} document{userDocuments.length !== 1 ? 's' : ''}
                </div>
              </div>
              
              {userDocuments.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No documents uploaded</h3>
                  <p className="text-gray-600 mb-4">Upload your first legal document to get started</p>
                  <button
                    onClick={() => setActiveTab('upload')}
                    className="bg-slate-900 text-white px-6 py-3 rounded-lg hover:bg-slate-800 transition-all font-medium"
                  >
                    Upload Document
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {userDocuments.map((doc: any) => (
                    <div key={doc.file_id} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{doc.filename}</h4>
                            <p className="text-sm text-gray-600">
                              Uploaded {new Date(doc.uploaded_at).toLocaleDateString()} • {doc.pages_processed} pages
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => deleteUserDocument(doc.file_id)}
                          className="text-red-600 hover:text-red-700 p-2 hover:bg-red-50 rounded-lg transition-all"
                          title="Delete document"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Analysis Tab */}
          {activeTab === 'analysis' && isBackendConfigured && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Analysis Tools</h2>
              
              {userDocuments.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No documents to analyze</h3>
                  <p className="text-gray-600 mb-4">Upload documents first to run analysis tools</p>
                  <button
                    onClick={() => setActiveTab('upload')}
                    className="bg-slate-900 text-white px-6 py-3 rounded-lg hover:bg-slate-800 transition-all font-medium"
                  >
                    Upload Documents
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {analysisTools.map((tool) => (
                    <div key={tool.id} className="border border-gray-200 rounded-xl p-6 hover:shadow-sm transition-all">
                      <div className="flex items-start gap-4 mb-4">
                        <div className="text-2xl">{tool.icon}</div>
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 mb-1">{tool.title}</h3>
                          <p className="text-sm text-gray-600 mb-3">{tool.description}</p>
                          
                          <div className="flex items-center gap-2 mb-3">
                            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
                              {tool.category}
                            </span>
                            <span className={`text-xs px-2 py-1 rounded-full ${
                              tool.riskLevel === 'low' ? 'bg-green-100 text-green-700' :
                              tool.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                              'bg-red-100 text-red-700'
                            }`}>
                              {tool.riskLevel} risk
                            </span>
                          </div>
                          
                          <div className="text-xs text-gray-500 mb-4">
                            <strong>Ideal for:</strong> {tool.idealFor.join(', ')}
                          </div>
                        </div>
                      </div>
                      
                      <button
                        onClick={() => runAnalysis(tool.id)}
                        disabled={isAnalyzing}
                        className="w-full bg-slate-900 text-white py-2 px-4 rounded-lg hover:bg-slate-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-medium text-sm"
                      >
                        {isAnalyzing ? 'Running Analysis...' : 'Run Analysis'}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Results Tab */}
          {activeTab === 'results' && isBackendConfigured && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold text-gray-900">Analysis Results</h2>
                <div className="text-sm text-gray-500">
                  {analysisResults.length} result{analysisResults.length !== 1 ? 's' : ''}
                </div>
              </div>
              
              {analysisResults.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No analysis results</h3>
                  <p className="text-gray-600 mb-4">Run analysis tools to see results here</p>
                  <button
                    onClick={() => setActiveTab('analysis')}
                    className="bg-slate-900 text-white px-6 py-3 rounded-lg hover:bg-slate-800 transition-all font-medium"
                  >
                    Go to Analysis Tools
                  </button>
                </div>
              ) : (
                <div className="space-y-6">
                  {analysisResults.map((result) => (
                    <div key={result.id} className="border border-gray-200 rounded-xl p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-semibold text-gray-900">{result.toolTitle}</h3>
                            <span className={`text-xs px-2 py-1 rounded-full font-medium ${getStatusColor(result.status)}`}>
                              {result.status}
                            </span>
                            {result.confidence && (
                              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                                {Math.round(result.confidence * 100)}% confidence
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-gray-600">
                            {result.timestamp} • {result.document}
                          </p>
                        </div>
                        <button
                          onClick={() => downloadResult(result.id)}
                          className="text-gray-600 hover:text-gray-700 p-2 hover:bg-gray-50 rounded-lg transition-all"
                          title="Download result"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </button>
                      </div>
                      
                      <div className="bg-gray-50 rounded-lg p-4 mb-4">
                        <div 
                          className="prose prose-sm max-w-none text-gray-800"
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(result.analysis) }}
                        />
                      </div>
                      
                      {result.warnings && result.warnings.length > 0 && (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                          <div className="flex items-start gap-2">
                            <svg className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                            <div>
                              <p className="text-sm font-medium text-amber-800 mb-1">Warnings:</p>
                              <ul className="text-sm text-amber-700 space-y-1">
                                {result.warnings.map((warning, i) => (
                                  <li key={i}>• {warning}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {result.sources && result.sources.length > 0 && (
                        <div className="border-t border-gray-200 pt-4">
                          <p className="text-sm font-medium text-gray-900 mb-2">Sources:</p>
                          <div className="space-y-1">
                            {result.sources.slice(0, 3).map((source: any, i: number) => (
                              <p key={i} className="text-xs text-gray-600">
                                • {source.file_name} {source.page && `(Page ${source.page})`}
                                {source.relevance && ` - ${Math.round(source.relevance * 100)}% relevant`}
                              </p>
                            ))}
                            {result.sources.length > 3 && (
                              <p className="text-xs text-gray-500">
                                +{result.sources.length - 3} more sources
                              </p>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
