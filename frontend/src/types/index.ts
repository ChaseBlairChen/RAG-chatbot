// types/index.ts
export type Message = {
  from: "user" | "bot";
  text: string;
  confidence?: number;
  expandAvailable?: boolean;
  sources?: any[];
};

export type AnalysisResult = {
  id: number;
  toolId: string;
  toolTitle: string;
  document: string;
  documentId: string;
  analysis: string;
  confidence?: number;
  timestamp: string;
  sources?: any[];
  status: 'completed' | 'failed' | 'processing';
  extractedData?: any;
  warnings?: string[];
  analysisType: string;
};

export type DocumentAnalysis = {
  id: string;
  filename: string;
  uploadedAt: string;
  pagesProcessed: number;
  analysisResults: {
    summary?: string;
    clauses?: string;
    risks?: string;
    timeline?: string;
    obligations?: string;
    missingClauses?: string;
  };
  lastAnalyzed?: string;
  confidence?: number;
};

export type User = {
  username: string;
  email?: string;
  role: string;
  subscription_tier: string;
  loginTime: string;
  user_id?: string;
};

export type UploadStatus = {
  fileId: string;
  filename: string;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  pagesProcessed?: number;
  chunksCreated?: number;
  processingTime?: number;
  errors?: string[];
};

export type BackendCapabilities = {
  hasChat: boolean;
  hasDocumentAnalysis: boolean;
  enhancedRag: boolean;
  userContainers: boolean;
  version: string;
  subscriptionTier: string;
};

export type AnalysisTool = {
  id: string;
  title: string;
  description: string;
  prompt: string;
  icon: string;
  category: string;
  idealFor: string[];
  riskLevel: string;
  isComprehensive?: boolean;
};
