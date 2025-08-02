// hooks/useAnalysis.ts
import { useState, useCallback } from 'react';
import { AnalysisResult, DocumentAnalysis } from '../types';
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

      const data = await apiService.post<any>('/comprehensive-analysis', requestBody);
      
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
        use_enhanced_rag: useEnhancedRag
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
