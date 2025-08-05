// src/hooks/useChat.ts (COMPLETE REPLACEMENT with External API forcing)
import { useState, useCallback } from 'react';
import type { Message } from '../types';
import { ApiService } from '../services/api';

export const useChat = (apiService: ApiService) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');

  const sendMessage = useCallback(async (
    input: string,
    responseStyle: string,
    searchScope: string,
    useEnhancedRag: boolean,
    expandRequest: boolean = false
  ) => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { from: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // 🎯 SMART SEARCH SCOPE DETECTION
      const congressionalTerms = ['bill', 'congress', 'house', 'senate', 'legislation', 'passed', 'law', 'h.r.', 's.', 'hr ', 'sb '];
      const legalTerms = ['case', 'court', 'decision', 'ruling', 'supreme court', 'circuit', 'district court'];
      const governmentTerms = ['sec enforcement', 'fda recall', 'epa violation', 'osha citation', 'recent enforcement'];
      
      const query = input.toLowerCase();
      
      // Detect query type and force appropriate search scope
      const isCongressionalQuery = congressionalTerms.some(term => query.includes(term));
      const isLegalQuery = legalTerms.some(term => query.includes(term));
      const isGovernmentQuery = governmentTerms.some(term => query.includes(term));
      
      // 🚀 FORCE EXTERNAL APIs for specific query types
      let finalSearchScope = searchScope;
      
      if (isCongressionalQuery || isLegalQuery || isGovernmentQuery) {
        finalSearchScope = 'all';
        console.log('🏛️ FORCING external search for query type:', {
          congressional: isCongressionalQuery,
          legal: isLegalQuery,
          government: isGovernmentQuery,
          originalScope: searchScope,
          forcedScope: finalSearchScope
        });
      }

      // Add external databases based on query type
      const externalDatabases: string[] = [];
      if (isCongressionalQuery) {
        externalDatabases.push('congress_gov', 'federal_register');
      }
      if (isLegalQuery) {
        externalDatabases.push('harvard_caselaw', 'courtlistener');
      }
      if (isGovernmentQuery) {
        externalDatabases.push('sec_edgar', 'epa_echo', 'fda_enforcement');
      }

      const requestBody = {
        question: expandRequest ? `Please provide more detailed information about: ${input}` : input,
        session_id: sessionId || undefined,
        response_style: responseStyle,
        search_scope: finalSearchScope,  // Use forced scope
        use_enhanced_rag: useEnhancedRag,
        external_databases: externalDatabases  // Add specific databases
      };

      console.log('📡 Frontend sending to backend:', requestBody);
      console.log('🎯 Query analysis:', {
        isCongressionalQuery,
        isLegalQuery, 
        isGovernmentQuery,
        finalSearchScope,
        externalDatabases
      });

      const data = await apiService.post<any>('/ask', requestBody);
      
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      const botText = data.response || data.error || 'No response received from server';
      
      // Add source information to response
      let enhancedBotText = botText;
      if (data.sources_searched && data.sources_searched.length > 0) {
        const sourcesSearched = data.sources_searched.join(', ');
        enhancedBotText += `\n\n---\n**Sources Searched:** ${sourcesSearched}`;
        
        if (data.sources_searched.includes('congress_gov') || data.sources_searched.includes('harvard_caselaw')) {
          enhancedBotText += '\n**External APIs:** ✅ Used';
        } else {
          enhancedBotText += '\n**External APIs:** ❌ Not used';
        }
      }

      const botMessage: Message = { 
        from: 'bot', 
        text: enhancedBotText,
        confidence: data.confidence_score,
        expandAvailable: data.expand_available,
        sources: data.sources || []
      };
      setMessages(prev => [...prev, botMessage]);
      
    } catch (error) {
      console.error('Chat error:', error);
      let errorMessage = 'Failed to connect to server.';
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timed out - the server may be busy.';
        } else if (error.message.includes('fetch')) {
          errorMessage = 'Network error - check your connection.';
        } else {
          errorMessage = `Error: ${error.message}`;
        }
      }
      
      const botMessage: Message = { from: 'bot', text: errorMessage };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [apiService, isLoading, sessionId]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setSessionId('');
  }, []);

  const addWelcomeMessage = useCallback((username: string) => {
    setMessages([{
      from: 'bot',
      text: `Welcome back, ${username}! I'm Legally, your AI-powered legal research assistant. 

🎯 **What I can help you with:**
• **Congressional Research:** Recent bills, voting records, H.R./S. status
• **Case Law:** Supreme Court decisions, circuit court rulings
• **Legal Analysis:** Document review, risk assessment, clause extraction
• **Government Data:** SEC filings, EPA violations, enforcement actions
• **Immigration Law:** Country conditions, case management, form guidance

💡 **Pro Tips:**
• Use "All Sources" to access 20+ external legal databases
• Ask about specific bills: "What is H.R.1 about?"
• Search case law: "recent Supreme Court immigration decisions"
• Research enforcement: "SEC enforcement actions 2025"

How can I help you today?`
    }]);
  }, []);

  return {
    messages,
    isLoading,
    sessionId,
    sendMessage,
    clearMessages,
    addWelcomeMessage
  };
};
