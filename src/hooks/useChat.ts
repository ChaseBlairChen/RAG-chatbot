// hooks/useChat.ts
import { useState, useCallback } from 'react';
import { Message } from '../types';
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
      const requestBody = {
        question: expandRequest ? `Please provide more detailed information about: ${input}` : input,
        session_id: sessionId || undefined,
        response_style: responseStyle,
        search_scope: searchScope,
        use_enhanced_rag: useEnhancedRag
      };

      const data = await apiService.post<any>('/ask', requestBody);
      
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      const botText = data.response || data.error || 'No response received from server';
      const botMessage: Message = { 
        from: 'bot', 
        text: botText,
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
      text: `Welcome back, ${username}! I'm Legally, your AI-powered legal document assistant. How can I help you today?`
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
