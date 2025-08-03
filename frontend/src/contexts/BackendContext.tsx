// src/contexts/BackendContext.tsx
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
      console.log('Backend health response:', healthData); // Debug log
      
      // Check if this is the legal assistant backend - match your actual response
      if (healthData.status === 'healthy' && healthData.features) {
        setBackendCapabilities({
          hasChat: true,
          hasDocumentAnalysis: true,
          enhancedRag: healthData.features.ai_enabled || false,
          userContainers: true,
          version: 'Legal Assistant Backend v10.0.0',
          subscriptionTier: currentUser?.subscription_tier || 'free'
        });
        setIsBackendConfigured(true);
        setConnectionError('');
        console.log('✅ Backend connected successfully');
        console.log('🔥 Backend features:', healthData.features);
      } else {
        throw new Error(`Backend response format: ${JSON.stringify(healthData)}`);
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
