// contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { User } from '../types';
import { TEST_ACCOUNTS } from '../utils/constants';

interface AuthContextType {
  isLoggedIn: boolean;
  currentUser: User | null;
  apiToken: string;
  login: (username: string, password: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [apiToken, setApiToken] = useState('');

  useEffect(() => {
    const savedUser = localStorage.getItem('legalAssistantUser');
    const savedToken = localStorage.getItem('legalAssistantToken');
    
    if (savedUser && savedToken) {
      try {
        const user = JSON.parse(savedUser);
        setCurrentUser(user);
        setApiToken(savedToken);
        setIsLoggedIn(true);
      } catch (error) {
        console.error('Error loading saved session:', error);
        logout();
      }
    }
  }, []);

  const login = async (username: string, password: string): Promise<{ success: boolean; error?: string }> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    const account = TEST_ACCOUNTS.find(acc => 
      acc.username === username && acc.password === password
    );

    if (account) {
      const user: User = {
        username: account.username,
        email: account.email,
        role: account.role,
        subscription_tier: account.subscription_tier,
        loginTime: new Date().toISOString(),
        user_id: `user_${account.username}`
      };

      const token = `user_${account.username}_${Date.now()}`;

      setCurrentUser(user);
      setApiToken(token);
      setIsLoggedIn(true);
      
      localStorage.setItem('legalAssistantUser', JSON.stringify(user));
      localStorage.setItem('legalAssistantToken', token);

      return { success: true };
    }

    return { success: false, error: 'Invalid username or password. Please try one of the test accounts.' };
  };

  const logout = () => {
    setCurrentUser(null);
    setApiToken('');
    setIsLoggedIn(false);
    localStorage.removeItem('legalAssistantUser');
    localStorage.removeItem('legalAssistantToken');
  };

  return (
    <AuthContext.Provider value={{ isLoggedIn, currentUser, apiToken, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};