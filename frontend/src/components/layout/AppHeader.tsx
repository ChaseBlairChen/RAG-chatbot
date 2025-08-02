// components/layout/AppHeader.tsx
import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useBackend } from '../../contexts/BackendContext';
import { getSubscriptionBadgeClass } from '../../utils/helpers';

interface AppHeaderProps {
  sessionId?: string;
}

export const AppHeader: React.FC<AppHeaderProps> = ({ sessionId }) => {
  const { currentUser, logout } = useAuth();
  const { isBackendConfigured, backendCapabilities } = useBackend();

  return (
    <header className="bg-white shadow-sm border-b border-gray-100">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-stone-200 rounded-xl flex items-center justify-center shadow-sm">
              <svg className="w-7 h-7 text-stone-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">
                Legally
              </h1>
              <p className="text-xs text-gray-500">— powered by AI • Welcome, {currentUser?.username}</p>
            </div>
          </div>
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

            {/* Backend Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
              isBackendConfigured 
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                : 'bg-rose-50 text-rose-700 border border-rose-200'
            }`}>
              <div className={`w-2 h-2 rounded-full ${isBackendConfigured ? 'bg-emerald-500' : 'bg-rose-500'} animate-pulse`} />
              <span>
                {isBackendConfigured 
                  ? `Connected` 
                  : 'Disconnected'
                }
              </span>
            </div>

            {/* Enhanced RAG Status */}
            {isBackendConfigured && backendCapabilities.enhancedRag && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-stone-100 text-stone-700 border border-stone-200">
                <span>🧠</span>
                <span>Smart RAG</span>
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
            
            {sessionId && (
              <div className="text-xs text-gray-500 font-mono bg-gray-50 px-3 py-1.5 rounded-lg">
                Session: {sessionId.substring(0, 8)}
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
