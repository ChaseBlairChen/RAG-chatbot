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
