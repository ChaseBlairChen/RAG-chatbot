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
    <header className="glass-effect border-b border-slate-200/50">
      <div className="px-8 py-6">
        <div className="flex items-center justify-between">
          {/* Left side - Welcome message */}
          <div className="space-y-1">
            <h2 className="text-2xl font-bold text-slate-900">
              Welcome back, {currentUser?.username}
            </h2>
            <p className="text-slate-600 font-medium">
              Your AI-powered legal research assistant
            </p>
          </div>

          {/* Right side - User info and controls */}
          <div className="flex items-center gap-4">
            {/* Status Indicators */}
            <div className="flex items-center gap-3">
              {/* Connection Status */}
              <div className="flex items-center gap-2 px-3 py-2 bg-green-50 rounded-xl border border-green-200">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-sm font-medium text-green-700">Connected</span>
              </div>

              {/* Enhanced RAG Status */}
              {backendCapabilities.enhancedRag && (
                <div className="flex items-center gap-2 px-3 py-2 bg-blue-50 rounded-xl border border-blue-200">
                  <span className="text-blue-600">🧠</span>
                  <span className="text-sm font-medium text-blue-700">Smart RAG</span>
                </div>
              )}
            </div>

            {/* User Profile */}
            <div className="flex items-center gap-3 px-4 py-2 bg-slate-50 rounded-xl border border-slate-200">
              <div className="w-8 h-8 bg-gradient-to-br from-slate-600 to-slate-800 rounded-full flex items-center justify-center">
                <span className="text-white font-semibold text-sm">
                  {currentUser?.username.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-semibold text-slate-900">{currentUser?.username}</span>
                <span className="text-xs text-slate-500">{currentUser?.email}</span>
              </div>
            </div>

            {/* Subscription Badge */}
            <div className={`px-4 py-2 rounded-xl text-sm font-semibold ${getSubscriptionBadgeClass(currentUser?.subscription_tier || 'free')}`}>
              {currentUser?.subscription_tier.toUpperCase()}
              {currentUser?.subscription_tier === 'premium' && (
                <span className="ml-2" title="Has access to external legal databases">🔗</span>
              )}
            </div>
            
            {/* Session Info */}
            {sessionId && (
              <div className="text-xs text-slate-500 font-mono bg-slate-100 px-3 py-2 rounded-lg border border-slate-200">
                Session: {sessionId.substring(0, 8)}
              </div>
            )}

            {/* Logout Button */}
            <button
              onClick={logout}
              className="p-3 bg-slate-100 hover:bg-slate-200 rounded-xl transition-all duration-200 text-slate-600 hover:text-slate-700 hover:shadow-soft"
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
