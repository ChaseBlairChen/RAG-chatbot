// src/components/layout/SidebarNavigation.tsx
import React from 'react';

interface Tab {
  id: string;
  label: string;
  icon: string;
  description: string;
  badge?: number | null;
  adminOnly?: boolean;
}

interface SidebarNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  userDocumentsCount: number;
  analysisResultsCount: number;
  isBackendConfigured: boolean;
}

export const SidebarNavigation: React.FC<SidebarNavigationProps> = ({
  activeTab,
  setActiveTab,
  userDocumentsCount,
  analysisResultsCount,
  isBackendConfigured
}) => {
  const tabs: Tab[] = [
    { 
      id: 'chat', 
      label: 'Smart Chat', 
      icon: '💬',
      description: 'AI-powered legal research'
    },
    { 
      id: 'upload', 
      label: 'Upload & Analyze', 
      icon: '📤',
      description: 'Process legal documents'
    },
    { 
      id: 'documents', 
      label: 'My Documents', 
      icon: '📁', 
      description: 'Manage your files',
      badge: userDocumentsCount > 0 ? userDocumentsCount : null 
    },
    { 
      id: 'analysis', 
      label: 'Analysis Tools', 
      icon: '🔍',
      description: 'Advanced document analysis'
    },
    { 
      id: 'results', 
      label: 'Results', 
      icon: '📊', 
      description: 'View analysis results',
      badge: analysisResultsCount > 0 ? analysisResultsCount : null 
    },
    { 
      id: 'legal-search', 
      label: 'Legal Search', 
      icon: '⚖️',
      description: 'Search legal databases'
    },
    { 
      id: 'immigration', 
      label: 'Immigration', 
      icon: '🗽',
      description: 'Immigration case tools'
    },
    { 
      id: 'system-health', 
      label: 'System Health', 
      icon: '🏥',
      description: 'Monitor system status'
    },
    { 
      id: 'admin', 
      label: 'Admin', 
      icon: '🔧', 
      description: 'System administration',
      adminOnly: true 
    },
  ];

  return (
    <nav className="w-72 bg-white/80 backdrop-blur-xl border-r border-slate-200/50 shadow-soft flex flex-col h-full">
      {/* Sidebar Header */}
      <div className="p-8 border-b border-slate-200/50">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl flex items-center justify-center shadow-medium">
            <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Legally</h1>
            <p className="text-sm text-slate-500 font-medium">— powered by AI</p>
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 py-8 px-6 space-y-2 overflow-y-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            disabled={!isBackendConfigured}
            className={`w-full group relative overflow-hidden ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-slate-900 to-slate-800 text-white shadow-strong'
                : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900'
            } ${!isBackendConfigured ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} ${
              tab.adminOnly && activeTab !== tab.id ? 'hover:bg-red-50 hover:text-red-700' : ''
            } rounded-2xl p-4 transition-all duration-300`}
          >
            {/* Active indicator */}
            {activeTab === tab.id && (
              <div className="absolute inset-0 bg-gradient-to-r from-slate-900 to-slate-800 opacity-10" />
            )}
            
            <div className="relative flex items-center gap-4">
              <span className={`text-2xl transition-all duration-300 ${
                activeTab === tab.id ? 'scale-110' : 'group-hover:scale-105'
              }`}>
                {tab.icon}
              </span>
              
              <div className="flex-1 min-w-0 text-left">
                <div className={`font-semibold text-sm ${
                  tab.adminOnly && activeTab !== tab.id ? 'text-red-600' : ''
                }`}>
                  {tab.label}
                </div>
                <div className={`text-xs mt-1 ${
                  activeTab === tab.id ? 'text-slate-300' : 'text-slate-500'
                }`}>
                  {tab.description}
                </div>
                {tab.adminOnly && (
                  <div className="text-xs text-red-500 opacity-75 mt-1">Admin Only</div>
                )}
              </div>

              {tab.badge && (
                <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                  activeTab === tab.id
                    ? 'bg-white text-slate-900'
                    : 'bg-slate-900 text-white'
                }`}>
                  {tab.badge}
                </span>
              )}

              {tab.adminOnly && (
                <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">
                  🔒
                </span>
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Backend Status Footer */}
      <div className="p-6 border-t border-slate-200/50">
        <div className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium ${
          isBackendConfigured 
            ? 'bg-green-50 text-green-700 border border-green-200' 
            : 'bg-red-50 text-red-700 border border-red-200'
        }`}>
          <div className={`w-3 h-3 rounded-full ${
            isBackendConfigured ? 'bg-green-500 animate-pulse' : 'bg-red-500 animate-pulse'
          }`} />
          <span>
            {isBackendConfigured ? 'Backend Connected' : 'Backend Disconnected'}
          </span>
        </div>
      </div>
    </nav>
  );
};
