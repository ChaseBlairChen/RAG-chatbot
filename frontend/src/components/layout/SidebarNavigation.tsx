// src/components/layout/SidebarNavigation.tsx
import React from 'react';

interface Tab {
  id: string;
  label: string;
  icon: string;
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
    { id: 'chat', label: 'Smart Chat', icon: '💬' },
    { id: 'upload', label: 'Upload & Analyze', icon: '📤' },
    { id: 'documents', label: 'My Documents', icon: '📁', badge: userDocumentsCount > 0 ? userDocumentsCount : null },
    { id: 'analysis', label: 'Analysis Tools', icon: '🔍' },
    { id: 'results', label: 'Results', icon: '📊', badge: analysisResultsCount > 0 ? analysisResultsCount : null },
    { id: 'legal-search', label: 'Legal Search', icon: '⚖️' },
    { id: 'immigration', label: 'Immigration', icon: '🗽' },
    { id: 'system-health', label: 'System Health', icon: '🏥' },
    { id: 'admin', label: 'Admin', icon: '🔧', adminOnly: true },
  ];

  return (
    <nav className="w-64 bg-white border-r border-gray-200 shadow-sm flex flex-col h-full">
      {/* Sidebar Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-stone-200 rounded-xl flex items-center justify-center shadow-sm">
            <svg className="w-6 h-6 text-stone-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Legally</h1>
            <p className="text-xs text-gray-500">— powered by AI</p>
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            disabled={!isBackendConfigured}
            className={`w-full flex items-center gap-3 px-3 py-3 text-left rounded-xl transition-all duration-200 group ${
              activeTab === tab.id
                ? 'bg-slate-900 text-white shadow-lg transform scale-[0.98]'
                : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
            } ${!isBackendConfigured ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} ${
              tab.adminOnly && activeTab !== tab.id ? 'hover:bg-red-50 hover:text-red-700' : ''
            }`}
          >
            <span className={`text-xl transition-transform duration-200 ${
              activeTab === tab.id ? 'scale-110' : 'group-hover:scale-105'
            }`}>
              {tab.icon}
            </span>
            
            <div className="flex-1 min-w-0">
              <span className={`text-sm font-medium block truncate ${
                tab.adminOnly && activeTab !== tab.id ? 'text-red-600' : ''
              }`}>
                {tab.label}
              </span>
              {tab.adminOnly && (
                <span className="text-xs text-red-500 opacity-75">Admin Only</span>
              )}
            </div>

            {tab.badge && (
              <span className={`text-xs font-medium px-2 py-1 rounded-full ${
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
          </button>
        ))}
      </div>

      {/* Backend Status Footer */}
      <div className="p-3 border-t border-gray-200">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${
          isBackendConfigured 
            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
            : 'bg-rose-50 text-rose-700 border border-rose-200'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            isBackendConfigured ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500 animate-pulse'
          }`} />
          <span className="font-medium">
            {isBackendConfigured ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
    </nav>
  );
};
