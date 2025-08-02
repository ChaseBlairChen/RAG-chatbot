// components/layout/TabNavigation.tsx
import React from 'react';

interface Tab {
  id: string;
  label: string;
  icon: string;
  badge?: number | null;
}

interface TabNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  userDocumentsCount: number;
  analysisResultsCount: number;
  isBackendConfigured: boolean;
}

export const TabNavigation: React.FC<TabNavigationProps> = ({
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
  ];

  return (
    <nav className="bg-white border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex space-x-8">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              disabled={!isBackendConfigured}
              className={`relative py-4 px-1 text-sm font-medium transition-all border-b-2 ${
                activeTab === tab.id
                  ? 'text-slate-900 border-slate-900'
                  : 'text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-300'
              } ${!isBackendConfigured ? 'cursor-not-allowed opacity-50' : ''}`}
            >
              <div className="flex items-center gap-2">
                <span className="text-base">{tab.icon}</span>
                <span>{tab.label}</span>
                {tab.badge && (
                  <span className="ml-1 bg-slate-900 text-white text-xs font-medium px-2 py-0.5 rounded-full">
                    {tab.badge}
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
};
