// components/immigration/ImmigrationTools.tsx
import React, { useState } from 'react';
import { ImmigrationCaseManager } from './ImmigrationCaseManager';
import { ImmigrationResearch } from './ImmigrationResearch';
import { AutonomousResearchAgent } from './AutonomousResearchAgent';

export const ImmigrationTools: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'cases' | 'research' | 'agent'>('cases');

  return (
    <div className="space-y-6">
      {/* Section Navigation */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">Immigration Law Center</h2>
          <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveSection('cases')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeSection === 'cases'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              📋 Case Management
            </button>
            <button
              onClick={() => setActiveSection('research')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeSection === 'research'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              🌍 Research Tools
            </button>
            <button
              onClick={() => setActiveSection('agent')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeSection === 'agent'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              🤖 AI Agent
            </button>
          </div>
        </div>
        
        <p className="text-gray-600">
          {activeSection === 'cases' && 'Manage immigration cases, track deadlines, and monitor case progress'}
          {activeSection === 'research' && 'Research country conditions and analyze testimonies for asylum cases'}
          {activeSection === 'agent' && 'Autonomous AI agent that continuously researches human rights conditions'}
        </p>
      </div>

      {/* Content */}
      {activeSection === 'cases' && <ImmigrationCaseManager />}
      {activeSection === 'research' && <ImmigrationResearch />}
      {activeSection === 'agent' && <AutonomousResearchAgent />}
    </div>
  );
};
