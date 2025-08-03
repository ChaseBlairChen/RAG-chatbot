// components/immigration/ImmigrationCaseManager.tsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

interface ImmigrationCase {
  case_id: string;
  case_type: string;
  client_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  deadlines: any[];
  forms: string[];
  notes?: string;
  language: string;
}

interface CaseDeadline {
  deadline_id: string;
  case_id: string;
  deadline_type: string;
  due_date: string;
  description: string;
  priority: string;
  completed: boolean;
}

export const ImmigrationCaseManager: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [loading, setLoading] = useState(false);
  const [cases, setCases] = useState<ImmigrationCase[]>([]);
  const [deadlines, setDeadlines] = useState<CaseDeadline[]>([]);
  const [selectedCase, setSelectedCase] = useState<string | null>(null);
  const [dashboardStats, setDashboardStats] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'cases' | 'deadlines' | 'create'>('dashboard');

  const [newCase, setNewCase] = useState({
    caseType: 'asylum',
    clientName: '',
    language: 'en',
    priorityDate: ''
  });

  const apiService = new ApiService(backendUrl, apiToken);

  // Load dashboard data
  const loadDashboard = async () => {
    setLoading(true);
    try {
      const stats = await apiService.get('/immigration/stats/dashboard');
      setDashboardStats(stats);
    } catch (error) {
      console.error('Failed to load dashboard:', error);
    }
    setLoading(false);
  };

  // Load deadlines
  const loadDeadlines = async () => {
    try {
      const data = await apiService.get('/immigration/deadlines/upcoming?days_ahead=30');
      setDeadlines(data.deadlines || []);
    } catch (error) {
      console.error('Failed to load deadlines:', error);
    }
  };

  // Create new case
  const createCase = async () => {
    if (!newCase.clientName.trim()) {
      alert('Please enter client name');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('case_type', newCase.caseType);
      formData.append('client_name', newCase.clientName);
      formData.append('language', newCase.language);
      if (newCase.priorityDate) {
        formData.append('priority_date', newCase.priorityDate);
      }

      const response = await fetch(`${backendUrl}/immigration/cases/create`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
        },
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        alert(`Case created successfully! Case ID: ${data.case_id}`);
        setNewCase({ caseType: 'asylum', clientName: '', language: 'en', priorityDate: '' });
        loadDashboard();
        setActiveTab('dashboard');
      } else {
        throw new Error(data.message || 'Failed to create case');
      }
    } catch (error) {
      console.error('Failed to create case:', error);
      alert(`Failed to create case: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
    setLoading(false);
  };

  // Generate evidence checklist
  const generateChecklist = async (caseType: string) => {
    try {
      const formData = new FormData();
      formData.append('case_type', caseType);

      const response = await fetch(`${backendUrl}/immigration/evidence/checklist`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
        },
        body: formData,
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Failed to generate checklist:', error);
      return null;
    }
  };

  useEffect(() => {
    loadDashboard();
    loadDeadlines();
  }, []);

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'normal': return 'bg-blue-100 text-blue-800 border-blue-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getCaseTypeIcon = (caseType: string) => {
    switch (caseType) {
      case 'asylum': return '🛡️';
      case 'family_based': return '👨‍👩‍👧‍👦';
      case 'employment_based': return '💼';
      case 'removal_defense': return '⚖️';
      case 'naturalization': return '🇺🇸';
      default: return '📄';
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h2 className="text-2xl font-semibold text-gray-900">Immigration Case Management</h2>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-gray-100 rounded-lg p-1">
        {[
          { id: 'dashboard', label: 'Dashboard', icon: '📊' },
          { id: 'create', label: 'Create Case', icon: '➕' },
          { id: 'deadlines', label: 'Deadlines', icon: '⏰' },
          { id: 'cases', label: 'All Cases', icon: '📋' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {dashboardStats && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-2xl font-bold text-blue-900">{dashboardStats.total_cases}</div>
                <div className="text-sm text-blue-700">Total Cases</div>
              </div>
              <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                <div className="text-2xl font-bold text-orange-900">{dashboardStats.pending_deadlines}</div>
                <div className="text-sm text-orange-700">Pending Deadlines</div>
              </div>
              <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                <div className="text-2xl font-bold text-red-900">{dashboardStats.critical_deadlines}</div>
                <div className="text-sm text-red-700">Critical Deadlines</div>
              </div>
              <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                <div className="text-2xl font-bold text-green-900">{dashboardStats.documents_pending_translation}</div>
                <div className="text-sm text-green-700">Pending Translations</div>
              </div>
            </div>
          )}

          {/* Cases by Type */}
          {dashboardStats?.by_type && (
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3">Cases by Type</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {Object.entries(dashboardStats.by_type).map(([type, count]) => (
                  <div key={type} className="text-center p-3 bg-white rounded border">
                    <div className="text-2xl mb-1">{getCaseTypeIcon(type)}</div>
                    <div className="font-semibold text-gray-900">{count as number}</div>
                    <div className="text-xs text-gray-600 capitalize">{type.replace('_', ' ')}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Deadlines */}
          {deadlines.length > 0 && (
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3">⏰ Upcoming Deadlines</h3>
              <div className="space-y-2">
                {deadlines.slice(0, 5).map((deadline, idx) => (
                  <div key={idx} className={`p-3 rounded border ${getPriorityColor(deadline.priority)}`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium">{deadline.description}</div>
                        <div className="text-sm opacity-75">Case ID: {deadline.case_id}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          {new Date(deadline.due_date).toLocaleDateString()}
                        </div>
                        <div className="text-xs capitalize">{deadline.priority} priority</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Create Case Tab */}
      {activeTab === 'create' && (
        <div className="space-y-6">
          <div className="p-6 bg-gray-50 rounded-lg">
            <h3 className="font-medium text-gray-900 mb-4">Create New Immigration Case</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Case Type</label>
                <select
                  value={newCase.caseType}
                  onChange={(e) => setNewCase(prev => ({...prev, caseType: e.target.value}))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="asylum">Asylum</option>
                  <option value="family_based">Family Based</option>
                  <option value="employment_based">Employment Based</option>
                  <option value="removal_defense">Removal Defense</option>
                  <option value="naturalization">Naturalization</option>
                  <option value="humanitarian">Humanitarian</option>
                  <option value="investor">Investor</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Client Name</label>
                <input
                  type="text"
                  value={newCase.clientName}
                  onChange={(e) => setNewCase(prev => ({...prev, clientName: e.target.value}))}
                  placeholder="Enter client full name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Language</label>
                <select
                  value={newCase.language}
                  onChange={(e) => setNewCase(prev => ({...prev, language: e.target.value}))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="zh">Chinese</option>
                  <option value="ar">Arabic</option>
                  <option value="fr">French</option>
                  <option value="hi">Hindi</option>
                  <option value="pt">Portuguese</option>
                  <option value="ru">Russian</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Priority Date (Optional)</label>
                <input
                  type="date"
                  value={newCase.priorityDate}
                  onChange={(e) => setNewCase(prev => ({...prev, priorityDate: e.target.value}))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <button
              onClick={createCase}
              disabled={loading || !newCase.clientName.trim()}
              className="w-full md:w-auto bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-all font-medium"
            >
              {loading ? 'Creating Case...' : 'Create Immigration Case'}
            </button>
          </div>

          {/* Evidence Checklist Generator */}
          <div className="p-6 bg-yellow-50 rounded-lg border border-yellow-200">
            <h3 className="font-medium text-yellow-900 mb-4">📋 Evidence Checklist Generator</h3>
            <p className="text-sm text-yellow-800 mb-4">
              Generate a customized evidence checklist based on case type to ensure all required documents are collected.
            </p>
            <button
              onClick={async () => {
                const checklist = await generateChecklist(newCase.caseType);
                if (checklist) {
                  alert(`Checklist generated for ${newCase.caseType} case:\n\n${Object.keys(checklist.checklist).join('\n• ')}`);
                }
              }}
              className="bg-yellow-600 text-white px-4 py-2 rounded hover:bg-yellow-700 transition-all"
            >
              Generate {newCase.caseType.replace('_', ' ')} Evidence Checklist
            </button>
          </div>
        </div>
      )}

      {/* Deadlines Tab */}
      {activeTab === 'deadlines' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-medium text-gray-900">Upcoming Deadlines (30 days)</h3>
            <button
              onClick={loadDeadlines}
              className="text-blue-600 hover:text-blue-700 text-sm font-medium"
            >
              Refresh
            </button>
          </div>

          {deadlines.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p>No upcoming deadlines</p>
            </div>
          ) : (
            <div className="space-y-3">
              {deadlines.map((deadline, idx) => (
                <div key={idx} className={`p-4 rounded-lg border ${getPriorityColor(deadline.priority)}`}>
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h4 className="font-medium mb-1">{deadline.description}</h4>
                      <p className="text-sm opacity-75">
                        Case ID: {deadline.case_id} • Type: {deadline.deadline_type}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">
                        {new Date(deadline.due_date).toLocaleDateString()}
                      </div>
                      <div className="text-sm">
                        {Math.ceil((new Date(deadline.due_date).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))} days
                      </div>
                      <span className="text-xs px-2 py-1 rounded-full bg-white bg-opacity-50 capitalize">
                        {deadline.priority}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Cases Tab */}
      {activeTab === 'cases' && (
        <div className="space-y-4">
          <h3 className="font-medium text-gray-900">All Cases</h3>
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p>Case listing feature coming soon</p>
            <p className="text-sm">Will display all created cases with status tracking</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Processing...</p>
        </div>
      )}

      {/* Immigration Law Notice */}
      <div className="mt-8 p-4 bg-amber-50 border border-amber-200 rounded-lg">
        <div className="flex items-start gap-2">
          <svg className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div>
            <h4 className="font-medium text-amber-800 mb-1">⚖️ Legal Disclaimer</h4>
            <p className="text-sm text-amber-700">
              This case management system is for organizational purposes only. Immigration law is complex 
              and changes frequently. Always consult with a qualified immigration attorney for legal advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ==================== Updated Immigration Tab Navigation ====================
// components/immigration/ImmigrationTools.tsx (Replace the existing file)
import React, { useState } from 'react';
import { ImmigrationCaseManager } from './ImmigrationCaseManager';
import { ImmigrationResearch } from './ImmigrationResearch';

export const ImmigrationTools: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'cases' | 'research'>('cases');

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
          </div>
        </div>
        
        <p className="text-gray-600">
          {activeSection === 'cases' 
            ? 'Manage immigration cases, track deadlines, and monitor case progress'
            : 'Research country conditions and analyze testimonies for asylum cases'
          }
        </p>
      </div>

      {/* Content */}
      {activeSection === 'cases' && <ImmigrationCaseManager />}
      {activeSection === 'research' && <ImmigrationResearch />}
    </div>
  );
};

// ==================== Immigration Research Component ====================
// components/immigration/ImmigrationResearch.tsx (Create this new file)
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const ImmigrationResearch: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [formData, setFormData] = useState({
    country: '',
    testimony: ''
  });

  const apiService = new ApiService(backendUrl, apiToken);

  const researchCountryConditions = async () => {
    if (!formData.country) {
      alert('Please enter country name');
      return;
    }

    setLoading(true);
    try {
      const requestData = {
        country: formData.country,
        topics: ['persecution', 'human_rights', 'government', 'violence'],
        date_range: 'last_2_years'
      };

      const data = await apiService.post('/immigration/country-conditions/research', requestData);
      setResults({ type: 'country_conditions', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'country_conditions', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  const analyzeTestimony = async () => {
    if (!formData.testimony || !formData.country) {
      alert('Please enter both testimony and country');
      return;
    }

    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('testimony', formData.testimony);
      postData.append('country', formData.country);

      const response = await fetch(`${backendUrl}/immigration/credible-fear/analyze`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
        },
        body: postData,
      });

      const data = await response.json();
      setResults({ type: 'testimony_analysis', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'testimony_analysis', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  // Enhanced result rendering (same as before but extracted to separate component)
  const renderResult = () => {
    if (!results) return null;

    if (results.error) {
      return (
        <div className="text-red-600 text-sm">
          <strong>Error:</strong> {results.error}
        </div>
      );
    }

    // Country Conditions Research - Enhanced formatting
    if (results.type === 'country_conditions' && results.data?.research) {
      const research = results.data.research;
      return (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">🌍 Country: {research.country}</h4>
            <p className="text-sm text-blue-800">Research Date: {new Date(research.research_date).toLocaleString()}</p>
          </div>

          {research.summary && (
            <div className="bg-white border rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Executive Summary
              </h4>
              <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed">
                {research.summary.split('\n').map((line: string, idx: number) => (
                  <p key={idx} className="mb-3">{line}</p>
                ))}
              </div>
            </div>
          )}

          {research.topics && Object.keys(research.topics).length > 0 && (
            <div className="bg-white border rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                Detailed Analysis by Topic
              </h4>
              <div className="space-y-6">
                {Object.entries(research.topics).map(([topic, content]) => (
                  <div key={topic} className="border-l-4 border-blue-300 pl-6 bg-gray-50 rounded-r-lg p-4">
                    <h5 className="font-semibold text-gray-800 capitalize mb-3 text-lg">
                      {topic === 'persecution' && '⚠️'}
                      {topic === 'government' && '🏛️'}
                      {topic === 'violence' && '⚡'}
                      {topic === 'human_rights' && '👥'}
                      {' '}
                      {topic.replace('_', ' ')}
                    </h5>
                    <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
                      {content as string}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {research.sources && research.sources.length > 0 && (
            <div className="bg-gray-50 border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                Source Documents
              </h4>
              <div className="space-y-2">
                {research.sources.map((source: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-sm text-gray-600 bg-white rounded p-3">
                    <div>
                      <span className="font-medium">{source.file_name}</span>
                      {source.page && <span className="ml-2 text-gray-500">(Page {source.page})</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                        {Math.round(source.relevance * 100)}% relevant
                      </span>
                      <span className="text-xs text-gray-500 capitalize">
                        {source.source_type.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Default rendering for other result types
    return (
      <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
        {JSON.stringify(results.data, null, 2)}
      </pre>
    );
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">Immigration Research Tools</h2>

      {/* Country Conditions Research */}
      <div className="mb-6 p-6 bg-green-50 rounded-lg border border-green-200">
        <h3 className="font-medium text-green-900 mb-4 flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Country Conditions Research
        </h3>
        <p className="text-sm text-green-800 mb-4">
          Research current human rights conditions, government persecution, and violence in specific countries for asylum cases.
        </p>
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={formData.country}
            onChange={(e) => setFormData(prev => ({...prev, country: e.target.value}))}
            placeholder="Country and topic (e.g., 'China LGBTQ', 'Afghanistan women')"
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
          />
          <button
            onClick={researchCountryConditions}
            disabled={loading || !formData.country}
            className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 transition-all disabled:bg-gray-400 font-medium"
          >
            {loading ? 'Researching...' : 'Research'}
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {['China LGBTQ', 'Afghanistan women', 'Myanmar military', 'Venezuela economy', 'Syria conflict'].map(example => (
            <button
              key={example}
              onClick={() => setFormData(prev => ({...prev, country: example}))}
              className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded hover:bg-green-200 transition-all"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Credible Fear Analysis */}
      <div className="mb-6 p-6 bg-purple-50 rounded-lg border border-purple-200">
        <h3 className="font-medium text-purple-900 mb-4 flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          Credible Fear Testimony Analysis
        </h3>
        <p className="text-sm text-purple-800 mb-4">
          Analyze client testimony for consistency, completeness, and strength for credible fear interviews.
        </p>
        <textarea
          value={formData.testimony}
          onChange={(e) => setFormData(prev => ({...prev, testimony: e.target.value}))}
          placeholder="Enter client testimony for analysis..."
          className="w-full px-3 py-2 border border-gray-300 rounded-md mb-4 h-32 focus:outline-none focus:ring-2 focus:ring-purple-500"
        />
        <button
          onClick={analyzeTestimony}
          disabled={loading || !formData.testimony || !formData.country}
          className="bg-purple-600 text-white px-6 py-2 rounded-md hover:bg-purple-700 transition-all disabled:bg-gray-400 font-medium"
        >
          {loading ? 'Analyzing...' : 'Analyze Testimony'}
        </button>
        <p className="text-xs text-purple-700 mt-2">
          ⚠️ Requires both country (from above) and testimony to be filled
        </p>
      </div>

      {/* Results Display with Enhanced Formatting */}
      {results && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-900 capitalize">
              {results.type.replace('_', ' ')} Results
            </h3>
            <span className="text-xs text-gray-500">{results.timestamp}</span>
          </div>
          {renderResult()}
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Processing immigration research...</p>
        </div>
      )}
    </div>
  );
};
