// components/immigration/ImmigrationTools.tsx
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const ImmigrationTools: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [formData, setFormData] = useState({
    caseType: 'asylum',
    clientName: '',
    country: '',
    testimony: ''
  });

  const apiService = new ApiService(backendUrl, apiToken);

  const createCase = async () => {
    if (!formData.clientName) {
      alert('Please enter client name');
      return;
    }

    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('case_type', formData.caseType);
      postData.append('client_name', formData.clientName);
      postData.append('language', 'en');

      const response = await fetch(`${backendUrl}/immigration/cases/create`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
        },
        body: postData,
      });

      const data = await response.json();
      setResults({ type: 'case_creation', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'case_creation', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

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

  const getUpcomingDeadlines = async () => {
    setLoading(true);
    try {
      const data = await apiService.get('/immigration/deadlines/upcoming?days_ahead=30');
      setResults({ type: 'deadlines', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'deadlines', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  // Enhanced result rendering function
  const renderResult = () => {
    if (!results) return null;

    if (results.error) {
      return (
        <div className="text-red-600 text-sm">
          <strong>Error:</strong> {results.error}
        </div>
      );
    }

    // Country Conditions Research - Special formatting
    if (results.type === 'country_conditions' && results.data?.research) {
      const research = results.data.research;
      return (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">Country: {research.country}</h4>
            <p className="text-sm text-blue-800">Research Date: {new Date(research.research_date).toLocaleString()}</p>
          </div>

          {research.summary && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">📊 Executive Summary</h4>
              <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-line">
                {research.summary}
              </div>
            </div>
          )}

          {research.topics && Object.keys(research.topics).length > 0 && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">📋 Topics Analysis</h4>
              <div className="space-y-4">
                {Object.entries(research.topics).map(([topic, content]) => (
                  <div key={topic} className="border-l-4 border-blue-300 pl-4">
                    <h5 className="font-medium text-gray-800 capitalize mb-2">{topic}</h5>
                    <div className="text-sm text-gray-600 whitespace-pre-line">
                      {content as string}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {research.sources && research.sources.length > 0 && (
            <div className="bg-gray-50 border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">📚 Sources</h4>
              <div className="space-y-2">
                {research.sources.map((source: any, idx: number) => (
                  <div key={idx} className="text-sm text-gray-600">
                    • {source.file_name} 
                    {source.page && ` (Page ${source.page})`}
                    {source.relevance && ` - ${Math.round(source.relevance * 100)}% relevant`}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Default JSON rendering for other types
    return (
      <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
        {JSON.stringify(results.data, null, 2)}
      </pre>
    );
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 className="text-2xl font-semibold text-gray-900">Immigration Law Tools</h2>
        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">SPECIALIZED</span>
      </div>

      {/* Case Management */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-medium text-blue-900 mb-3">📋 Case Management</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <select
            value={formData.caseType}
            onChange={(e) => setFormData(prev => ({...prev, caseType: e.target.value}))}
            className="px-3 py-2 border border-gray-200 rounded"
          >
            <option value="asylum">Asylum</option>
            <option value="family_based">Family Based</option>
            <option value="employment_based">Employment Based</option>
            <option value="removal_defense">Removal Defense</option>
            <option value="naturalization">Naturalization</option>
          </select>
          <input
            type="text"
            value={formData.clientName}
            onChange={(e) => setFormData(prev => ({...prev, clientName: e.target.value}))}
            placeholder="Client Name"
            className="px-3 py-2 border border-gray-200 rounded"
          />
        </div>
        <button
          onClick={createCase}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-all disabled:bg-gray-400"
        >
          {loading ? 'Creating...' : 'Create Immigration Case'}
        </button>
      </div>

      {/* Country Conditions Research */}
      <div className="mb-6 p-4 bg-green-50 rounded-lg">
        <h3 className="font-medium text-green-900 mb-3">🌍 Country Conditions Research</h3>
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={formData.country}
            onChange={(e) => setFormData(prev => ({...prev, country: e.target.value}))}
            placeholder="Country (e.g., China LGBTQ, Afghanistan women)"
            className="flex-1 px-3 py-2 border border-gray-200 rounded"
          />
          <button
            onClick={researchCountryConditions}
            disabled={loading}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-all disabled:bg-gray-400"
          >
            {loading ? 'Researching...' : 'Research Conditions'}
          </button>
        </div>
        <p className="text-xs text-green-700">
          💡 Try: "China LGBTQ", "Afghanistan women", "Myanmar military", "Venezuela economy"
        </p>
      </div>

      {/* Credible Fear Analysis */}
      <div className="mb-6 p-4 bg-purple-50 rounded-lg">
        <h3 className="font-medium text-purple-900 mb-3">🗣️ Credible Fear Analysis</h3>
        <textarea
          value={formData.testimony}
          onChange={(e) => setFormData(prev => ({...prev, testimony: e.target.value}))}
          placeholder="Enter client testimony for analysis..."
          className="w-full px-3 py-2 border border-gray-200 rounded mb-4 h-24"
        />
        <button
          onClick={analyzeTestimony}
          disabled={loading || !formData.testimony || !formData.country}
          className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-all disabled:bg-gray-400"
        >
          {loading ? 'Analyzing...' : 'Analyze Testimony'}
        </button>
        <p className="text-xs text-purple-700 mt-2">
          ⚠️ Requires both country and testimony to be filled
        </p>
      </div>

      {/* Deadlines */}
      <div className="mb-6 p-4 bg-orange-50 rounded-lg">
        <h3 className="font-medium text-orange-900 mb-3">⏰ Deadline Management</h3>
        <button
          onClick={getUpcomingDeadlines}
          disabled={loading}
          className="bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700 transition-all disabled:bg-gray-400"
        >
          {loading ? 'Loading...' : 'Get Upcoming Deadlines'}
        </button>
      </div>

      {/* Results Display with Enhanced Formatting */}
      {results && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900 capitalize">
              {results.type.replace('_', ' ')} Results
            </h3>
            <span className="text-xs text-gray-500">{results.timestamp}</span>
          </div>
          {renderResult()}
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Processing immigration request...</p>
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
              This information is for research purposes only and does not constitute legal advice. 
              Immigration law is complex and changes frequently. Always consult with a qualified 
              immigration attorney for advice specific to individual cases.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
