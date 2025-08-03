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
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-all"
        >
          Create Immigration Case
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
            placeholder="Country (e.g., Afghanistan, Myanmar)"
            className="flex-1 px-3 py-2 border border-gray-200 rounded"
          />
          <button
            onClick={researchCountryConditions}
            disabled={loading}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-all"
          >
            Research Conditions
          </button>
        </div>
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
          className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-all"
        >
          Analyze Testimony
        </button>
      </div>

      {/* Deadlines */}
      <div className="mb-6 p-4 bg-orange-50 rounded-lg">
        <h3 className="font-medium text-orange-900 mb-3">⏰ Deadline Management</h3>
        <button
          onClick={getUpcomingDeadlines}
          disabled={loading}
          className="bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700 transition-all"
        >
          Get Upcoming Deadlines
        </button>
      </div>

      {/* Results Display */}
      {results && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900">Results: {results.type}</h3>
            <span className="text-xs text-gray-500">{results.timestamp}</span>
          </div>
          {results.error ? (
            <div className="text-red-600 text-sm">
              <strong>Error:</strong> {results.error}
            </div>
          ) : (
            <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
              {JSON.stringify(results.data, null, 2)}
            </pre>
          )}
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Processing immigration request...</p>
        </div>
      )}
    </div>
  );
};
