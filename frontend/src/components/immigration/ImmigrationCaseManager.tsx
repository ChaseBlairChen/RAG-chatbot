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
