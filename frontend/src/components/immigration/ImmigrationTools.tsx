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
      
      // Clear the form on success
      if (data.success) {
        setFormData(prev => ({ ...prev, clientName: '' }));
      }
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
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <strong className="text-red-800">Error:</strong>
            <span className="text-red-700">{results.error}</span>
          </div>
        </div>
      );
    }

    // Case Creation Results - Enhanced formatting
    if (results.type === 'case_creation' && results.data) {
      return (
        <div className="space-y-4">
          {results.data.success ? (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h4 className="font-semibold text-green-900">✅ Case Created Successfully!</h4>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-3 border border-green-200">
                  <div className="text-sm text-green-700 font-medium">Case ID</div>
                  <div className="text-lg font-mono text-green-900">{results.data.case_id}</div>
                </div>
                <div className="bg-white rounded-lg p-3 border border-green-200">
                  <div className="text-sm text-green-700 font-medium">Status</div>
                  <div className="text-lg text-green-900">{results.data.message}</div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
                <h5 className="font-medium text-blue-900 mb-2">📋 Next Steps:</h5>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Document case details and client information</li>
                  <li>• Set up deadline tracking for key milestones</li>
                  <li>• Begin evidence collection process</li>
                  <li>• Schedule initial client consultation</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <strong className="text-red-800">Case Creation Failed</strong>
              </div>
              <p className="text-red-700 mt-2">{results.data.message || 'Unknown error occurred'}</p>
            </div>
          )}
        </div>
      );
    }

    // Country Conditions Research - Enhanced formatting
    if (results.type === 'country_conditions' && results.data?.research) {
      const research = results.data.research;
      return (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              🌍 Country: {research.country}
            </h4>
            <p className="text-sm text-blue-800">Research Date: {new Date(research.research_date).toLocaleString()}</p>
          </div>

          {research.summary && (
            <div className="bg-white border rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                📊 Executive Summary
              </h4>
              <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed">
                {research.summary.split('\n').map((line: string, idx: number) => (
                  line.trim() && <p key={idx} className="mb-3">{line}</p>
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
                📋 Detailed Analysis by Topic
              </h4>
              <div className="space-y-6">
                {Object.entries(research.topics).map(([topic, content]) => (
                  <div key={topic} className="border-l-4 border-blue-300 pl-6 bg-gray-50 rounded-r-lg p-4">
                    <h5 className="font-semibold text-gray-800 capitalize mb-3 text-lg flex items-center gap-2">
                      {topic === 'persecution' && '⚠️'}
                      {topic === 'government' && '🏛️'}
                      {topic === 'violence' && '⚡'}
                      {topic === 'human_rights' && '👥'}
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
                📚 Source Documents
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

    // Testimony Analysis Results
    if (results.type === 'testimony_analysis' && results.data) {
      return (
        <div className="space-y-4">
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h4 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              🗣️ Testimony Analysis Complete
            </h4>
          </div>

          {results.data.analysis && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">📝 Analysis</h4>
              <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-line">
                {results.data.analysis.analysis || results.data.analysis}
              </div>
            </div>
          )}

          {results.data.recommendations && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-900 mb-3">💡 Recommendations</h4>
              <ul className="space-y-1">
                {results.data.recommendations.map((rec: string, idx: number) => (
                  <li key={idx} className="text-sm text-green-800">• {rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    }

    // Deadlines Results
    if (results.type === 'deadlines' && results.data) {
      return (
        <div className="space-y-4">
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
            <h4 className="font-semibold text-orange-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              ⏰ Upcoming Deadlines ({results.data.total || 0})
            </h4>
          </div>

          {results.data.deadlines && results.data.deadlines.length > 0 ? (
            <div className="space-y-3">
              {results.data.deadlines.map((deadline: any, idx: number) => (
                <div key={idx} className={`p-4 rounded-lg border ${
                  deadline.priority === 'critical' ? 'bg-red-50 border-red-200' :
                  deadline.priority === 'high' ? 'bg-orange-50 border-orange-200' :
                  'bg-blue-50 border-blue-200'
                }`}>
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h5 className="font-medium text-gray-900">{deadline.description}</h5>
                      <p className="text-sm text-gray-600">
                        Case ID: {deadline.case_id} • Type: {deadline.deadline_type}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-medium text-gray-900">
                        {new Date(deadline.due_date).toLocaleDateString()}
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        deadline.priority === 'critical' ? 'bg-red-100 text-red-700' :
                        deadline.priority === 'high' ? 'bg-orange-100 text-orange-700' :
                        'bg-blue-100 text-blue-700'
                      }`}>
                        {deadline.priority} priority
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-500">
              <svg className="w-12 h-12 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p>No upcoming deadlines found</p>
            </div>
          )}
        </div>
      );
    }

    // Default JSON rendering for unknown types
    return (
      <div className="bg-gray-50 border rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-3">Raw Response</h4>
        <pre className="text-xs overflow-auto max-h-96 bg-white p-3 rounded border">
          {JSON.stringify(results.data, null, 2)}
        </pre>
      </div>
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
            className="px-3 py-2 border border-gray-200 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            className="px-3 py-2 border border-gray-200 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <button
          onClick={createCase}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition-all disabled:bg-gray-400 font-medium"
        >
          {loading ? 'Creating Case...' : 'Create Immigration Case'}
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
            placeholder="Country and topic (e.g., 'China LGBTQ', 'Afghanistan women')"
            className="flex-1 px-3 py-2 border border-gray-200 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
          />
          <button
            onClick={researchCountryConditions}
            disabled={loading}
            className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 transition-all disabled:bg-gray-400 font-medium"
          >
            {loading ? 'Researching...' : 'Research Conditions'}
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
      <div className="mb-6 p-4 bg-purple-50 rounded-lg">
        <h3 className="font-medium text-purple-900 mb-3">🗣️ Credible Fear Analysis</h3>
        <textarea
          value={formData.testimony}
          onChange={(e) => setFormData(prev => ({...prev, testimony: e.target.value}))}
          placeholder="Enter client testimony for analysis..."
          className="w-full px-3 py-2 border border-gray-200 rounded mb-4 h-24 focus:outline-none focus:ring-2 focus:ring-purple-500"
        />
        <button
          onClick={analyzeTestimony}
          disabled={loading || !formData.testimony || !formData.country}
          className="bg-purple-600 text-white px-6 py-2 rounded hover:bg-purple-700 transition-all disabled:bg-gray-400 font-medium"
        >
          {loading ? 'Analyzing...' : 'Analyze Testimony'}
        </button>
        <p className="text-xs text-purple-700 mt-2">
          ⚠️ Requires both country (from above) and testimony to be filled
        </p>
      </div>

      {/* Deadlines */}
      <div className="mb-6 p-4 bg-orange-50 rounded-lg">
        <h3 className="font-medium text-orange-900 mb-3">⏰ Deadline Management</h3>
        <button
          onClick={getUpcomingDeadlines}
          disabled={loading}
          className="bg-orange-600 text-white px-6 py-2 rounded hover:bg-orange-700 transition-all disabled:bg-gray-400 font-medium"
        >
          {loading ? 'Loading...' : 'Get Upcoming Deadlines'}
        </button>
      </div>

      {/* Results Display with Enhanced Formatting */}
      {results && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-900 capitalize flex items-center gap-2">
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              {results.type.replace('_', ' ')} Results
            </h3>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {results.timestamp}
            </span>
          </div>
          {renderResult()}
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 font-medium">Processing immigration request...</p>
          <p className="text-sm text-gray-500 mt-1">This may take a few moments</p>
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
