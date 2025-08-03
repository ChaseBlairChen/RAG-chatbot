// components/legal/LegalDatabaseSearch.tsx
import React, { useState } from 'react';
import { useBackend } from '../../contexts/BackendContext';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';

export const LegalDatabaseSearch: React.FC = () => {
  const { backendUrl } = useBackend();
  const { apiToken, currentUser } = useAuth();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [debugInfo, setDebugInfo] = useState<any>(null);
  
  const apiService = new ApiService(backendUrl, apiToken);

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError('');
    setResults([]);
    setDebugInfo(null);
    
    try {
      console.log(`🔍 Searching for: "${query}"`);
      console.log(`📡 Backend URL: ${backendUrl}`);
      console.log(`🔑 API Token: ${apiToken.substring(0, 20)}...`);
      
      // Try the search
      const response = await apiService.searchFreeLegalDatabases(query);
      
      console.log('✅ Full backend response:', response);
      setDebugInfo(response);
      
      // Extract results - handle different response formats
      let searchResults = [];
      
      if (response.results && Array.isArray(response.results)) {
        searchResults = response.results;
      } else if (Array.isArray(response)) {
        searchResults = response;
      } else if (response.data && Array.isArray(response.data)) {
        searchResults = response.data;
      }
      
      console.log(`📊 Extracted ${searchResults.length} results:`, searchResults);
      setResults(searchResults);
      
      if (searchResults.length === 0) {
        setError('No results found. The external legal databases may be unavailable or your query needs refinement.');
      }
      
    } catch (error) {
      console.error('❌ Search failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown search error';
      setError(`Search failed: ${errorMessage}`);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const testQueries = [
    'miranda rights',
    'contract breach',
    'negligence',
    'first amendment',
    'Brown v Board'
  ];

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Legal Database Search</h2>
        <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
          FREE DATABASES
        </span>
      </div>
      
      <p className="text-gray-600 mb-4">
        Search through millions of cases from Harvard Caselaw, CourtListener, and Federal Register
      </p>

      {/* Search Input */}
      <div className="flex gap-3 mb-6">
        <input 
          type="text"
          value={query} 
          onChange={(e) => {
            setQuery(e.target.value);
            setError('');
          }}
          onKeyPress={handleKeyPress}
          placeholder="Search cases... (e.g., 'miranda rights', 'contract breach')"
          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500"
          disabled={loading}
        />
        <button 
          onClick={handleSearch} 
          disabled={loading || !query.trim()}
          className="bg-slate-900 text-white px-6 py-3 rounded-xl hover:bg-slate-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-medium"
        >
          {loading ? (
            <div className="flex items-center">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              Searching...
            </div>
          ) : 'Search'}
        </button>
      </div>

      {/* Test Queries */}
      <div className="mb-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
        <h4 className="font-medium text-yellow-900 mb-2">🧪 Try These Test Queries:</h4>
        <div className="flex flex-wrap gap-2">
          {testQueries.map(testQuery => (
            <button
              key={testQuery}
              onClick={() => {
                setQuery(testQuery);
                setError('');
              }}
              className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded hover:bg-yellow-200 transition-all"
            >
              {testQuery}
            </button>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Debug Info */}
      {debugInfo && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-2">🔧 Debug Info (Raw Backend Response):</h4>
          <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-40">
            {JSON.stringify(debugInfo, null, 2)}
          </pre>
        </div>
      )}
      
      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">
              Found {results.length} results
            </h3>
            <span className="text-sm text-gray-500">
              Free legal databases
            </span>
          </div>
          
          {results.map((result, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-semibold text-gray-900 flex-1 pr-4">
                  {result.title || result.name || 'Untitled Document'}
                </h4>
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full whitespace-nowrap">
                  {result.source_database?.replace('_', ' ').toUpperCase() || result.source || 'Unknown Source'}
                </span>
              </div>
              
              <div className="text-sm text-gray-600 mb-3 space-y-1">
                {result.court && (
                  <div><strong>Court:</strong> {result.court}</div>
                )}
                {result.date && (
                  <div><strong>Date:</strong> {new Date(result.date).toLocaleDateString()}</div>
                )}
                {result.citation && (
                  <div><strong>Citation:</strong> {result.citation}</div>
                )}
                {result.jurisdiction && (
                  <div><strong>Jurisdiction:</strong> {result.jurisdiction}</div>
                )}
              </div>
              
              {(result.preview || result.snippet || result.description) && (
                <p className="text-gray-700 mb-3 text-sm leading-relaxed">
                  {result.preview || result.snippet || result.description}
                </p>
              )}
              
              <div className="flex items-center gap-4 pt-2 border-t border-gray-100">
                {result.url && (
                  <a 
                    href={result.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-700 font-medium text-sm flex items-center gap-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    View Full Document
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* No Results State */}
      {results.length === 0 && query && !loading && !error && (
        <div className="text-center py-8 text-gray-500">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <p className="text-lg font-medium mb-2">No results found</p>
          <p className="text-sm">Try different search terms or check the debug info above</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Searching legal databases...</p>
        </div>
      )}

      {/* Search Tips */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">💡 Search Tips:</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use specific legal terms: "miranda rights", "contract breach"</li>
            <li>• Include case names: "Brown v. Board"</li>
            <li>• Try constitutional terms: "first amendment", "due process"</li>
            <li>• Search tort law: "negligence", "liability"</li>
          </ul>
        </div>
        
        <div className="p-4 bg-green-50 rounded-lg">
          <h4 className="font-medium text-green-900 mb-2">📚 Backend Status:</h4>
          <ul className="text-sm text-green-800 space-y-1">
            <li>• <strong>Backend URL:</strong> {backendUrl}</li>
            <li>• <strong>User Tier:</strong> {currentUser?.subscription_tier}</li>
            <li>• <strong>API Token:</strong> {apiToken ? 'Present' : 'Missing'}</li>
            <li>• <strong>Endpoint:</strong> /external/search-free</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
