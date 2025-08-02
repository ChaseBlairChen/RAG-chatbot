// components/legal/LegalDatabaseSearch.tsx
import React, { useState } from 'react';
import { useBackend } from '../../contexts/BackendContext';
import { ApiService } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

export const LegalDatabaseSearch: React.FC = () => {
  const { backendUrl } = useBackend();
  const { apiToken } = useAuth();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  
  const apiService = new ApiService(backendUrl, apiToken);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await apiService.searchFreeLegalDatabases(query);
      setResults(response.results || []);
    } catch (error) {
      console.error('Search failed:', error);
      alert('Search failed. Please try again.');
    }
    setLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">Search Free Legal Databases</h2>
      <p className="text-gray-600 mb-4">
        Search through millions of cases from Harvard Caselaw, CourtListener, and Justia
      </p>
      
      <div className="flex gap-3 mb-6">
        <input 
          type="text"
          value={query} 
          onChange={(e) => setQuery(e.target.value)}
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
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      {results.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900">
            Found {results.length} results:
          </h3>
          {results.map((result, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
              <h4 className="font-semibold text-gray-900 mb-1">{result.title}</h4>
              <p className="text-sm text-gray-600 mb-2">
                {result.court} {result.court && result.date && '•'} {result.date}
              </p>
              {result.citation && (
                <p className="text-sm text-gray-500 mb-2">Citation: {result.citation}</p>
              )}
              <p className="text-gray-700 mb-3">
                {result.preview || result.snippet || result.description || 'No preview available'}
              </p>
              <div className="flex items-center justify-between">
                <a 
                  href={result.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 font-medium text-sm"
                >
                  View Full Case →
                </a>
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                  Source: {result.source_database}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {results.length === 0 && query && !loading && (
        <div className="text-center py-8 text-gray-500">
          No results found for "{query}". Try different search terms.
        </div>
      )}
    </div>
  );
};
