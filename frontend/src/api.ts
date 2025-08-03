// ==================== ./src/services/api.ts ====================
export class ApiService {
  private baseUrl: string;
  private apiToken: string;

  constructor(baseUrl: string, apiToken: string) {
    this.baseUrl = baseUrl;
    this.apiToken = apiToken;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  async uploadFile(endpoint: string, formData: FormData): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5min for uploads

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
        },
        body: formData,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (${response.status}): ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  // Fixed method for searching free legal databases
  async searchFreeLegalDatabases(query: string): Promise<any> {
    const formData = new FormData();
    formData.append('query', query);
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(`${this.baseUrl}/external/search-free`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
        },
        body: formData,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed (${response.status}): ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      throw error;
    }
  }

  // Method for searching premium legal databases
  async searchLegalDatabases(query: string, databases: string[]): Promise<any> {
    const formData = new FormData();
    formData.append('query', query);
    databases.forEach(db => formData.append('databases', db));
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(`${this.baseUrl}/external/search`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
        },
        body: formData,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed (${response.status}): ${errorText || response.statusText}`);
      }

      return response.json();
    } catch (error) {
      throw error;
    }
  }
}

// ==================== ./src/components/legal/LegalDatabaseSearch.tsx ====================
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
  const [searchType, setSearchType] = useState<'free' | 'premium'>('free');
  const [selectedDatabases, setSelectedDatabases] = useState<string[]>(['harvard_caselaw', 'courtlistener']);
  const [error, setError] = useState<string>('');
  
  const apiService = new ApiService(backendUrl, apiToken);

  const availableDatabases = {
    free: [
      { id: 'harvard_caselaw', name: 'Harvard Caselaw Access Project', description: 'Comprehensive case law database' },
      { id: 'courtlistener', name: 'CourtListener', description: 'Federal and state court data' },
      { id: 'federal_register', name: 'Federal Register', description: 'Government regulations and notices' }
    ],
    premium: [
      { id: 'lexisnexis', name: 'LexisNexis', description: 'Premium legal research platform' },
      { id: 'westlaw', name: 'Westlaw', description: 'Premium legal database' }
    ]
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError('');
    setResults([]);
    
    try {
      console.log(`🔍 Searching ${searchType} databases for: "${query}"`);
      
      let response;
      
      if (searchType === 'free') {
        response = await apiService.searchFreeLegalDatabases(query);
        console.log('Free database search response:', response);
      } else {
        if (selectedDatabases.length === 0) {
          throw new Error('Please select at least one database');
        }
        response = await apiService.searchLegalDatabases(query, selectedDatabases);
        console.log('Premium database search response:', response);
      }
      
      const searchResults = response.results || [];
      setResults(searchResults);
      
      if (searchResults.length === 0) {
        setError('No results found. Try different search terms or check your query.');
      }
      
    } catch (error) {
      console.error('Search failed:', error);
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

  const toggleDatabase = (dbId: string) => {
    setSelectedDatabases(prev => 
      prev.includes(dbId) 
        ? prev.filter(id => id !== dbId)
        : [...prev, dbId]
    );
  };

  const canUsePremium = currentUser?.subscription_tier === 'premium';

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Legal Database Search</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Search Mode:</span>
          <select 
            value={searchType} 
            onChange={(e) => {
              setSearchType(e.target.value as 'free' | 'premium');
              setError('');
              setResults([]);
            }}
            className="bg-white border border-gray-200 rounded px-3 py-1 text-sm"
            disabled={!canUsePremium}
          >
            <option value="free">Free Databases</option>
            <option value="premium" disabled={!canUsePremium}>
              Premium Databases {!canUsePremium && '(Premium Only)'}
            </option>
          </select>
        </div>
      </div>
      
      <p className="text-gray-600 mb-4">
        {searchType === 'free' 
          ? 'Search through millions of cases from Harvard Caselaw, CourtListener, and Federal Register'
          : 'Access premium legal databases including LexisNexis and Westlaw'
        }
      </p>

      {/* Database Selection for Premium */}
      {searchType === 'premium' && canUsePremium && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-2">Select Databases:</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {availableDatabases.premium.map(db => (
              <label key={db.id} className="flex items-center gap-2 p-2 hover:bg-gray-100 rounded">
                <input
                  type="checkbox"
                  checked={selectedDatabases.includes(db.id)}
                  onChange={() => toggleDatabase(db.id)}
                  className="w-4 h-4"
                />
                <div>
                  <div className="font-medium text-sm">{db.name}</div>
                  <div className="text-xs text-gray-600">{db.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Free Database Info */}
      {searchType === 'free' && (
        <div className="mb-4 p-4 bg-green-50 rounded-lg border border-green-200">
          <h4 className="font-medium text-green-900 mb-2">📚 Free Databases Included:</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {availableDatabases.free.map(db => (
              <div key={db.id} className="p-2">
                <div className="font-medium text-sm text-green-800">✓ {db.name}</div>
                <div className="text-xs text-green-700">{db.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      
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
          placeholder="Search cases... (e.g., 'miranda rights', 'contract breach', 'negligence')"
          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500"
          disabled={loading}
        />
        <button 
          onClick={handleSearch} 
          disabled={loading || !query.trim() || (searchType === 'premium' && selectedDatabases.length === 0)}
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
      
      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">
              Found {results.length} results
            </h3>
            <span className="text-sm text-gray-500">
              Searched: {searchType === 'free' ? 'Free databases' : selectedDatabases.join(', ')}
            </span>
          </div>
          
          {results.map((result, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-all">
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-semibold text-gray-900 flex-1 pr-4">{result.title || 'Untitled Document'}</h4>
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full whitespace-nowrap">
                  {result.source_database?.replace('_', ' ').toUpperCase() || 'Unknown Source'}
                </span>
              </div>
              
              <div className="text-sm text-gray-600 mb-3 space-y-1">
                {result.court && (
                  <div className="flex gap-1">
                    <strong>Court:</strong> 
                    <span>{result.court}</span>
                  </div>
                )}
                {result.date && (
                  <div className="flex gap-1">
                    <strong>Date:</strong> 
                    <span>{new Date(result.date).toLocaleDateString()}</span>
                  </div>
                )}
                {result.citation && (
                  <div className="flex gap-1">
                    <strong>Citation:</strong> 
                    <span>{result.citation}</span>
                  </div>
                )}
                {result.docket_number && (
                  <div className="flex gap-1">
                    <strong>Docket:</strong> 
                    <span>{result.docket_number}</span>
                  </div>
                )}
                {result.agency && (
                  <div className="flex gap-1">
                    <strong>Agency:</strong> 
                    <span>{result.agency}</span>
                  </div>
                )}
                {result.type && (
                  <div className="flex gap-1">
                    <strong>Type:</strong> 
                    <span className="capitalize">{result.type}</span>
                  </div>
                )}
              </div>
              
              {(result.preview || result.snippet || result.description || result.summary) && (
                <p className="text-gray-700 mb-3 text-sm leading-relaxed">
                  {result.preview || result.snippet || result.description || result.summary}
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
                {result.pdf_url && (
                  <a 
                    href={result.pdf_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-green-600 hover:text-green-700 font-medium text-sm flex items-center gap-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Download PDF
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
          <p className="text-sm">Try different search terms or broader keywords</p>
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
            <li>• Try statute citations: "42 USC 1983"</li>
            <li>• Use quotation marks for exact phrases</li>
            <li>• Search constitutional terms: "first amendment", "due process"</li>
          </ul>
        </div>
        
        <div className="p-4 bg-green-50 rounded-lg">
          <h4 className="font-medium text-green-900 mb-2">📚 Available Sources:</h4>
          <ul className="text-sm text-green-800 space-y-1">
            <li>• <strong>Harvard Caselaw:</strong> Historical and recent cases</li>
            <li>• <strong>CourtListener:</strong> Federal and state courts</li>
            <li>• <strong>Federal Register:</strong> Government regulations</li>
            {canUsePremium && (
              <>
                <li>• <strong>LexisNexis:</strong> Premium legal research (Premium)</li>
                <li>• <strong>Westlaw:</strong> Comprehensive case law (Premium)</li>
              </>
            )}
          </ul>
        </div>
      </div>

      {/* Test Queries */}
      <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
        <h4 className="font-medium text-yellow-900 mb-2">🧪 Try These Test Queries:</h4>
        <div className="flex flex-wrap gap-2">
          {[
            'miranda rights',
            'contract breach',
            'negligence',
            'first amendment',
            'Brown v Board',
            'search and seizure',
            'due process'
          ].map(testQuery => (
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
    </div>
  );
};

// ==================== ./src/utils/constants.ts ====================
import type { AnalysisTool } from '../types';

export const ANALYSIS_TOOLS: AnalysisTool[] = [
  {
    id: 'comprehensive',
    title: 'Complete Document Analysis',
    description: 'Run all analysis tools at once for comprehensive insights',
    prompt: 'Provide a comprehensive legal analysis including: summary, key clauses, risks, timeline, obligations, and missing clauses.',
    icon: '🔍',
    category: 'Complete',
    idealFor: ['Any legal document'],
    riskLevel: 'low',
    isComprehensive: true
  },
  {
    id: 'summarize',
    title: 'Legal Document Summarization',
    description: 'Get plain English summaries while keeping legal tone intact',
    prompt: 'Summarize this legal document in plain English, keeping the legal tone intact. Highlight purpose, parties involved, and key terms.',
    icon: '📄',
    category: 'Analysis',
    idealFor: ['Contracts', 'Case briefs', 'Discovery documents'],
    riskLevel: 'low'
  },
  {
    id: 'extract-clauses',
    title: 'Key Clause Extraction',
    description: 'Extract termination, indemnification, liability clauses automatically',
    prompt: 'Extract and list the clauses related to termination, indemnification, liability, governing law, and confidentiality.',
    icon: '📋',
    category: 'Extraction',
    idealFor: ['NDAs', 'Employment agreements', 'Service contracts'],
    riskLevel: 'low'
  },
  {
    id: 'missing-clauses',
    title: 'Missing Clause Detection',
    description: 'Flag commonly expected clauses that might be missing',
    prompt: 'Analyze this contract and flag any commonly expected legal clauses that are missing, such as limitation of liability or dispute resolution.',
    icon: '⚠️',
    category: 'Risk Assessment',
    idealFor: ['Startup contracts', 'Vendor agreements'],
    riskLevel: 'medium'
  },
  {
    id: 'risk-flagging',
    title: 'Legal Risk Flagging',
    description: 'Identify clauses that may pose legal risks to signing party',
    prompt: 'Identify any clauses that may pose legal risks to the signing party, such as unilateral termination, broad indemnity, or vague obligations.',
    icon: '🚩',
    category: 'Risk Assessment',
    idealFor: ['Lease agreements', 'IP transfer agreements'],
    riskLevel: 'high'
  },
  {
    id: 'timeline-extraction',
    title: 'Timeline & Deadline Extraction',
    description: 'Extract all dates, deadlines, and renewal periods',
    prompt: 'Extract and list all dates, deadlines, renewal periods, and notice periods mentioned in this document.',
    icon: '📅',
    category: 'Extraction',
    idealFor: ['Leases', 'Licensing deals'],
    riskLevel: 'low'
  },
  {
    id: 'obligations',
    title: 'Obligation Summary',
    description: 'List all required actions and obligations with deadlines',
    prompt: 'List all actions or obligations the signing party is required to perform, along with associated deadlines or conditions.',
    icon: '✅',
    category: 'Analysis',
    idealFor: ['Service contracts', 'Compliance agreements'],
    riskLevel: 'low'
  }
];

export const TEST_ACCOUNTS = [
  { username: 'demo', password: 'demo123', email: 'demo@legalassistant.ai', role: 'user', subscription_tier: 'free' },
  { username: 'tester1', password: 'test123', email: 'tester1@company.com', role: 'user', subscription_tier: 'premium' },
  { username: 'tester2', password: 'test456', email: 'tester2@company.com', role: 'user', subscription_tier: 'free' },
  { username: 'lawyer1', password: 'legal123', email: 'lawyer1@lawfirm.com', role: 'user', subscription_tier: 'premium' }
];

// UPDATE THIS to your actual Cloudflare tunnel URL
export const DEFAULT_BACKEND_URL = "https://accurately-feb-distinguished-optical.trycloudflare.com";
