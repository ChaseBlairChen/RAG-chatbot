// src/components/chat/ChatInput.tsx (COMPLETE REPLACEMENT)
import React from 'react';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  responseStyle: string;
  setResponseStyle: (value: string) => void;
  searchScope: string;
  setSearchScope: (value: string) => void;
  useEnhancedRag: boolean;
  setUseEnhancedRag: (value: boolean) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  onSend,
  isLoading,
  responseStyle,
  setResponseStyle,
  searchScope,
  setSearchScope,
  useEnhancedRag,
  setUseEnhancedRag
}) => {
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  // Detect if this is a query that should use external APIs
  const shouldUseExternalAPIs = () => {
    const query = input.toLowerCase();
    const congressionalTerms = ['bill', 'congress', 'house', 'senate', 'legislation', 'passed', 'law'];
    const legalTerms = ['case', 'court', 'decision', 'ruling', 'supreme court'];
    const governmentTerms = ['sec', 'enforcement', 'violation', 'fda', 'recent'];
    
    return [...congressionalTerms, ...legalTerms, ...governmentTerms].some(term => query.includes(term));
  };

  const getSearchScopeDescription = (scope: string) => {
    switch (scope) {
      case 'all':
        return 'External APIs + Your Documents (Recommended for legal research)';
      case 'user_only':
        return 'Only Your Uploaded Documents';
      case 'default_only':
        return 'Only Default Legal Database';
      default:
        return scope;
    }
  };

  return (
    <div className="p-6 border-t border-gray-100 bg-gray-50">
      {/* External API Indicator */}
      {shouldUseExternalAPIs() && searchScope === 'all' && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center gap-2 text-sm text-green-800">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="font-medium">External Legal APIs Enabled</span>
            <span className="text-green-600">
              - Will search Congress.gov, Harvard Caselaw, CourtListener, and Government databases
            </span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-4 mb-4 text-sm">
        <div className="flex items-center gap-2">
          <label className="text-gray-600">Style:</label>
          <select 
            value={responseStyle} 
            onChange={(e) => setResponseStyle(e.target.value)}
            className="bg-white border border-gray-200 rounded px-2 py-1 text-sm"
          >
            <option value="concise">Concise</option>
            <option value="balanced">Balanced</option>
            <option value="detailed">Detailed</option>
          </select>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="text-gray-600">Search:</label>
          <select 
            value={searchScope} 
            onChange={(e) => setSearchScope(e.target.value)}
            className="bg-white border border-gray-200 rounded px-2 py-1 text-sm"
            title={getSearchScopeDescription(searchScope)}
          >
            <option value="all">🌐 All Sources (External APIs + Documents)</option>
            <option value="user_only">📁 My Documents Only</option>
            <option value="default_only">📚 Legal Database Only</option>
          </select>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-gray-600">
            <input 
              type="checkbox" 
              checked={useEnhancedRag} 
              onChange={(e) => setUseEnhancedRag(e.target.checked)}
              className="w-4 h-4"
            />
            Enhanced RAG
          </label>
        </div>

        {/* Search Scope Indicator */}
        <div className="flex items-center gap-2 ml-auto">
          {searchScope === 'all' && (
            <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
              ✅ External APIs Active
            </span>
          )}
          {searchScope === 'user_only' && (
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
              📁 Documents Only
            </span>
          )}
          {searchScope === 'default_only' && (
            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
              📚 Database Only
            </span>
          )}
        </div>
      </div>

      {/* Query Suggestions */}
      {input.length === 0 && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">💡 Try These External API Queries:</h4>
          <div className="flex flex-wrap gap-2">
            {[
              'recent bills passed in Congress 2025',
              'Supreme Court immigration decisions',
              'H.R.1 One Big Beautiful Bill status',
              'recent SEC enforcement actions',
              'EPA violations in Washington state'
            ].map(suggestion => (
              <button
                key={suggestion}
                onClick={() => {
                  setInput(suggestion);
                  setSearchScope('all'); // Force external APIs
                }}
                className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded hover:bg-blue-200 transition-all"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Input and Send */}
      <div className="flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about bills, cases, laws, or your documents..."
          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent"
          disabled={isLoading}
        />
        <button
          onClick={onSend}
          disabled={isLoading || !input.trim()}
          className="bg-slate-900 text-white px-6 py-3 rounded-xl hover:bg-slate-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-medium"
        >
          {isLoading ? (
            <div className="flex items-center">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              Searching...
            </div>
          ) : (
            'Send'
          )}
        </button>
      </div>

      {/* Search Scope Help */}
      <div className="mt-3 text-xs text-gray-500">
        <strong>Search Scope:</strong> {getSearchScopeDescription(searchScope)}
        {shouldUseExternalAPIs() && searchScope !== 'all' && (
          <span className="ml-2 text-orange-600 font-medium">
            ⚠️ This query would benefit from External APIs - consider switching to "All Sources"
          </span>
        )}
      </div>
    </div>
  );
};
