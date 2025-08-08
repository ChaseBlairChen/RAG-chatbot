// src/components/chat/ChatInput.tsx (COMPLETE REPLACEMENT)
import React, { useState } from 'react';

interface ChatInputProps {
  input: string;
  setInput: (input: string) => void;
  onSend: () => void;
  isLoading: boolean;
  responseStyle: string;
  setResponseStyle: (style: string) => void;
  searchScope: string;
  setSearchScope: (scope: string) => void;
  useEnhancedRag: boolean;
  setUseEnhancedRag: (enabled: boolean) => void;
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
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="border-t border-slate-200 bg-slate-50 p-6">
      {/* Advanced Options Toggle */}
      <div className="mb-4">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-slate-600 hover:text-slate-800 font-medium transition-colors"
        >
          {showAdvanced ? '▼' : '▶'} Advanced Options
        </button>
      </div>

      {/* Advanced Options */}
      {showAdvanced && (
        <div className="mb-4 p-4 bg-white rounded-xl border border-slate-200 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Response Style */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Response Style
              </label>
              <select
                value={responseStyle}
                onChange={(e) => setResponseStyle(e.target.value)}
                className="input-field"
              >
                <option value="balanced">Balanced</option>
                <option value="detailed">Detailed</option>
                <option value="concise">Concise</option>
                <option value="technical">Technical</option>
              </select>
            </div>

            {/* Search Scope */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Search Scope
              </label>
              <select
                value={searchScope}
                onChange={(e) => setSearchScope(e.target.value)}
                className="input-field"
              >
                <option value="all">All Sources</option>
                <option value="user_only">User Documents</option>
                <option value="external_only">External APIs</option>
              </select>
            </div>

            {/* Enhanced RAG Toggle */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Smart RAG
              </label>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={useEnhancedRag}
                  onChange={(e) => setUseEnhancedRag(e.target.checked)}
                  className="w-4 h-4 text-slate-600 bg-slate-100 border-slate-300 rounded focus:ring-slate-500"
                />
                <span className="ml-2 text-sm text-slate-600">Enhanced Search</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="flex gap-4">
        <div className="flex-1">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about legal research, document analysis, or case law..."
            className="input-field resize-none h-20"
            disabled={isLoading}
          />
        </div>
        <button
          onClick={onSend}
          disabled={!input.trim() || isLoading}
          className="btn-primary h-20 px-8 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Thinking...</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span>Send</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </div>
          )}
        </button>
      </div>
    </div>
  );
};
