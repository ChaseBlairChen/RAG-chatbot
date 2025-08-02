// components/chat/ChatInput.tsx
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

  return (
    <div className="p-6 border-t border-gray-100 bg-gray-50">
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
          >
            <option value="all">All Sources</option>
            <option value="user_only">My Documents Only</option>
            <option value="default_only">Default Database Only</option>
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
      </div>
      
      <div className="flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a legal question..."
          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent"
          disabled={isLoading}
        />
        <button
          onClick={onSend}
          disabled={isLoading || !input.trim()}
          className="bg-slate-900 text-white px-6 py-3 rounded-xl hover:bg-slate-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-medium"
        >
          Send
        </button>
      </div>
    </div>
  );
};
