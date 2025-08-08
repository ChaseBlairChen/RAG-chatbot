// components/chat/ChatTab.tsx
import React, { useState } from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import type { Message } from '../../types';

interface ChatTabProps {
  messages: Message[];
  isLoading: boolean;
  sessionId: string;
  sendMessage: (input: string, responseStyle: string, searchScope: string, useEnhancedRag: boolean, expandRequest?: boolean) => Promise<void>;
}

export const ChatTab: React.FC<ChatTabProps> = ({ messages, isLoading, sendMessage }) => {
  const [input, setInput] = useState('');
  const [responseStyle, setResponseStyle] = useState('balanced');
  const [searchScope, setSearchScope] = useState('all');
  const [useEnhancedRag, setUseEnhancedRag] = useState(true);

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      sendMessage(input, responseStyle, searchScope, useEnhancedRag);
      setInput('');
    }
  };

  const requestExpansion = (messageText: string) => {
    const expansionInput = `Please expand on: ${messageText.slice(0, 100)}...`;
    sendMessage(expansionInput, responseStyle, searchScope, useEnhancedRag, true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-slate-900">Legal Research Assistant</h1>
        <p className="text-slate-600 max-w-2xl mx-auto">
          Ask me anything about legal research, document analysis, case law, or get help with your legal documents.
        </p>
      </div>

      {/* Chat Container */}
      <div className="card h-[calc(100vh-300px)] flex flex-col overflow-hidden">
        <MessageList 
          messages={messages} 
          isLoading={isLoading}
          onRequestExpansion={requestExpansion}
        />
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={handleSend}
          isLoading={isLoading}
          responseStyle={responseStyle}
          setResponseStyle={setResponseStyle}
          searchScope={searchScope}
          setSearchScope={setSearchScope}
          useEnhancedRag={useEnhancedRag}
          setUseEnhancedRag={setUseEnhancedRag}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button 
          onClick={() => sendMessage("What are the latest Supreme Court decisions?", "detailed", "all", true)}
          className="card-hover p-4 text-left group"
        >
          <div className="text-2xl mb-2">⚖️</div>
          <h3 className="font-semibold text-slate-900 group-hover:text-slate-700">Recent Cases</h3>
          <p className="text-sm text-slate-600">Latest Supreme Court decisions</p>
        </button>
        
        <button 
          onClick={() => sendMessage("Help me analyze a contract", "detailed", "all", true)}
          className="card-hover p-4 text-left group"
        >
          <div className="text-2xl mb-2">📄</div>
          <h3 className="font-semibold text-slate-900 group-hover:text-slate-700">Contract Analysis</h3>
          <p className="text-sm text-slate-600">Get help with contract review</p>
        </button>
        
        <button 
          onClick={() => sendMessage("What are the immigration law updates?", "detailed", "all", true)}
          className="card-hover p-4 text-left group"
        >
          <div className="text-2xl mb-2">🗽</div>
          <h3 className="font-semibold text-slate-900 group-hover:text-slate-700">Immigration Law</h3>
          <p className="text-sm text-slate-600">Latest immigration updates</p>
        </button>
      </div>
    </div>
  );
};