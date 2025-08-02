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
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 h-[calc(100vh-200px)] flex flex-col overflow-hidden">
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
  );
};