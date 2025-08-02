// components/chat/MessageList.tsx
import React, { useRef, useEffect } from 'react';
import { Message } from '../../types';
import { MessageItem } from './MessageItem';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  onRequestExpansion: (text: string) => void;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, isLoading, onRequestExpansion }) => {
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-grow overflow-y-auto p-6 space-y-4">
      {messages.map((msg, idx) => (
        <MessageItem 
          key={idx} 
          message={msg} 
          onRequestExpansion={onRequestExpansion}
        />
      ))}
      
      {isLoading && (
        <div className="flex justify-start">
          <div className="bg-gray-50 rounded-2xl px-4 py-3 border border-gray-100">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              <span className="text-sm text-gray-500 ml-2">AI is thinking...</span>
            </div>
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} />
    </div>
  );
};
