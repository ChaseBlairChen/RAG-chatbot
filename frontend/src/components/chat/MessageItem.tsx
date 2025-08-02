// components/chat/MessageItem.tsx
import React from 'react';
import type { Message } from '../../types';
import { renderMarkdown } from '../../utils/markdown';

interface MessageItemProps {
  message: Message;
  onRequestExpansion?: (text: string) => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onRequestExpansion }) => {
  return (
    <div className={`flex ${message.from === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-3xl rounded-2xl px-4 py-3 ${
        message.from === 'user' 
          ? 'bg-slate-900 text-white' 
          : 'bg-gray-50 text-gray-900 border border-gray-100'
      }`}>
        <div 
          className="prose prose-sm max-w-none"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(message.text) }}
        />
        
        {message.from === 'bot' && (message.confidence || (message.sources && message.sources.length > 0)) && (
          <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
            {message.confidence && (
              <div className="flex items-center gap-2 mb-2">
                <span>Confidence:</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-20">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all" 
                    style={{ width: `${(message.confidence * 100)}%` }}
                  />
                </div>
                <span>{Math.round((message.confidence || 0) * 100)}%</span>
              </div>
            )}
            {message.sources && message.sources.length > 0 && (
              <div>
                <span className="font-medium">Sources: </span>
                {message.sources.slice(0, 3).map((source: any, i: number) => (
                  <span key={i} className="mr-2">
                    {source.file_name}
                    {i < Math.min(message.sources!.length - 1, 2) ? ',' : ''}
                  </span>
                ))}
                {message.sources.length > 3 && <span>+{message.sources.length - 3} more</span>}
              </div>
            )}
          </div>
        )}
        
        {message.from === 'bot' && message.expandAvailable && onRequestExpansion && (
          <button
            onClick={() => onRequestExpansion(message.text)}
            className="mt-2 text-xs bg-blue-50 text-blue-600 px-3 py-1 rounded-full hover:bg-blue-100 transition-all"
          >
            Expand Answer
          </button>
        )}
      </div>
    </div>
  );
};