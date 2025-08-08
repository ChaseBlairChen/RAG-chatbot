// components/chat/MessageItem.tsx
import React from 'react';
import type { Message } from '../../types';

interface MessageItemProps {
  message: Message;
  onRequestExpansion?: (text: string) => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onRequestExpansion }) => {
  const isUser = message.from === 'user';
  
  return (
    <div className={`flex gap-4 p-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-10 h-10 bg-gradient-to-br from-slate-800 to-slate-900 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-white font-semibold text-sm">AI</span>
        </div>
      )}
      
      <div className={`max-w-3xl ${isUser ? 'order-first' : ''}`}>
        <div className={`rounded-2xl p-4 ${
          isUser 
            ? 'bg-slate-900 text-white' 
            : 'bg-slate-50 text-slate-900 border border-slate-200'
        }`}>
          <div className="prose prose-sm max-w-none">
            {message.text.split('\n').map((line, index) => (
              <p key={index} className="mb-2 last:mb-0">
                {line}
              </p>
            ))}
          </div>
          
          {message.confidence && (
            <div className="mt-3 pt-3 border-t border-slate-200/50">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span>Confidence:</span>
                <div className="flex-1 bg-slate-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${message.confidence * 100}%` }}
                  />
                </div>
                <span>{Math.round(message.confidence * 100)}%</span>
              </div>
            </div>
          )}
        </div>
        
        {!isUser && onRequestExpansion && (
          <div className="mt-2 flex gap-2">
            <button
              onClick={() => onRequestExpansion(message.text)}
              className="text-xs text-slate-500 hover:text-slate-700 font-medium transition-colors"
            >
               Expand
            </button>
            {message.sources && message.sources.length > 0 && (
              <span className="text-xs text-slate-400">
                📚 {message.sources.length} sources
              </span>
            )}
          </div>
        )}
      </div>
      
      {isUser && (
        <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-orange-600 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-white font-semibold text-sm">You</span>
        </div>
      )}
    </div>
  );
};