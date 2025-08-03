// components/immigration/AutonomousResearchAgent.tsx
import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

interface ResearchTask {
  id: string;
  country: string;
  topics: string[];
  status: 'queued' | 'researching' | 'completed' | 'failed';
  progress: number;
  results?: any;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export const AutonomousResearchAgent: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [researchQueue, setResearchQueue] = useState<ResearchTask[]>([]);
  const [completedResearch, setCompletedResearch] = useState<ResearchTask[]>([]);
  const [currentTask, setCurrentTask] = useState<ResearchTask | null>(null);
  const [agentSettings, setAgentSettings] = useState({
    interval: 30, // minutes between auto-research
    maxConcurrent: 3,
    autoMode: false,
    priorityCountries: ['Afghanistan', 'Myanmar', 'Syria', 'Venezuela', 'China']
  });

  const apiService = new ApiService(backendUrl, apiToken);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Add research task to queue
  const addResearchTask = (country: string, topics: string[] = ['persecution', 'human_rights', 'government', 'violence']) => {
    const task: ResearchTask = {
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      country,
      topics,
      status: 'queued',
      progress: 0
    };
    
    setResearchQueue(prev => [...prev, task]);
    return task.id;
  };

  // Execute research task
  const executeResearchTask = async (task: ResearchTask) => {
    setCurrentTask(task);
    
    // Update task status
    setResearchQueue(prev => prev.map(t => 
      t.id === task.id 
        ? { ...t, status: 'researching', progress: 10, startedAt: new Date().toISOString() }
        : t
    ));

    try {
      // Step 1: Research country conditions
      setResearchQueue(prev => prev.map(t => 
        t.id === task.id ? { ...t, progress: 30 } : t
      ));

      const requestData = {
        country: task.country,
        topics: task.topics,
        date_range: 'last_2_years'
      };

      const researchResults = await apiService.post('/immigration/country-conditions/research', requestData);
      
      // Step 2: Analyze and summarize
      setResearchQueue(prev => prev.map(t => 
        t.id === task.id ? { ...t, progress: 70 } : t
      ));

      // Step 3: Generate AI summary
      const summaryRequest = {
        question: `Based on the research data for ${task.country}, provide a comprehensive asylum case assessment focusing on: ${task.topics.join(', ')}. Include specific recommendations for evidence gathering and case strategy.`,
        session_id: `research_agent_${task.id}`,
        response_style: "detailed",
        search_scope: "all",
        use_enhanced_rag: true
      };

      const aiSummary = await apiService.post('/ask', summaryRequest);

      // Complete the task
      const completedTask = {
        ...task,
        status: 'completed' as const,
        progress: 100,
        completedAt: new Date().toISOString(),
        results: {
          countryConditions: researchResults,
          aiSummary: aiSummary,
          recommendations: extractRecommendations(aiSummary.response)
        }
      };

      setResearchQueue(prev => prev.filter(t => t.id !== task.id));
      setCompletedResearch(prev => [completedTask, ...prev]);
      setCurrentTask(null);

    } catch (error) {
      console.error('Research task failed:', error);
      
      const failedTask = {
        ...task,
        status: 'failed' as const,
        progress: 0,
        completedAt: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      };

      setResearchQueue(prev => prev.filter(t => t.id !== task.id));
      setCompletedResearch(prev => [failedTask, ...prev]);
      setCurrentTask(null);
    }
  };

  // Extract actionable recommendations from AI response
  const extractRecommendations = (aiResponse: string): string[] => {
    const recommendations = [];
    
    // Look for recommendation patterns
    const recPatterns = [
      /recommend[s]?\s+([^.]+)/gi,
      /suggest[s]?\s+([^.]+)/gi,
      /should\s+([^.]+)/gi,
      /evidence\s+needed[:\s]+([^.]+)/gi
    ];

    recPatterns.forEach(pattern => {
      const matches = aiResponse.match(pattern);
      if (matches) {
        recommendations.push(...matches.slice(0, 3)); // Limit to 3 per pattern
      }
    });

    return recommendations.slice(0, 10); // Max 10 recommendations
  };

  // Start autonomous agent
  const startAgent = () => {
    setIsAgentRunning(true);
    
    if (agentSettings.autoMode) {
      intervalRef.current = setInterval(() => {
        // Auto-queue priority countries if queue is empty
        if (researchQueue.length === 0 && currentTask === null) {
          const randomCountry = agentSettings.priorityCountries[
            Math.floor(Math.random() * agentSettings.priorityCountries.length)
          ];
          addResearchTask(randomCountry);
        }
      }, agentSettings.interval * 60 * 1000);
    }

    // Process queue
    const processQueue = async () => {
      while (isAgentRunning && researchQueue.length > 0) {
        const nextTask = researchQueue.find(t => t.status === 'queued');
        if (nextTask && !currentTask) {
          await executeResearchTask(nextTask);
        }
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds between tasks
      }
    };

    processQueue();
  };

  // Stop autonomous agent
  const stopAgent = () => {
    setIsAgentRunning(false);
    setCurrentTask(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Auto-process queue when new tasks are added
  useEffect(() => {
    if (isAgentRunning && researchQueue.length > 0 && !currentTask) {
      const nextTask = researchQueue.find(t => t.status === 'queued');
      if (nextTask) {
        executeResearchTask(nextTask);
      }
    }
  }, [researchQueue, isAgentRunning, currentTask]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            isAgentRunning ? 'bg-green-100 animate-pulse' : 'bg-gray-100'
          }`}>
            <svg className={`w-5 h-5 ${isAgentRunning ? 'text-green-600' : 'text-gray-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-semibold text-gray-900">Autonomous Research Agent</h2>
            <p className="text-sm text-gray-600">
              Status: {isAgentRunning ? '🟢 Active' : '🔴 Stopped'} • 
              Queue: {researchQueue.length} • 
              Completed: {completedResearch.length}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {!isAgentRunning ? (
            <button
              onClick={startAgent}
              className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-all font-medium"
            >
              🚀 Start Agent
            </button>
          ) : (
            <button
              onClick={stopAgent}
              className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-all font-medium"
            >
              ⏹️ Stop Agent
            </button>
          )}
        </div>
      </div>

      {/* Agent Settings */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium text-gray-900 mb-3">⚙️ Agent Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Auto-Research Interval (minutes)</label>
            <input
              type="number"
              value={agentSettings.interval}
              onChange={(e) => setAgentSettings(prev => ({...prev, interval: parseInt(e.target.value)}))}
              className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
              min="5"
              max="1440"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Max Concurrent Tasks</label>
            <input
              type="number"
              value={agentSettings.maxConcurrent}
              onChange={(e) => setAgentSettings(prev => ({...prev, maxConcurrent: parseInt(e.target.value)}))}
              className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
              min="1"
              max="10"
            />
          </div>
          <div className="flex items-center gap-2 pt-6">
            <input
              type="checkbox"
              checked={agentSettings.autoMode}
              onChange={(e) => setAgentSettings(prev => ({...prev, autoMode: e.target.checked}))}
              className="w-4 h-4"
            />
            <label className="text-sm text-gray-700">Auto-mode (continuous research)</label>
          </div>
        </div>
      </div>

      {/* Quick Add Research */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h3 className="font-medium text-blue-900 mb-3">➕ Add Research Task</h3>
        <div className="flex flex-wrap gap-2 mb-3">
          {agentSettings.priorityCountries.map(country => (
            <button
              key={country}
              onClick={() => addResearchTask(country)}
              disabled={isAgentRunning && researchQueue.some(t => t.country === country)}
              className="bg-blue-100 text-blue-800 px-3 py-1 rounded text-sm hover:bg-blue-200 transition-all disabled:opacity-50"
            >
              + {country}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Custom country (e.g., 'Ethiopia protests')"
            className="flex-1 px-3 py-2 border border-gray-300 rounded text-sm"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                const target = e.target as HTMLInputElement;
                if (target.value.trim()) {
                  addResearchTask(target.value.trim());
                  target.value = '';
                }
              }
            }}
          />
          <button
            onClick={() => {
              const input = document.querySelector('input[placeholder*="Custom country"]') as HTMLInputElement;
              if (input?.value.trim()) {
                addResearchTask(input.value.trim());
                input.value = '';
              }
            }}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-all text-sm"
          >
            Add to Queue
          </button>
        </div>
      </div>

      {/* Current Task Status */}
      {currentTask && (
        <div className="mb-6 p-4 bg-green-50 rounded-lg border border-green-200">
          <h3 className="font-medium text-green-900 mb-3">🔍 Currently Researching</h3>
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium">{currentTask.country}</span>
            <span className="text-sm text-green-700">{currentTask.progress}%</span>
          </div>
          <div className="bg-green-200 rounded-full h-2 mb-2">
            <div 
              className="bg-green-600 h-2 rounded-full transition-all duration-500" 
              style={{ width: `${currentTask.progress}%` }}
            />
          </div>
          <div className="text-sm text-green-700">
            Topics: {currentTask.topics.join(', ')}
          </div>
        </div>
      )}

      {/* Research Queue */}
      {researchQueue.length > 0 && (
        <div className="mb-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
          <h3 className="font-medium text-yellow-900 mb-3">⏳ Research Queue ({researchQueue.length})</h3>
          <div className="space-y-2">
            {researchQueue.slice(0, 5).map((task, idx) => (
              <div key={task.id} className="flex items-center justify-between bg-white rounded p-3 border">
                <div>
                  <span className="font-medium text-gray-900">{task.country}</span>
                  <span className="ml-2 text-sm text-gray-600">#{idx + 1} in queue</span>
                </div>
                <button
                  onClick={() => setResearchQueue(prev => prev.filter(t => t.id !== task.id))}
                  className="text-red-600 hover:text-red-700 text-sm"
                >
                  Remove
                </button>
              </div>
            ))}
            {researchQueue.length > 5 && (
              <div className="text-sm text-yellow-700 text-center">
                +{researchQueue.length - 5} more tasks in queue
              </div>
            )}
          </div>
        </div>
      )}

      {/* Completed Research */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-gray-900">📊 Completed Research</h3>
          {completedResearch.length > 0 && (
            <button
              onClick={() => setCompletedResearch([])}
              className="text-gray-600 hover:text-gray-800 text-sm"
            >
              Clear All
            </button>
          )}
        </div>

        {completedResearch.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <p>No autonomous research completed yet</p>
            <p className="text-sm">Add countries to the queue and start the agent</p>
          </div>
        ) : (
          <div className="space-y-4">
            {completedResearch.map((task) => (
              <div key={task.id} className={`border rounded-lg p-4 ${
                task.status === 'completed' ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
              }`}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full ${
                      task.status === 'completed' ? 'bg-green-500' : 'bg-red-500'
                    }`} />
                    <h4 className="font-semibold text-gray-900">{task.country}</h4>
                    <span className="text-xs bg-white px-2 py-1 rounded">
                      {task.completedAt && new Date(task.completedAt).toLocaleString()}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      if (task.results) {
                        const content = `Autonomous Research Report: ${task.country}\n\nGenerated: ${task.completedAt}\nTopics: ${task.topics.join(', ')}\n\nAI SUMMARY:\n${task.results.aiSummary?.response || 'No summary available'}\n\nRECOMMENDATIONS:\n${task.results.recommendations?.join('\n') || 'No recommendations available'}\n\n---\nGenerated by Legally Autonomous Research Agent`;
                        
                        const blob = new Blob([content], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `research-${task.country.replace(/\s+/g, '-')}-${Date.now()}.txt`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }
                    }}
                    className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                  >
                    📥 Download
                  </button>
                </div>

                {task.status === 'failed' && task.error && (
                  <div className="text-red-700 text-sm mb-3">
                    <strong>Error:</strong> {task.error}
                  </div>
                )}

                {task.status === 'completed' && task.results && (
                  <div className="space-y-3">
                    {/* AI Summary Preview */}
                    {task.results.aiSummary?.response && (
                      <div className="bg-white border rounded p-3">
                        <h5 className="font-medium text-gray-900 mb-2">🤖 AI Analysis Summary</h5>
                        <div className="text-sm text-gray-700 line-clamp-3">
                          {task.results.aiSummary.response.substring(0, 200)}...
                        </div>
                      </div>
                    )}

                    {/* Key Recommendations */}
                    {task.results.recommendations && task.results.recommendations.length > 0 && (
                      <div className="bg-white border rounded p-3">
                        <h5 className="font-medium text-gray-900 mb-2">💡 Key Recommendations</h5>
                        <ul className="space-y-1">
                          {task.results.recommendations.slice(0, 3).map((rec: string, idx: number) => (
                            <li key={idx} className="text-sm text-gray-700">• {rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Research Confidence */}
                    {task.results.aiSummary?.confidence_score && (
                      <div className="bg-white border rounded p-3">
                        <h5 className="font-medium text-gray-900 mb-2">📊 Research Confidence</h5>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-500 h-2 rounded-full transition-all" 
                              style={{ width: `${(task.results.aiSummary.confidence_score * 100)}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium">
                            {Math.round(task.results.aiSummary.confidence_score * 100)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Agent Instructions */}
      <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-medium text-blue-900 mb-2">🤖 How the Autonomous Agent Works</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• <strong>Queues Research:</strong> Add countries to research queue manually or automatically</li>
          <li>• <strong>Conducts Research:</strong> Uses your backend's country conditions research API</li>
          <li>• <strong>AI Analysis:</strong> Generates detailed asylum case assessments using your AI service</li>
          <li>• <strong>Extracts Recommendations:</strong> Identifies actionable advice for case building</li>
          <li>• <strong>Auto-Downloads:</strong> Saves comprehensive reports for each country</li>
          <li>• <strong>Continuous Operation:</strong> Can run autonomously with configurable intervals</li>
        </ul>
      </div>
    </div>
  );
};
