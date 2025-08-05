// src/components/immigration/ImmigrationCaseManager.tsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const ImmigrationCaseManager: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [loading, setLoading] = useState(false);
  const [cases, setCases] = useState<any[]>([]);
  const [deadlines, setDeadlines] = useState<any[]>([]);
  const [dashboardStats, setDashboardStats] = useState<any>(null);
  const [formData, setFormData] = useState({
    caseType: 'asylum',
    clientName: '',
    language: 'en',
    priorityDate: ''
  });

  const apiService = new ApiService(backendUrl, apiToken);

  const loadDashboardStats = async () => {
    try {
      const data = await apiService.get('/immigration/stats/dashboard');
      setDashboardStats(data);
    } catch (error) {
      console.error('Failed to load dashboard stats:', error);
    }
  };

  const loadDeadlines = async () => {
    try {
      const data = await apiService.get('/immigration/deadlines/upcoming?days_ahead=60');
      setDeadlines(data.deadlines || []);
    } catch (error) {
      console.error('Failed to load deadlines:', error);
    }
  };

  const createCase = async () => {
    if (!formData.clientName.trim()) {
      alert('Please enter client name');
      return;
    }

    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('case_type', formData.caseType);
      postData.append('client_name', formData.clientName);
      postData.append('language', formData.language);
      if (formData.priorityDate) {
        postData.append('priority_date', formData.priorityDate);
      }

      const response = await fetch(`${backendUrl}/immigration/cases/create`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiToken}` },
        body: postData,
      });

      const data = await response.json();
      
      if (data.success) {
        alert(`✅ Case created successfully!\n\nCase ID: ${data.case_id}\nType: ${formData.caseType}\nClient: ${formData.clientName}\n\nNext steps:\n• Upload client documents\n• Research country conditions\n• Set up deadline tracking`);
        setFormData(prev => ({ ...prev, clientName: '', priorityDate: '' }));
        loadDeadlines();
        loadDashboardStats();
      } else {
        alert(`❌ Case creation failed: ${data.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Case creation failed:', error);
      alert(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
    setLoading(false);
  };

  const generateEvidenceChecklist = async (caseType: string) => {
    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('case_type', caseType);

      const response = await fetch(`${backendUrl}/immigration/evidence/checklist`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiToken}` },
        body: postData,
      });

      const data = await response.json();
      
      // Create a more user-friendly checklist display
      const checklistItems = Object.entries(data.checklist || {});
      const checklistText = checklistItems
        .map(([item, completed]) => `${completed ? '✅' : '☐'} ${item.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}`)
        .join('\n');
      
      const checklistInfo = `📋 EVIDENCE CHECKLIST FOR ${caseType.toUpperCase()} CASE

${checklistText}

📊 PROGRESS: ${data.completed}/${data.total_required} items completed

💡 NEXT STEPS:
• Review unchecked items with client
• Begin gathering missing evidence
• Prepare evidence summary for filing
• Set deadlines for evidence collection

⚖️ LEGAL NOTE: This checklist is based on standard requirements. Specific cases may require additional evidence.`;
      
      alert(checklistInfo);
      
    } catch (error) {
      console.error('Checklist generation failed:', error);
      alert('❌ Failed to generate evidence checklist. Please try again.');
    }
    setLoading(false);
  };

  const classifyDocument = async () => {
    // This would integrate with file upload for document classification
    alert('📎 Document Classification Feature\n\nTo classify immigration documents:\n1. Go to Upload tab\n2. Upload your documents\n3. The system will automatically classify them\n4. Return here to see classification results');
  };

  useEffect(() => {
    loadDashboardStats();
    loadDeadlines();
  }, []);

  return (
    <div className="space-y-6">
      {/* Dashboard Stats */}
      {dashboardStats && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Immigration Practice Dashboard
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-900">{dashboardStats.total_cases || 0}</div>
              <div className="text-sm text-blue-700">Total Cases</div>
            </div>
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-orange-900">{dashboardStats.pending_deadlines || 0}</div>
              <div className="text-sm text-orange-700">Pending Deadlines</div>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-red-900">{dashboardStats.critical_deadlines || 0}</div>
              <div className="text-sm text-red-700">Critical Deadlines</div>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-green-900">{dashboardStats.documents_pending_translation || 0}</div>
              <div className="text-sm text-green-700">Need Translation</div>
            </div>
          </div>
          
          {dashboardStats.by_type && Object.keys(dashboardStats.by_type).length > 0 && (
            <div className="border-t border-gray-200 pt-4">
              <h4 className="font-medium text-gray-900 mb-3">Cases by Type:</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(dashboardStats.by_type).map(([type, count]) => (
                  <span key={type} className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm">
                    {type.replace('_', ' ')}: {count as number}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Case Creation */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span className="text-2xl">📋</span>
          Create New Immigration Case
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Case Type</label>
            <select
              value={formData.caseType}
              onChange={(e) => setFormData(prev => ({...prev, caseType: e.target.value}))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="asylum">Asylum</option>
              <option value="family_based">Family Based</option>
              <option value="employment_based">Employment Based</option>
              <option value="removal_defense">Removal Defense</option>
              <option value="naturalization">Naturalization</option>
              <option value="humanitarian">Humanitarian</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Client Name</label>
            <input
              type="text"
              value={formData.clientName}
              onChange={(e) => setFormData(prev => ({...prev, clientName: e.target.value}))}
              placeholder="Enter client full name"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Client Language</label>
            <select
              value={formData.language}
              onChange={(e) => setFormData(prev => ({...prev, language: e.target.value}))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="en">English</option>
              <option value="es">Español (Spanish)</option>
              <option value="zh">中文 (Chinese)</option>
              <option value="ar">العربية (Arabic)</option>
              <option value="fr">Français (French)</option>
              <option value="hi">हिन्दी (Hindi)</option>
              <option value="pt">Português (Portuguese)</option>
              <option value="ru">Русский (Russian)</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Priority Date (if applicable)</label>
            <input
              type="date"
              value={formData.priorityDate}
              onChange={(e) => setFormData(prev => ({...prev, priorityDate: e.target.value}))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        
        <div className="flex gap-3 mb-4">
          <button
            onClick={createCase}
            disabled={loading || !formData.clientName.trim()}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium flex items-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Creating Case...
              </>
            ) : (
              <>
                <span>📋</span>
                Create Case
              </>
            )}
          </button>
          
          <button
            onClick={() => generateEvidenceChecklist(formData.caseType)}
            disabled={loading}
            className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:bg-gray-400 font-medium flex items-center gap-2"
          >
            {loading ? 'Generating...' : (
              <>
                <span>📝</span>
                Generate Evidence Checklist
              </>
            )}
          </button>
          
          <button
            onClick={classifyDocument}
            className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 font-medium flex items-center gap-2"
          >
            <span>📎</span>
            Classify Documents
          </button>
        </div>
        
        {/* Case Type Info */}
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="font-medium text-blue-900 mb-2">💡 {formData.caseType.replace('_', ' ').toUpperCase()} Case Information:</h4>
          <div className="text-sm text-blue-800">
            {formData.caseType === 'asylum' && (
              <div>
                <p><strong>Key Requirements:</strong> One-year filing deadline, credible fear of persecution, country conditions evidence</p>
                <p><strong>Important Forms:</strong> I-589, supporting evidence, country conditions research</p>
                <p><strong>Typical Timeline:</strong> 6-24 months depending on court backlog</p>
              </div>
            )}
            {formData.caseType === 'family_based' && (
              <div>
                <p><strong>Key Requirements:</strong> Relationship evidence, financial support, petitioner eligibility</p>
                <p><strong>Important Forms:</strong> I-130, I-485, I-864, supporting evidence</p>
                <p><strong>Typical Timeline:</strong> 12-36 months depending on category and country</p>
              </div>
            )}
            {formData.caseType === 'employment_based' && (
              <div>
                <p><strong>Key Requirements:</strong> Job offer, labor certification, employer petition</p>
                <p><strong>Important Forms:</strong> I-140, I-485, PERM labor certification</p>
                <p><strong>Typical Timeline:</strong> 12-60 months depending on category and country</p>
              </div>
            )}
            {formData.caseType === 'removal_defense' && (
              <div>
                <p><strong>Key Requirements:</strong> Legal basis for relief, timely filing, court deadlines</p>
                <p><strong>Important Forms:</strong> EOIR forms, application for relief, evidence</p>
                <p><strong>Typical Timeline:</strong> 12-48 months depending on court and case complexity</p>
              </div>
            )}
            {formData.caseType === 'naturalization' && (
              <div>
                <p><strong>Key Requirements:</strong> 5 years permanent residence, English/civics test, good moral character</p>
                <p><strong>Important Forms:</strong> N-400, supporting evidence, interview preparation</p>
                <p><strong>Typical Timeline:</strong> 8-18 months from filing to oath ceremony</p>
              </div>
            )}
            {formData.caseType === 'humanitarian' && (
              <div>
                <p><strong>Key Requirements:</strong> Qualifying humanitarian circumstances, admissibility</p>
                <p><strong>Important Forms:</strong> I-131, I-730, supporting evidence</p>
                <p><strong>Typical Timeline:</strong> 6-36 months depending on program and circumstances</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Upcoming Deadlines */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
            <span className="text-2xl">⏰</span>
            Upcoming Deadlines
          </h3>
          <button
            onClick={loadDeadlines}
            disabled={loading}
            className="text-blue-600 hover:text-blue-700 font-medium text-sm flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>
        
        {deadlines.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">📅</span>
            </div>
            <p className="font-medium">No upcoming deadlines</p>
            <p className="text-sm">Create cases to track important deadlines automatically</p>
          </div>
        ) : (
          <div className="space-y-3">
            {deadlines.map((deadline, idx) => (
              <div key={idx} className={`p-4 rounded-lg border ${
                deadline.priority === 'critical' ? 'bg-red-50 border-red-200' :
                deadline.priority === 'high' ? 'bg-orange-50 border-orange-200' :
                'bg-blue-50 border-blue-200'
              }`}>
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 flex items-center gap-2">
                      {deadline.priority === 'critical' && '🚨'}
                      {deadline.priority === 'high' && '⚠️'}
                      {deadline.priority === 'normal' && '📌'}
                      {deadline.description}
                    </h4>
                    <p className="text-sm text-gray-600 mt-1">
                      Case: {deadline.case_id} • Type: {deadline.deadline_type?.replace('_', ' ')}
                    </p>
                    
                    {/* Days until deadline */}
                    {(() => {
                      const dueDate = new Date(deadline.due_date);
                      const today = new Date();
                      const daysUntil = Math.ceil((dueDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
                      
                      return (
                        <p className={`text-sm mt-1 font-medium ${
                          daysUntil <= 7 ? 'text-red-600' :
                          daysUntil <= 30 ? 'text-orange-600' :
                          'text-blue-600'
                        }`}>
                          {daysUntil > 0 ? `${daysUntil} days remaining` : 
                           daysUntil === 0 ? 'Due TODAY' : 
                           `${Math.abs(daysUntil)} days overdue`}
                        </p>
                      );
                    })()}
                  </div>
                  <div className="text-right">
                    <div className="font-medium text-gray-900">
                      {new Date(deadline.due_date).toLocaleDateString()}
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                      deadline.priority === 'critical' ? 'bg-red-100 text-red-700' :
                      deadline.priority === 'high' ? 'bg-orange-100 text-orange-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {deadline.priority}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Tools */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span className="text-2xl">🛠️</span>
          Immigration Quick Tools
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <button 
            onClick={() => generateEvidenceChecklist('asylum')}
            className="p-4 bg-green-50 hover:bg-green-100 rounded-lg border border-green-200 text-left transition-all"
          >
            <div className="text-xl mb-2">📝</div>
            <div className="font-medium text-green-900">Asylum Evidence Checklist</div>
            <div className="text-sm text-green-700">Generate comprehensive evidence checklist</div>
          </button>
          
          <button 
            onClick={() => generateEvidenceChecklist('family_based')}
            className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200 text-left transition-all"
          >
            <div className="text-xl mb-2">👨‍👩‍👧‍👦</div>
            <div className="font-medium text-blue-900">Family-Based Checklist</div>
            <div className="text-sm text-blue-700">Family petition evidence requirements</div>
          </button>
          
          <button 
            onClick={classifyDocument}
            className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200 text-left transition-all"
          >
            <div className="text-xl mb-2">📎</div>
            <div className="font-medium text-purple-900">Document Classification</div>
            <div className="text-sm text-purple-700">Auto-classify uploaded documents</div>
          </button>
          
          <button 
            onClick={() => generateEvidenceChecklist('employment_based')}
            className="p-4 bg-orange-50 hover:bg-orange-100 rounded-lg border border-orange-200 text-left transition-all"
          >
            <div className="text-xl mb-2">💼</div>
            <div className="font-medium text-orange-900">Employment Checklist</div>
            <div className="text-sm text-orange-700">Employment-based petition requirements</div>
          </button>
          
          <button 
            onClick={() => alert('🚀 Batch Processing\n\nSelect multiple cases to:\n• Generate forms in bulk\n• Update statuses\n• Send notifications\n• Export reports\n\nFeature coming soon!')}
            className="p-4 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 text-left transition-all"
          >
            <div className="text-xl mb-2">⚡</div>
            <div className="font-medium text-gray-900">Batch Processing</div>
            <div className="text-sm text-gray-700">Process multiple cases at once</div>
          </button>
          
          <button 
            onClick={() => alert('📚 Resource Library\n\nAccess to:\n• Form instructions in multiple languages\n• Country conditions templates\n• Legal research guides\n• Case law summaries\n\nFeature coming soon!')}
            className="p-4 bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200 text-left transition-all"
          >
            <div className="text-xl mb-2">📚</div>
            <div className="font-medium text-indigo-900">Resource Library</div>
            <div className="text-sm text-indigo-700">Multilingual resources and guides</div>
          </button>
        </div>
      </div>

      {/* Immigration Law Notice */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <svg className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div>
            <h4 className="font-medium text-amber-800 mb-1">⚖️ Legal Disclaimer</h4>
            <p className="text-sm text-amber-700">
              This case management system is for organizational purposes only and does not constitute legal advice. 
              Immigration law is complex and changes frequently. Always consult with a qualified immigration attorney 
              for advice specific to individual cases. All client information is encrypted and protected.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
