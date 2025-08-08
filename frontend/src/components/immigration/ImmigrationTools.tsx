// src/components/immigration/ImmigrationTools.tsx (COMPLETE REPLACEMENT)
import React, { useState } from 'react';
import { ImmigrationCaseManager } from './ImmigrationCaseManager';
import { useAuth } from '../../contexts/AuthContext';
import { ApiService } from '../../services/api';
import { useBackend } from '../../contexts/BackendContext';

export const ImmigrationTools: React.FC = () => {
  const { apiToken } = useAuth();
  const { backendUrl } = useBackend();
  const [activeSection, setActiveSection] = useState<'cases' | 'research' | 'documents'>('cases');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [formData, setFormData] = useState({
    country: '',
    testimony: ''
  });
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [documentResults, setDocumentResults] = useState<any[]>([]);

  const apiService = new ApiService(backendUrl, apiToken);

  const researchCountryConditions = async () => {
    if (!formData.country) {
      alert('Please enter country name');
      return;
    }

    setLoading(true);
    setResults(null); // Clear previous results
    
    try {
      const requestData = {
        country: formData.country,
        topics: ['persecution', 'human_rights', 'government', 'violence'],
        date_range: 'last_2_years'
      };

      console.log('🔍 Starting country research for:', formData.country);
      const data = await apiService.post('/immigration/country-conditions/research', requestData);
      console.log('✅ Country research completed');
      setResults({ type: 'country_conditions', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      console.error('❌ Country research failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Check if it's a timeout error
      if (errorMessage.includes('timeout') || errorMessage.includes('aborted')) {
        setResults({ 
          type: 'country_conditions', 
          error: 'Research timed out. This can happen with complex country research. Please try again or try a different country.' 
        });
      } else {
        setResults({ type: 'country_conditions', error: errorMessage });
      }
    }
    setLoading(false);
  };

  const analyzeTestimony = async () => {
    if (!formData.testimony || !formData.country) {
      alert('Please enter both testimony and country');
      return;
    }

    setLoading(true);
    try {
      const postData = new FormData();
      postData.append('testimony', formData.testimony);
      postData.append('country', formData.country);

      const response = await fetch(`${backendUrl}/immigration/credible-fear/analyze`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiToken}` },
        body: postData,
      });

      const data = await response.json();
      setResults({ type: 'testimony_analysis', data, timestamp: new Date().toLocaleString() });
    } catch (error) {
      setResults({ type: 'testimony_analysis', error: error instanceof Error ? error.message : 'Unknown error' });
    }
    setLoading(false);
  };

  // Document processing functions
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(files);
  };

  const classifyDocuments = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select files to classify');
      return;
    }

    setLoading(true);
    setDocumentResults([]);
    
    try {
      const results = [];
      
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${backendUrl}/immigration/documents/classify`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${apiToken}` },
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          results.push({
            filename: file.name,
            ...result,
            success: true
          });
        } else {
          results.push({
            filename: file.name,
            success: false,
            error: `HTTP ${response.status}: ${response.statusText}`
          });
        }
      }
      
      setDocumentResults(results);
      setResults({ 
        type: 'document_classification', 
        data: { results }, 
        timestamp: new Date().toLocaleString() 
      });
    } catch (error) {
      setResults({ 
        type: 'document_classification', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
    setLoading(false);
  };

  const checkTranslationNeeds = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select files to check for translation needs');
      return;
    }

    setLoading(true);
    try {
      const results = [];
      
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${backendUrl}/immigration/documents/classify`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${apiToken}` },
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          results.push({
            filename: file.name,
            language: result.classification?.language || 'unknown',
            requires_translation: result.classification?.requires_translation || false,
            success: true
          });
        } else {
          results.push({
            filename: file.name,
            success: false,
            error: `HTTP ${response.status}: ${response.statusText}`
          });
        }
      }
      
      setResults({ 
        type: 'translation_check', 
        data: { results }, 
        timestamp: new Date().toLocaleString() 
      });
    } catch (error) {
      setResults({ 
        type: 'translation_check', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
    setLoading(false);
  };

  const runCompletenessCheck = async () => {
    setLoading(true);
    try {
      // This would typically check against a case's required documents
      // For now, we'll provide a general checklist
      const checklist = {
        asylum_case: [
          'I-589 Application for Asylum',
          'Personal Statement/Affidavit',
          'Country Conditions Evidence',
          'Identity Documents',
          'Medical Records (if applicable)',
          'Police Reports (if applicable)',
          'Witness Statements (if applicable)'
        ],
        family_based: [
          'I-130 Petition',
          'I-485 Application',
          'Birth Certificates',
          'Marriage Certificate',
          'Financial Support Documents',
          'Medical Examination'
        ]
      };
      
      setResults({ 
        type: 'completeness_check', 
        data: { checklist }, 
        timestamp: new Date().toLocaleString() 
      });
    } catch (error) {
      setResults({ 
        type: 'completeness_check', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
    setLoading(false);
  };

  const generatePrefilledForms = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select documents to extract information from');
      return;
    }

    setLoading(true);
    try {
      const results = [];
      
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${backendUrl}/immigration/documents/classify`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${apiToken}` },
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          results.push({
            filename: file.name,
            extracted_data: result.classification?.extracted_data || {},
            form_type: result.classification?.form_type || 'unknown',
            success: true
          });
        } else {
          results.push({
            filename: file.name,
            success: false,
            error: `HTTP ${response.status}: ${response.statusText}`
          });
        }
      }
      
      setResults({ 
        type: 'form_prefill', 
        data: { results }, 
        timestamp: new Date().toLocaleString() 
      });
    } catch (error) {
      setResults({ 
        type: 'form_prefill', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
    setLoading(false);
  };

  const renderResult = () => {
    if (!results) return null;

    if (results.error) {
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <strong className="text-red-800">Error:</strong>
            <span className="text-red-700">{results.error}</span>
          </div>
        </div>
      );
    }

    // Country Conditions Research - Enhanced formatting
    if (results.type === 'country_conditions' && results.data?.research) {
      const research = results.data.research;
      return (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              🌍 Country: {research.country}
            </h4>
            <p className="text-sm text-blue-800">Research Date: {new Date(research.research_date).toLocaleString()}</p>
          </div>

          {research.summary && (
            <div className="bg-white border rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                📊 Executive Summary
              </h4>
              <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed">
                {research.summary.split('\n').map((line: string, idx: number) => (
                  line.trim() && <p key={idx} className="mb-3">{line}</p>
                ))}
              </div>
            </div>
          )}

          {research.topics && Object.keys(research.topics).length > 0 && (
            <div className="bg-white border rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                📋 Detailed Analysis by Topic
              </h4>
              <div className="space-y-6">
                {Object.entries(research.topics).map(([topic, content]) => (
                  <div key={topic} className="border-l-4 border-blue-300 pl-6 bg-gray-50 rounded-r-lg p-4">
                    <h5 className="font-semibold text-gray-800 capitalize mb-3 text-lg flex items-center gap-2">
                      {topic === 'persecution' && '⚠️'}
                      {topic === 'government' && '🏛️'}
                      {topic === 'violence' && '⚡'}
                      {topic === 'human_rights' && '👥'}
                      {topic.replace('_', ' ')}
                    </h5>
                    <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
                      {content as string}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {research.sources && research.sources.length > 0 && (
            <div className="bg-gray-50 border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                📚 Source Documents
              </h4>
              <div className="space-y-2">
                {research.sources.map((source: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-sm text-gray-600 bg-white rounded p-3">
                    <div>
                      <span className="font-medium">{source.file_name}</span>
                      {source.page && <span className="ml-2 text-gray-500">(Page {source.page})</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                        {Math.round(source.relevance * 100)}% relevant
                      </span>
                      <span className="text-xs text-gray-500 capitalize">
                        {source.source_type.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Testimony Analysis Results
    if (results.type === 'testimony_analysis' && results.data) {
      return (
        <div className="space-y-4">
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h4 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              🗣️ Testimony Analysis Complete
            </h4>
          </div>

          {results.data.analysis && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">📝 Analysis</h4>
              <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-line">
                {results.data.analysis.analysis || results.data.analysis}
              </div>
            </div>
          )}

          {results.data.recommendations && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-900 mb-3">💡 Recommendations</h4>
              <ul className="space-y-1">
                {results.data.recommendations.map((rec: string, idx: number) => (
                  <li key={idx} className="text-sm text-green-700">• {rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    }

    // Document Classification Results
    if (results.type === 'document_classification' && results.data?.results) {
      return (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              📄 Document Classification Complete
            </h4>
          </div>

          <div className="space-y-3">
            {results.data.results.map((result: any, index: number) => (
              <div key={index} className={`border rounded-lg p-4 ${result.success ? 'bg-white' : 'bg-red-50 border-red-200'}`}>
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-medium text-gray-900">{result.filename}</h5>
                  <span className={`text-xs px-2 py-1 rounded-full ${result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {result.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                
                {result.success ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">Category:</span>
                      <span className="text-sm font-medium">{result.classification?.category || 'Unknown'}</span>
                    </div>
                    {result.classification?.form_type && (
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600">Form Type:</span>
                        <span className="text-sm font-medium">{result.classification.form_type}</span>
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">Language:</span>
                      <span className="text-sm font-medium">{result.classification?.language || 'Unknown'}</span>
                    </div>
                    {result.classification?.requires_translation && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                          ⚠️ Requires Translation
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-red-600">{result.error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }

    // Translation Check Results
    if (results.type === 'translation_check' && results.data?.results) {
      return (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="font-semibold text-green-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
              </svg>
              🌐 Translation Check Complete
            </h4>
          </div>

          <div className="space-y-3">
            {results.data.results.map((result: any, index: number) => (
              <div key={index} className={`border rounded-lg p-4 ${result.success ? 'bg-white' : 'bg-red-50 border-red-200'}`}>
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-medium text-gray-900">{result.filename}</h5>
                  <span className={`text-xs px-2 py-1 rounded-full ${result.requires_translation ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                    {result.requires_translation ? 'Needs Translation' : 'No Translation Needed'}
                  </span>
                </div>
                
                {result.success ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">Language:</span>
                      <span className="text-sm font-medium">{result.language}</span>
                    </div>
                    {result.requires_translation && (
                      <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                        <p className="text-sm text-yellow-800">
                          ⚠️ This document requires certified translation for USCIS submission.
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-red-600">{result.error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }

    // Completeness Check Results
    if (results.type === 'completeness_check' && results.data?.checklist) {
      return (
        <div className="space-y-4">
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h4 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              ✅ Completeness Checklist
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white border rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 mb-3">Asylum Case Requirements</h5>
              <ul className="space-y-2">
                {results.data.checklist.asylum_case.map((item: string, index: number) => (
                  <li key={index} className="flex items-center gap-2 text-sm text-gray-700">
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="bg-white border rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 mb-3">Family-Based Case Requirements</h5>
              <ul className="space-y-2">
                {results.data.checklist.family_based.map((item: string, index: number) => (
                  <li key={index} className="flex items-center gap-2 text-sm text-gray-700">
                    <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      );
    }

    // Form Pre-fill Results
    if (results.type === 'form_prefill' && results.data?.results) {
      return (
        <div className="space-y-4">
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
            <h4 className="font-semibold text-orange-900 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              📋 Form Pre-fill Data Extracted
            </h4>
          </div>

          <div className="space-y-3">
            {results.data.results.map((result: any, index: number) => (
              <div key={index} className={`border rounded-lg p-4 ${result.success ? 'bg-white' : 'bg-red-50 border-red-200'}`}>
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-medium text-gray-900">{result.filename}</h5>
                  <span className={`text-xs px-2 py-1 rounded-full ${result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {result.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                
                {result.success ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">Form Type:</span>
                      <span className="text-sm font-medium">{result.form_type}</span>
                    </div>
                    {Object.keys(result.extracted_data).length > 0 ? (
                      <div className="bg-gray-50 rounded p-3">
                        <h6 className="text-sm font-medium text-gray-900 mb-2">Extracted Data:</h6>
                        <div className="space-y-1">
                          {Object.entries(result.extracted_data).map(([key, value]: [string, any]) => (
                            <div key={key} className="flex justify-between text-xs">
                              <span className="text-gray-600">{key}:</span>
                              <span className="text-gray-900">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500">No data extracted from this document.</p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-red-600">{result.error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
        {JSON.stringify(results.data, null, 2)}
      </pre>
    );
  };

  return (
    <div className="space-y-6">
      {/* Section Navigation */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">Immigration Law Center</h2>
          <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveSection('cases')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeSection === 'cases'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              📋 Case Management
            </button>
            <button
              onClick={() => setActiveSection('research')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeSection === 'research'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              🌍 Country Research
            </button>
            <button
              onClick={() => setActiveSection('documents')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeSection === 'documents'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              📎 Document Tools
            </button>
          </div>
        </div>
        
        <p className="text-gray-600">
          {activeSection === 'cases' && 'Create and manage immigration cases, track deadlines, and monitor case progress'}
          {activeSection === 'research' && 'Research country conditions and analyze testimonies for asylum cases'}
          {activeSection === 'documents' && 'Classify and analyze immigration documents automatically'}
        </p>
      </div>

      {/* Case Management Section */}
      {activeSection === 'cases' && <ImmigrationCaseManager />}

      {/* Country Research Section */}
      {activeSection === 'research' && (
        <div className="space-y-6">
          {/* Country Conditions Research */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🌍</span>
              Country Conditions Research
            </h3>
            
            <div className="p-4 bg-green-50 rounded-lg border border-green-200 mb-6">
              <p className="text-sm text-green-800 mb-4">
                Research current human rights conditions, government persecution, and violence in specific countries for asylum cases.
                <br />
                <span className="text-xs text-green-700 mt-1 block">
                  ⏱️ Research may take 1-2 minutes for comprehensive analysis
                </span>
              </p>
              <div className="flex gap-4 mb-4">
                <input
                  type="text"
                  value={formData.country}
                  onChange={(e) => setFormData(prev => ({...prev, country: e.target.value}))}
                  placeholder="Country and topic (e.g., 'China LGBTQ', 'Afghanistan women')"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <button
                  onClick={researchCountryConditions}
                  disabled={loading || !formData.country}
                  className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 transition-all disabled:bg-gray-400 font-medium"
                >
                  {loading ? 'Researching...' : 'Research'}
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {['China LGBTQ', 'Afghanistan women', 'Myanmar military', 'Venezuela economy', 'Syria conflict'].map(example => (
                  <button
                    key={example}
                    onClick={() => setFormData(prev => ({...prev, country: example}))}
                    className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded hover:bg-green-200 transition-all"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            {/* Credible Fear Analysis */}
            <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
              <h4 className="font-medium text-purple-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Credible Fear Testimony Analysis
              </h4>
              <p className="text-sm text-purple-800 mb-4">
                Analyze client testimony for consistency, completeness, and strength for credible fear interviews.
              </p>
              <textarea
                value={formData.testimony}
                onChange={(e) => setFormData(prev => ({...prev, testimony: e.target.value}))}
                placeholder="Enter client testimony for analysis..."
                className="w-full px-3 py-2 border border-gray-300 rounded-md mb-4 h-32 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <button
                onClick={analyzeTestimony}
                disabled={loading || !formData.testimony || !formData.country}
                className="bg-purple-600 text-white px-6 py-2 rounded-md hover:bg-purple-700 transition-all disabled:bg-gray-400 font-medium"
              >
                {loading ? 'Analyzing...' : 'Analyze Testimony'}
              </button>
              <p className="text-xs text-purple-700 mt-2">
                ⚠️ Requires both country (from above) and testimony to be filled
              </p>
            </div>

            {/* Results Display */}
            {results && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-gray-900 capitalize">
                    {results.type.replace('_', ' ')} Results
                  </h3>
                  <span className="text-xs text-gray-500">{results.timestamp}</span>
                </div>
                {renderResult()}
              </div>
            )}

            {loading && (
              <div className="text-center py-8">
                <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-600 font-medium mb-2">Researching country conditions...</p>
                <p className="text-sm text-gray-500">
                  This may take 1-2 minutes for comprehensive research. 
                  <br />
                  Searching documents and analyzing current conditions...
                </p>
                <div className="mt-4 flex justify-center space-x-2">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Document Tools Section */}
      {activeSection === 'documents' && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="text-2xl">📎</span>
            Immigration Document Tools
          </h3>
          
          {/* File Upload Section */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-3">📁 Select Documents</h4>
            <div className="flex items-center gap-4">
              <input
                type="file"
                multiple
                accept=".pdf,.doc,.docx,.txt"
                onChange={handleFileSelect}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-600">
                {selectedFiles.length} file(s) selected
              </span>
            </div>
            {selectedFiles.length > 0 && (
              <div className="mt-3">
                <p className="text-sm text-gray-700 mb-2">Selected files:</p>
                <div className="space-y-1">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between text-sm bg-white rounded p-2">
                      <span className="text-gray-900">{file.name}</span>
                      <span className="text-gray-500">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h4 className="font-medium text-blue-900 mb-3">📄 Document Classification</h4>
              <p className="text-sm text-blue-800 mb-4">
                Automatically classify uploaded documents by type (I-589, evidence, identity docs, etc.)
              </p>
              <button 
                onClick={classifyDocuments}
                disabled={loading || selectedFiles.length === 0}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Classifying...' : 'Upload & Classify Documents'}
              </button>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
              <h4 className="font-medium text-green-900 mb-3">🌐 Translation Detection</h4>
              <p className="text-sm text-green-800 mb-4">
                Detect document language and identify documents that need certified translation
              </p>
              <button 
                onClick={checkTranslationNeeds}
                disabled={loading || selectedFiles.length === 0}
                className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Checking...' : 'Check Translation Needs'}
              </button>
            </div>
            
            <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
              <h4 className="font-medium text-purple-900 mb-3">✅ Completeness Check</h4>
              <p className="text-sm text-purple-800 mb-4">
                Verify all required documents are present for specific case types
              </p>
              <button 
                onClick={runCompletenessCheck}
                disabled={loading}
                className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Checking...' : 'Run Completeness Check'}
              </button>
            </div>
            
            <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
              <h4 className="font-medium text-orange-900 mb-3">📋 Form Pre-fill</h4>
              <p className="text-sm text-orange-800 mb-4">
                Extract information from documents to pre-fill USCIS forms
              </p>
              <button 
                onClick={generatePrefilledForms}
                disabled={loading || selectedFiles.length === 0}
                className="bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Extracting...' : 'Generate Pre-filled Forms'}
              </button>
            </div>
          </div>
          
          {/* Results Display */}
          {results && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-gray-900 capitalize">
                  {results.type.replace('_', ' ')} Results
                </h3>
                <span className="text-xs text-gray-500">{results.timestamp}</span>
              </div>
              {renderResult()}
            </div>
          )}

          {loading && (
            <div className="text-center py-8">
              <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-600 font-medium mb-2">Processing documents...</p>
              <p className="text-sm text-gray-500">
                Analyzing immigration documents and extracting information...
              </p>
            </div>
          )}
          
          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h4 className="font-medium text-yellow-900 mb-2">💡 Document Processing Tips:</h4>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• Upload all case documents at once for comprehensive analysis</li>
              <li>• The system will automatically detect USCIS forms and evidence documents</li>
              <li>• Documents in foreign languages will be flagged for translation requirements</li>
              <li>• Classification results help organize case files efficiently</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};
