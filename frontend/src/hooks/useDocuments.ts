// hooks/useDocuments.ts
import { useState, useCallback, useEffect } from 'react';
import type { DocumentAnalysis, UploadStatus } from '../types';
import { ApiService } from '../services/api';
import { validateFileBeforeUpload } from '../utils/fileValidation';

export const useDocuments = (apiService: ApiService, isBackendConfigured: boolean) => {
  const [userDocuments, setUserDocuments] = useState<any[]>([]);
  const [documentAnalyses, setDocumentAnalyses] = useState<DocumentAnalysis[]>([]);
  const [uploadQueue, setUploadQueue] = useState<File[]>([]);
  const [currentlyUploading, setCurrentlyUploading] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResults, setUploadResults] = useState<any[]>([]);
  const [uploadStatuses, setUploadStatuses] = useState<Map<string, UploadStatus>>(new Map());
  const [statusCheckIntervals, setStatusCheckIntervals] = useState<Map<string, NodeJS.Timeout>>(new Map());

  // Cleanup intervals on unmount
  useEffect(() => {
    return () => {
      statusCheckIntervals.forEach(interval => clearInterval(interval));
    };
  }, [statusCheckIntervals]);

  const loadUserDocuments = useCallback(async () => {
    if (!isBackendConfigured) return;

    try {
      const data = await apiService.get<any>('/user/documents');
      setUserDocuments(data.documents || []);
      
      const docAnalyses = (data.documents || []).map((doc: any) => ({
        id: doc.file_id,
        filename: doc.filename,
        uploadedAt: doc.uploaded_at,
        pagesProcessed: doc.pages_processed,
        analysisResults: {},
        lastAnalyzed: null,
        confidence: null
      }));
      setDocumentAnalyses(docAnalyses);
    } catch (error) {
      console.error('Failed to load user documents:', error);
    }
  }, [apiService, isBackendConfigured]);

  const checkDocumentStatus = useCallback(async (fileId: string): Promise<UploadStatus | null> => {
    try {
      const data = await apiService.get<any>(`/user/documents/${fileId}/status`);
      return {
        fileId: data.file_id,
        filename: data.filename,
        status: data.status,
        progress: data.progress || 0,
        message: data.message || '',
        pagesProcessed: data.pages_processed,
        chunksCreated: data.chunks_created,
        processingTime: data.processing_time,
        errors: data.errors || []
      };
    } catch (error) {
      console.error('Status check error:', error);
      return null;
    }
  }, [apiService]);

  const startProgressTracking = useCallback((fileId: string, filename: string) => {
    const initialStatus: UploadStatus = {
      fileId,
      filename,
      status: 'uploading',
      progress: 0,
      message: 'Uploading document...'
    };
    
    setUploadStatuses(prev => new Map(prev).set(fileId, initialStatus));

    const interval = setInterval(async () => {
      const status = await checkDocumentStatus(fileId);
      
      if (status) {
        setUploadStatuses(prev => new Map(prev).set(fileId, status));
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          setStatusCheckIntervals(prev => {
            const newMap = new Map(prev);
            newMap.delete(fileId);
            return newMap;
          });
          
          if (status.status === 'completed') {
            setTimeout(() => {
              setUploadStatuses(prev => {
                const newMap = new Map(prev);
                newMap.delete(fileId);
                return newMap;
              });
            }, 5000);
          }
        }
      }
    }, 2000);

    setStatusCheckIntervals(prev => new Map(prev).set(fileId, interval));

    setTimeout(() => {
      if (statusCheckIntervals.has(fileId)) {
        clearInterval(interval);
        setStatusCheckIntervals(prev => {
          const newMap = new Map(prev);
          newMap.delete(fileId);
          return newMap;
        });
        
        setUploadStatuses(prev => new Map(prev).set(fileId, {
          ...initialStatus,
          status: 'failed',
          message: 'Document processing timed out',
          errors: ['Processing took too long']
        }));
      }
    }, 5 * 60 * 1000);
  }, [checkDocumentStatus, statusCheckIntervals]);

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return;

    const fileArray = Array.from(files);
    const validFiles: File[] = [];
    const errors: string[] = [];

    fileArray.forEach(file => {
      const validationError = validateFileBeforeUpload(file);
      if (validationError) {
        errors.push(validationError);
      } else {
        validFiles.push(file);
      }
    });

    if (errors.length > 0) {
      alert('Some files had issues:\n\n' + errors.join('\n\n'));
    }

    if (validFiles.length > 0) {
      setUploadQueue(prev => [...prev, ...validFiles]);
    }
  }, []);

  const uploadAllDocuments = useCallback(async (runAnalysisAfter = false): Promise<string[]> => {
    if (uploadQueue.length === 0) return [];

    setUploadResults([]);
    const results: any[] = [];
    const uploadedDocIds: string[] = [];

    for (let i = 0; i < uploadQueue.length; i++) {
      const file = uploadQueue[i];
      setCurrentlyUploading(file);
      setUploadProgress(((i) / uploadQueue.length) * 100);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const data = await apiService.uploadFile('/user/upload', formData);
        
        if (data.status === 'processing' && data.file_id) {
          startProgressTracking(data.file_id, file.name);
        }
        
        results.push({
          filename: file.name,
          success: true,
          pages_processed: data.pages_processed,
          file_id: data.file_id,
          processing_time: data.processing_time,
          warnings: data.warnings || [],
          status: data.status
        });

        uploadedDocIds.push(data.file_id);
      } catch (error) {
        console.error(`Upload failed for ${file.name}:`, error);
        
        let errorMessage = 'Upload failed';
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        results.push({
          filename: file.name,
          success: false,
          error: errorMessage
        });
      }
    }

    setUploadProgress(100);
    setUploadResults(results);
    setCurrentlyUploading(null);
    setUploadQueue([]);
    
    setTimeout(async () => {
      await loadUserDocuments();
    }, 3000);

    return uploadedDocIds;
  }, [uploadQueue, apiService, startProgressTracking, loadUserDocuments]);

  const deleteDocument = useCallback(async (fileId: string) => {
    try {
      await apiService.delete(`/user/documents/${fileId}`);
      await loadUserDocuments();
      alert('Document deleted successfully');
    } catch (error) {
      console.error('Delete failed:', error);
      alert(`Delete failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [apiService, loadUserDocuments]);

  const removeFromQueue = useCallback((index: number) => {
    setUploadQueue(prev => prev.filter((_, i) => i !== index));
  }, []);

  const clearQueue = useCallback(() => {
    setUploadQueue([]);
    setUploadResults([]);
  }, []);

  const clearStatuses = useCallback(() => {
    statusCheckIntervals.forEach(interval => clearInterval(interval));
    setStatusCheckIntervals(new Map());
    setUploadStatuses(new Map());
  }, [statusCheckIntervals]);

  return {
    userDocuments,
    documentAnalyses,
    uploadQueue,
    currentlyUploading,
    uploadProgress,
    uploadResults,
    uploadStatuses,
    setDocumentAnalyses,
    loadUserDocuments,
    handleFileUpload,
    uploadAllDocuments,
    deleteDocument,
    removeFromQueue,
    clearQueue,
    clearStatuses
  };
};