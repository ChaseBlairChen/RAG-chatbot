"""Pydantic models for API requests and responses"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from .enums import AnalysisType

class User(BaseModel):
    user_id: str
    email: Optional[str] = None
    container_id: Optional[str] = None
    subscription_tier: str = "free"
    external_db_access: List[str] = []

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"
    user_id: Optional[str] = None
    search_scope: Optional[str] = "all"
    external_databases: Optional[List[str]] = []
    use_enhanced_rag: Optional[bool] = True
    document_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None
    session_id: str
    confidence_score: float = 0.0
    expand_available: bool = False
    sources_searched: List[str] = []
    retrieval_method: Optional[str] = None

class ComprehensiveAnalysisRequest(BaseModel):
    document_id: Optional[str] = None
    analysis_types: List[AnalysisType] = [AnalysisType.COMPREHENSIVE]
    user_id: str
    session_id: Optional[str] = None
    response_style: str = "detailed"

class StructuredAnalysisResponse(BaseModel):
    document_summary: Optional[str] = None
    key_clauses: Optional[str] = None
    risk_assessment: Optional[str] = None
    timeline_deadlines: Optional[str] = None
    party_obligations: Optional[str] = None
    missing_clauses: Optional[str] = None
    confidence_scores: Dict[str, float] = {}
    sources_by_section: Dict[str, List[Dict]] = {}
    overall_confidence: float = 0.0
    processing_time: float = 0.0
    warnings: List[str] = []
    retrieval_method: str = "comprehensive_analysis"

class UserDocumentUpload(BaseModel):
    user_id: str
    file_id: str
    filename: str
    upload_timestamp: str
    pages_processed: int
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    message: str
    file_id: str
    pages_processed: int
    processing_time: float
    warnings: List[str]
    session_id: str
    user_id: str
    container_id: str
    status: str = "completed"  # NEW: Add status field

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
