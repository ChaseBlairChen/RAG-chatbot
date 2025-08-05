from datetime import datetime
from typing import Optional, List, Dict, Any
from beanie import Document
from pydantic import Field
from bson import ObjectId

class UserDocument(Document):
    """User information in MongoDB"""
    user_id: str = Field(..., index=True)
    email: Optional[str] = None
    container_id: Optional[str] = None
    subscription_tier: str = "free"
    external_db_access: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Settings:
        name = "users"
        indexes = [
            "user_id",
            "email",
            "last_active"
        ]

class UploadedFileDocument(Document):
    """Uploaded file metadata in MongoDB"""
    file_id: str = Field(..., index=True)
    filename: str
    user_id: str = Field(..., index=True)
    container_id: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    file_size: int
    file_ext: str
    status: str = "processing"  # queued, processing, completed, failed
    progress: int = 0
    pages_processed: int = 0
    chunks_created: int = 0
    processing_time: float = 0.0
    content_length: int = 0
    extraction_quality: float = 1.0
    processing_method: Optional[str] = None
    validation_warnings: List[str] = Field(default_factory=list)
    processing_warnings: List[str] = Field(default_factory=list)
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    
    class Settings:
        name = "uploaded_files"
        indexes = [
            "file_id",
            "user_id", 
            "status",
            "uploaded_at"
        ]

class ProcessingStatusDocument(Document):
    """Document processing status in MongoDB"""
    file_id: str = Field(..., index=True)
    status: str
    progress: int = Field(ge=0, le=100)
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    stage: Optional[str] = None
    
    class Settings:
        name = "processing_status"
        indexes = [
            "file_id",
            "status",
            "started_at"
        ]

class ConversationDocument(Document):
    """Conversation history in MongoDB"""
    session_id: str = Field(..., index=True)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = Field(None, index=True)
    
    class Settings:
        name = "conversations"
        indexes = [
            "session_id",
            "user_id",
            "last_accessed"
        ]

class ImmigrationCaseDocument(Document):
    """Immigration cases in MongoDB"""
    case_id: str = Field(..., index=True)
    case_type: str
    client_id: str  # Encrypted
    priority_date: Optional[datetime] = None
    filing_date: Optional[datetime] = None
    status: str = "pending"
    assigned_attorney: Optional[str] = Field(None, index=True)
    deadlines: List[Dict[str, Any]] = Field(default_factory=list)
    forms: List[str] = Field(default_factory=list)
    evidence_checklist: Dict[str, bool] = Field(default_factory=dict)
    notes: Optional[str] = None
    language: str = "en"
    requires_translation: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "immigration_cases"
        indexes = [
            "case_id",
            "assigned_attorney",
            "status",
            "created_at"
        ]
