"""Custom exceptions for the application"""

class LegalAssistantException(Exception):
    """Base exception for all custom exceptions"""
    pass

class DocumentProcessingError(LegalAssistantException):
    """Raised when document processing fails"""
    pass

class ContainerError(LegalAssistantException):
    """Raised when container operations fail"""
    pass

class RetrievalError(LegalAssistantException):
    """Raised when document retrieval fails"""
    pass

class AnalysisError(LegalAssistantException):
    """Raised when analysis operations fail"""
    pass

class AuthenticationError(LegalAssistantException):
    """Raised when authentication fails"""
    pass
