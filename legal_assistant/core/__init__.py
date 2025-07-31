"""Core functionality package"""
from .dependencies import (
    initialize_nlp_models,
    initialize_feature_flags,
    get_nlp,
    get_sentence_model,
    get_embeddings
)
from .security import get_current_user, security
from .exceptions import (
    LegalAssistantException,
    DocumentProcessingError,
    ContainerError,
    RetrievalError,
    AnalysisError,
    AuthenticationError
)

__all__ = [
    'initialize_nlp_models',
    'initialize_feature_flags',
    'get_nlp',
    'get_sentence_model',
    'get_embeddings',
    'get_current_user',
    'security',
    'LegalAssistantException',
    'DocumentProcessingError',
    'ContainerError',
    'RetrievalError',
    'AnalysisError',
    'AuthenticationError'
]
