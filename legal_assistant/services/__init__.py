"""Services package"""
from .document_processor import SafeDocumentProcessor
from .container_manager import UserContainerManager, get_container_manager, initialize_container_manager
from .rag_service import (
    enhanced_retrieval_v2,
    combined_search,
    load_database,
    remove_duplicate_documents,
    calculate_confidence_score
)
from .analysis_service import ComprehensiveAnalysisProcessor
from .ai_service import call_openrouter_api

# Import only what exists in external_db_service
try:
    from .external_db_service import (
        search_external_databases,
        search_free_legal_databases,
        search_free_legal_databases_enhanced,
        get_database_status
    )
except ImportError as e:
    print(f"Warning: Some external database functions not available: {e}")

__all__ = [
    'SafeDocumentProcessor',
    'UserContainerManager',
    'get_container_manager',
    'initialize_container_manager',
    'enhanced_retrieval_v2',
    'combined_search',
    'load_database',
    'remove_duplicate_documents',
    'calculate_confidence_score',
    'ComprehensiveAnalysisProcessor',
    'call_openrouter_api',
    'search_external_databases',
    'search_free_legal_databases',
    'search_free_legal_databases_enhanced',
    'get_database_status'
]
