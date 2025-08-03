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
from .external_db_service import (
    LegalDatabaseInterface,
    LexisNexisInterface,
    WestlawInterface,
    search_external_databases
)

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
    'LegalDatabaseInterface',
    'LexisNexisInterface',
    'WestlawInterface',
    'search_external_databases'
]

