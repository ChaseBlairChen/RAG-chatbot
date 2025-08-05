# legal_assistant/processors/query_processor.py - COMPLETE MERGED VERSION
"""
Complete Query Processing Service - Merges class-based architecture with enhanced timeout handling,
anti-hallucination measures, progress tracking, and all original functionality.
"""
import re
import logging
import traceback
import asyncio
import hashlib
import time
from typing import Optional, Dict, List, Tuple, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..models import QueryResponse, ComprehensiveAnalysisRequest, AnalysisType
from ..config import FeatureFlags, OPENROUTER_API_KEY, MIN_RELEVANCE_SCORE
from ..services import (
    ComprehensiveAnalysisProcessor,
    combined_search,
    calculate_confidence_score,
    call_openrouter_api
)
from ..storage.managers import add_to_conversation, get_conversation_context
from ..utils import (
    parse_multiple_questions,
    extract_bill_information,
    extract_universal_information,
    format_context_for_llm
)

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classifications"""
    STATUTORY = "statutory"
    IMMIGRATION = "immigration"
    LEGAL_SEARCH = "legal_search"
    GOVERNMENT_DATA = "government_data"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    GENERAL = "general"

class ProcessingStage(Enum):
    """Processing stages for progress tracking"""
    INITIALIZING = "initializing"
    ANALYZING_QUERY = "analyzing_query"
    SEARCHING_INTERNAL = "searching_internal"
    SEARCHING_EXTERNAL = "searching_external"
    EXTRACTING_INFO = "extracting_info"
    GENERATING_RESPONSE = "generating_response"
    VALIDATING_RESPONSE = "validating_response"
    COMPLETING = "completing"

@dataclass
class QueryContext:
    """Container for query processing context"""
    original_question: str
    expanded_questions: List[str]
    query_types: List[str]
    search_scope: str
    user_id: Optional[str]
    session_id: str
    document_id: Optional[str]
    use_enhanced_rag: bool
    response_style: str
    conversation_context: str

@dataclass
class SearchResults:
    """Container for search results"""
    internal_results: List[Tuple]
    external_context: Optional[str]
    external_source_info: List[Dict]
    sources_searched: List[str]
    retrieval_method: str

@dataclass
class ProcessingResult:
    """Container for processing results"""
    response_text: str
    confidence_score: float
    sources: List[Dict]
    warnings: List[str]
    processing_time: float

@dataclass
class ProcessingProgress:
    """Track processing progress"""
    stage: ProcessingStage
    progress_percent: int
    message: str
    partial_results: Optional[Dict] = None
    started_at: datetime = None
    stage_start: datetime = None

class AdaptiveTimeoutHandler:
    """Enhanced timeout handler with adaptive timeouts and progress tracking"""
    
    def __init__(self):
        self.base_timeouts = {
            ProcessingStage.INITIALIZING: 2,
            ProcessingStage.ANALYZING_QUERY: 3,
            ProcessingStage.SEARCHING_INTERNAL: 8,
            ProcessingStage.SEARCHING_EXTERNAL: 12,
            ProcessingStage.EXTRACTING_INFO: 3,
            ProcessingStage.GENERATING_RESPONSE: 15,
            ProcessingStage.VALIDATING_RESPONSE: 2,
            ProcessingStage.COMPLETING: 2
        }
        
        self.complexity_multipliers = {
            'simple': 0.7,      # Short questions, basic queries
            'medium': 1.0,      # Normal complexity
            'complex': 1.5,     # Long questions, multiple parts
            'comprehensive': 2.0 # Analysis requests, complex legal research
        }
    
    def calculate_adaptive_timeout(self, query: str, query_types: List[str], 
                                 search_scope: str) -> int:
        """Calculate adaptive timeout based on query complexity"""
        
        # Base timeout
        base_timeout = sum(self.base_timeouts.values())
        
        # Determine complexity
        complexity = self._assess_query_complexity(query, query_types, search_scope)
        multiplier = self.complexity_multipliers.get(complexity, 1.0)
        
        # Calculate adaptive timeout
        adaptive_timeout = int(base_timeout * multiplier)
        
        # Apply bounds (min 15s, max 60s)
        return max(15, min(60, adaptive_timeout))
    
    def _assess_query_complexity(self, query: str, query_types: List[str], 
                                search_scope: str) -> str:
        """Assess query complexity for adaptive timeout calculation"""
        
        complexity_factors = 0
        
        # Length factor
        if len(query) > 200:
            complexity_factors += 1
        elif len(query) > 100:
            complexity_factors += 0.5
        
        # Multiple questions
        if query.count('?') > 1 or ';' in query:
            complexity_factors += 1
        
        # Complex query types
        complex_types = ['statutory', 'comprehensive_analysis', 'immigration']
        if any(qt in query_types for qt in complex_types):
            complexity_factors += 1
        
        # External search requirement
        if search_scope == "all":
            complexity_factors += 0.5
        
        # Legal complexity indicators
        legal_indicators = ['bill', 'statute', 'regulation', 'case law', 'precedent']
        if sum(1 for indicator in legal_indicators if indicator in query.lower()) > 2:
            complexity_factors += 0.5
        
        # Classify complexity
        if complexity_factors >= 3:
            return 'comprehensive'
        elif complexity_factors >= 2:
            return 'complex'
        elif complexity_factors >= 1:
            return 'medium'
        else:
            return 'simple'

class ProgressTracker:
    """Track and report processing progress"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.current_progress = ProcessingProgress(
            stage=ProcessingStage.INITIALIZING,
            progress_percent=0,
            message="Initializing query processing...",
            started_at=datetime.utcnow()
        )
    
    def update_stage(self, stage: ProcessingStage, message: str, 
                    progress_percent: int, partial_results: Dict = None):
        """Update processing stage with progress"""
        
        self.current_progress = ProcessingProgress(
            stage=stage,
            progress_percent=progress_percent,
            message=message,
            partial_results=partial_results,
            started_at=self.current_progress.started_at,
            stage_start=datetime.utcnow()
        )
        
        # Call progress callback if provided
        if self.progress_callback:
            try:
                self.progress_callback(self.current_progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def get_current_progress(self) -> ProcessingProgress:
        """Get current processing progress"""
        return self.current_progress

class QueryProcessor:
    """
    Complete query processing service with enhanced timeout handling, class-based architecture,
    anti-hallucination measures, and comprehensive legal research capabilities.
    """
    
    def __init__(self):
        """Initialize the query processor with all dependencies"""
        self.logger = logging.getLogger(f"{__name__}.QueryProcessor")
        
        # Initialize services
        self._init_services()
        
        # Initialize feature flags
        self._init_feature_availability()
        
        # Initialize enhanced timeout handling
        self.timeout_handler = AdaptiveTimeoutHandler()
        self.active_queries = {}  # Track active queries for monitoring
        
        # Processing metrics
        self.processing_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'timeout_queries': 0,
            'error_queries': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("🚀 Complete QueryProcessor initialized successfully")
    
    def _init_services(self):
        """Initialize all external services"""
        try:
            self.analysis_processor = ComprehensiveAnalysisProcessor()
            self.logger.info("✅ Analysis processor initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize analysis processor: {e}")
            self.analysis_processor = None
        
        # Initialize external API services with error handling
        self._init_external_services()
    
    def _init_external_services(self):
        """Initialize external API services with graceful degradation"""
        self.comprehensive_apis_available = False
        self.comprehensive_researcher = None
        
        try:
            from ..services.comprehensive_legal_apis import (
                comprehensive_legal_hub, 
                search_comprehensive_legal_databases
            )
            self.comprehensive_legal_hub = comprehensive_legal_hub
            self.search_comprehensive_legal_databases = search_comprehensive_legal_databases
            self.comprehensive_apis_available = True
            self.logger.info("✅ Comprehensive legal APIs available")
        except ImportError as e:
            self.logger.warning(f"⚠️ Comprehensive legal APIs not available: {e}")
        
        try:
            from ..services.news_country_apis import comprehensive_researcher
            self.comprehensive_researcher = comprehensive_researcher
            self.logger.info("✅ Country conditions researcher available")
        except ImportError as e:
            self.logger.warning(f"⚠️ News country APIs not available: {e}")
    
    def _init_feature_availability(self):
        """Initialize feature availability flags"""
        self.features = {
            'ai_enabled': bool(FeatureFlags.AI_ENABLED and OPENROUTER_API_KEY),
            'comprehensive_apis': self.comprehensive_apis_available,
            'country_research': self.comprehensive_researcher is not None,
            'analysis_processor': self.analysis_processor is not None
        }
        
        self.logger.info(f"Feature availability: {self.features}")
    
    # === ENHANCED TIMEOUT HANDLING METHODS ===
    
    async def process_query_with_enhanced_timeout(
        self, 
        question: str, 
        session_id: str, 
        user_id: Optional[str],
        search_scope: str, 
        response_style: str = "balanced", 
        use_enhanced_rag: bool = True,
        document_id: str = None, 
        search_external: bool = None,
        progress_callback: Optional[Callable] = None
    ) -> QueryResponse:
        """
        MAIN METHOD: Process query with enhanced timeout handling, progress tracking,
        and partial result recovery.
        """
        
        # Step 1: Initialize progress tracking
        progress_tracker = ProgressTracker(progress_callback)
        query_id = hashlib.md5(f"{question}{session_id}{time.time()}".encode()).hexdigest()[:16]
        
        # Step 2: Calculate adaptive timeout
        query_types = self._detect_query_types(question)
        adaptive_timeout = self.timeout_handler.calculate_adaptive_timeout(
            question, query_types, search_scope
        )
        
        self.logger.info(f"🕐 Processing query with adaptive timeout: {adaptive_timeout}s")
        
        # Track active query
        self.active_queries[query_id] = {
            'question': question[:100] + "..." if len(question) > 100 else question,
            'started_at': datetime.utcnow(),
            'timeout': adaptive_timeout,
            'progress_tracker': progress_tracker
        }
        
        try:
            # Step 3: Process with staged timeouts and progress tracking
            result = await self._process_with_staged_timeouts(
                question, session_id, user_id, search_scope,
                response_style, use_enhanced_rag, document_id, 
                search_external, progress_tracker, adaptive_timeout
            )
            
            # Update stats
            processing_time = (datetime.utcnow() - progress_tracker.current_progress.started_at).total_seconds()
            self._update_stats('success', processing_time)
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"⏰ Query timeout after {adaptive_timeout}s")
            return await self._handle_timeout_with_partial_results(
                progress_tracker, session_id, query_id
            )
            
        except Exception as e:
            self.logger.error(f"❌ Query processing error: {e}")
            return await self._handle_processing_error(
                e, progress_tracker, session_id, query_id
            )
            
        finally:
            # Clean up active query tracking
            self.active_queries.pop(query_id, None)
    
    async def _process_with_staged_timeouts(
        self, 
        question: str, 
        session_id: str, 
        user_id: Optional[str],
        search_scope: str, 
        response_style: str, 
        use_enhanced_rag: bool,
        document_id: str, 
        search_external: bool, 
        progress_tracker: ProgressTracker,
        total_timeout: int
    ) -> QueryResponse:
        """Process query with staged timeouts and progress updates"""
        
        # Stage 1: Query Analysis (5% - 15%)
        progress_tracker.update_stage(
            ProcessingStage.ANALYZING_QUERY, 
            "Analyzing query and detecting types...", 
            5
        )
        
        query_context = await asyncio.wait_for(
            self._build_query_context_async(
                question, session_id, user_id, search_scope,
                response_style, use_enhanced_rag, document_id
            ),
            timeout=self.timeout_handler.base_timeouts[ProcessingStage.ANALYZING_QUERY]
        )
        
        query_context.query_types = self._detect_query_types(question)
        
        progress_tracker.update_stage(
            ProcessingStage.ANALYZING_QUERY,
            f"Query analysis complete - detected types: {', '.join(query_context.query_types)}",
            15,
            {'detected_types': query_context.query_types}
        )
        
        # Stage 2: Handle special cases (15% - 25%)
        if self._is_comprehensive_analysis_request(question):
            progress_tracker.update_stage(
                ProcessingStage.SEARCHING_INTERNAL,
                "Processing comprehensive analysis request...",
                20
            )
            return await asyncio.wait_for(
                self._handle_comprehensive_analysis(query_context),
                timeout=total_timeout * 0.8  # 80% of remaining time
            )
        
        # Stage 3: Internal Search (25% - 50%)
        progress_tracker.update_stage(
            ProcessingStage.SEARCHING_INTERNAL,
            "Searching your uploaded documents...",
            25
        )
        
        if search_external is None:
            search_external = self._should_search_external(question, search_scope, query_context.query_types)
        
        internal_search_timeout = self.timeout_handler.base_timeouts[ProcessingStage.SEARCHING_INTERNAL]
        if search_scope == "user_only":
            internal_search_timeout *= 1.5  # More time if only searching user docs
        
        try:
            search_results = await asyncio.wait_for(
                self._perform_searches_with_progress(query_context, search_external, progress_tracker),
                timeout=internal_search_timeout
            )
        except asyncio.TimeoutError:
            # Create partial search results
            search_results = SearchResults(
                internal_results=[],
                external_context=None,
                external_source_info=[],
                sources_searched=["timeout_during_search"],
                retrieval_method="search_timeout"
            )
        
        progress_tracker.update_stage(
            ProcessingStage.SEARCHING_INTERNAL,
            f"Search complete - found {len(search_results.internal_results)} results",
            50,
            {'results_found': len(search_results.internal_results)}
        )
        
        # Stage 4: Check for results (50% - 55%)
        if not search_results.internal_results and not search_results.external_context:
            progress_tracker.update_stage(
                ProcessingStage.COMPLETING,
                "No results found - completing response",
                90
            )
            return self._create_no_results_response(session_id, search_results.sources_searched)
        
        # Stage 5: Context Processing (55% - 65%)
        progress_tracker.update_stage(
            ProcessingStage.EXTRACTING_INFO,
            "Processing and formatting context...",
            55
        )
        
        context_text, source_info = self._process_context(search_results)
        
        # Add specialized context with timeout protection
        try:
            specialized_context = await asyncio.wait_for(
                self._get_specialized_context(question, query_context.query_types),
                timeout=3  # Quick timeout for specialized context
            )
            if specialized_context:
                context_text = specialized_context + "\n\n" + context_text
        except asyncio.TimeoutError:
            self.logger.warning("Specialized context search timed out, continuing without it")
        
        progress_tracker.update_stage(
            ProcessingStage.EXTRACTING_INFO,
            "Context processing complete",
            65,
            {'context_length': len(context_text)}
        )
        
        # Stage 6: Response Generation (65% - 85%)
        progress_tracker.update_stage(
            ProcessingStage.GENERATING_RESPONSE,
            "Generating AI response...",
            65
        )
        
        response_timeout = self.timeout_handler.base_timeouts[ProcessingStage.GENERATING_RESPONSE]
        # Adjust timeout based on context length
        if len(context_text) > 5000:
            response_timeout *= 1.3
        
        try:
            processing_result = await asyncio.wait_for(
                self._generate_response_with_progress(
                    query_context, search_results, context_text, source_info, progress_tracker
                ),
                timeout=response_timeout
            )
        except asyncio.TimeoutError:
            # Create partial response with available context
            self.logger.warning("Response generation timed out, creating partial response")
            return self._create_partial_response_from_context(
                context_text, source_info, query_context, search_results
            )
        
        progress_tracker.update_stage(
            ProcessingStage.GENERATING_RESPONSE,
            "AI response generated successfully",
            85
        )
        
        # Stage 7: Post-processing (85% - 100%)
        progress_tracker.update_stage(
            ProcessingStage.VALIDATING_RESPONSE,
            "Validating and finalizing response...",
            85
        )
        
        try:
            final_response = await asyncio.wait_for(
                self._post_process_response_async(processing_result, query_context, search_results),
                timeout=3  # Quick timeout for post-processing
            )
        except asyncio.TimeoutError:
            # Use unvalidated response if post-processing times out
            self.logger.warning("Post-processing timed out, using unvalidated response")
            final_response = self._create_basic_response_from_processing_result(
                processing_result, query_context, search_results
            )
        
        progress_tracker.update_stage(
            ProcessingStage.COMPLETING,
            "Query processing completed successfully",
            100
        )
        
        # Add to conversation history
        self._add_to_conversation_history(
            session_id, question, final_response.response, source_info
        )
        
        return final_response
    
    # === LEGACY ASYNC METHOD FOR BACKWARD COMPATIBILITY ===
    
    async def process_query_async(self, question: str, session_id: str, user_id: Optional[str], 
                                search_scope: str, response_style: str = "balanced", 
                                use_enhanced_rag: bool = True, document_id: str = None, 
                                search_external: bool = None, timeout_seconds: int = 25) -> QueryResponse:
        """
        Legacy async method - redirects to enhanced timeout handling
        """
        return await self.process_query_with_enhanced_timeout(
            question, session_id, user_id, search_scope, response_style,
            use_enhanced_rag, document_id, search_external
        )
    
    # === CORE PROCESSING METHODS ===
    
    def _build_query_context(self, question: str, session_id: str, user_id: Optional[str],
                           search_scope: str, response_style: str, use_enhanced_rag: bool,
                           document_id: str) -> QueryContext:
        """Build query processing context"""
        
        # Parse multiple questions if enhanced RAG is enabled
        expanded_questions = parse_multiple_questions(question) if use_enhanced_rag else [question]
        
        # Get conversation context
        conversation_context = get_conversation_context(session_id)
        
        return QueryContext(
            original_question=question,
            expanded_questions=expanded_questions,
            query_types=[],  # Will be filled later
            search_scope=search_scope,
            user_id=user_id,
            session_id=session_id,
            document_id=document_id,
            use_enhanced_rag=use_enhanced_rag,
            response_style=response_style,
            conversation_context=conversation_context
        )
    
    async def _build_query_context_async(self, question: str, session_id: str, user_id: Optional[str],
                                       search_scope: str, response_style: str, use_enhanced_rag: bool,
                                       document_id: str) -> QueryContext:
        """Async version of query context building"""
        return await asyncio.to_thread(
            self._build_query_context,
            question, session_id, user_id, search_scope,
            response_style, use_enhanced_rag, document_id
        )
    
    def _detect_query_types(self, question: str) -> List[str]:
        """Detect query types for intelligent processing"""
        query_types = []
        question_lower = question.lower()
        
        # Statutory/regulatory patterns
        if self._detect_statutory_question(question):
            query_types.append(QueryType.STATUTORY.value)
        
        # Immigration patterns
        if self._detect_immigration_query(question):
            query_types.append(QueryType.IMMIGRATION.value)
        
        # Legal search patterns
        if self._detect_legal_search_intent(question):
            query_types.append(QueryType.LEGAL_SEARCH.value)
        
        # Government data patterns
        if self._detect_government_data_need(question):
            query_types.append(QueryType.GOVERNMENT_DATA.value)
        
        # Enhanced query type detection for comprehensive API routing
        comprehensive_types = self._detect_comprehensive_query_types(question)
        query_types.extend(comprehensive_types)
        
        # Default to general if no specific types detected
        if not query_types:
            query_types.append(QueryType.GENERAL.value)
        
        return query_types
    
    def _detect_statutory_question(self, question: str) -> bool:
        """Detect if this is a statutory/regulatory question"""
        statutory_indicators = [
            r'\bUSC\s+\d+', r'\bU\.S\.C\.\s*§?\s*\d+', r'\bCFR\s+\d+', r'\bC\.F\.R\.\s*§?\s*\d+',
            r'\bFed\.\s*R\.\s*Civ\.\s*P\.', r'\bRCW\s+\d+\.\d+\.\d+', r'\bWAC\s+\d+',
            r'\bstatute[s]?', r'\bregulation[s]?', r'\bcode\s+section[s]?',
            r'\bminimum\s+standards?', r'\brequirements?', r'\bmust\s+meet',
            r'\bshall\s+(?:meet|comply|maintain)', r'\bmust\s+(?:include|contain|provide)',
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in statutory_indicators)
    
    def _detect_immigration_query(self, question: str) -> bool:
        """Detect if this is an immigration-related query"""
        immigration_indicators = [
            r'\basylum\b', r'\brefugee\b', r'\bgreen\s*card\b', r'\bvisa\b',
            r'\bimmigration\b', r'\bnaturalization\b', r'\bcitizenship\b',
            r'\bI-\d{3}\b', r'\bN-\d{3}\b', r'\bUSCIS\b',
            r'\bcredible\s*fear\b', r'\bremoval\b', r'\bdeportation\b',
            r'\bwork\s*permit\b', r'\bEAD\b', r'\bpriority\s*date\b',
            r'\bcountry\s*conditions?\b', r'\bpersecution\b', r'\bhuman\s*rights\b',
            r'\bcase\s*status\b', r'\bprocessing\s*time\b', r'\binterview\b',
            r'\b[A-Z]{3}\d{10}\b'  # USCIS receipt number format
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in immigration_indicators)
    
    def _detect_legal_search_intent(self, question: str) -> bool:
        """Detect if this question would benefit from external legal database search"""
        legal_search_indicators = [
            r'\bcase\s*law\b', r'\bcases?\b', r'\bprecedent\b', r'\bruling\b',
            r'\bdecision\b', r'\bcourt\s*opinion\b', r'\bjudgment\b',
            r'\bmiranda\b', r'\bconstitutional\b', r'\bamendment\b',
            r'\bsupreme\s*court\b', r'\bappellate\b', r'\bdistrict\s*court\b',
            r'\blegal\s*research\b', r'\bfind\s*cases?\b', r'\blook\s*up\s*law\b',
            r'\bsearch\s*(?:for\s*)?(?:cases?|law|precedent)\b',
            r'\bliability\b', r'\bnegligence\b', r'\bcontract\s*law\b', r'\btort\b',
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in legal_search_indicators)
    
    def _detect_government_data_need(self, question: str) -> bool:
        """Detect if question needs government enforcement/statistical data"""
        government_indicators = [
            r'\bviolation\b', r'\benforcement\b', r'\bcitation\b', r'\bpenalty\b',
            r'\bfine\b', r'\binspection\b', r'\bcompliance\b',
            r'\bepa\b', r'\bosha\b', r'\bsec\b', r'\bfda\b', r'\buscis\b',
            r'\bfbi\b', r'\bdol\b', r'\bhud\b',
            r'\bstatistics\b', r'\bdata\b', r'\brates?\b', r'\btrends?\b',
            r'\bnumbers?\b', r'\bfigures?\b', r'\breport\b',
            r'\bstatus\b', r'\bprocessing\s*time\b', r'\bcase\s*status\b',
            r'\bcurrent\b', r'\brecent\b', r'\blatest\b'
        ]
        
        return any(re.search(pattern, question.lower()) for pattern in government_indicators)
    
    def _detect_comprehensive_query_types(self, question: str) -> List[str]:
        """Enhanced query type detection for comprehensive API routing"""
        query_types = []
        question_lower = question.lower()
        
        # Environmental law patterns
        environmental_patterns = [
            r'\bepa\b', r'\benvironmental\b', r'\bpollution\b', r'\bclimate\b',
            r'\bemissions\b', r'\bclean\s+air\b', r'\bclean\s+water\b',
            r'\bwetlands\b', r'\bendangered\s+species\b', r'\bnepa\b',
        ]
        
        # Business law patterns
        business_patterns = [
            r'\bsec\b', r'\bsecurities\b', r'\bcorporate\b', r'\bfilings?\b',
            r'\bipo\b', r'\bmerger\b', r'\bacquisition\b', r'\b10-k\b', r'\b10-q\b',
        ]
        
        # Labor law patterns
        labor_patterns = [
            r'\bosha\b', r'\bworkplace\s+safety\b', r'\blabor\b', r'\bemployment\b',
            r'\bwage\b', r'\bovertime\b', r'\bdiscrimination\b', r'\bharassment\b',
        ]
        
        # Healthcare patterns
        healthcare_patterns = [
            r'\bfda\b', r'\bdrug\b', r'\bmedical\s+device\b', r'\brecall\b',
            r'\bhipaa\b', r'\bpatient\s+privacy\b', r'\bmedicare\b', r'\bmedicaid\b',
        ]
        
        # Criminal justice patterns
        criminal_patterns = [
            r'\bcriminal\b', r'\bcrime\b', r'\bfbi\b', r'\barrest\b',
            r'\bprosecution\b', r'\bsentencing\b', r'\bprison\b', r'\bfelony\b',
        ]
        
        # Housing patterns
        housing_patterns = [
            r'\bhousing\b', r'\brental\b', r'\blandlord\b', r'\btenant\b',
            r'\beviction\b', r'\bfair\s+housing\b', r'\bhud\b', r'\bsection\s+8\b',
        ]
        
        # Check each pattern type
        pattern_map = {
            'environmental': environmental_patterns,
            'business': business_patterns,
            'labor': labor_patterns,
            'healthcare': healthcare_patterns,
            'criminal': criminal_patterns,
            'housing': housing_patterns
        }
        
        for area, patterns in pattern_map.items():
            if any(re.search(pattern, question_lower) for pattern in patterns):
                query_types.append(area)
        
        return query_types
    
    def _is_comprehensive_analysis_request(self, question: str) -> bool:
        """Check if this is a comprehensive analysis request"""
        analysis_phrases = [
            "comprehensive analysis", "complete analysis", "full analysis",
            "detailed analysis", "thorough analysis", "analyze this document"
        ]
        return any(phrase in question.lower() for phrase in analysis_phrases)
    
    async def _handle_comprehensive_analysis(self, query_context: QueryContext) -> QueryResponse:
        """Handle comprehensive analysis requests"""
        if not self.analysis_processor:
            return QueryResponse(
                response="Comprehensive analysis is not available at this time.",
                error="Analysis processor not initialized",
                context_found=False,
                sources=[],
                session_id=query_context.session_id,
                confidence_score=0.0,
                sources_searched=[],
                retrieval_method="analysis_unavailable"
            )
        
        try:
            comp_request = ComprehensiveAnalysisRequest(
                document_id=query_context.document_id,
                analysis_types=[AnalysisType.COMPREHENSIVE],
                user_id=query_context.user_id or "default_user",
                session_id=query_context.session_id,
                response_style=query_context.response_style
            )
            
            comp_result = self.analysis_processor.process_comprehensive_analysis(comp_request)
            
            formatted_response = self._format_comprehensive_analysis_response(comp_result)
            
            add_to_conversation(query_context.session_id, "user", query_context.original_question)
            add_to_conversation(query_context.session_id, "assistant", formatted_response)
            
            return QueryResponse(
                response=formatted_response,
                error=None,
                context_found=True,
                sources=comp_result.sources_by_section.get('summary', []),
                session_id=query_context.session_id,
                confidence_score=comp_result.overall_confidence,
                expand_available=False,
                sources_searched=["comprehensive_analysis"],
                retrieval_method=comp_result.retrieval_method
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return QueryResponse(
                response="Comprehensive analysis encountered an error. Please try a simpler question.",
                error=str(e),
                context_found=False,
                sources=[],
                session_id=query_context.session_id,
                confidence_score=0.0,
                sources_searched=[],
                retrieval_method="analysis_error"
            )
    
    def _should_search_external(self, question: str, search_scope: str, query_types: List[str]) -> bool:
        """Determine if comprehensive government databases should be searched"""
        
        # Don't search external if user explicitly wants only their documents
        if search_scope == "user_only":
            return False
        
        # Check for government data needs
        if self._detect_government_data_need(question):
            return True
        
        # Check for legal areas that benefit from comprehensive search
        comprehensive_areas = ['environmental', 'business', 'labor', 'healthcare', 'criminal', 'immigration', 'housing']
        if any(area in query_types for area in comprehensive_areas):
            return True
        
        # Check for enforcement/compliance questions
        enforcement_keywords = ['violation', 'enforcement', 'compliance', 'citation', 'penalty', 'recall']
        if any(keyword in question.lower() for keyword in enforcement_keywords):
            return True
        
        return self._detect_legal_search_intent(question) or self._detect_statutory_question(question)
    
    async def _perform_searches_with_progress(
        self, 
        query_context: QueryContext, 
        search_external: bool,
        progress_tracker: ProgressTracker
    ) -> SearchResults:
        """Perform searches with progress updates and timeout protection"""
        
        # Initialize search results
        internal_results = []
        external_context = None
        external_source_info = []
        sources_searched = []
        retrieval_method = "enhanced_search"
        
        # Internal search with progress
        progress_tracker.update_stage(
            ProcessingStage.SEARCHING_INTERNAL,
            "Searching uploaded documents...",
            30
        )
        
        try:
            combined_query = " ".join(query_context.expanded_questions)
            search_k = 20 if 'statutory' in query_context.query_types else 15
            
            internal_results, internal_sources, retrieval_method = await asyncio.wait_for(
                asyncio.to_thread(
                    combined_search,
                    combined_query,
                    query_context.user_id,
                    query_context.search_scope,
                    query_context.conversation_context,
                    query_context.use_enhanced_rag,
                    search_k,
                    query_context.document_id
                ),
                timeout=8  # 8 second timeout for internal search
            )
            
            sources_searched.extend(internal_sources)
            
            progress_tracker.update_stage(
                ProcessingStage.SEARCHING_INTERNAL,
                f"Internal search complete - {len(internal_results)} results found",
                40,
                {'internal_results': len(internal_results)}
            )
            
        except asyncio.TimeoutError:
            self.logger.warning("Internal search timed out")
            sources_searched.append("internal_search_timeout")
            progress_tracker.update_stage(
                ProcessingStage.SEARCHING_INTERNAL,
                "Internal search timed out - continuing with external search",
                35
            )
        
        # External search with progress
        if search_external:
            progress_tracker.update_stage(
                ProcessingStage.SEARCHING_EXTERNAL,
                "Searching external legal databases...",
                45
            )
            
            try:
                external_context, external_source_info = await asyncio.wait_for(
                    self._search_external_databases(
                        query_context.original_question, query_context.query_types
                    ),
                    timeout=10  # 10 second timeout for external search
                )
                
                if external_context:
                    sources_searched.append("comprehensive_legal_databases")
                
                progress_tracker.update_stage(
                    ProcessingStage.SEARCHING_EXTERNAL,
                    f"External search complete - {len(external_source_info)} sources found",
                    50,
                    {'external_sources': len(external_source_info)}
                )
                
            except asyncio.TimeoutError:
                self.logger.warning("External search timed out")
                progress_tracker.update_stage(
                    ProcessingStage.SEARCHING_EXTERNAL,
                    "External search timed out - using internal results only",
                    45
                )
        
        return SearchResults(
            internal_results=internal_results,
            external_context=external_context,
            external_source_info=external_source_info,
            sources_searched=sources_searched,
            retrieval_method=retrieval_method
        )
    
    async def _search_external_databases(self, question: str, query_types: List[str]) -> Tuple[Optional[str], List[Dict]]:
    """FIXED: Fast external search with timeout protection"""
    
    # Skip external search for EPA queries - they're slow and return no useful results
    if any(term in question.lower() for term in ['epa', 'environmental', 'air quality', 'violation']):
        self.logger.info("🚀 EPA query detected - skipping slow external APIs")
        return None, []
    
    try:
        from ..services.external_db_service import get_fast_external_optimizer
        
        optimizer = get_fast_external_optimizer()
        external_results = await optimizer.search_external_fast(question, None)
        
        if external_results:
            self.logger.info(f"🏛️ Found {len(external_results)} results from fast external search")
            
            # Simple formatting for speed
            formatted_text = "\n\n## External Sources Found:\n"
            for result in external_results[:3]:
                formatted_text += f"• {result.get('title', 'Unknown')} ({result.get('source_database', 'Unknown')})\n"
            
            source_info = [{
                'file_name': result.get('title', 'Unknown'),
                'source_type': 'external_fast',
                'database': result.get('source_database', 'unknown'),
                'relevance': 0.7,
                'url': result.get('url', '')
            } for result in external_results[:3]]
            
            return formatted_text, source_info
        
    except Exception as e:
        self.logger.error(f"Fast external search failed: {e}")
    
    return None, []    
    
    async def _search_traditional_legal_databases(self, question: str) -> Tuple[Optional[str], List[Dict]]:
        """Fallback search using traditional legal databases"""
        try:
            from ..services.external_db_service import search_free_legal_databases_enhanced
            
            traditional_results = search_free_legal_databases_enhanced(
                question, None, []
            )
            
            if traditional_results:
                self.logger.info(f"📚 Found {len(traditional_results)} results from traditional legal databases")
                
                # Format traditional results
                traditional_context, traditional_source_info = self._format_external_results_with_citations(
                    traditional_results
                )
                
                return traditional_context, traditional_source_info
                
        except Exception as e:
            self.logger.error(f"Traditional legal database search failed: {e}")
        
        return None, []
    
    async def _get_specialized_context(self, question: str, query_types: List[str]) -> Optional[str]:
        """Get specialized context for immigration and other specific queries"""
        specialized_context = ""
        
        # Handle country conditions for immigration
        if QueryType.IMMIGRATION.value in query_types and self.features['country_research']:
            country_context = await self._get_country_conditions_context(question)
            if country_context:
                specialized_context += country_context
        
        # Handle immigration form queries
        if QueryType.IMMIGRATION.value in query_types:
            form_context = self._get_immigration_form_context(question)
            if form_context:
                specialized_context += form_context
        
        return specialized_context if specialized_context else None
    
    async def _get_country_conditions_context(self, question: str) -> Optional[str]:
        """Get country conditions context for immigration queries"""
        if not self.comprehensive_researcher:
            return None
        
        country_conditions_match = re.search(
            r'(?:country\s*conditions?|human\s*rights?|persecution|asylum\s*claim)\s*(?:for|in|about)?\s*([A-Z][a-zA-Z\s]+)',
            question, re.IGNORECASE
        )
        
        if country_conditions_match:
            country = country_conditions_match.group(1).strip()
            self.logger.info(f"📍 Country conditions query detected for: {country}")
            
            try:
                country_results = self.comprehensive_researcher.research_all_sources(
                    country=country,
                    topics=['persecution', 'human_rights', 'government', 'violence'],
                    include_multilingual=True
                )
                
                return f"""
COMPREHENSIVE COUNTRY CONDITIONS RESEARCH FOR {country.upper()}:
{country_results.get('summary', '')}
"""
            except Exception as e:
                self.logger.error(f"Country conditions research failed: {e}")
        
        return None
    
    def _get_immigration_form_context(self, question: str) -> Optional[str]:
        """Get immigration form context"""
        form_match = re.search(r'\b(I-\d{3}|N-\d{3})\b', question, re.IGNORECASE)
        if form_match:
            form_number = form_match.group(1).upper()
            self.logger.info(f"📋 Form-specific query for: {form_number}")
            
            try:
                from ..utils.immigration_helpers import get_form_info
                form_info = get_form_info(form_number)
                return f"\nFORM {form_number} INFORMATION:\n{form_info}\n"
            except ImportError:
                return f"\nFORM {form_number} INFORMATION:\nForm {form_number} information not available\n"
        
        return None
    
    def _process_context(self, search_results: SearchResults) -> Tuple[str, List[Dict]]:
        """Process and format context for LLM"""
        max_context_length = 8000 if QueryType.STATUTORY.value in search_results.internal_results else 6000
        
        context_text, source_info = format_context_for_llm(
            search_results.internal_results, 
            max_length=max_context_length
        )
        
        # Add metadata to source_info
        for i, (doc, score) in enumerate(search_results.internal_results[:len(source_info)]):
            if hasattr(doc, 'metadata'):
                source_info[i]['metadata'] = doc.metadata
                if 'chunk_index' in doc.metadata:
                    source_info[i]['chunk_index'] = doc.metadata['chunk_index']
        
        # Combine source info from internal and external sources
        all_source_info = source_info + search_results.external_source_info
        
        return context_text, all_source_info
    
    async def _generate_response_with_progress(
        self,
        query_context: QueryContext,
        search_results: SearchResults,
        context_text: str,
        source_info: List[Dict],
        progress_tracker: ProgressTracker
    ) -> ProcessingResult:
        """Generate response with progress updates and timeout protection"""
        
        start_time = datetime.utcnow()
        warnings = []
        
        # Extract information with progress
        progress_tracker.update_stage(
            ProcessingStage.EXTRACTING_INFO,
            "Extracting key information...",
            70
        )
        
        extracted_info = self._extract_information(query_context.original_question, context_text)
        
        # Add extracted information to context
        if extracted_info:
            enhancement = self._format_extracted_info(extracted_info)
            if enhancement:
                context_text += enhancement
        
        # Combine contexts
        if search_results.external_context:
            full_context = f"{context_text}\n\n{search_results.external_context}"
        else:
            full_context = context_text
        
        # Generate response with timeout protection
        progress_tracker.update_stage(
            ProcessingStage.GENERATING_RESPONSE,
            "Generating AI response...",
            75
        )
        
        if self.features['ai_enabled']:
            try:
                prompt = self._create_prompt(query_context, search_results, full_context)
                
                # AI generation with timeout
                response_text = await asyncio.wait_for(
                    asyncio.to_thread(call_openrouter_api, prompt, OPENROUTER_API_KEY),
                    timeout=12  # 12 second timeout for AI generation
                )
                
            except asyncio.TimeoutError:
                self.logger.warning("AI response generation timed out")
                response_text = self._create_fallback_response_from_context(
                    full_context, query_context.original_question
                )
                warnings.append("AI response timed out - using fallback response")
        else:
            response_text = self._create_fallback_response_from_context(
                full_context, query_context.original_question
            )
        
        # Validation with progress
        progress_tracker.update_stage(
            ProcessingStage.VALIDATING_RESPONSE,
            "Validating response against sources...",
            90
        )
        
        try:
            validated_response, validation_confidence = await asyncio.wait_for(
                asyncio.to_thread(
                    self._validate_response_against_context,
                    response_text, full_context, query_context.original_question
                ),
                timeout=2  # Quick validation timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Response validation timed out")
            validated_response = response_text
            validation_confidence = 0.5
            warnings.append("Response validation timed out")
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ProcessingResult(
            response_text=validated_response,
            confidence_score=validation_confidence,
            sources=source_info,
            warnings=warnings,
            processing_time=processing_time
        )
    
    def _extract_information(self, question: str, context_text: str) -> Dict[str, Any]:
        """Extract specific information from context based on question type"""
        # Enhanced pattern to find bill information
        bill_match = re.search(r"(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)", question, re.IGNORECASE)
        statute_match = re.search(r"(RCW|USC|CFR|WAC)\s+(\d+\.\d+\.\d+|\d+)", question, re.IGNORECASE)
        
        if bill_match:
            bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
            self.logger.info(f"Searching for bill: {bill_number}")
            return extract_bill_information(context_text, bill_number)
        elif statute_match:
            statute_citation = f"{statute_match.group(1)} {statute_match.group(2)}"
            self.logger.info(f"🏛️ Searching for statute: {statute_citation}")
            return self._extract_statutory_information(context_text, statute_citation)
        else:
            return extract_universal_information(context_text, question)
    
    def _extract_statutory_information(self, context_text: str, statute_citation: str) -> Dict[str, Any]:
        """Extract specific information from statutory text"""
        extracted_info = {
            "requirements": [],
            "durations": [],
            "numbers": [],
            "procedures": []
        }
        
        try:
            # Look for duration patterns
            duration_patterns = [
                r"minimum of (\d+) (?:minutes?|hours?)",
                r"at least (\d+) (?:minutes?|hours?)",
                r"(\d+)-(?:minute|hour) (?:minimum|maximum)",
                r"(?:shall|must) be (\d+) (?:minutes?|hours?)"
            ]
            
            for pattern in duration_patterns:
                matches = re.findall(pattern, context_text, re.IGNORECASE)
                for match in matches:
                    extracted_info["durations"].append(f"{match} minutes/hours")
            
            # Look for numerical requirements
            number_patterns = [
                r"(?:maximum|minimum) of (\d+) (?:participants?|people|individuals?)",
                r"at least (\d+) (?:speakers?|facilitators?|members?)",
                r"no more than (\d+) (?:participants?|attendees?)"
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, context_text, re.IGNORECASE)
                for match in matches:
                    extracted_info["numbers"].append(match)
            
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"Error extracting statutory information: {e}")
            return extracted_info
    
    def _format_extracted_info(self, extracted_info: Dict[str, Any]) -> str:
        """Format extracted information for context enhancement"""
        if not extracted_info:
            return ""
        
        enhancement = "\n\nKEY INFORMATION EXTRACTED:\n"
        for key, value in extracted_info.items():
            if value:
                if isinstance(value, list):
                    enhancement += f"- {key.replace('_', ' ').title()}: {', '.join(value[:5])}\n"
                else:
                    enhancement += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return enhancement if enhancement.strip() != "KEY INFORMATION EXTRACTED:" else ""
    
    def _create_prompt(self, query_context: QueryContext, search_results: SearchResults, 
                      context_text: str) -> str:
        """Create appropriate prompt based on query type and context"""
        
        # Determine detected areas for comprehensive prompts
        detected_areas = [area for area in query_context.query_types 
                         if area in ['environmental', 'business', 'labor', 'healthcare', 'criminal', 'immigration', 'housing']]
        
        # Choose prompt template based on query characteristics
        if (search_results.external_context and 
            (self._detect_government_data_need(query_context.original_question) or detected_areas)):
            return self._create_comprehensive_legal_prompt(
                context_text, query_context, search_results, detected_areas
            )
        elif QueryType.IMMIGRATION.value in query_context.query_types:
            return self._create_immigration_prompt(context_text, query_context, search_results)
        elif QueryType.STATUTORY.value in query_context.query_types:
            return self._create_statutory_prompt(context_text, query_context, search_results)
        else:
            return self._create_regular_prompt(context_text, query_context, search_results)
    
    def _create_comprehensive_legal_prompt(self, context_text: str, query_context: QueryContext,
                                         search_results: SearchResults, detected_areas: List[str]) -> str:
        """Create prompt for comprehensive legal research questions with government data"""
        
        legal_areas_text = ", ".join(detected_areas) if detected_areas else "general legal research"
        
        return f"""You are a comprehensive legal research assistant with access to official government databases, enforcement data, and authoritative legal sources.

DETECTED LEGAL AREAS: {legal_areas_text}

SOURCE HIERARCHY & AUTHORITY:
- **HIGHEST AUTHORITY** (Official Government Sources): 
  🏛️ Federal enforcement databases (EPA, SEC, DOL, FDA, USCIS, FBI)
  🏛️ Official legislative sources (Congress.gov, Federal Register)
  🏛️ Government statistical and regulatory data

- **VERY HIGH AUTHORITY** (Legal Authorities):
  🎓 Federal and state statutes from authoritative academic sources
  🎓 Court decisions from Harvard Caselaw Access Project and CourtListener
  📚 Federal regulations and administrative guidance

- **HIGH AUTHORITY** (Reliable Legal Sources):
  📊 Professional legal databases (Justia, Cornell Law)
  📊 State legislative tracking (OpenStates)
  📊 Legal academic resources

ANTI-HALLUCINATION REQUIREMENTS:
🚫 ONLY answer based on the provided context and sources
🚫 If information is NOT in the sources, say "This information is not available in the provided documents"
🚫 NEVER make up facts, dates, case names, citations, or enforcement actions
🚫 When uncertain, use phrases like "Based on the available information..." or "The documents suggest..."

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}
RETRIEVAL METHOD: {search_results.retrieval_method}
{f"DOCUMENT FILTER: Specific document {query_context.document_id}" if query_context.document_id else "DOCUMENT SCOPE: All available sources including government databases"}

RESPONSE STYLE: {query_context.response_style}

CONVERSATION HISTORY:
{query_context.conversation_context}

COMPREHENSIVE LEGAL CONTEXT (Government databases, enforcement data, legal authorities):
{context_text}

USER QUESTION:
{query_context.original_question}

REQUIRED RESPONSE FORMAT:
## 📋 Direct Answer
[Your main answer based ONLY on the provided sources]

## 🔑 Key Supporting Points
[Bullet points from the sources that support your answer]

## 📚 Sources Referenced
[List the specific documents/databases that support your answer]

## 🎯 Confidence Assessment
[High/Medium/Low] - [Brief explanation based on source quality and coverage]

RESPONSE INSTRUCTIONS:
- Lead with government data when available (enforcement actions, violations, official statistics)
- Provide specific, actionable information from official sources with concrete examples
- Include enforcement patterns and compliance guidance from government databases
- Explain the underlying legal framework using primary legal authorities
- Note jurisdictional differences and enforcement variations by agency/state
- Include practical compliance steps based on real enforcement examples
- Cite official sources with authority indicators for maximum credibility

RESPONSE:"""
    
    def _create_immigration_prompt(self, context_text: str, query_context: QueryContext,
                                 search_results: SearchResults) -> str:
        """Create prompt specifically for immigration queries"""
        
        return f"""You are an immigration legal assistant with access to official USCIS data, visa bulletins, and immigration law databases.

IMMIGRATION-SPECIFIC AUTHORITY SOURCES:
🏛️ **Official Government**: USCIS case status, State Dept visa bulletins, immigration court data
🎓 **Legal Authorities**: Immigration statutes, regulations, and federal court decisions
📊 **Professional Resources**: Immigration law databases and practice guides

ANTI-HALLUCINATION REQUIREMENTS:
🚫 ONLY answer based on the provided context and sources
🚫 If information is NOT in the sources, say "This information is not available in the provided documents"
🚫 NEVER make up processing times, case outcomes, or legal requirements
🚫 When uncertain about immigration law, clearly state limitations

IMPORTANT IMMIGRATION CONTEXT:
- Always note that immigration law is complex and changes frequently
- Processing times and requirements vary by service center and case type
- Country conditions can change rapidly affecting asylum claims
- Each case is unique and requires individual assessment

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}
RETRIEVAL METHOD: {search_results.retrieval_method}

CONVERSATION HISTORY:
{query_context.conversation_context}

IMMIGRATION CONTEXT (including official government data):
{context_text}

USER QUESTION:
{query_context.original_question}

REQUIRED RESPONSE FORMAT:
## 📋 Direct Answer
[Your main answer based ONLY on the provided sources]

## 🔑 Key Immigration Points
[Specific points from immigration sources]

## 📚 Official Sources Referenced
[Government sources and legal authorities cited]

## ⚠️ Important Immigration Disclaimers
[Relevant warnings about processing times, legal complexity, etc.]

RESPONSE INSTRUCTIONS:
- Provide practical, actionable guidance based on official sources
- Include specific form numbers, deadlines, and current processing information
- Note any recent policy changes or enforcement priorities
- Always include disclaimer about consulting an immigration attorney

RESPONSE:"""
    
    def _create_statutory_prompt(self, context_text: str, query_context: QueryContext,
                               search_results: SearchResults) -> str:
        """Create an enhanced prompt specifically for statutory analysis"""
        
        return f"""You are a legal research assistant specializing in statutory analysis. Your job is to extract COMPLETE, SPECIFIC information from legal documents.

STRICT SOURCE REQUIREMENTS:
- Answer ONLY based on the retrieved documents provided in the context
- Do NOT use general legal knowledge, training data, assumptions, or inferences beyond what's explicitly stated
- If information is not in the provided documents, state: "This information is not available in the provided documents"
- When citing external database results, use the full citation format provided

🔴 CRITICAL: You MUST extract actual numbers, durations, and specific requirements. NEVER use placeholders like "[duration not specified]" or "[requirements not listed]".

🔥 CRITICAL FAILURE PREVENTION:
- If you write "[duration not specified]" you have FAILED at your job
- If you write "[duties not specified]" you have FAILED at your job  
- If you write "[requirements not listed]" you have FAILED at your job
- READ EVERY WORD of the context before claiming information is missing
- The human is counting on you to find the actual requirements

MANDATORY EXTRACTION RULES:
1. 📖 READ EVERY WORD of the provided context before claiming anything is missing
2. 🔢 EXTRACT ALL NUMBERS: durations (60 minutes), quantities (25 people), percentages, dollar amounts
3. 📝 QUOTE EXACT LANGUAGE: Use quotation marks for statutory text
4. 📋 LIST ALL REQUIREMENTS: Number each requirement found (1., 2., 3., etc.)
5. 🎯 BE SPECIFIC: Include section numbers, subsection letters, paragraph numbers
6. ⚠️ ONLY claim information is "missing" after thorough analysis of ALL provided text

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}
RETRIEVAL METHOD: {search_results.retrieval_method}
{f"DOCUMENT FILTER: Specific document {query_context.document_id}" if query_context.document_id else "DOCUMENT SCOPE: All available documents"}

RESPONSE FORMAT REQUIRED FOR STATUTORY QUESTIONS:

## SPECIFIC REQUIREMENTS FOUND:
[List each requirement with exact quotes and citations]

## NUMERICAL STANDARDS:
[Extract ALL numbers, durations, thresholds, limits]

## PROCEDURAL RULES:
[Detail composition, conduct, attendance, record-keeping rules]

## ROLES AND RESPONSIBILITIES:
[Define each party's specific duties with citations]

## ENFORCEMENT DATA (if available):
[Include any government enforcement examples, violations, or compliance guidance]

## INFORMATION NOT FOUND:
[Only list what is genuinely absent after thorough review]

CONVERSATION HISTORY:
{query_context.conversation_context}

DOCUMENT CONTEXT TO ANALYZE WORD-BY-WORD:
{context_text}

USER QUESTION REQUIRING COMPLETE EXTRACTION:
{query_context.original_question}

RESPONSE:"""
    
    def _create_regular_prompt(self, context_text: str, query_context: QueryContext,
                             search_results: SearchResults) -> str:
        """Create the regular prompt for non-statutory questions"""
        
        return f"""You are a legal research assistant with access to comprehensive legal databases and government enforcement data.

SOURCE HIERARCHY:
- **PRIMARY**: Information from the retrieved documents provided in the context
- **GOVERNMENT DATA**: Official enforcement actions, violations, and regulatory guidance (highest authority)
- **LEGAL AUTHORITIES**: Statutes, regulations, and case law from authoritative sources
- **SECONDARY**: General legal knowledge ONLY when documents are unavailable

ANTI-HALLUCINATION REQUIREMENTS:
🚫 ONLY answer based on the provided context and sources
🚫 If information is NOT in the sources, say "This information is not available in the provided documents"
🚫 NEVER make up facts, dates, case names, or legal citations
🚫 When uncertain, use phrases like "Based on the available information..." or "The documents suggest..."

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}
RETRIEVAL METHOD: {search_results.retrieval_method}
{f"DOCUMENT FILTER: Specific document {query_context.document_id}" if query_context.document_id else "DOCUMENT SCOPE: All available documents"}

RESPONSE STYLE: {query_context.response_style}

CONVERSATION HISTORY:
{query_context.conversation_context}

COMPREHENSIVE LEGAL CONTEXT:
{context_text}

USER QUESTION:
{query_context.original_question}

REQUIRED RESPONSE FORMAT:
## 📋 Direct Answer
[Your main answer based ONLY on the provided sources]

## 🔑 Key Supporting Points
[Bullet points from the sources that support your answer]

## 📚 Sources Referenced
[List the specific documents that support your answer]

## 🎯 Confidence Assessment
[High/Medium/Low] - [Brief explanation based on source quality]

RESPONSE APPROACH:
- **FIRST**: Check for official government data (enforcement actions, violations, compliance guidance)
- **SECOND**: Provide legal framework from authoritative sources (statutes, regulations, case law)
- **THIRD**: Include practical compliance guidance based on enforcement patterns
- **FOURTH**: Note jurisdictional variations and recent developments
- **ALWAYS**: Cite sources with authority indicators (🏛️ government, 🎓 academic, 📊 legal databases)

RESPONSE:"""
    
    def _validate_response_against_context(self, response: str, context: str, question: str) -> Tuple[str, float]:
        """Validate response to reduce hallucination and calculate confidence"""
        
        # Extract factual claims from response
        claims = self._extract_factual_claims(response)
        
        # Check each claim against context
        verified_claims = 0
        total_claims = len(claims)
        
        if total_claims == 0:
            return response, 0.5
        
        for claim in claims:
            if self._verify_claim_in_context(claim, context):
                verified_claims += 1
        
        confidence = verified_claims / total_claims if total_claims > 0 else 0.0
        
        # Add uncertainty markers for low confidence
        if confidence < 0.6:
            response = self._add_uncertainty_markers(response)
        
        # Enhance confidence with context overlap scoring
        context_overlap_score = self._calculate_context_overlap(response, context)
        final_confidence = (confidence * 0.7 + context_overlap_score * 0.3)
        
        return response, final_confidence
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from response"""
        # Split into sentences and filter out uncertainty phrases
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        uncertainty_phrases = [
            'i think', 'i believe', 'it seems', 'maybe', 'perhaps', 
            'based on the available information', 'the documents suggest',
            'according to the provided sources', 'from what i can determine'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # Skip sentences that start with uncertainty markers
                sentence_lower = sentence.lower()
                if not any(phrase in sentence_lower[:50] for phrase in uncertainty_phrases):
                    claims.append(sentence)
        
        return claims
    
    def _verify_claim_in_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context using keyword overlap"""
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
        
        claim_words -= stop_words
        context_words -= stop_words
        
        if not claim_words:
            return True  # Empty claim after filtering
        
        # Require at least 60% of claim words to be in context
        overlap = len(claim_words.intersection(context_words))
        overlap_ratio = overlap / len(claim_words)
        
        return overlap_ratio >= 0.6
    
    def _calculate_context_overlap(self, response: str, context: str) -> float:
        """Calculate overall overlap between response and context"""
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words.intersection(context_words))
        return min(1.0, overlap / len(response_words))
    
    def _add_uncertainty_markers(self, response: str) -> str:
        """Add uncertainty markers to low-confidence responses"""
        
        # Check if response already has uncertainty markers
        uncertainty_phrases = [
            "based on the available information",
            "the documents suggest that",
            "according to the provided sources",
            "from what i can determine"
        ]
        
        response_lower = response.lower()
        has_uncertainty = any(phrase in response_lower for phrase in uncertainty_phrases)
        
        if not has_uncertainty:
            # Add uncertainty marker at the beginning of the main content
            lines = response.split('\n')
            main_content_started = False
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and not main_content_started:
                    if not any(phrase in line.lower() for phrase in uncertainty_phrases):
                        lines[i] = f"Based on the available information, {line}"
                    main_content_started = True
                    break
            
            response = '\n'.join(lines)
        
        return response
    
    async def _post_process_response_async(
        self, 
        processing_result: ProcessingResult,
        query_context: QueryContext, 
        search_results: SearchResults
    ) -> QueryResponse:
        """Async version of response post-processing"""
        return await asyncio.to_thread(
            self._post_process_response,
            processing_result, query_context, search_results
        )
    
    def _post_process_response(self, processing_result: ProcessingResult, 
                             query_context: QueryContext, search_results: SearchResults) -> QueryResponse:
        """Post-process and validate response"""
        
        # Add context-specific notices
        notices = []
        
        if QueryType.IMMIGRATION.value in query_context.query_types:
            notices.append("**Immigration Law Notice**: This information is general guidance only. Immigration law is complex and changes frequently. Please consult with an immigration attorney for advice specific to your situation.")
        
        if (self._detect_government_data_need(query_context.original_question) and 
            search_results.external_source_info):
            notices.append("**Government Data Notice**: This response includes official government enforcement data and regulatory information. Always verify current requirements with the relevant agencies.")
        
        # Add notices to response
        final_response = processing_result.response_text
        if notices:
            final_response += "\n\n" + "\n\n".join(notices)
        
        # Filter and consolidate sources
        relevant_sources = [s for s in processing_result.sources if s.get('relevance', 0) >= MIN_RELEVANCE_SCORE]
        consolidated_sources = self._consolidate_sources(relevant_sources)
        
        # Add source citations to response if sources exist
        if consolidated_sources:
            citations = self._format_source_citations(consolidated_sources)
            final_response += f"\n\n{citations}"
        
        # Calculate final confidence score
        final_confidence = calculate_confidence_score(
            search_results.internal_results, 
            len(final_response)
        )
        
        # Combine with validation confidence
        combined_confidence = (processing_result.confidence_score * 0.6 + final_confidence * 0.4)
        
        return QueryResponse(
            response=final_response,
            error=None,
            context_found=bool(search_results.internal_results or search_results.external_context),
            sources=consolidated_sources,
            session_id=query_context.session_id,
            confidence_score=float(combined_confidence),
            sources_searched=search_results.sources_searched,
            expand_available=len(query_context.expanded_questions) > 1 if query_context.use_enhanced_rag else False,
            retrieval_method=search_results.retrieval_method
        )
    
    # === CITATION AND FORMATTING METHODS ===
    
    def _format_comprehensive_analysis_response(self, comp_result) -> str:
        """Format comprehensive analysis response"""
        return f"""# Comprehensive Legal Document Analysis

## Document Summary
{comp_result.document_summary or 'No summary available'}

## Key Clauses Analysis
{comp_result.key_clauses or 'No clauses analysis available'}

## Risk Assessment
{comp_result.risk_assessment or 'No risk assessment available'}

## Timeline & Deadlines
{comp_result.timeline_deadlines or 'No timeline information available'}

## Party Obligations
{comp_result.party_obligations or 'No obligations analysis available'}

## Missing Clauses Analysis
{comp_result.missing_clauses or 'No missing clauses analysis available'}

---
**Analysis Confidence:** {comp_result.overall_confidence:.1%}
**Processing Time:** {comp_result.processing_time:.2f} seconds"""
    
    def _format_comprehensive_results_with_citations(self, comprehensive_results: List[Dict]) -> Tuple[str, List[Dict]]:
        """Format comprehensive API results with proper citations"""
        if not comprehensive_results:
            return "", []
        
        # Group results by legal area and source type
        results_by_category = {}
        source_info = []
        
        for idx, result in enumerate(comprehensive_results):
            legal_area = result.get('legal_area', result.get('enforcement_area', 'general'))
            source_db = result.get('source_database', 'unknown')
            detected_state = result.get('detected_state', 'Federal')
            
            if legal_area not in results_by_category:
                results_by_category[legal_area] = []
            
            results_by_category[legal_area].append(result)
            
            # Create source info entry
            source_info.append({
                'file_name': self._format_legal_citation(result),
                'page': None,
                'relevance': result.get('relevance_score', 0.9),
                'source_type': 'comprehensive_government_database',
                'database': source_db,
                'legal_area': legal_area,
                'jurisdiction': detected_state,
                'url': result.get('url', ''),
                'date': result.get('date', ''),
                'authority_level': 'very_high' if any(gov in source_db for gov in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi']) else 'high'
            })
        
        # Format the context text
        formatted_text = "\n\n## COMPREHENSIVE LEGAL & GOVERNMENT DATABASE RESULTS:\n"
        formatted_text += "(Official government enforcement data, regulatory information, and authoritative legal sources)\n\n"
        
        for legal_area, area_results in results_by_category.items():
            if not area_results:
                continue
                
            # Add section header for this legal area
            area_title = legal_area.replace('_', ' ').title()
            if area_title.lower() != 'general':
                formatted_text += f"### {area_title} - Official Data & Legal Framework\n"
            
            for idx, result in enumerate(area_results, 1):
                # Create proper citation
                citation = self._format_legal_citation(result)
                
                formatted_text += f"#### {idx}. {citation}\n"
                
                # Add database attribution with authority indicator
                source_db = result.get('source_database', 'Unknown')
                if any(gov in source_db for gov in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi', 'congress', 'federal_register']):
                    authority_indicator = "🏛️"
                    authority_text = "Official U.S. Government Database"
                elif any(academic in source_db for academic in ['harvard', 'cornell']):
                    authority_indicator = "🎓"
                    authority_text = "Academic Legal Institution"
                else:
                    authority_indicator = "📊"
                    authority_text = "Legal Database"
                
                formatted_text += f"**Source:** {authority_indicator} {authority_text} - {self._format_database_name(source_db)}\n"
                
                # Add jurisdiction info
                jurisdiction = result.get('detected_state') or result.get('state', 'Federal')
                formatted_text += f"**Jurisdiction:** {jurisdiction}\n"
                
                # Add date if available
                date = (result.get('date') or result.get('filing_date') or 
                       result.get('report_date') or result.get('citation_date'))
                if date:
                    formatted_text += f"**Date:** {date}\n"
                
                # Add government-specific data
                if 'violation_type' in result:
                    formatted_text += f"**Violation Type:** {result['violation_type']}\n"
                if 'penalty' in result and result['penalty']:
                    formatted_text += f"**Penalty:** {result['penalty']}\n"
                if 'status' in result and result['status']:
                    formatted_text += f"**Status:** {result['status']}\n"
                if 'classification' in result:
                    formatted_text += f"**Classification:** {result['classification']}\n"
                if 'company' in result and result['company']:
                    formatted_text += f"**Company/Facility:** {result['company']}\n"
                
                # Add description/summary
                description = (result.get('description') or result.get('summary') or 
                             result.get('reason_for_recall') or result.get('snippet', ''))
                
                if description:
                    # Clean and truncate description
                    clean_description = re.sub(r'\s+', ' ', description).strip()
                    if len(clean_description) > 400:
                        clean_description = clean_description[:397] + "..."
                    formatted_text += f"**Details:** {clean_description}\n"
                
                # Add URL for full details
                if result.get('url'):
                    formatted_text += f"**Full Information:** [View Official Source]({result['url']})\n"
                
                formatted_text += "\n"
        
        return formatted_text, source_info
    
    def _format_external_results_with_citations(self, external_results: List[Dict]) -> Tuple[str, List[Dict]]:
        """Format external results with proper citations"""
        if not external_results:
            return "", []
        
        formatted_text = "\n\n## LEGAL DATABASE RESULTS:\n"
        formatted_text += "(Results from authoritative legal databases and academic institutions)\n\n"
        
        source_info = []
        
        for idx, result in enumerate(external_results[:5], 1):  # Limit to top 5 results
            # Create proper citation
            citation = self._format_legal_citation(result)
            
            formatted_text += f"### {idx}. {citation}\n"
            
            # Add database attribution
            source_db = result.get('source_database', 'Unknown')
            formatted_text += f"**Database:** {self._format_database_name(source_db)}\n"
            
            # Add court and jurisdiction info
            court = result.get('court', '')
            jurisdiction = result.get('jurisdiction', '')
            if court:
                formatted_text += f"**Court:** {court}\n"
            if jurisdiction:
                formatted_text += f"**Jurisdiction:** {jurisdiction}\n"
            
            # Add date decided
            date = result.get('date', '')
            if date:
                formatted_text += f"**Date:** {date}\n"
            
            # Add relevant excerpt/preview
            preview = result.get('preview') or result.get('snippet', '') or result.get('description', '')
            if preview:
                # Clean up the preview text
                preview = re.sub(r'\s+', ' ', preview).strip()
                if len(preview) > 500:
                    preview = preview[:497] + "..."
                formatted_text += f"**Relevant Excerpt:** {preview}\n"
            
            # Add URL for full text
            if result.get('url'):
                formatted_text += f"**Full Text:** [View Source]({result['url']})\n"
            
            formatted_text += "\n"
            
            # Create source info entry with proper citation
            source_info.append({
                'file_name': citation,  # Use the formatted citation as the file name
                'page': None,
                'relevance': result.get('relevance_score', 0.8),
                'source_type': 'external_legal_database',
                'database': source_db,
                'url': result.get('url', ''),
                'court': court,
                'date': date
            })
        
        return formatted_text, source_info
    
    def _format_legal_citation(self, result: Dict) -> str:
        """Format a legal database result into a proper citation"""
        source_db = result.get('source_database', '').lower()
        
        # Harvard Library Caselaw Access Project format
        if 'harvard' in source_db or 'caselaw' in source_db:
            case_name = result.get('title', 'Unknown Case')
            court = result.get('court', '')
            date = result.get('date', '')
            citation = result.get('citation', '')
            
            if citation:
                year = date.split('-')[0] if date else ''
                return f"{case_name}, {citation} ({court} {year})"
            else:
                return f"{case_name} ({court} {date})"
        
        # CourtListener format
        elif 'courtlistener' in source_db:
            case_name = result.get('title', 'Unknown Case')
            docket = result.get('docket_number', '')
            court = result.get('court', '')
            date = result.get('date', '')
            
            if docket:
                return f"{case_name}, No. {docket} ({court} {date})"
            else:
                return f"{case_name} ({court} {date})"
        
        # Government database formats
        elif 'epa' in source_db:
            facility = result.get('facility_name', result.get('title', 'EPA Action'))
            location = result.get('location', '')
            violation = result.get('violation_type', '')
            date = result.get('date', result.get('citation_date', ''))
            return f"EPA Enforcement: {facility} {location} - {violation} ({date})"
        
        elif 'sec' in source_db:
            company = result.get('company', result.get('title', 'SEC Filing'))
            form_type = result.get('form_type', 'Filing')
            date = result.get('filing_date', result.get('date', ''))
            return f"SEC {form_type}: {company} ({date})"
        
        elif 'osha' in source_db or 'dol' in source_db:
            company = result.get('company', result.get('title', 'DOL Action'))
            violation = result.get('violation_type', 'Violation')
            date = result.get('citation_date', result.get('date', ''))
            return f"OSHA Citation: {company} - {violation} ({date})"
        
        elif 'uscis' in source_db:
            receipt = result.get('receipt_number', '')
            status = result.get('status', result.get('title', 'Case Status'))
            date = result.get('checked_date', result.get('date', ''))
            return f"USCIS Case {receipt}: {status} ({date})"
        
        elif 'fda' in source_db:
            product = result.get('product_description', result.get('title', 'FDA Action'))
            company = result.get('company', result.get('recalling_firm', ''))
            date = result.get('report_date', result.get('date', ''))
            return f"FDA Action: {company} - {product} ({date})"
        
        # Google Scholar format
        elif 'scholar' in source_db or 'google' in source_db:
            case_name = result.get('title', 'Unknown Case')
            citation = result.get('citation', '')
            year = result.get('date', '').split('-')[0] if result.get('date') else ''
            
            if citation:
                return f"{case_name}, {citation} ({year})"
            else:
                return f"{case_name} ({year})"
        
        # Generic format for other sources
        else:
            title = result.get('title', 'Unknown')
            source = result.get('source_database', 'Unknown Source').replace('_', ' ').title()
            date = result.get('date', '')
            
            if date:
                return f"{source}: {title} ({date})"
            else:
                return f"{source}: {title}"
    
    def _format_database_name(self, source_db: str) -> str:
        """Format database name for display"""
        database_names = {
            'epa_echo': 'EPA Enforcement & Compliance History Online',
            'epa_air_quality': 'EPA Air Quality System',
            'sec_edgar': 'SEC EDGAR Corporate Filings Database',
            'dol_osha': 'Department of Labor OSHA Enforcement Database',
            'fda_drug_enforcement': 'FDA Drug Enforcement Reports',
            'uscis_case_status': 'USCIS Case Status System',
            'state_dept_visa_bulletin': 'State Department Visa Bulletin',
            'hud_fair_market_rents': 'HUD Fair Market Rent Database',
            'census_housing': 'U.S. Census Bureau Housing Statistics',
            'fbi_crime_data': 'FBI Crime Data Explorer',
            'uspto_patents': 'USPTO Patent Database',
            'congress_gov': 'Congress.gov Official Legislative Database',
            'openstates': 'OpenStates Legislative Tracking',
            'justia': 'Justia Free Law Database',
            'cornell_law': 'Cornell Law School Legal Information Institute',
            'harvard_caselaw': 'Harvard Law School Caselaw Access Project',
            'courtlistener': 'CourtListener Federal & State Court Database',
            'federal_register': 'Federal Register - Official U.S. Government Regulations'
        }
        
        return database_names.get(source_db, source_db.replace('_', ' ').title())
    
    def _consolidate_sources(self, source_info: List[Dict]) -> List[Dict]:
        """Consolidate multiple chunks from the same document into meaningful source entries"""
        from collections import defaultdict
        
        # Group sources by document
        document_groups = defaultdict(list)
        
        for source in source_info:
            key = (source['file_name'], source['source_type'])
            document_groups[key].append(source)
        
        consolidated_sources = []
        
        for (file_name, source_type), sources in document_groups.items():
            if source_type in ['external_legal_database', 'comprehensive_government_database']:
                # External sources should not be consolidated - each is unique
                consolidated_sources.extend(sources)
            else:
                # For internal documents, consolidate chunks
                relevance_scores = [s['relevance'] for s in sources]
                max_relevance = max(relevance_scores)
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                
                # Get unique pages if available
                pages = [s['page'] for s in sources if s.get('page') is not None]
                unique_pages = sorted(set(pages)) if pages else []
                
                consolidated_source = {
                    'file_name': file_name,
                    'source_type': source_type,
                    'relevance': max_relevance,
                    'avg_relevance': avg_relevance,
                    'num_chunks': len(sources),
                    'pages': unique_pages,
                    'authority_level': sources[0].get('authority_level', 'medium')
                }
                
                consolidated_sources.append(consolidated_source)
        
        # Sort by authority level first, then relevance
        authority_order = {'very_high': 4, 'high': 3, 'medium_high': 2, 'medium': 1, 'low': 0}
        consolidated_sources.sort(
            key=lambda x: (authority_order.get(x.get('authority_level', 'medium'), 1), x.get('relevance', 0)), 
            reverse=True
        )
        
        return consolidated_sources
    
    def _format_source_citations(self, consolidated_sources: List[Dict]) -> str:
        """Format source citations for response"""
        if not consolidated_sources:
            return ""
        
        citations = "\n\n**SOURCES:**"
        citations += "\n*🏛️ = Official Government, 🎓 = Academic/Authoritative, 📊 = Legal Database*"
        
        for source in consolidated_sources:
            citations += "\n" + self._format_source_display(source)
        
        return citations
    
    def _format_source_display(self, source: Dict) -> str:
        """Format a source for display in the response with authority indicators"""
        source_type = source['source_type'].replace('_', ' ').title()
        authority_level = source.get('authority_level', 'medium')
        
        # Authority indicators
        if authority_level == 'very_high':
            authority_icon = "🏛️"
        elif authority_level == 'high':
            authority_icon = "🎓"
        else:
            authority_icon = "📊"
        
        if source_type in ['External Legal Database', 'Comprehensive Government Database']:
            # External sources with authority indicators
            display = f"- {authority_icon} {source['file_name']}"
            
            if source.get('database'):
                db_name = self._format_database_name(source['database'])
                display += f" [{db_name}]"
            
            if source.get('url'):
                display += f" - [Full Text]({source['url']})"
            
            return display
        else:
            # Internal document format
            display = f"- {authority_icon} {source['file_name']}"
            
            # Add page information if available
            if source.get('pages'):
                if len(source['pages']) == 1:
                    display += f" (Page {source['pages'][0]})"
                elif len(source['pages']) > 1:
                    if len(source['pages']) <= 3:
                        display += f" (Pages {', '.join(map(str, source['pages']))})"
                    else:
                        display += f" (Pages {source['pages'][0]}-{source['pages'][-1]})"
            
            # Add relevance score with authority context
            relevance = source['relevance']
            if relevance <= 1.0:
                relevance_pct = relevance * 100
            else:
                relevance_pct = relevance
                
            if relevance_pct >= 80:
                relevance_label = "Excellent match"
            elif relevance_pct >= 60:
                relevance_label = "Good match"
            elif relevance_pct >= 40:
                relevance_label = "Fair match"
            else:
                relevance_label = "Partial match"
            
            display += f" ({relevance_label}: {relevance_pct:.0f}/100, {authority_level.replace('_', ' ').title()} Authority)"
            
            return display
    
    # === TIMEOUT AND ERROR HANDLING METHODS ===
    
    async def _handle_timeout_with_partial_results(
        self, 
        progress_tracker: ProgressTracker, 
        session_id: str,
        query_id: str
    ) -> QueryResponse:
        """Handle timeout by providing partial results if available"""
        
        current_progress = progress_tracker.get_current_progress()
        partial_results = current_progress.partial_results or {}
        
        self._update_stats('timeout', 25)  # Default timeout duration
        
        # Create response based on how far we got
        if current_progress.stage in [ProcessingStage.SEARCHING_INTERNAL, ProcessingStage.SEARCHING_EXTERNAL]:
            if partial_results.get('internal_results', 0) > 0:
                response = f"""⚠️ **Query Timeout - Partial Results Available**

Your query was complex and timed out during processing, but I was able to find {partial_results.get('internal_results', 0)} relevant documents in your uploaded files.

**What I found so far:**
- Searched your uploaded documents
- Found {partial_results.get('internal_results', 0)} potentially relevant results
- Processing stopped at: {current_progress.message}

**To get complete results:**
- Try asking a more specific question
- Break complex questions into smaller parts
- Upload fewer documents if you have many large files

**Tip:** Questions about specific bills, statutes, or documents typically process faster than broad research queries."""
            else:
                response = "⚠️ Your query timed out during document search. Please try a more specific question or check if your documents are properly uploaded."
        
        elif current_progress.stage == ProcessingStage.GENERATING_RESPONSE:
            response = f"""⚠️ **Query Timeout During Response Generation**

I found relevant information in your documents but timed out while generating the response.

**What I found:**
- Successfully searched your documents
- Found relevant content from {len(partial_results.get('sources', []))} sources
- Timeout occurred during AI response generation

**To get complete results:**
- Try asking a more focused question
- Ask about specific sections or topics rather than broad analysis

**Note:** The information is available in your documents - a more specific question will process faster."""
        
        else:
            response = f"""⚠️ **Query Processing Timeout**

Your query timed out after {current_progress.progress_percent}% completion.

**Processing stage when timeout occurred:** {current_progress.message}

**To resolve:**
- Try a more specific question
- Break complex questions into parts
- Reduce the scope of your search

**Tip:** Specific questions about particular documents or legal concepts process much faster than broad research queries."""
        
        return QueryResponse(
            response=response,
            error="timeout_with_partial_results",
            context_found=bool(partial_results.get('internal_results', 0) > 0),
            sources=partial_results.get('sources', []),
            session_id=session_id,
            confidence_score=0.2,  # Low confidence for partial results
            sources_searched=partial_results.get('sources_searched', []),
            retrieval_method="timeout_partial"
        )
    
    async def _handle_processing_error(
        self, 
        error: Exception, 
        progress_tracker: ProgressTracker,
        session_id: str, 
        query_id: str
    ) -> QueryResponse:
        """Handle processing errors with helpful user feedback"""
        
        current_progress = progress_tracker.get_current_progress()
        self._update_stats('error', 25)
        
        # Provide stage-specific error messages
        error_messages = {
            ProcessingStage.ANALYZING_QUERY: "Error analyzing your question. Please try rephrasing it.",
            ProcessingStage.SEARCHING_INTERNAL: "Error searching your documents. Please check if documents are properly uploaded.",
            ProcessingStage.SEARCHING_EXTERNAL: "Error searching external databases. Your documents are still searchable.",
            ProcessingStage.GENERATING_RESPONSE: "Error generating response. Please try a simpler question.",
            ProcessingStage.VALIDATING_RESPONSE: "Error validating response. Please try again."
        }
        
        user_message = error_messages.get(
            current_progress.stage,
            "An unexpected error occurred. Please try again with a simpler question."
        )
        
        return QueryResponse(
            response=f"❌ {user_message}",
            error=str(error),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method=f"error_at_{current_progress.stage.value}"
        )
    
    def _create_fallback_response_from_context(self, context: str, question: str) -> str:
        """Create fallback response when AI generation fails"""
        
        # Extract key information from context
        context_preview = context[:1500] + "..." if len(context) > 1500 else context
        
        return f"""**Based on the retrieved documents:**

{context_preview}

---
*Note: This is a direct excerpt from your documents. AI response generation was unavailable, so I'm showing you the relevant content directly.*

**To get a more refined answer:** Try asking a more specific question about the content above."""
    
    def _create_partial_response_from_context(
        self, 
        context_text: str, 
        source_info: List[Dict],
        query_context: QueryContext, 
        search_results: SearchResults
    ) -> QueryResponse:
        """Create partial response when full processing isn't possible"""
        
        # Create basic response from context
        response = f"""**Partial Results - Processing Timeout**

I found relevant information but couldn't complete full processing. Here's what I found:

## Key Information Found:
{context_text[:1000]}{'...' if len(context_text) > 1000 else ''}

## Sources Found:
{len(source_info)} relevant document sections

**To get complete analysis:** Try asking a more specific question about the information above."""
        
        return QueryResponse(
            response=response,
            error="partial_processing_timeout",
            context_found=bool(search_results.internal_results),
            sources=source_info[:5],  # Limit sources
            session_id=query_context.session_id,
            confidence_score=0.3,  # Low confidence for partial
            sources_searched=search_results.sources_searched,
            retrieval_method="partial_timeout"
        )
    
    def _create_basic_response_from_processing_result(
        self,
        processing_result: ProcessingResult,
        query_context: QueryContext,
        search_results: SearchResults
    ) -> QueryResponse:
        """Create basic response when post-processing fails"""
        
        return QueryResponse(
            response=processing_result.response_text,
            error=None,
            context_found=bool(search_results.internal_results),
            sources=processing_result.sources,
            session_id=query_context.session_id,
            confidence_score=processing_result.confidence_score,
            sources_searched=search_results.sources_searched,
            retrieval_method=search_results.retrieval_method
        )
    
    def _create_no_results_response(self, session_id: str, sources_searched: List[str]) -> QueryResponse:
        """Create response when no results are found"""
        return QueryResponse(
            response="I couldn't find any relevant information to answer your question in the searched sources. Try rephrasing your question or uploading relevant documents.",
            error=None,
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.1,
            sources_searched=sources_searched,
            retrieval_method="no_results"
        )
    
    def _add_to_conversation_history(self, session_id: str, question: str, response: str, source_info: List[Dict]):
        """Add query and response to conversation history"""
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", response, source_info)
    
    # === MONITORING AND STATISTICS ===
    
    def _update_stats(self, result_type: str, processing_time: float):
        """Update processing statistics"""
        self.processing_stats['total_queries'] += 1
        
        if result_type == 'success':
            self.processing_stats['successful_queries'] += 1
        elif result_type == 'timeout':
            self.processing_stats['timeout_queries'] += 1
        elif result_type == 'error':
            self.processing_stats['error_queries'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_queries']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = ((current_avg * (total - 1)) + processing_time) / total
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.processing_stats,
            'success_rate': (self.processing_stats['successful_queries'] / 
                           max(1, self.processing_stats['total_queries'])),
            'timeout_rate': (self.processing_stats['timeout_queries'] / 
                           max(1, self.processing_stats['total_queries'])),
            'error_rate': (self.processing_stats['error_queries'] / 
                         max(1, self.processing_stats['total_queries'])),
            'features_available': self.features
        }
    
    def get_active_queries(self) -> Dict[str, Any]:
        """Get information about currently active queries"""
        
        current_time = datetime.utcnow()
        active_info = {}
        
        for query_id, query_info in self.active_queries.items():
            started_at = query_info['started_at']
            elapsed = (current_time - started_at).total_seconds()
            progress = query_info['progress_tracker'].get_current_progress()
            
            active_info[query_id] = {
                'question_preview': query_info['question'],
                'elapsed_seconds': round(elapsed, 1),
                'timeout_in_seconds': query_info['timeout'],
                'current_stage': progress.stage.value,
                'progress_percent': progress.progress_percent,
                'current_message': progress.message,
                'time_remaining': max(0, query_info['timeout'] - elapsed)
            }
        
        return active_info
    
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get timeout and performance statistics"""
        
        stats = self.get_processing_stats()
        
        return {
            'processing_stats': stats,
            'timeout_analysis': {
                'timeout_rate': stats['timeout_rate'],
                'avg_processing_time': stats['avg_processing_time'],
                'success_rate': stats['success_rate']
            },
            'active_queries': len(self.active_queries),
            'timeout_thresholds': self.timeout_handler.base_timeouts,
            'complexity_multipliers': self.timeout_handler.complexity_multipliers,
            'recommendations': self._get_performance_recommendations(stats)
        }
    
    def _get_performance_recommendations(self, stats: Dict) -> List[str]:
        """Get performance improvement recommendations"""
        recommendations = []
        
        if stats['timeout_rate'] > 0.1:  # More than 10% timeouts
            recommendations.append("High timeout rate detected - consider reducing document count or query complexity")
        
        if stats['avg_processing_time'] > 20:
            recommendations.append("High average processing time - consider optimizing document chunking")
        
        if stats['error_rate'] > 0.05:  # More than 5% errors
            recommendations.append("High error rate detected - check document processing and external API status")
        
        if len(self.active_queries) > 5:
            recommendations.append("Many concurrent queries - consider implementing query queuing")
        
        return recommendations

# === BACKWARD COMPATIBLE FUNCTIONS ===

# Global instance for backward compatibility
_query_processor = None

def get_query_processor() -> QueryProcessor:
    """Get or create global query processor instance"""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor

# Legacy function for backward compatibility
def process_query(question: str, session_id: str, user_id: Optional[str], search_scope: str, 
                 response_style: str = "balanced", use_enhanced_rag: bool = True, 
                 document_id: str = None, search_external: bool = None) -> QueryResponse:
    """
    Legacy synchronous query processing function for backward compatibility
    
    WARNING: This function blocks. Use QueryProcessor.process_query_with_enhanced_timeout() for new code.
    """
    processor = get_query_processor()
    
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            processor.process_query_with_enhanced_timeout(
                question, session_id, user_id, search_scope,
                response_style, use_enhanced_rag, document_id, search_external
            )
        )
        
        loop.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in legacy sync query processing: {e}")
        return QueryResponse(
            response="An error occurred processing your query. Please try again.",
            error=str(e),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="sync_error"
        )

# === ROUTER INTEGRATION EXAMPLE ===

"""
COMPLETE ROUTER INTEGRATION:

from ..processors.query_processor import get_query_processor

# Initialize once
query_processor = get_query_processor()

@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: Query, current_user: User = Depends(get_current_user)):
    '''Enhanced async ask endpoint with timeout protection'''
    
    # Optional: Set up progress callback for real-time updates
    def progress_callback(progress: ProcessingProgress):
        # Could emit to WebSocket, store in Redis, etc.
        logger.info(f"Progress: {progress.progress_percent}% - {progress.message}")
    
    return await query_processor.process_query_with_enhanced_timeout(
        question=query.question,
        session_id=query.session_id or str(uuid.uuid4()),
        user_id=current_user.user_id,
        search_scope=query.search_scope or "all",
        response_style=query.response_style or "balanced",
        use_enhanced_rag=query.use_enhanced_rag if query.use_enhanced_rag is not None else True,
        document_id=query.document_id,
        progress_callback=progress_callback
    )

@router.get("/processing/active-queries")
async def get_active_queries():
    '''Monitor active queries'''
    return query_processor.get_active_queries()

@router.get("/processing/timeout-stats") 
async def get_timeout_stats():
    '''Get timeout and performance statistics'''
    return query_processor.get_timeout_statistics()

COMPLETE FEATURE SET:
✅ Enhanced timeout handling with adaptive timeouts (15-60s)
✅ Staged processing with individual stage timeouts
✅ Progress tracking with real-time updates
✅ Partial result recovery on timeouts
✅ Anti-hallucination validation
✅ Professional response formatting
✅ Comprehensive legal database integration
✅ Immigration and country conditions support
✅ Government enforcement data integration
✅ Performance monitoring and statistics
✅ Complete backward compatibility
✅ Production-ready error handling
"""

