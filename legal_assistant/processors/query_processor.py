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
    
    # === MAIN PROCESSING METHOD ===
    
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

    async def _search_case_law_databases_direct(self, question: str) -> Tuple[Optional[str], List[Dict]]:
        """Direct case law database search with timeout protection"""
        
        try:
            from ..services.external_db_service import external_databases
            
            # Use Harvard Caselaw and CourtListener for case law
            case_law_dbs = ['harvard_caselaw', 'courtlistener']
            available_dbs = [db for db in case_law_dbs if db in external_databases and external_databases[db]]
            
            if not available_dbs:
                self.logger.warning("❌ No case law databases available")
                return None, []
            
            self.logger.info(f"🏛️ Searching case law databases: {available_dbs}")
            
            all_results = []
            
            for db_name in available_dbs:
                try:
                    db_interface = external_databases[db_name]
                    
                    # Search with federal court filters
                    filters = {'court_level': 'federal'} if 'federal' in question.lower() else {}
                    
                    db_results = await asyncio.wait_for(
                        asyncio.to_thread(db_interface.search, question, filters),
                        timeout=5  # 5 second timeout
                    )
                    
                    if db_results:
                        self.logger.info(f"✅ {db_name}: Found {len(db_results)} results")
                        all_results.extend(db_results[:5])
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"⏰ {db_name}: Search timed out")
                except Exception as e:
                    self.logger.error(f"❌ {db_name}: Search failed - {e}")
            
            if all_results:
                # Format case law results
                formatted_text = "\n\n## FEDERAL COURT CASE LAW:\n"
                
                for result in all_results[:6]:
                    title = result.get('title', 'Unknown Case')
                    court = result.get('court', 'Unknown Court')
                    date = result.get('date', '')
                    preview = result.get('preview', result.get('snippet', ''))
                    
                    formatted_text += f"\n**{title}**\n"
                    formatted_text += f"Court: {court}\n"
                    if date:
                        formatted_text += f"Date: {date}\n"
                    if preview:
                        formatted_text += f"Summary: {preview[:200]}...\n"
                    formatted_text += "---\n"
                
                source_info = [{
                    'file_name': result.get('title', 'Federal Court Case'),
                    'source_type': 'external_case_law',
                    'database': result.get('source_database', 'unknown'),
                    'relevance': 0.9,
                    'url': result.get('url', ''),
                    'court': result.get('court', ''),
                    'date': result.get('date', '')
                } for result in all_results[:6]]
                
                self.logger.info(f"🏛️ Case law search SUCCESS: {len(all_results)} results found")
                return formatted_text, source_info
            
            else:
                self.logger.warning("❌ No case law results found")
                return None, []
                
        except Exception as e:
            self.logger.error(f"❌ Case law search failed: {e}")
            return None, []
    
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


    def _detect_federal_court_query(self, question: str) -> bool:
        """Detect queries specifically about federal court cases"""
        federal_court_patterns = [
            r'\bfederal\s+(?:district\s+)?court\b',
            r'\bU\.S\.\s+District\s+Court\b',
            r'\bfederal\s+(?:judge|ruling|decision|case)\b',
            r'\bdistrict\s+court\s+(?:decision|ruling)\b',
            r'\bcircuit\s+court\s+(?:decision|ruling)\b',
            r'\bfederal\s+court\s+(?:decision|ruling|case)\b',
            r'\bfederal\s+appeals\s+court\b',
            r'\bU\.S\.\s+Court\s+of\s+Appeals\b'
        ]
        
        found_patterns = []
        for pattern in federal_court_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                found_patterns.append(pattern)
        
        if found_patterns:
            self.logger.info(f"🏛️ Federal court patterns found: {len(found_patterns)} matches")
            return True
        
        return False
    
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
            # Keep all existing patterns
            r'\bcase\s*law\b', r'\bcases?\b', r'\bprecedent\b', r'\bruling\b',
            r'\bdecision\b', r'\bcourt\s*opinion\b', r'\bjudgment\b',
            r'\bmiranda\b', r'\bconstitutional\b', r'\bamendment\b',
            r'\bsupreme\s*court\b', r'\bappellate\b', r'\bdistrict\s*court\b',
            r'\blegal\s*research\b', r'\bfind\s*cases?\b', r'\blook\s*up\s*law\b',
            r'\bsearch\s*(?:for\s*)?(?:cases?|law|precedent)\b',
            r'\bliability\b', r'\bnegligence\b', r'\bcontract\s*law\b', r'\btort\b',
            
            # Add these new patterns:
            r'\bfederal\s+court\b', r'\bfederal\s+district\b', r'\bfederal\s+rulings?\b',
            r'\bcourt\s+decisions?\b', r'\bcourt\s+rulings?\b', r'\bjudicial\s+decisions?\b',
            r'\bfederal\s+cases?\b', r'\bdistrict\s+court\s+decisions?\b',
            
            # Specific legal areas:
            r'\bcontract\s+disputes?\b', r'\bintellectual\s+property\s+(?:rulings?|cases?)\b',
            r'\bemployment\s+discrimination\s+cases?\b', r'\bcivil\s+rights\s+(?:violations?|cases?)\b',
            r'\bantitrust\s+(?:cases?|violations?|rulings?)\b',
            
            # Recent case law:
            r'\brecent\s+(?:federal\s+)?court\s+rulings?\b',
            r'\brecent\s+(?:federal\s+)?(?:cases?|decisions?)\b'
        ]
        
        # Rest of method stays the same
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
        """FIXED: Enhanced decision making for external search"""
        
        # Don't search external if user explicitly wants only their documents
        if search_scope == "user_only":
            return False
        
        # FIXED: More specific EPA filtering instead of broad "violation" blocking
        epa_specific_queries = [
            'epa violation', 'environmental violation', 'epa enforcement data', 
            'environmental compliance', 'epa citation database', 'environmental penalty data'
        ]
        if any(term in question.lower() for term in epa_specific_queries):
            self.logger.info("🌱 EPA-specific query detected - skipping external search")
            return False
        
        # POSITIVE: Force external search for federal court queries
        if self._detect_federal_court_query(question):
            self.logger.info("🏛️ Federal court query detected - enabling external search")
            return True
        
        # POSITIVE: Force external search for legal research areas
        legal_research_areas = [
            'contract disputes', 'intellectual property', 'employment discrimination',
            'civil rights violations', 'antitrust', 'securities law', 'patent law',
            'copyright law', 'constitutional law', 'tort law', 'criminal law'
        ]
        
        if any(area in question.lower() for area in legal_research_areas):
            self.logger.info("⚖️ Legal research area detected - enabling external search")
            return True
        
        # Check for legal areas that benefit from comprehensive search
        comprehensive_areas = ['business', 'labor', 'healthcare', 'criminal', 'immigration', 'housing']
        if any(area in query_types for area in comprehensive_areas):
            return True
        
        # Check for enforcement/compliance questions (but not EPA)
        enforcement_keywords = ['enforcement', 'compliance', 'citation', 'penalty', 'recall']
        if any(keyword in question.lower() for keyword in enforcement_keywords) and 'epa' not in question.lower():
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
        
        # Only skip for very specific EPA data queries
        if any(term in question.lower() for term in ['epa enforcement data', 'epa violation database']):
            self.logger.info("🌱 EPA data query - skipping external APIs")
            return None, []
        
        # PRIORITY: Case law queries get special treatment
        case_law_indicators = ['court', 'ruling', 'decision', 'case law', 'judicial']
        is_case_law_query = any(indicator in question.lower() for indicator in case_law_indicators)
        
        if is_case_law_query:
            self.logger.info("🏛️ Case law query - using dedicated case law search")
            return await self._search_case_law_databases_direct(question)
        
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
        max_context_length = 8000 if QueryType.STATUTORY.value in [t for t in search_results.sources_searched if 'statutory' in t] else 6000
        
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
        
        # UPDATED SECTION: Combine contexts with limits
        if search_results.external_context:
            # Limit total context size and prioritize most relevant content
            max_user_context = 4000  # Reduced from unlimited
            max_external_context = 2000  # Limit external context
            
            # Truncate contexts if too long
            limited_user_context = context_text[:max_user_context] + "..." if len(context_text) > max_user_context else context_text
            limited_external_context = search_results.external_context[:max_external_context] + "..." if len(search_results.external_context) > max_external_context else search_results.external_context
            
            # Combine with clear sections
            full_context = f"""=== EXTERNAL LEGAL SOURCES ===
{limited_external_context}

=== YOUR UPLOADED DOCUMENTS ===
{limited_user_context}"""
            
            self.logger.info(f"📖 Combined context: {len(limited_external_context)} chars external + {len(limited_user_context)} chars user")
        else:
            full_context = context_text
            self.logger.info(f"📖 Using user documents only: {len(context_text)} chars")
        
        # Generate response with timeout protection
        progress_tracker.update_stage(
            ProcessingStage.GENERATING_RESPONSE,
            "Generating AI response...",
            75
        )
        
        if self.features['ai_enabled']:
            try:
                prompt = self._create_prompt(query_context, search_results, full_context)
                
                # UPDATED: AI generation with increased timeout
                response_text = await asyncio.wait_for(
                    asyncio.to_thread(call_openrouter_api, prompt, OPENROUTER_API_KEY),
                    timeout=30  # 30 second timeout for complex legal analysis
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

ANTI-HALLUCINATION REQUIREMENTS:
🚫 ONLY answer based on the provided context and sources
🚫 If information is NOT in the sources, say "This information is not available in the provided documents"
🚫 NEVER make up facts, dates, case names, citations, or enforcement actions
🚫 When uncertain, use phrases like "Based on the available information..." or "The documents suggest..."

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}
RETRIEVAL METHOD: {search_results.retrieval_method}

CONVERSATION HISTORY:
{query_context.conversation_context}

LEGAL CONTEXT:
{context_text}

USER QUESTION:
{query_context.original_question}

RESPONSE FORMAT:
## Direct Answer
[Your main answer based ONLY on the provided sources]

## Key Supporting Points
[Bullet points from the sources that support your answer]

## Sources Referenced
[List the specific documents/databases that support your answer]

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
            "according to the provided sources"
        ]
        
        response_lower = response.lower()
        has_uncertainty = any(phrase in response_lower for phrase in uncertainty_phrases)
        
        if not has_uncertainty:
            # Add uncertainty marker at the beginning
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    lines[i] = f"Based on the available information, {line}"
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
        
        # Add notices to response
        final_response = processing_result.response_text
        if notices:
            final_response += "\n\n" + "\n\n".join(notices)
        
        # Filter relevant sources
        relevant_sources = [s for s in processing_result.sources if s.get('relevance', 0) >= MIN_RELEVANCE_SCORE]
        
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
            sources=relevant_sources,
            session_id=query_context.session_id,
            confidence_score=float(combined_confidence),
            sources_searched=search_results.sources_searched,
            expand_available=len(query_context.expanded_questions) > 1 if query_context.use_enhanced_rag else False,
            retrieval_method=search_results.retrieval_method
        )
    
    # === FORMATTING AND CITATION METHODS ===
    
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
- Ask about specific sections or topics rather than broad analysis"""
        
        else:
            response = f"""⚠️ **Query Processing Timeout**

Your query timed out after {current_progress.progress_percent}% completion.

**Processing stage when timeout occurred:** {current_progress.message}

**To resolve:**
- Try a more specific question
- Break complex questions into parts
- Reduce the scope of your search"""
        
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
    
    This function maintains the exact interface your existing router expects.
    """
    
    try:
        logger.info(f"🔍 Processing query: '{question[:100]}...' for user: {user_id}")
        
        # Step 1: Parse and analyze question
        if use_enhanced_rag:
            questions = parse_multiple_questions(question)
            combined_query = " ".join(questions)
            logger.info(f"Enhanced RAG: Parsed {len(questions)} questions")
        else:
            combined_query = question
        
        # Step 2: Get conversation context
        conversation_context = get_conversation_context(session_id)
        
        # Step 3: Search for relevant documents
        logger.info(f"🔎 Searching documents with scope: {search_scope}")
        search_results, sources_searched, retrieval_method = combined_search(
            combined_query, 
            user_id, 
            search_scope, 
            conversation_context, 
            use_enhanced_rag, 
            k=15,  # Get more results for better context
            document_id=document_id
        )
        
        # Step 4: Check if we found relevant documents
        if not search_results:
            logger.warning("No relevant documents found")
            return QueryResponse(
                response="I couldn't find any relevant information to answer your question in your uploaded documents. Please try rephrasing your question or upload relevant documents.",
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.1,
                sources_searched=sources_searched,
                retrieval_method=retrieval_method
            )
        
        logger.info(f"📚 Found {len(search_results)} relevant document chunks")
        
        # Step 5: Format context for LLM processing
        max_context = 8000 if _is_statutory_question(question) else 6000
        context_text, source_info = format_context_for_llm(search_results, max_length=max_context)
        
        # Step 6: Extract specific information based on question type
        extracted_info = _extract_information_by_type(question, context_text)
        
        # Step 7: Enhance context with extracted information
        if extracted_info:
            enhancement = _format_extracted_info(extracted_info)
            if enhancement:
                context_text += enhancement
        
        # Step 8: Generate AI response
        if FeatureFlags.AI_ENABLED and call_openrouter_api:
            logger.info("🤖 Generating AI response...")
            
            # Create appropriate prompt based on question type
            prompt = _create_prompt(question, context_text, response_style, conversation_context)
            
            try:
                ai_response = call_openrouter_api(prompt, OPENROUTER_API_KEY)
                response_text = ai_response or "I found relevant documents but couldn't generate a response."
                logger.info("✅ AI response generated successfully")
            except Exception as e:
                logger.error(f"AI response generation failed: {e}")
                response_text = _create_fallback_response(context_text, question)
        else:
            logger.info("📄 AI not available, creating fallback response")
            response_text = _create_fallback_response(context_text, question)
        
        # Step 9: Calculate confidence score
        confidence = calculate_confidence_score(search_results, len(response_text))
        
        # Step 10: Add conversation to history
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", response_text, source_info)
        
        # Step 11: Return comprehensive response
        return QueryResponse(
            response=response_text,
            context_found=True,
            sources=source_info,
            session_id=session_id,
            confidence_score=confidence,
            sources_searched=sources_searched,
            expand_available=len(questions) > 1 if use_enhanced_rag else False,
            retrieval_method=retrieval_method
        )
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        
        return QueryResponse(
            response=f"An error occurred while processing your question: {str(e)}",
            error=str(e),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="error"
        )

# === HELPER FUNCTIONS FOR LEGACY COMPATIBILITY ===

def _is_statutory_question(question: str) -> bool:
    """Detect if this is a statutory/regulatory question"""
    statutory_indicators = [
        r'\bUSC\s+\d+', r'\bU\.S\.C\.\s*§?\s*\d+', r'\bCFR\s+\d+', r'\bC\.F\.R\.\s*§?\s*\d+',
        r'\bRCW\s+\d+\.\d+\.\d+', r'\bWAC\s+\d+',
        r'\bstatute[s]?', r'\bregulation[s]?', r'\bcode\s+section[s]?',
        r'\brequirements?', r'\bmust\s+meet',
        r'\bshall\s+(?:meet|comply|maintain)', r'\bmust\s+(?:include|contain|provide)',
    ]
    
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in statutory_indicators)

def _extract_information_by_type(question: str, context_text: str) -> Dict[str, Any]:
    """Extract specific information from context based on question type"""
    
    # Enhanced pattern to find bill information
    bill_match = re.search(r"(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)", question, re.IGNORECASE)
    statute_match = re.search(r"(RCW|USC|CFR|WAC)\s+(\d+\.\d+\.\d+|\d+)", question, re.IGNORECASE)
    
    if bill_match:
        bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
        logger.info(f"🏛️ Searching for bill: {bill_number}")
        return extract_bill_information(context_text, bill_number)
    elif statute_match:
        statute_citation = f"{statute_match.group(1)} {statute_match.group(2)}"
        logger.info(f"📖 Searching for statute: {statute_citation}")
        return _extract_statutory_information(context_text, statute_citation)
    else:
        return extract_universal_information(context_text, question)

def _extract_statutory_information(context_text: str, statute_citation: str) -> Dict[str, Any]:
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
        
        # Look for requirement patterns
        requirement_patterns = [
            r"(?:shall|must|required)\s+([^.]{10,100})",
            r"(?:requirement|standard)\s*:\s*([^\n]{10,200})"
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, context_text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:
                    extracted_info["requirements"].append(match.strip())
        
        return extracted_info
        
    except Exception as e:
        logger.error(f"Error extracting statutory information: {e}")
        return extracted_info

def _format_extracted_info(extracted_info: Dict[str, Any]) -> str:
    """Format extracted information for context enhancement"""
    if not extracted_info:
        return ""
    
    enhancement = "\n\n=== KEY INFORMATION EXTRACTED ===\n"
    for key, value in extracted_info.items():
        if value:
            if isinstance(value, list):
                if value:  # Only add if list is not empty
                    enhancement += f"• {key.replace('_', ' ').title()}: {', '.join(str(v) for v in value[:5])}\n"
            else:
                enhancement += f"• {key.replace('_', ' ').title()}: {value}\n"
    
    return enhancement if enhancement.strip() != "=== KEY INFORMATION EXTRACTED ===" else ""

def _create_prompt(question: str, context_text: str, response_style: str, conversation_context: str) -> str:
    """Create appropriate prompt based on query type and context"""
    
    if _is_statutory_question(question):
        return _create_statutory_prompt(context_text, question, response_style, conversation_context)
    else:
        return _create_regular_prompt(context_text, question, response_style, conversation_context)

def _create_statutory_prompt(context_text: str, question: str, response_style: str, conversation_context: str) -> str:
    """Create enhanced prompt for statutory analysis using your complex prompt"""
    
    return f"""You are a legal research assistant. Provide thorough, accurate responses based on the provided documents.

SOURCE HIERARCHY:
- **PRIMARY**: Information from the retrieved documents provided in the context
- **SECONDARY**: General legal knowledge ONLY when documents are unavailable or contain insufficient information
- **STRICT LIMITATIONS**: 
  - Only use well-established, fundamental legal principles (e.g., basic elements of crimes, standard procedural rules)
  - Do NOT invent case law, specific precedents, or detailed statutory provisions
  - Clearly label all general knowledge with disclaimers
  - When in doubt, default to "information not available"

HALLUCINATION CHECK - Before responding, verify:
1. Is each claim supported by the retrieved documents?
2. Am I adding information not present in the sources?
3. If uncertain, default to "information not available"

INSTRUCTIONS FOR THOROUGH ANALYSIS:
1. **READ CAREFULLY**: Scan the entire context for information that answers the user's question
2. **EXTRACT COMPLETELY**: When extracting requirements, include FULL details (e.g., "60 minutes" not just "minimum of")
3. **QUOTE VERBATIM**: For statutory standards, use exact quotes: `"[Exact Text]" (Source)`
4. **ENUMERATE EXPLICITLY**: Present listed requirements as numbered points with full quotes
5. **CITE SOURCES**: Reference the document name or case citation for each fact
6. **BE COMPLETE**: Explicitly note missing standards: "Documents lack full subsection [X]"
7. **USE DECISIVE PHRASING**: State facts directly ("The statute requires...") - NEVER "documents indicate"

RESPONSE STYLE: {response_style}

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT (ANALYZE THOROUGHLY):
{context_text}

USER QUESTION:
{question}

RESPONSE APPROACH:
- **FIRST**: Identify what specific information the user is asking for. Do not reference any statute, case law, or principle unless it appears verbatim in the context.
- **SECOND**: Search the context thoroughly for that information  
- **THIRD**: Present any information found clearly and completely. At the end of your response, list all facts provided and their source documents for verification.
- **FOURTH**: Note what information is not available (if any)
- **FIFTH**: When documents lack specific guidance but user requests legal analysis, provide response based on fundamental legal principles with clear disclaimers
- **ALWAYS**: Cite the source document or case for each fact provided

LEGAL ANALYSIS FRAMEWORK:
- When documents lack specific guidance, provide analysis based on fundamental legal principles
- Focus on established concepts, not novel interpretations
- Structure responses around: "Based on general legal principles, typical approaches include..."
- Avoid making definitive statements about jurisdiction-specific rules not in the documents
- Clearly distinguish between document-based facts and general legal knowledge

GENERAL LEGAL KNOWLEDGE GUIDANCE:
When the provided documents contain little or no information relevant to the user's question:
1. **FIRST ACKNOWLEDGE**: State clearly that the documents do not contain sufficient information
2. **THEN PROVIDE**: Offer helpful general legal information from pre-trained knowledge
3. **USE CLEAR MARKERS**: Begin such sections with phrases like:
   - "While not found in your documents, general legal principles suggest..."
   - "Based on standard legal practice (not from your documents)..."
   - "From general legal knowledge..."
4. **STRUCTURE THE RESPONSE**:
   - Start with what IS in the documents (even if minimal)
   - Clearly transition to general knowledge section
   - Provide useful, accurate legal information
   - End with appropriate disclaimers

ADDITIONAL GUIDANCE:
- After fully answering based on the provided documents, if relevant key legal principles under Washington state law, any other U.S. state law, or U.S. federal law are not found in the sources, you may add a clearly labeled general legal principles section.
- This section should help answer the user's question using your pre-trained legal knowledge
- Be explicit that this information is NOT from the provided documents
- Include standard legal disclaimers about seeking professional legal advice
- Format this section distinctly under a heading such as:

**GENERAL LEGAL INFORMATION** 
*(Not from provided documents - based on general legal knowledge)*

RESPONSE:"""

def _create_regular_prompt(context_text: str, question: str, response_style: str, conversation_context: str) -> str:
    """Create simple prompt for general questions"""
    
    return f"""You are a legal research assistant with access to comprehensive legal databases.

PRIMARY RESPONSE REQUIREMENTS:
✅ FIRST answer based on the provided context and sources
✅ If documents contain relevant information, cite it specifically
✅ Be clear about what comes from documents vs. general knowledge

WHEN DOCUMENTS LACK INFORMATION:
If the provided documents don't contain sufficient information to answer the question:
1. **STATE CLEARLY**: "The provided documents do not contain [specific information requested]"
2. **OFFER HELP**: Provide useful general legal information from your training
3. **USE MARKERS**: Clearly mark general knowledge sections with phrases like:
   - "Based on general legal principles (not from your documents)..."
   - "From standard legal practice..."
   - "General legal information suggests..."
4. **INCLUDE DISCLAIMERS**: Always note that general information should be verified with legal counsel

RESPONSE STYLE: {response_style}

CONVERSATION HISTORY:
{conversation_context}

LEGAL CONTEXT:
{context_text}

USER QUESTION:
{question}

RESPONSE FORMAT:
## Information from Your Documents
[What the provided documents say about this topic - even if limited]

## Answer Based on Documents
[Your main answer based on the provided sources, or note if documents don't contain the answer]

## General Legal Information
*(If documents are insufficient)*
[Helpful legal information from general knowledge, clearly marked as not from documents]

## Sources Referenced
[List the specific documents that support your answer, if any]

## Legal Disclaimer
*This response includes general legal information where your documents did not contain specific answers. For legal advice specific to your situation, please consult with a qualified attorney.*

RESPONSE:"""

def _create_fallback_response(context_text: str, question: str) -> str:
    """Create fallback response when AI generation fails"""
    
    # Extract key information from context
    context_preview = context_text[:1000] + "..." if len(context_text) > 1000 else context_text
    
    return f"""**Based on the retrieved documents:**

{context_preview}

---
*Note: AI response generation was unavailable, so I'm showing you the relevant content directly.*

**To get a more refined answer:** Try asking a more specific question about the content above."""
    
    def _create_immigration_prompt(self, context_text: str, query_context: QueryContext,
                                 search_results: SearchResults) -> str:
        """Create prompt specifically for immigration queries"""
        
        return f"""You are an immigration legal assistant with access to official USCIS data, visa bulletins, and immigration law databases.

ANTI-HALLUCINATION REQUIREMENTS:
🚫 ONLY answer based on the provided context and sources
🚫 If information is NOT in the sources, say "This information is not available in the provided documents"
🚫 NEVER make up processing times, case outcomes, or legal requirements

IMPORTANT IMMIGRATION CONTEXT:
- Immigration law is complex and changes frequently
- Processing times vary by service center and case type
- Each case is unique and requires individual assessment

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}

CONVERSATION HISTORY:
{query_context.conversation_context}

IMMIGRATION CONTEXT:
{context_text}

USER QUESTION:
{query_context.original_question}

RESPONSE FORMAT:
## Direct Answer
[Your main answer based ONLY on the provided sources]

## Key Immigration Points
[Specific points from immigration sources]

## Important Disclaimers
[Relevant warnings about legal complexity, etc.]

RESPONSE:"""
    
    def _create_statutory_prompt(self, context_text: str, query_context: QueryContext,
                               search_results: SearchResults) -> str:
        """Create enhanced prompt for statutory analysis using your complex prompt"""
        
        return f"""You are a legal research assistant. Provide thorough, accurate responses based on the provided documents.

SOURCE HIERARCHY:
- **PRIMARY**: Information from the retrieved documents provided in the context
- **SECONDARY**: General legal knowledge ONLY when documents are unavailable
- **STRICT LIMITATIONS**: 
  - Only use well-established, fundamental legal principles (e.g., basic elements of crimes, standard procedural rules)
  - Do NOT invent case law, specific precedents, or detailed statutory provisions
  - Clearly label all general knowledge with disclaimers
  - When in doubt, default to "information not available"

SOURCES SEARCHED: {', '.join(search_results.sources_searched)}
RETRIEVAL METHOD: {search_results.retrieval_method}
{f"DOCUMENT FILTER: Specific document {query_context.document_id}" if query_context.document_id else "DOCUMENT SCOPE: All available documents"}

HALLUCINATION CHECK - Before responding, verify:
1. Is each claim supported by the retrieved documents?
2. Am I adding information not present in the sources?
3. If uncertain, default to "information not available"

INSTRUCTIONS FOR THOROUGH ANALYSIS:
1. **READ CAREFULLY**: Scan the entire context for information that answers the user's question
2. **EXTRACT COMPLETELY**: When extracting requirements, include FULL details (e.g., "60 minutes" not just "minimum of")
3. **QUOTE VERBATIM**: For statutory standards, use exact quotes: `"[Exact Text]" (Source)`
4. **ENUMERATE EXPLICITLY**: Present listed requirements as numbered points with full quotes
5. **CITE SOURCES**: Reference the document name or case citation for each fact
6. **BE COMPLETE**: Explicitly note missing standards: "Documents lack full subsection [X]"
7. **USE DECISIVE PHRASING**: State facts directly ("The statute requires...") - NEVER "documents indicate"

RESPONSE STYLE: {query_context.response_style}

CONVERSATION HISTORY:
{query_context.conversation_context}

DOCUMENT CONTEXT (ANALYZE THOROUGHLY):
{context_text}

USER QUESTION:
{query_context.original_question}

RESPONSE APPROACH:
- **FIRST**: Identify what specific information the user is asking for. Do not reference any statute, case law, or principle unless it appears verbatim in the context.
- **SECOND**: Search the context thoroughly for that information  
- **THIRD**: Present any information found clearly and completely. At the end of your response, list all facts provided and their source documents for verification.
- **FOURTH**: Note what information is not available (if any)
- **FIFTH**: When documents lack specific guidance but user requests legal analysis, provide response based on fundamental legal principles with clear disclaimers
- **ALWAYS**: Cite the source document or case for each fact provided

LEGAL ANALYSIS FRAMEWORK:
- When documents lack specific guidance, provide analysis based on fundamental legal principles
- Focus on established concepts, not novel interpretations
- Structure responses around: "Based on general legal principles, typical approaches include..."
- Avoid making definitive statements about jurisdiction-specific rules not in the documents
- Clearly distinguish between document-based facts and general legal knowledge

ADDITIONAL GUIDANCE:
- After fully answering based on the provided documents, if relevant key legal principles under Washington state law, any other U.S. state law, or U.S. federal law are not found in the sources, you may add a clearly labeled general legal principles disclaimer.
- This disclaimer must clearly state it is NOT based on the provided documents but represents general background knowledge of applicable Washington state, other state, and federal law.
- Do NOT use this disclaimer to answer the user's question directly; it serves only as supplementary context.
- This disclaimer must explicitly state that these principles are not found in the provided documents but are usually relevant legal background.
- Format this disclaimer distinctly at the end of the response under a heading such as "GENERAL LEGAL PRINCIPLES DISCLAIMER."

RESPONSE:"""
