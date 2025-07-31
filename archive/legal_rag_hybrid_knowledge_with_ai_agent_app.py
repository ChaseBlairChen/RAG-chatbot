# Enhanced AI Agent with Comprehensive Case Law Analysis Capabilities
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
# Third-party library imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from chromadb.config import Settings
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this section after your imports (around line 30-35):
# Create FastAPI app instance
app = FastAPI(
    title="Enhanced Legal AI Agent", 
    description="Comprehensive Legal Analysis with Case Law Research", 
    version="3.0.0"
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global variables and configuration
CHROMA_PATH = "chroma"  # Update this to your actual Chroma database path

# --- FIXED: Updated Chroma Settings to match the working version ---
CHROMA_CLIENT_SETTINGS = Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)
# --- END OF FIX ---

# In-memory conversation storage
conversations = {}

# --- Case Law Analysis Data Models ---
class CaseType(Enum):
    CRIMINAL = "criminal"
    CIVIL = "civil"
    CONSTITUTIONAL = "constitutional"
    ADMINISTRATIVE = "administrative"
    IMMIGRATION = "immigration"
    CORPORATE = "corporate"
    FAMILY = "family"
    PROPERTY = "property"
    CONTRACT = "contract"
    TORT = "tort"

class JurisdictionLevel(Enum):
    SUPREME_COURT = "supreme_court"
    APPELLATE = "appellate"
    FEDERAL_DISTRICT = "federal_district"
    STATE_SUPREME = "state_supreme"
    STATE_APPELLATE = "state_appellate"
    STATE_TRIAL = "state_trial"
    ADMINISTRATIVE = "administrative"

@dataclass
class CaseCitation:
    case_name: str
    citation: str
    year: int
    court: str
    jurisdiction_level: JurisdictionLevel
    case_type: CaseType
    relevance_score: float = 0.0

@dataclass
class LegalPrinciple:
    principle: str
    supporting_cases: List[CaseCitation]
    jurisdiction: str
    confidence: float
    legal_area: str

@dataclass
class PrecedentAnalysis:
    binding_precedents: List[CaseCitation]
    persuasive_precedents: List[CaseCitation]
    conflicting_precedents: List[CaseCitation]
    trend_analysis: str
    jurisdiction_hierarchy: Dict[str, List[CaseCitation]]

# --- Enhanced Pydantic Models ---
class LegalQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"
    analysis_type: Optional[str] = "comprehensive"  # "case_law", "precedent", "comprehensive", "quick"
    jurisdiction: Optional[str] = "federal"  # "federal", "state", "specific_state"
    case_types: Optional[List[str]] = None  # Filter by case types
    time_period: Optional[str] = "all"  # "recent", "historical", "specific_years", "all"

# --- NEW: Pydantic Model for /ask endpoint ---
class AskQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
# --- END OF NEW MODEL ---

class LegalAnalysisResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[List[Dict]] = None
    session_id: str
    confidence_score: float = 0.0
    expand_available: bool = False
    # Enhanced legal analysis fields
    case_citations: Optional[List[Dict]] = None
    legal_principles: Optional[List[Dict]] = None
    precedent_analysis: Optional[Dict] = None
    jurisdiction_analysis: Optional[Dict] = None
    case_law_trends: Optional[str] = None
    recommended_cases: Optional[List[Dict]] = None

# --- Case Law Analysis Engine ---
class CaseLawAnalyzer:
    def __init__(self):
        self.case_patterns = self._compile_case_patterns()
        self.citation_patterns = self._compile_citation_patterns()
        self.legal_concepts = self._load_legal_concepts()

    def _compile_case_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for case identification"""
        return {
            'case_name': re.compile(r'([A-Z][a-zA-Z\s&]+)\s+v\.?\s+([A-Z][a-zA-Z\s&]+)', re.IGNORECASE),
            'citation_full': re.compile(r'(\d+)\s+([A-Z][a-zA-Z\.]*)\s+(\d+)(?:\s*\((\d{4})\))?'),
            'court_mention': re.compile(r'(Supreme Court|Court of Appeals|District Court|Circuit Court)', re.IGNORECASE),
            'year': re.compile(r'\b(19|20)\d{2}\b'),
            'legal_holding': re.compile(r'(held|holding|ruled|decided|found|concluded)\s+that', re.IGNORECASE),
            'legal_reasoning': re.compile(r'(because|since|therefore|thus|consequently|as a result)', re.IGNORECASE)
        }

    def _compile_citation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for legal citation formats"""
        return {
            'us_reports': re.compile(r'(\d+)\s+U\.S\.\s+(\d+)\s*\((\d{4})\)'),
            'supreme_court': re.compile(r'(\d+)\s+S\.\s*Ct\.\s+(\d+)\s*\((\d{4})\)'),
            'federal_reporter': re.compile(r'(\d+)\s+F\.\s*(?:2d|3d)?\s+(\d+)\s*\([^)]*(\d{4})[^)]*\)'),
            'state_reports': re.compile(r'(\d+)\s+[A-Z]{2,4}\.?\s*(?:2d|3d)?\s+(\d+)\s*\([^)]*(\d{4})[^)]*\)'),
            'westlaw': re.compile(r'(\d{4})\s+WL\s+(\d+)'),
            'lexis': re.compile(r'(\d{4})\s+[A-Z]+\s+LEXIS\s+(\d+)')
        }

    def _load_legal_concepts(self) -> Dict[str, List[str]]:
        """Load legal concepts and their related terms"""
        return {
            'constitutional_law': [
                'due process', 'equal protection', 'first amendment', 'fourth amendment',
                'commerce clause', 'supremacy clause', 'establishment clause', 'free speech'
            ],
            'criminal_law': [
                'mens rea', 'actus reus', 'burden of proof', 'reasonable doubt',
                'miranda rights', 'probable cause', 'search and seizure', 'self-incrimination'
            ],
            'civil_procedure': [
                'jurisdiction', 'venue', 'standing', 'class action', 'summary judgment',
                'discovery', 'pleadings', 'motion to dismiss', 'res judicata'
            ],
            'contract_law': [
                'offer', 'acceptance', 'consideration', 'breach', 'damages',
                'specific performance', 'statute of frauds', 'parol evidence'
            ],
            'tort_law': [
                'negligence', 'duty of care', 'causation', 'damages', 'strict liability',
                'intentional torts', 'defamation', 'privacy', 'product liability'
            ]
        }

    def extract_case_citations(self, text: str) -> List[CaseCitation]:
        """Extract and parse case citations from text"""
        citations = []
        # Extract case names
        case_matches = self.case_patterns['case_name'].findall(text)
        citation_matches = self.citation_patterns['us_reports'].findall(text)
        for i, (plaintiff, defendant) in enumerate(case_matches):
            case_name = f"{plaintiff.strip()} v. {defendant.strip()}"
            # Try to find corresponding citation
            citation = ""
            year = 0
            court = "Unknown"
            jurisdiction_level = JurisdictionLevel.STATE_TRIAL
            if i < len(citation_matches):
                vol, page, yr = citation_matches[i]
                citation = f"{vol} U.S. {page} ({yr})"
                year = int(yr)
                court = "U.S. Supreme Court"
                jurisdiction_level = JurisdictionLevel.SUPREME_COURT
            # Determine case type based on context
            case_type = self._determine_case_type(text, case_name)
            citations.append(CaseCitation(
                case_name=case_name,
                citation=citation,
                year=year,
                court=court,
                jurisdiction_level=jurisdiction_level,
                case_type=case_type,
                relevance_score=0.8  # Will be updated based on context
            ))
        return citations

    def _determine_case_type(self, text: str, case_name: str) -> CaseType:
        """Determine case type based on context and content"""
        text_lower = text.lower()
        case_lower = case_name.lower()
        # Criminal law indicators
        criminal_indicators = ['criminal', 'prosecution', 'defendant', 'guilty', 'sentence', 'conviction']
        if any(indicator in text_lower for indicator in criminal_indicators):
            return CaseType.CRIMINAL
        # Constitutional law indicators
        constitutional_indicators = ['constitutional', 'amendment', 'due process', 'equal protection']
        if any(indicator in text_lower for indicator in constitutional_indicators):
            return CaseType.CONSTITUTIONAL
        # Civil indicators
        civil_indicators = ['damages', 'plaintiff', 'liability', 'negligence', 'contract']
        if any(indicator in text_lower for indicator in civil_indicators):
            return CaseType.CIVIL
        return CaseType.CIVIL  # Default

    def analyze_precedents(self, cases: List[CaseCitation], query_context: str) -> PrecedentAnalysis:
        """Analyze precedential value of cases"""
        binding_precedents = []
        persuasive_precedents = []
        conflicting_precedents = []
        jurisdiction_hierarchy = {}
        # Sort cases by jurisdiction hierarchy and date
        supreme_court_cases = [c for c in cases if c.jurisdiction_level == JurisdictionLevel.SUPREME_COURT]
        appellate_cases = [c for c in cases if c.jurisdiction_level in [JurisdictionLevel.APPELLATE, JurisdictionLevel.STATE_SUPREME]]
        trial_cases = [c for c in cases if c.jurisdiction_level in [JurisdictionLevel.FEDERAL_DISTRICT, JurisdictionLevel.STATE_TRIAL]]
        # Supreme Court cases are generally binding
        binding_precedents.extend(supreme_court_cases)
        # Appellate cases may be binding or persuasive depending on jurisdiction
        for case in appellate_cases:
            if case.relevance_score > 0.7:
                binding_precedents.append(case)
            else:
                persuasive_precedents.append(case)
        # Trial cases are generally persuasive
        persuasive_precedents.extend(trial_cases)
        # Build jurisdiction hierarchy
        jurisdiction_hierarchy = {
            "Supreme Court": supreme_court_cases,
            "Appellate Courts": appellate_cases,
            "Trial Courts": trial_cases
        }
        # Analyze trends
        trend_analysis = self._analyze_case_trends(cases)
        return PrecedentAnalysis(
            binding_precedents=binding_precedents,
            persuasive_precedents=persuasive_precedents,
            conflicting_precedents=conflicting_precedents,
            trend_analysis=trend_analysis,
            jurisdiction_hierarchy=jurisdiction_hierarchy
        )

    def _analyze_case_trends(self, cases: List[CaseCitation]) -> str:
        """Analyze trends in case law over time"""
        if not cases:
            return "No cases available for trend analysis."
        # Sort cases by year
        dated_cases = [c for c in cases if c.year > 0]
        if not dated_cases:
            return "Insufficient temporal data for trend analysis."
        dated_cases.sort(key=lambda x: x.year)
        recent_cases = [c for c in dated_cases if c.year >= 2010]
        older_cases = [c for c in dated_cases if c.year < 2010]
        trend_analysis = f"Analyzed {len(cases)} cases spanning from {dated_cases[0].year} to {dated_cases[-1].year}. "
        if len(recent_cases) > len(older_cases):
            trend_analysis += "Recent decisions show increased attention to this legal area."
        elif len(recent_cases) < len(older_cases):
            trend_analysis += "This appears to be a more established area of law with foundational precedents."
        else:
            trend_analysis += "Consistent judicial attention across time periods."
        return trend_analysis

    def extract_legal_principles(self, text: str, cases: List[CaseCitation]) -> List[LegalPrinciple]:
        """Extract legal principles from text and associated cases"""
        principles = []
        # Look for holdings and legal rules
        holding_matches = self.case_patterns['legal_holding'].finditer(text)
        for match in holding_matches:
            # Get the sentence containing the holding
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 200)
            context = text[start:end]
            # Extract the principle
            principle_text = context.strip()
            # Determine legal area
            legal_area = self._categorize_legal_area(principle_text)
            # Associate with relevant cases
            supporting_cases = [c for c in cases if c.case_name.lower() in context.lower()]
            principles.append(LegalPrinciple(
                principle=principle_text,
                supporting_cases=supporting_cases,
                jurisdiction="Federal",  # Could be refined
                confidence=0.8,
                legal_area=legal_area
            ))
        return principles

    def _categorize_legal_area(self, text: str) -> str:
        """Categorize text into legal area based on keywords"""
        text_lower = text.lower()
        for area, keywords in self.legal_concepts.items():
            if any(keyword in text_lower for keyword in keywords):
                return area.replace('_', ' ').title()
        return "General Law"

# --- Enhanced Retrieval with Case Law Focus ---
def legal_enhanced_retrieval(db, query_text: str, conversation_history_context: str, 
                           analysis_type: str = "comprehensive", jurisdiction: str = "federal",
                           case_types: List[str] = None, k: int = 15) -> Tuple[List, Any]:
    """
    Enhanced retrieval specifically designed for legal case law analysis
    """
    logger.info(f"[LEGAL_RETRIEVAL] Analysis type: {analysis_type}, Jurisdiction: {jurisdiction}")
    try:
        # Strategy 1: Direct query with legal expansion
        legal_expanded_query = expand_legal_query_advanced(query_text, analysis_type, case_types)
        results_with_scores = db.similarity_search_with_relevance_scores(legal_expanded_query, k=k)
        # Strategy 2: Case-specific searches
        if analysis_type in ["case_law", "precedent", "comprehensive"]:
            case_queries = generate_case_law_queries(query_text)
            for case_query in case_queries[:3]:
                case_results = db.similarity_search_with_relevance_scores(case_query, k=k//3)
                results_with_scores.extend(case_results)
        # Strategy 3: Jurisdiction-specific search
        if jurisdiction != "general":
            jurisdiction_query = f"{query_text} {jurisdiction} court jurisdiction"
            jurisdiction_results = db.similarity_search_with_relevance_scores(jurisdiction_query, k=k//3)
            results_with_scores.extend(jurisdiction_results)
        # Strategy 4: Legal principle extraction
        principle_queries = extract_legal_principle_queries(query_text)
        for principle_query in principle_queries[:2]:
            principle_results = db.similarity_search_with_relevance_scores(principle_query, k=k//4)
            results_with_scores.extend(principle_results)
        # Remove duplicates and apply legal-specific filtering
        unique_results = remove_duplicate_documents(results_with_scores)
        # Legal-specific threshold (more lenient for case law)
        min_threshold = 0.25 if analysis_type == "comprehensive" else 0.3
        filtered_results = [(doc, score) for doc, score in unique_results if score > min_threshold]
        # Prioritize case law documents
        case_law_results = prioritize_case_law_documents(filtered_results)
        logger.info(f"[LEGAL_RETRIEVAL] Found {len(unique_results)} unique results, {len(case_law_results)} prioritized")
        final_results = case_law_results if case_law_results else unique_results[:k]
        docs, scores = zip(*final_results) if final_results else ([], [])
        return list(docs), {
            "query_used": legal_expanded_query,
            "scores": list(scores),
            "threshold_used": min_threshold,
            "strategies_used": ["legal_expansion", "case_law", "jurisdiction", "principles"],
            "analysis_type": analysis_type,
            "jurisdiction": jurisdiction
        }
    except Exception as e:
        logger.error(f"[LEGAL_RETRIEVAL] Search failed: {e}")
        return [], {"error": str(e)}

def expand_legal_query_advanced(query: str, analysis_type: str, case_types: List[str] = None) -> str:
    """Advanced legal query expansion"""
    # Base legal expansions
    legal_expansions = {
        "precedent": "precedent stare decisis binding authority case law judicial decision",
        "holding": "holding rule decision ruling court found established",
        "reasoning": "reasoning rationale analysis basis justification logic",
        "constitutional": "constitutional amendment due process equal protection fundamental rights",
        "statute": "statute law code regulation rule provision section",
        "case": "case decision opinion judgment ruling court",
        "court": "court tribunal judge judicial hearing proceeding",
        "appeal": "appeal appellate review reversal affirmation",
        "jurisdiction": "jurisdiction authority power venue competence territorial",
        "liability": "liability responsibility obligation duty accountable damages"
    }
    # Analysis-type specific expansions
    if analysis_type == "case_law":
        legal_expansions.update({
            "opinion": "opinion decision judgment holding majority dissent concurring",
            "citation": "citation reference authority support precedent case law"
        })
    elif analysis_type == "precedent":
        legal_expansions.update({
            "binding": "binding mandatory authoritative controlling precedential",
            "persuasive": "persuasive influential guidance advisory recommendatory"
        })
    # Case type specific expansions
    if case_types:
        for case_type in case_types:
            if case_type == "criminal":
                legal_expansions["criminal"] = "criminal prosecution defendant guilt sentence conviction"
            elif case_type == "civil":
                legal_expansions["civil"] = "civil plaintiff defendant damages liability tort"
            elif case_type == "constitutional":
                legal_expansions["constitutional"] = "constitutional amendment rights liberty due process"
    expanded_terms = []
    query_lower = query.lower()
    for term, expansion in legal_expansions.items():
        if term in query_lower:
            expanded_terms.extend(expansion.split())
    if expanded_terms:
        return f"{query} {' '.join(set(expanded_terms))}"
    return query

def generate_case_law_queries(query: str) -> List[str]:
    """Generate case law specific queries"""
    case_queries = []
    # Add case law specific terms
    case_terms = ["case law", "precedent", "judicial decision", "court ruling", "legal authority"]
    for term in case_terms:
        case_queries.append(f"{query} {term}")
    # Add jurisdiction variations
    jurisdictions = ["federal", "supreme court", "appellate", "circuit court"]
    for jurisdiction in jurisdictions:
        case_queries.append(f"{query} {jurisdiction}")
    return case_queries[:5]  # Limit to avoid too many queries

def extract_legal_principle_queries(query: str) -> List[str]:
    """Extract queries focused on legal principles"""
    principle_queries = []
    # Transform query to focus on principles
    if "what" in query.lower():
        principle_queries.append(query.replace("what", "legal principle"))
    if "how" in query.lower():
        principle_queries.append(query.replace("how", "legal standard"))
    # Add principle-focused variations
    principle_queries.append(f"legal doctrine {query}")
    principle_queries.append(f"legal standard {query}")
    return principle_queries

def prioritize_case_law_documents(results_with_scores: List[Tuple]) -> List[Tuple]:
    """Prioritize documents that appear to contain case law"""
    case_law_indicators = [
        "v.", "vs.", "case", "court", "opinion", "judgment", "ruling", 
        "precedent", "citation", "holding", "decided", "appeal"
    ]
    prioritized_results = []
    regular_results = []
    for doc, score in results_with_scores:
        content_lower = doc.page_content.lower()
        case_law_score = sum(1 for indicator in case_law_indicators if indicator in content_lower)
        if case_law_score >= 3:  # Has multiple case law indicators
            # Boost the score for case law documents
            boosted_score = min(1.0, score * 1.2)
            prioritized_results.append((doc, boosted_score))
        else:
            regular_results.append((doc, score))
    # Combine prioritized first, then regular
    return prioritized_results + regular_results

# --- Enhanced Legal Prompt Templates ---
def get_legal_analysis_prompt_template(analysis_type: str) -> str:
    """Get specialized prompt template for legal analysis"""
    if analysis_type == "case_law":
        return """You are an expert legal researcher specializing in case law analysis. Your responses must be STRICTLY based on the provided legal documents and case law.
CRITICAL REQUIREMENTS:
1. **ONLY use information from the provided context below**
2. **Extract and cite ALL relevant case names, citations, and holdings**
3. **Identify legal principles and their supporting precedents**
4. **Analyze precedential value and binding authority**
5. **Maintain precise legal terminology and citation format**
CASE LAW ANALYSIS FOCUS:
- Identify key cases and their holdings
- Extract legal principles and doctrines
- Analyze precedential relationships
- Note jurisdictional considerations
- Highlight conflicting or evolving precedents
RESPONSE STYLE: {response_style}
CONVERSATION HISTORY:
{conversation_history}
LEGAL DOCUMENTS AND CASE LAW (USE ONLY THIS INFORMATION):
{context}
USER QUESTION:
{questions}
RESPONSE FORMAT:
1. **Direct Answer**: Provide the main legal answer
2. **Key Cases**: List relevant cases with citations and brief holdings
3. **Legal Principles**: Identify applicable legal doctrines
4. **Precedential Analysis**: Discuss binding vs. persuasive authority
5. **Citations**: [document_name.pdf] for each source used
If the context doesn't contain sufficient case law information, state: "Based on the available legal documents, I can provide limited case law analysis..."
RESPONSE:"""
    elif analysis_type == "precedent":
        return """You are an expert in legal precedent analysis. Focus on precedential relationships, binding authority, and the evolution of legal doctrine.
CRITICAL REQUIREMENTS:
1. **ONLY use information from the provided context below**
2. **Analyze precedential hierarchy and binding authority**
3. **Identify evolution of legal doctrine over time**
4. **Note circuit splits or conflicting precedents**
5. **Distinguish binding from persuasive authority**
PRECEDENT ANALYSIS FOCUS:
- Hierarchy of courts and binding authority
- Evolution of legal doctrine
- Circuit splits and conflicting precedents
- Overruling and distinguishing of cases
- Trend analysis in judicial decisions
CONVERSATION HISTORY:
{conversation_history}
LEGAL DOCUMENTS AND PRECEDENTS (USE ONLY THIS INFORMATION):
{context}
USER QUESTION:
{questions}
RESPONSE FORMAT:
1. **Binding Precedents**: Cases that are controlling authority
2. **Persuasive Precedents**: Influential but not binding cases
3. **Conflicting Authority**: Cases with different holdings
4. **Doctrinal Evolution**: How the law has developed
5. **Current State**: Present legal landscape
RESPONSE:"""
    else:  # comprehensive
        return """You are a comprehensive legal research expert with deep knowledge of case law, statutes, regulations, and legal doctrine.
CRITICAL REQUIREMENTS:
1. **ONLY use information from the provided context below**
2. **Provide thorough legal analysis with case law support**
3. **Integrate statutory, regulatory, and common law sources**
4. **Analyze practical implications and applications**
5. **Maintain highest standards of legal accuracy**
COMPREHENSIVE LEGAL ANALYSIS:
- Primary sources (cases, statutes, regulations)
- Secondary authority and commentary
- Practical applications and implications
- Risk analysis and recommendations
- Jurisdictional variations
RESPONSE STYLE: {response_style}
CONVERSATION HISTORY:
{conversation_history}
COMPREHENSIVE LEGAL CONTEXT (USE ONLY THIS INFORMATION):
{context}
USER QUESTION:
{questions}
RESPONSE FORMAT:
1. **Executive Summary**: Key legal conclusions
2. **Primary Authority**: Controlling cases and statutes
3. **Legal Analysis**: Detailed examination of issues
4. **Practical Implications**: Real-world applications
5. **Recommendations**: Suggested approaches or considerations
6. **Supporting Citations**: [document_name.pdf] for all sources
Provide comprehensive analysis while staying strictly within the provided context.
RESPONSE:"""

# --- Enhanced Legal Processing Function ---
def process_legal_query_enhanced(question: str, session_id: str, response_style: str = "balanced",
                               analysis_type: str = "comprehensive", jurisdiction: str = "federal",
                               case_types: List[str] = None, time_period: str = "all") -> LegalAnalysisResponse:
    """
    Enhanced legal query processing with comprehensive case law analysis
    """
    try:
        # Initialize case law analyzer
        case_analyzer = CaseLawAnalyzer()
        # Load Database
        db = load_database()
        # Parse Question
        questions = parse_multiple_questions(question)
        combined_query = " ".join(questions)
        # Get Conversation History
        conversation_history_context = get_conversation_context(session_id, max_messages=10)
        # Enhanced legal retrieval
        results, search_result = legal_enhanced_retrieval(
            db, combined_query, conversation_history_context, 
            analysis_type, jurisdiction, case_types, k=15
        )
        if not results:
            logger.warning("No relevant legal documents found")
            no_info_response = f"I couldn't find any relevant legal documents to provide {analysis_type} analysis for your question about {jurisdiction} law."
            add_to_conversation(session_id, "assistant", no_info_response)
            return LegalAnalysisResponse(
                response=no_info_response,
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.0,
                expand_available=False
            )
        # Create Enhanced Legal Context
        context_text, source_info = create_enhanced_legal_context(results, search_result, questions)
        # Perform Case Law Analysis
        case_citations = case_analyzer.extract_case_citations(context_text)
        precedent_analysis = case_analyzer.analyze_precedents(case_citations, combined_query)
        legal_principles = case_analyzer.extract_legal_principles(context_text, case_citations)
        # Calculate enhanced confidence score
        confidence_score = calculate_legal_confidence_score(
            results, search_result, len(context_text), case_citations, legal_principles
        )
        # Get specialized legal prompt template
        prompt_template = get_legal_analysis_prompt_template(analysis_type)
        # Format questions
        formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions)) if len(questions) > 1 else questions[0]
        # Create enhanced prompt with legal analysis context
        enhanced_context = create_legal_analysis_context(
            context_text, case_citations, legal_principles, precedent_analysis
        )
        formatted_prompt = prompt_template.format(
            response_style=response_style.capitalize(),
            conversation_history=conversation_history_context if conversation_history_context else "No previous legal consultation.",
            context=enhanced_context,
            questions=formatted_questions
        )
        # Call LLM with enhanced prompt
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        if not api_key:
            error_msg = "OPENAI_API_KEY environment variable not set."
            return LegalAnalysisResponse(
                response=None,
                error=f"Configuration Error: {error_msg}",
                context_found=True,
                sources=source_info,
                session_id=session_id,
                confidence_score=0.0,
                expand_available=False
            )
        raw_response = call_openrouter_api(formatted_prompt, api_key, api_base)
        # Format response with legal analysis
        formatted_response, expand_available = format_legal_response(
            raw_response, source_info, response_style, case_citations, legal_principles
        )
        # Add comprehensive source information
        if source_info:
            formatted_response += f"\n**LEGAL SOURCES** (Confidence: {confidence_score:.1%}, Analysis: {analysis_type.title()}):\n"
            for source in source_info:
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                formatted_response += f"- {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
        # Add case citations if found
        if case_citations:
            formatted_response += f"\n**KEY CASE CITATIONS** ({len(case_citations)} cases identified):\n"
            for citation in case_citations[:10]:  # Limit to top 10
                formatted_response += f"- {citation.case_name}"
                if citation.citation:
                    formatted_response += f", {citation.citation}"
                formatted_response += f" ({citation.court})\n"
        # Update conversation
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", formatted_response, source_info)
        # Prepare enhanced response data
        case_citations_dict = [
            {
                "case_name": citation.case_name,
                "citation": citation.citation,
                "year": citation.year,
                "court": citation.court,
                "jurisdiction_level": citation.jurisdiction_level.value,
                "case_type": citation.case_type.value,
                "relevance_score": citation.relevance_score
            }
            for citation in case_citations
        ]
        legal_principles_dict = [
            {
                "principle": principle.principle,
                "supporting_cases": [case.case_name for case in principle.supporting_cases],
                "jurisdiction": principle.jurisdiction,
                "confidence": principle.confidence,
                "legal_area": principle.legal_area
            }
            for principle in legal_principles
        ]
        precedent_analysis_dict = {
            "binding_precedents": [
                {"case_name": case.case_name, "citation": case.citation, "court": case.court}
                for case in precedent_analysis.binding_precedents
            ],
            "persuasive_precedents": [
                {"case_name": case.case_name, "citation": case.citation, "court": case.court}
                for case in precedent_analysis.persuasive_precedents
            ],
            "trend_analysis": precedent_analysis.trend_analysis,
            "jurisdiction_hierarchy": {
                level: [{"case_name": case.case_name, "year": case.year} for case in cases]
                for level, cases in precedent_analysis.jurisdiction_hierarchy.items()
            }
        }
        # Generate recommended cases for further research
        recommended_cases = generate_recommended_cases(case_citations, legal_principles, combined_query)
        return LegalAnalysisResponse(
            response=formatted_response,
            error=None,
            context_found=True,
            sources=source_info,
            session_id=session_id,
            confidence_score=confidence_score,
            expand_available=expand_available,
            case_citations=case_citations_dict,
            legal_principles=legal_principles_dict,
            precedent_analysis=precedent_analysis_dict,
            jurisdiction_analysis={
                "primary_jurisdiction": jurisdiction,
                "case_distribution": analyze_jurisdiction_distribution(case_citations),
                "binding_authority_strength": calculate_binding_authority_strength(precedent_analysis)
            },
            case_law_trends=precedent_analysis.trend_analysis,
            recommended_cases=recommended_cases
        )
    except Exception as e:
        logger.error(f"Legal query processing failed: {e}", exc_info=True)
        error_msg = f"Failed to process legal analysis request: {str(e)}"
        return LegalAnalysisResponse(
            response=None,
            error=error_msg,
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            expand_available=False
        )

def create_enhanced_legal_context(results: List, search_result: Dict, questions: List[str]) -> Tuple[str, List[Dict]]:
    """Create enhanced legal context with case law prioritization"""
    if not results:
        return "", []
    context_parts = []
    source_info = []
    seen_sources = set()
    # Higher threshold for legal documents
    MIN_RELEVANCE_FOR_LEGAL_CONTEXT = 0.3
    for i, (doc, score) in enumerate(zip(results, search_result.get("scores", [0.0]*len(results)))):
        if score < MIN_RELEVANCE_FOR_LEGAL_CONTEXT:
            continue
        content = doc.page_content.strip()
        if not content:
            continue
        source_path = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', None)
        source_id = (source_path, page)
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        display_source = os.path.basename(source_path)
        page_info = f" (Page {page})" if page is not None else ""
        # Enhanced content processing for legal documents
        processed_content = enhance_legal_content(content)
        context_part = f"[{display_source}{page_info}] (Legal Relevance: {score:.2f}):\n{processed_content}"
        context_parts.append(context_part)
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source_path,
            'document_type': identify_legal_document_type(content)
        })
    context_text = "\n---LEGAL DOCUMENT SEPARATOR---\n".join(context_parts)
    return context_text, source_info

def enhance_legal_content(content: str) -> str:
    """Enhance legal content by highlighting key legal elements"""
    # Highlight case names
    content = re.sub(r'([A-Z][a-zA-Z\s&]+)\s+v\.?\s+([A-Z][a-zA-Z\s&]+)', 
                    r'**CASE: \1 v. \2**', content)
    # Highlight legal holdings
    content = re.sub(r'(held|holding|ruled|decided|found|concluded)\s+that', 
                    r'**HOLDING:** \1 that', content, flags=re.IGNORECASE)
    # Highlight citations
    content = re.sub(r'(\d+)\s+([A-Z][a-zA-Z\.]*)\s+(\d+)\s*\((\d{4})\)', 
                    r'**CITATION:** \1 \2 \3 (\4)', content)
    return content

def identify_legal_document_type(content: str) -> str:
    """Identify the type of legal document based on content"""
    content_lower = content.lower()
    if any(term in content_lower for term in ['opinion', 'judgment', 'decided', 'appeal']):
        return "case_law"
    elif any(term in content_lower for term in ['statute', 'code', 'section', 'chapter']):
        return "statute"
    elif any(term in content_lower for term in ['regulation', 'rule', 'cfr', 'federal register']):
        return "regulation"
    elif any(term in content_lower for term in ['constitution', 'amendment', 'article']):
        return "constitutional"
    else:
        return "general_legal"

def create_legal_analysis_context(context_text: str, case_citations: List[CaseCitation], 
                                legal_principles: List[LegalPrinciple], 
                                precedent_analysis: PrecedentAnalysis) -> str:
    """Create enhanced context with legal analysis annotations"""
    enhanced_context = f"LEGAL DOCUMENT CONTEXT:\n{context_text}\n"
    if case_citations:
        enhanced_context += "IDENTIFIED CASE CITATIONS:\n"
        for citation in case_citations[:5]:  # Top 5 cases
            enhanced_context += f"- {citation.case_name}"
            if citation.citation:
                enhanced_context += f" [{citation.citation}]"
            enhanced_context += f" ({citation.court}, {citation.year})\n"
        enhanced_context += "\n"
    if legal_principles:
        enhanced_context += "EXTRACTED LEGAL PRINCIPLES:\n"
        for principle in legal_principles[:3]:  # Top 3 principles
            enhanced_context += f"- {principle.legal_area}: {principle.principle[:200]}...\n"
        enhanced_context += "\n"
    if precedent_analysis.binding_precedents:
        enhanced_context += "BINDING PRECEDENTS IDENTIFIED:\n"
        for precedent in precedent_analysis.binding_precedents[:3]:
            enhanced_context += f"- {precedent.case_name} ({precedent.court})\n"
        enhanced_context += "\n"
    return enhanced_context

def calculate_legal_confidence_score(results: List, search_result: Dict, response_length: int,
                                   case_citations: List[CaseCitation], 
                                   legal_principles: List[LegalPrinciple]) -> float:
    """Calculate confidence score with legal-specific factors"""
    if not results:
        return 0.1
    scores = search_result.get("scores", [])
    if not scores:
        return 0.2
    # Base factors
    avg_relevance = np.mean(scores)
    doc_factor = min(1.0, len(results) / 8.0)  # Legal research benefits from more sources
    # Legal-specific factors
    case_citation_factor = min(1.0, len(case_citations) / 5.0)  # Bonus for case citations
    legal_principle_factor = min(1.0, len(legal_principles) / 3.0)  # Bonus for legal principles
    # Document type diversity bonus
    doc_types = set()
    for doc, _ in zip(results, scores):
        content = doc.page_content
        doc_types.add(identify_legal_document_type(content))
    diversity_factor = min(1.0, len(doc_types) / 3.0)  # Bonus for diverse source types
    # Weighted combination with legal emphasis
    confidence = (
        avg_relevance * 0.3 +
        doc_factor * 0.2 +
        case_citation_factor * 0.2 +
        legal_principle_factor * 0.15 +
        diversity_factor * 0.15
    )
    return min(1.0, max(0.0, confidence))

def format_legal_response(content: str, sources: List[Dict], style: str, 
                         case_citations: List[CaseCitation], 
                         legal_principles: List[LegalPrinciple]) -> Tuple[str, bool]:
    """Format response with legal-specific enhancements"""
    if style == "concise":
        # Create concise legal response with key points
        concise_response = create_concise_legal_response(content, case_citations, legal_principles)
        return concise_response, True
    elif style == "detailed":
        # Return comprehensive legal analysis
        detailed_response = create_detailed_legal_response(content, case_citations, legal_principles)
        return detailed_response, False
    else:  # balanced
        # Provide structured legal response
        balanced_response = create_balanced_legal_response(content, case_citations, legal_principles)
        return balanced_response, True

def create_concise_legal_response(content: str, case_citations: List[CaseCitation], 
                                legal_principles: List[LegalPrinciple]) -> str:
    """Create concise legal response highlighting key elements"""
    # Extract key legal points
    key_points = extract_key_legal_points(content)
    concise = f"""**LEGAL SUMMARY:**
{'\n'.join(f"• {point}" for point in key_points[:5])}
**KEY AUTHORITY:** {len(case_citations)} case(s), {len(legal_principles)} principle(s) identified
💡 *Need comprehensive analysis? Ask for detailed legal research.*
🔍 *Want specific precedent analysis? Request precedent breakdown.*"""
    return concise

def create_balanced_legal_response(content: str, case_citations: List[CaseCitation], 
                                 legal_principles: List[LegalPrinciple]) -> str:
    """Create balanced legal response with clear structure"""
    if len(content) > 1200:
        preview = content[:1000] + "..."
        balanced = f"""{preview}
🔍 **LEGAL RESEARCH DEPTH:** {len(case_citations)} cases analyzed, {len(legal_principles)} principles identified
🔍 **Want comprehensive analysis?** Ask for detailed legal opinion with full case law review.
🔍 **Need precedent analysis?** Request binding vs. persuasive authority breakdown.
🔍 **Specific jurisdiction focus?** Ask about federal vs. state law applications."""
    else:
        balanced = content
    return balanced

def create_detailed_legal_response(content: str, case_citations: List[CaseCitation], 
                                 legal_principles: List[LegalPrinciple]) -> str:
    """Return comprehensive legal analysis"""
    detailed = f"""{content}
**COMPREHENSIVE LEGAL ANALYSIS COMPLETE**
- Case Citations Analyzed: {len(case_citations)}
- Legal Principles Identified: {len(legal_principles)}
- Precedential Authority Assessed: {len([c for c in case_citations if c.jurisdiction_level in [JurisdictionLevel.SUPREME_COURT, JurisdictionLevel.APPELLATE]])} binding sources"""
    return detailed

def extract_key_legal_points(content: str) -> List[str]:
    """Extract key legal points from content"""
    sentences = content.split('.')
    key_points = []
    legal_keywords = ['held', 'ruled', 'found', 'established', 'precedent', 'authority', 'binding']
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in legal_keywords):
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                key_points.append(clean_sentence)
    return key_points[:10]  # Limit to top 10

def generate_recommended_cases(case_citations: List[CaseCitation], 
                             legal_principles: List[LegalPrinciple], 
                             query: str) -> List[Dict]:
    """Generate recommended cases for further research"""
    recommended = []
    # Recommend high-authority cases
    high_authority_cases = [
        case for case in case_citations 
        if case.jurisdiction_level in [JurisdictionLevel.SUPREME_COURT, JurisdictionLevel.APPELLATE]
        and case.relevance_score > 0.7
    ]
    for case in high_authority_cases[:5]:
        recommended.append({
            "case_name": case.case_name,
            "citation": case.citation,
            "reason": "High precedential value",
            "court": case.court,
            "relevance_score": case.relevance_score
        })
    # Recommend cases from different time periods for trend analysis
    if len(case_citations) > 3:
        recent_cases = [c for c in case_citations if c.year >= 2010]
        historical_cases = [c for c in case_citations if c.year < 2010 and c.year > 0]
        if recent_cases and historical_cases:
            recommended.append({
                "suggestion": "Comparative Analysis",
                "recent_count": len(recent_cases),
                "historical_count": len(historical_cases),
                "reason": "Analyze doctrinal evolution over time"
            })
    return recommended

def analyze_jurisdiction_distribution(case_citations: List[CaseCitation]) -> Dict[str, int]:
    """Analyze distribution of cases across jurisdictions"""
    distribution = {}
    for case in case_citations:
        level = case.jurisdiction_level.value
        distribution[level] = distribution.get(level, 0) + 1
    return distribution

def calculate_binding_authority_strength(precedent_analysis: PrecedentAnalysis) -> float:
    """Calculate the strength of binding authority"""
    if not precedent_analysis.binding_precedents:
        return 0.0
    total_precedents = (
        len(precedent_analysis.binding_precedents) + 
        len(precedent_analysis.persuasive_precedents)
    )
    if total_precedents == 0:
        return 0.0
    binding_ratio = len(precedent_analysis.binding_precedents) / total_precedents
    # Adjust for Supreme Court cases (stronger authority)
    supreme_court_cases = sum(
        1 for case in precedent_analysis.binding_precedents 
        if case.jurisdiction_level == JurisdictionLevel.SUPREME_COURT
    )
    supreme_court_bonus = min(0.3, supreme_court_cases * 0.1)
    return min(1.0, binding_ratio + supreme_court_bonus)

# --- Updated API Endpoint ---
@app.post("/legal-analysis", response_model=LegalAnalysisResponse)
async def legal_analysis_endpoint(query: LegalQuery):
    """
    Comprehensive legal analysis endpoint with case law capabilities
    """
    cleanup_expired_conversations()
    session_id = query.session_id or str(uuid.uuid4())
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    user_question = query.question.strip()
    if not user_question:
        return LegalAnalysisResponse(
            response=None,
            error="Legal question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            expand_available=False
        )
    # Process with comprehensive legal analysis
    response = process_legal_query_enhanced(
        user_question, 
        session_id, 
        query.response_style, 
        query.analysis_type,
        query.jurisdiction,
        query.case_types,
        query.time_period
    )
    return response

# --- NEW /ask endpoint that delegates to /legal-analysis ---
@app.post("/ask", response_model=LegalAnalysisResponse)
async def ask_endpoint(query: AskQuery):
    """
    Simplified ask endpoint that delegates to the legal analysis endpoint
    """
    # Map the AskQuery to LegalQuery with default values
    legal_query = LegalQuery(
        question=query.question,
        session_id=query.session_id,
        response_style="balanced",
        analysis_type="comprehensive",
        jurisdiction="federal", # Default, user can change if needed via /legal-analysis
        case_types=None,
        time_period="all"
    )
    
    # Reuse the existing legal analysis endpoint logic
    return await legal_analysis_endpoint(legal_query)
# --- END OF NEW ENDPOINT ---

# [Include all utility functions from previous versions...]
# Additional utility functions needed
def cleanup_expired_conversations():
    """Remove conversations older than 2 hours for legal consultations"""
    now = datetime.utcnow()
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if now - data['last_accessed'] > timedelta(hours=2)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]

# --- FIXED: Updated load_database function to match the working version ---
def load_database():
    """Load the Chroma database"""
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # --- FIXED: Updated collection_name to match the working version ---
        db = Chroma(
            collection_name="default", # Changed from "legal_documents"
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=CHROMA_CLIENT_SETTINGS # Uses the corrected settings
        )
        # --- END OF FIX ---
        return db
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        raise
# --- END OF FIXED FUNCTION ---

def get_conversation_context(session_id: str, max_messages: int = 10) -> str:
    """Get conversation context for legal consultations"""
    if session_id not in conversations:
        return ""
    messages = conversations[session_id]['messages'][-max_messages:]
    context_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content'][:600] + "..." if len(msg['content']) > 600 else msg['content']
        context_parts.append(f"{role}: {content}")
    return "\n".join(context_parts) if context_parts else ""

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """Add message to legal consultation history"""
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'created_at': datetime.utcnow(),
            'last_accessed': datetime.utcnow()
        }
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow().isoformat(),
        'sources': sources or []
    }
    conversations[session_id]['messages'].append(message)
    conversations[session_id]['last_accessed'] = datetime.utcnow()
    if len(conversations[session_id]['messages']) > 50:  # Keep more history for legal consultations
        conversations[session_id]['messages'] = conversations[session_id]['messages'][-50:]

def parse_multiple_questions(query_text: str) -> List[str]:
    """Parse multiple legal questions from input"""
    questions = []
    query_text = query_text.strip()
    if '?' in query_text and not query_text.endswith('?'):
        parts = query_text.split('?')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part + '?')
    else:
        final_question = query_text
        if not final_question.endswith('?') and '?' not in final_question:
            final_question += '?'
        questions = [final_question]
    return questions

def remove_duplicate_documents(results_with_scores: List[Tuple]) -> List[Tuple]:
    """Remove duplicate documents based on content similarity"""
    if not results_with_scores:
        return []
    unique_results = []
    seen_content = set()
    for doc, score in results_with_scores:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append((doc, score))
    return sorted(unique_results, key=lambda x: x[1], reverse=True)

def call_openrouter_api(prompt: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1") -> str:
    """Call OpenRouter API with legal analysis optimizations"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Legal Analysis Assistant"
    }
    # Prioritize models good for legal analysis
    models_to_try = [
        "anthropic/claude-3-sonnet",  # Good for legal reasoning
        "openai/gpt-4-turbo-preview",  # Strong analytical capabilities
        "deepseek/deepseek-chat-v3-0324:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free"
    ]
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # Lower temperature for legal precision
                "max_tokens": 4000,  # More tokens for comprehensive legal analysis
                "top_p": 0.9
            }
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=90)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    if content and content.strip():
                        return content.strip()
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            continue
    return "I apologize, but I'm experiencing technical difficulties with the legal analysis system. Please try again."

# Additional endpoints for legal-specific functionality
@app.get("/legal-health")
def legal_health_check():
    """Legal system health check"""
    return {
        "status": "healthy",
        "version": "3.0.0-legal",
        "features": [
            "case_law_analysis", 
            "precedent_research", 
            "legal_principle_extraction",
            "jurisdiction_analysis",
            "citation_parsing"
        ],
        "supported_analysis_types": ["case_law", "precedent", "comprehensive"],
        "supported_jurisdictions": ["federal", "state", "specific_state"],
        "active_legal_consultations": len(conversations)
    }

@app.get("/case-citations/{session_id}")
def get_case_citations(session_id: str):
    """Get extracted case citations from a legal consultation"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Legal consultation not found")
    # This would be enhanced to return actual extracted citations
    return {"message": "Case citation extraction endpoint - implementation depends on stored analysis results"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Enhanced Legal AI Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
