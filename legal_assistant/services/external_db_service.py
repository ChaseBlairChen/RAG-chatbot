# legal_assistant/services/external_db_service.py - COMPLETE ENHANCED VERSION
"""Enhanced external database service with comprehensive state law API integration"""
import asyncio
import aiohttp
import requests
import logging
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import time
import re
from urllib.parse import urlencode, quote

# Import the new state law APIs
from .state_law_apis import (
    CornellLegalAPI, OpenStatesAPI, JustiaLegalAPI, GoogleScholarLegalAPI,
    StateLawSearchService, state_law_service
)

# Import comprehensive APIs
from .comprehensive_legal_apis import (
    comprehensive_legal_hub,
    search_comprehensive_legal_databases
)

logger = logging.getLogger(__name__)

# Rate limiting helper (enhanced)
class RateLimiter:
    def __init__(self, requests_per_hour: int = 100, burst: int = 10):
        self.requests_per_hour = requests_per_hour
        self.burst = burst
        self.requests = []
        self.last_request = 0
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        now = time.time()
        
        # Remove requests older than 1 hour
        cutoff = now - 3600
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        # Check burst limit (last minute)
        recent_requests = [req_time for req_time in self.requests if req_time > now - 60]
        if len(recent_requests) >= self.burst:
            sleep_time = 60 / self.burst
            time.sleep(sleep_time)
        
        # Check hourly limit
        if len(self.requests) >= self.requests_per_hour:
            sleep_time = 3600 / self.requests_per_hour
            time.sleep(sleep_time)
        
        self.requests.append(now)

@dataclass
class SearchResult:
    title: str
    content: str
    source: str
    url: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = None
    relevance_score: float = 0.0

# Enhanced LegalDatabaseInterface with state law support
class LegalDatabaseInterface(ABC):
    """Abstract base class for legal database interfaces"""
    
    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with the legal database"""
        pass
    
    @abstractmethod
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search the legal database"""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Dict:
        """Retrieve a specific document"""
        pass

# State Law Database Interfaces
class StateLawInterface(LegalDatabaseInterface):
    """Interface for state law databases"""
    
    def __init__(self, state_service: StateLawSearchService):
        self.state_service = state_service
        self.authenticated = True  # No auth needed for free services
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search state law databases"""
        state = filters.get('state', 'Washington') if filters else 'Washington'
        search_type = filters.get('search_type', 'comprehensive') if filters else 'comprehensive'
        
        try:
            results = self.state_service.search_state_specific(state, query, search_type)
            
            # Flatten and format results
            formatted_results = []
            for source_type, sources in results.get('sources', {}).items():
                if isinstance(sources, list):
                    for source in sources:
                        formatted_result = {
                            'title': source.get('title', ''),
                            'url': source.get('url', ''),
                            'description': source.get('description', ''),
                            'source_database': f"state_law_{source_type}",
                            'state': state,
                            'category': source_type,
                            'access': source.get('access', 'Free'),
                            'date': source.get('updated_at', '')
                        }
                        formatted_results.append(formatted_result)
            
            logger.info(f"State law search returned {len(formatted_results)} results for {state}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"State law search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from state law database"""
        return {'id': document_id, 'source': 'state_law', 'note': 'Full text available at source URL'}

class HarvardCaselawInterface(LegalDatabaseInterface):
    """Harvard Caselaw Access Project - Enhanced with state filtering"""
    
    def __init__(self):
        self.api_endpoint = "https://api.case.law/v1"
        self.authenticated = True
        self.rate_limiter = RateLimiter(500, 20)
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced search with state filtering"""
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {
                "search": query,
                "page_size": 10
            }
            
            # Enhanced filtering
            if filters:
                if "state" in filters:
                    # Map state name to jurisdiction
                    state_jurisdictions = {
                        'california': 'cal',
                        'new_york': 'ny',
                        'texas': 'tex',
                        'florida': 'fla',
                        'washington': 'wash',
                        'illinois': 'ill',
                        'pennsylvania': 'pa',
                        'ohio': 'ohio',
                        'georgia': 'ga',
                        'north_carolina': 'nc',
                        'michigan': 'mich',
                        'new_jersey': 'nj',
                        'virginia': 'va',
                        'massachusetts': 'mass'
                    }
                    state_key = filters["state"].lower().replace(' ', '_')
                    if state_key in state_jurisdictions:
                        params["jurisdiction"] = state_jurisdictions[state_key]
                
                if "court_level" in filters:
                    court_mappings = {
                        'supreme': 'supreme',
                        'appellate': 'appellate',
                        'trial': 'trial'
                    }
                    if filters["court_level"] in court_mappings:
                        params["court"] = court_mappings[filters["court_level"]]
                
                if "after_date" in filters:
                    params["decision_date_min"] = filters["after_date"]
                if "before_date" in filters:
                    params["decision_date_max"] = filters["before_date"]
                
                # Enhanced search by case type
                if "case_type" in filters:
                    case_type = filters["case_type"].lower()
                    if case_type in ['criminal', 'civil', 'constitutional']:
                        params["search"] = f"{query} {case_type}"
            
            response = requests.get(
                f"{self.api_endpoint}/cases/",
                params=params,
                timeout=15,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                data = response.json()
                results = []
                for case in data.get('results', []):
                    # Enhanced result formatting
                    result = {
                        'title': case.get('name', ''),
                        'court': case.get('court', {}).get('name', '') if case.get('court') else '',
                        'date': case.get('decision_date', ''),
                        'citation': self._format_multiple_citations(case.get('citations', [])),
                        'url': case.get('frontend_url', ''),
                        'preview': self._extract_better_preview(case),
                        'source_database': 'harvard_caselaw',
                        'id': case.get('id', ''),
                        'jurisdiction': case.get('jurisdiction', {}).get('name', '') if case.get('jurisdiction') else '',
                        'court_level': self._determine_court_level(case.get('court', {})),
                        'case_type': self._classify_case_type(case.get('name', '')),
                        'relevance_indicators': self._extract_relevance_indicators(case, query)
                    }
                    results.append(result)
                
                logger.info(f"Harvard Caselaw found {len(results)} results")
                return results
            else:
                logger.error(f"Harvard search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Harvard Caselaw: {e}")
            return []
    
    def _format_multiple_citations(self, citations: List[Dict]) -> str:
        """Format multiple citations properly"""
        if not citations:
            return ""
        
        formatted_citations = []
        for citation in citations[:3]:  # Limit to top 3 citations
            cite_text = citation.get('cite', '')
            if cite_text:
                formatted_citations.append(cite_text)
        
        return '; '.join(formatted_citations)
    
    def _extract_better_preview(self, case: Dict) -> str:
        """Extract better preview text from case"""
        # Try multiple preview sources
        preview_sources = [
            case.get('preview', []),
            case.get('casebody', {}).get('text', ''),
            case.get('summary', '')
        ]
        
        for source in preview_sources:
            if isinstance(source, list) and source:
                return source[0].get('text', '')[:300] + '...'
            elif isinstance(source, str) and source:
                return source[:300] + '...'
        
        return f"Case decided {case.get('decision_date', 'date unknown')} by {case.get('court', {}).get('name', 'unknown court')}"
    
    def _determine_court_level(self, court: Dict) -> str:
        """Determine court level from court data"""
        court_name = court.get('name', '').lower()
        
        if 'supreme' in court_name:
            return 'supreme'
        elif any(term in court_name for term in ['appeal', 'appellate']):
            return 'appellate'
        elif any(term in court_name for term in ['trial', 'district', 'superior']):
            return 'trial'
        else:
            return 'unknown'
    
    def _classify_case_type(self, case_name: str) -> str:
        """Classify case type from case name"""
        case_lower = case_name.lower()
        
        # Criminal case indicators
        if any(term in case_lower for term in ['state v.', 'people v.', 'commonwealth v.', 'united states v.']):
            return 'criminal'
        
        # Civil case indicators
        if any(term in case_lower for term in [' v. ', 'inc.', 'corp.', 'llc']):
            return 'civil'
        
        return 'unknown'
    
    def _extract_relevance_indicators(self, case: Dict, query: str) -> List[str]:
        """Extract indicators of why this case is relevant"""
        indicators = []
        
        # Check if query terms appear in case name
        query_terms = query.lower().split()
        case_name = case.get('name', '').lower()
        
        for term in query_terms:
            if term in case_name:
                indicators.append(f"Case name contains '{term}'")
        
        # Check court level relevance
        court_name = case.get('court', {}).get('name', '')
        if 'supreme' in court_name.lower():
            indicators.append("Supreme Court decision (binding precedent)")
        
        return indicators
    
    def get_document(self, document_id: str) -> Dict:
        """Get full case text with enhanced parsing"""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.api_endpoint}/cases/{document_id}/",
                params={"full_case": "true"},
                timeout=20,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                case_data = response.json()
                
                # Enhanced document parsing
                enhanced_doc = {
                    'id': document_id,
                    'case_name': case_data.get('name', ''),
                    'court': case_data.get('court', {}).get('name', ''),
                    'date': case_data.get('decision_date', ''),
                    'citations': case_data.get('citations', []),
                    'full_text': self._extract_full_text(case_data),
                    'headnotes': self._extract_headnotes(case_data),
                    'holding': self._extract_holding(case_data),
                    'source': 'harvard_caselaw',
                    'retrieved_at': datetime.now().isoformat()
                }
                
                return enhanced_doc
            else:
                logger.error(f"Failed to get Harvard case {document_id}: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting case from Harvard: {e}")
            return {}
    
    def _extract_full_text(self, case_data: Dict) -> str:
        """Extract full case text"""
        casebody = case_data.get('casebody', {})
        if casebody.get('status') == 'ok':
            return casebody.get('data', {}).get('text', '')
        return ''
    
    def _extract_headnotes(self, case_data: Dict) -> List[str]:
        """Extract headnotes from case"""
        # Implementation depends on Harvard API structure
        return []
    
    def _extract_holding(self, case_data: Dict) -> str:
        """Extract case holding"""
        # Would parse full text to find holding
        return ""

class CourtListenerInterface(LegalDatabaseInterface):
    """Enhanced CourtListener with state court support"""
    
    def __init__(self):
        self.api_endpoint = "https://www.courtlistener.com/api/rest/v4"
        self.api_key = os.environ.get("COURTLISTENER_API_KEY", "")
        self.authenticated = True
        self.rate_limiter = RateLimiter(100, 10)
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced search with state court filtering"""
        self.rate_limiter.wait_if_needed()
        
        try:
            headers = {'User-Agent': 'LegalAssistant/1.0'}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            # Enhanced search parameters
            params = {
                "q": query,
                "format": "json",
                "order_by": "score desc"
            }
            
            # Add enhanced filtering
            if filters:
                if "state" in filters:
                    # CourtListener court filtering by state
                    state_courts = self._get_state_courts(filters["state"])
                    if state_courts:
                        params["court"] = state_courts
                
                if "court_level" in filters:
                    court_level = filters["court_level"]
                    if court_level == "supreme":
                        params["court__jurisdiction"] = "S"  # State supreme courts
                    elif court_level == "appellate":
                        params["court__jurisdiction"] = "SA"  # State appellate courts
                    elif court_level == "federal":
                        params["court__jurisdiction"] = "F"  # Federal courts
                
                if "case_type" in filters:
                    case_type = filters["case_type"]
                    if case_type in ["criminal", "civil"]:
                        params["q"] = f"{query} {case_type}"
                
                if "after_date" in filters:
                    params["filed_after"] = filters["after_date"]
                if "before_date" in filters:
                    params["filed_before"] = filters["before_date"]
            
            # Search opinions
            response = requests.get(
                f"{self.api_endpoint}/search/",
                params={**params, "type": "o"},  # opinions
                headers=headers,
                timeout=15
            )
            
            results = []
            
            if response.ok:
                data = response.json()
                opinions = data.get('results', [])
                
                for opinion in opinions:
                    # Enhanced result formatting
                    result = {
                        'title': opinion.get('caseName', 'Unknown Case'),
                        'court': opinion.get('court', ''),
                        'date': opinion.get('dateFiled', ''),
                        'snippet': opinion.get('text', '')[:300] + '...' if opinion.get('text') else '',
                        'url': f"https://www.courtlistener.com{opinion.get('absolute_url', '')}" if opinion.get('absolute_url') else '',
                        'source_database': 'courtlistener',
                        'id': opinion.get('id', ''),
                        'type': 'opinion',
                        'docket_number': opinion.get('docketNumber', ''),
                        'citation': opinion.get('citation', ''),
                        'state': self._extract_state_from_court(opinion.get('court', '')),
                        'court_level': self._classify_court_level(opinion.get('court', '')),
                        'case_type': self._infer_case_type(opinion),
                        'relevance_score': opinion.get('score', 0.5)
                    }
                    results.append(result)
                
                logger.info(f"CourtListener found {len(results)} results")
                return results
            else:
                logger.error(f"CourtListener search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching CourtListener: {e}")
            return []
    
    def _get_state_courts(self, state: str) -> str:
        """Get CourtListener court IDs for a state"""
        # This would require querying CourtListener's court API
        # For now, return a search modifier
        return f"{state.lower()}"
    
    def _extract_state_from_court(self, court_name: str) -> str:
        """Extract state from court name"""
        # Pattern matching for state courts
        state_patterns = [
            r'California|Cal\.', r'New York|N\.Y\.', r'Texas|Tex\.', 
            r'Florida|Fla\.', r'Washington|Wash\.', r'Illinois|Ill\.',
            # Add more patterns
        ]
        
        for pattern in state_patterns:
            if re.search(pattern, court_name, re.IGNORECASE):
                return pattern.split('|')[0]
        
        return 'Unknown'
    
    def _classify_court_level(self, court_name: str) -> str:
        """Classify court level from name"""
        court_lower = court_name.lower()
        
        if 'supreme' in court_lower:
            return 'supreme'
        elif any(term in court_lower for term in ['appeal', 'appellate']):
            return 'appellate'
        elif any(term in court_lower for term in ['district', 'superior', 'trial']):
            return 'trial'
        elif 'circuit' in court_lower:
            return 'federal_appellate'
        else:
            return 'unknown'
    
    def _infer_case_type(self, opinion: Dict) -> str:
        """Infer case type from opinion data"""
        case_name = opinion.get('caseName', '').lower()
        
        # Criminal case patterns
        criminal_indicators = ['state v.', 'people v.', 'commonwealth v.', 'united states v.']
        if any(indicator in case_name for indicator in criminal_indicators):
            return 'criminal'
        
        # Civil case patterns
        if ' v. ' in case_name and not any(indicator in case_name for indicator in criminal_indicators):
            return 'civil'
        
        return 'unknown'
    
    def get_document(self, document_id: str) -> Dict:
        """Enhanced document retrieval"""
        self.rate_limiter.wait_if_needed()
        
        try:
            headers = {'User-Agent': 'LegalAssistant/1.0'}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            response = requests.get(
                f"{self.api_endpoint}/opinions/{document_id}/",
                headers=headers,
                timeout=15
            )
            
            if response.ok:
                return response.json()
            else:
                logger.error(f"Failed to get CourtListener opinion {document_id}: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting opinion from CourtListener: {e}")
            return {}

class FederalRegisterInterface(LegalDatabaseInterface):
    """Federal Register API - Free Government Regulations"""
    
    def __init__(self):
        self.api_endpoint = "https://www.federalregister.gov/api/v1"
        self.authenticated = True
        self.rate_limiter = RateLimiter(1000, 50)  # Government API, good limits
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Federal Register"""
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {
                "conditions[term]": query,
                "per_page": 20,
                "format": "json"
            }
            
            if filters:
                if "agency" in filters:
                    params["conditions[agency]"] = filters["agency"]
                if "after_date" in filters:
                    params["conditions[publication_date][gte]"] = filters["after_date"]
            
            response = requests.get(
                f"{self.api_endpoint}/documents.json",
                params=params,
                timeout=10,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                data = response.json()
                results = []
                
                for doc in data.get('results', []):
                    results.append({
                        'title': doc.get('title', ''),
                        'agency': ', '.join(doc.get('agencies', [])) if doc.get('agencies') else '',
                        'date': doc.get('publication_date', ''),
                        'type': doc.get('type', ''),
                        'preview': doc.get('abstract', '')[:200] + '...' if doc.get('abstract') else '',
                        'url': doc.get('html_url', ''),
                        'pdf_url': doc.get('pdf_url', ''),
                        'source_database': 'federal_register',
                        'id': doc.get('document_number', ''),
                        'citation': doc.get('citation', '')
                    })
                
                logger.info(f"Federal Register found {len(results)} results")
                return results
            else:
                logger.error(f"Federal Register search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Federal Register: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get full regulation text"""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.api_endpoint}/documents/{document_id}.json",
                timeout=10,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting Federal Register document: {e}")
            return {}

class CongressInterface(LegalDatabaseInterface):
    """Congress.gov API - Free Legislative Information"""
    
    def __init__(self):
        self.api_key = os.environ.get("CONGRESS_API_KEY", "7J5Bfj6i0F3tg4VZleZ4SyQmVbG0QyIM9tPMQA2M")
        self.api_endpoint = "https://api.congress.gov/v3"
        self.authenticated = bool(self.api_key)
        self.rate_limiter = RateLimiter(5000, 100)  # Government API
    
    def authenticate(self, credentials: Dict) -> bool:
        return self.authenticated
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Congress.gov"""
        if not self.authenticated:
            logger.warning("Congress API key required")
            return []
        
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {
                "q": query,
                "format": "json",
                "limit": 20
            }
            
            headers = {
                'X-API-Key': self.api_key,
                'User-Agent': 'LegalAssistant/1.0'
            }
            
            # Search bills
            response = requests.get(
                f"{self.api_endpoint}/bill",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                results = []
                
                for bill in data.get('bills', []):
                    results.append({
                        'title': bill.get('title', ''),
                        'number': bill.get('number', ''),
                        'congress': bill.get('congress', ''),
                        'type': bill.get('type', ''),
                        'date': bill.get('introducedDate', ''),
                        'sponsor': bill.get('sponsors', [{}])[0].get('fullName', '') if bill.get('sponsors') else '',
                        'url': bill.get('url', ''),
                        'source_database': 'congress_gov',
                        'id': f"{bill.get('congress', '')}-{bill.get('type', '')}-{bill.get('number', '')}",
                        'latest_action': bill.get('latestAction', {}).get('text', '') if bill.get('latestAction') else ''
                    })
                
                logger.info(f"Congress.gov found {len(results)} results")
                return results
            else:
                logger.error(f"Congress search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Congress.gov: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get bill details"""
        if not self.authenticated:
            return {}
        
        self.rate_limiter.wait_if_needed()
        
        try:
            # Parse document_id to get congress, type, number
            parts = document_id.split('-')
            if len(parts) >= 3:
                congress, bill_type, number = parts[0], parts[1], parts[2]
                
                headers = {
                    'X-API-Key': self.api_key,
                    'User-Agent': 'LegalAssistant/1.0'
                }
                
                response = requests.get(
                    f"{self.api_endpoint}/bill/{congress}/{bill_type}/{number}",
                    headers=headers,
                    timeout=10
                )
                
                if response.ok:
                    return response.json()
            
            return {}
                
        except Exception as e:
            logger.error(f"Error getting Congress document: {e}")
            return {}

# Enhanced search functions
def search_free_legal_databases_enhanced(query: str, user=None, source_types: List[str] = None, 
                                       state: str = None, filters: Dict = None) -> List[Dict]:
    """Enhanced search with comprehensive API integration"""
    try:
        all_results = []
        
        # First, try comprehensive search for government data and specialized APIs
        try:
            comprehensive_results = search_comprehensive_legal_databases(query, user, True)
            if comprehensive_results:
                logger.info(f"Found {len(comprehensive_results)} results from comprehensive APIs")
                all_results.extend(comprehensive_results)
        except Exception as e:
            logger.error(f"Comprehensive search failed: {e}")
        
        # If state is specified or detected in query, search state law databases
        if state or _detect_state_in_query(query):
            detected_state = state or _detect_state_in_query(query)
            logger.info(f"Searching state law databases for: {detected_state}")
            
            # Search state-specific databases
            state_results = search_state_law_databases(query, detected_state, source_types)
            all_results.extend(state_results)
        
        # Search existing free databases
        existing_results = search_free_legal_databases(query, user)
        all_results.extend(existing_results)
        
        # Enhanced filtering by source types
        if source_types:
            filtered_results = []
            for result in all_results:
                source_db = result.get('source_database', '').lower()
                category = result.get('category', '').lower()
                legal_area = result.get('legal_area', '').lower()
                
                should_include = False
                
                if 'cases' in source_types and any(term in source_db for term in ['harvard', 'courtlistener', 'scholar', 'cases']):
                    should_include = True
                elif 'statutes' in source_types and any(term in source_db for term in ['cornell', 'justia', 'codes', 'statutes']):
                    should_include = True
                elif 'regulations' in source_types and any(term in source_db for term in ['federal_register', 'cfr', 'regulations']):
                    should_include = True
                elif 'legislation' in source_types and any(term in source_db for term in ['congress', 'openstates', 'bills']):
                    should_include = True
                elif 'business' in source_types and any(term in source_db for term in ['sec', 'sba', 'uspto', 'business']):
                    should_include = True
                elif 'environmental' in source_types and any(term in source_db for term in ['epa', 'environmental']):
                    should_include = True
                elif 'immigration' in source_types and any(term in source_db for term in ['uscis', 'immigration', 'visa']):
                    should_include = True
                elif 'labor' in source_types and any(term in source_db for term in ['osha', 'dol', 'labor']):
                    should_include = True
                elif not source_types:  # If no specific types, include all
                    should_include = True
                
                if should_include:
                    filtered_results.append(result)
            
            all_results = filtered_results
        
        # Remove duplicates and sort by relevance
        all_results = _remove_duplicates_and_rank(all_results, query)
        
        logger.info(f"Enhanced search returned {len(all_results)} total results")
        return all_results
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        return search_free_legal_databases(query, user)

def search_state_law_databases(query: str, state: str = None, law_types: List[str] = None) -> List[Dict]:
    """Search state law databases using new APIs"""
    
    if not state:
        state = _detect_state_in_query(query) or "Washington"
    
    logger.info(f"Searching state law databases for {state}: {query}")
    
    try:
        # Use the state law service
        search_results = state_law_service.search_state_specific(state, query, "comprehensive")
        
        # Flatten and format results
        flattened_results = []
        
        for source_type, sources in search_results.get('sources', {}).items():
            if isinstance(sources, list):
                for source in sources:
                    enhanced_result = {
                        'title': source.get('title', ''),
                        'url': source.get('url', ''),
                        'description': source.get('description', ''),
                        'source_database': f"state_law_{source_type}",
                        'state': state,
                        'category': source_type,
                        'access': source.get('access', 'Free'),
                        'date': source.get('updated_at', ''),
                        'code_type': source.get('code_type', ''),
                        'court_level': source.get('court_level', ''),
                        'relevance_score': _calculate_relevance_score(source, query)
                    }
                    flattened_results.append(enhanced_result)
        
        # Filter by law types if specified
        if law_types:
            type_mapping = {
                'statutes': ['codes', 'statutes'],
                'cases': ['cases', 'case_law'],
                'legislation': ['bills', 'legislation'],
                'regulations': ['regulations', 'administrative']
            }
            
            filtered_results = []
            for result in flattened_results:
                category = result.get('category', '').lower()
                for law_type in law_types:
                    if law_type in type_mapping and any(t in category for t in type_mapping[law_type]):
                        filtered_results.append(result)
                        break
            
            flattened_results = filtered_results
        
        logger.info(f"State law search returned {len(flattened_results)} results for {state}")
        return flattened_results
        
    except Exception as e:
        logger.error(f"State law database search failed: {e}")
        return []

def _detect_state_in_query(query: str) -> Optional[str]:
    """Enhanced state detection in query"""
    # Import enhanced state detection patterns
    from ..config import STATE_DETECTION_PATTERNS
    
    query_upper = query.upper()
    
    # Check for specific state codes first (most reliable)
    for state, patterns in STATE_DETECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.upper() in query_upper:
                return state.replace('_', ' ').title()
    
    # Fallback to basic state name detection
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
        'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
        'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
        'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
        'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
    ]
    
    for state in states:
        if state.lower() in query.lower():
            return state
    
    return None

def _calculate_relevance_score(result: Dict, query: str) -> float:
    """Calculate relevance score for search result"""
    score = 0.5  # Base score
    
    title = result.get('title', '').lower()
    description = result.get('description', '').lower()
    query_lower = query.lower()
    
    # Title match bonus
    if query_lower in title:
        score += 0.3
    
    # Description match bonus
    if query_lower in description:
        score += 0.2
    
    # Source authority bonus
    source_db = result.get('source_database', '').lower()
    if 'cornell' in source_db or 'harvard' in source_db:
        score += 0.1  # Academic sources
    elif 'gov' in source_db:
        score += 0.15  # Government sources
    
    # Recency bonus
    date_str = result.get('date', '')
    if date_str:
        try:
            result_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            days_old = (datetime.now() - result_date.replace(tzinfo=None)).days
            if days_old < 365:  # Within last year
                score += 0.1
        except:
            pass
    
    return min(1.0, score)

def _remove_duplicates_and_rank(results: List[Dict], query: str) -> List[Dict]:
    """Remove duplicates and rank results by relevance and authority"""
    
    # Remove duplicates based on URL and title
    seen_urls = set()
    seen_titles = set()
    unique_results = []
    
    for result in results:
        url = result.get('url', '')
        title = result.get('title', '').lower()
        
        # Create a unique key
        unique_key = f"{url}:{title}"
        
        if unique_key not in seen_urls and title not in seen_titles:
            seen_urls.add(unique_key)
            seen_titles.add(title)
            unique_results.append(result)
    
    # Calculate enhanced relevance scores
    for result in unique_results:
        if 'relevance_score' not in result:
            result['relevance_score'] = _calculate_relevance_score(result, query)
    
    # Authority weights for different sources
    authority_weights = {
        'harvard_caselaw': 0.95,
        'courtlistener': 0.90,
        'cornell_law': 0.85,
        'congress_gov': 0.95,
        'federal_register': 0.90,
        'state_law_codes': 0.80,
        'openstates': 0.75,
        'justia': 0.70,
        'epa_echo': 0.85,
        'sec_edgar': 0.85,
        'dol_osha': 0.85,
        'uscis_case_status': 0.90,
        'fbi_crime_data': 0.85,
        'google_scholar_legal': 0.65
    }
    
    # Apply authority weighting
    for result in unique_results:
        source_db = result.get('source_database', '')
        authority_weight = authority_weights.get(source_db, 0.5)
        
        # Combine relevance and authority
        final_score = (result['relevance_score'] * 0.7) + (authority_weight * 0.3)
        result['final_score'] = final_score
    
    # Sort by final score
    unique_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    return unique_results

# Dictionary of available external databases (COMPLETE VERSION)
external_databases = {
    # Premium databases
    "lexisnexis": None,  # Premium - would need actual implementation
    "westlaw": None,     # Premium - would need actual implementation
    
    # Core free legal databases
    "harvard_caselaw": HarvardCaselawInterface(),
    "courtlistener": CourtListenerInterface(),
    "federal_register": FederalRegisterInterface(),
    "congress_gov": CongressInterface(),
    
    # State law databases
    "cornell_law": CornellLegalAPI(),
    "openstates": OpenStatesAPI(),
    "justia": JustiaLegalAPI(),
    "google_scholar_legal": GoogleScholarLegalAPI(),
    "state_law_comprehensive": StateLawInterface(state_law_service)
}

def search_external_databases(query: str, databases: List[str], user=None) -> List[Dict]:
    """Enhanced external database search with comprehensive API support"""
    all_results = []
    
    # Determine user tier
    user_tier = getattr(user, 'subscription_tier', 'free') if user else 'free'
    
    # Enhanced tier access with all new databases
    tier_access = {
        "free": [
            "harvard_caselaw", "courtlistener", "federal_register", "congress_gov",
            "cornell_law", "openstates", "justia", "google_scholar_legal", 
            "state_law_comprehensive"
        ],
        "basic": [
            "harvard_caselaw", "courtlistener", "federal_register", "congress_gov",
            "cornell_law", "openstates", "justia", "google_scholar_legal", 
            "state_law_comprehensive"
        ],
        "premium": [
            "lexisnexis", "westlaw", "harvard_caselaw", "courtlistener", 
            "federal_register", "congress_gov", "cornell_law", "openstates", 
            "justia", "google_scholar_legal", "state_law_comprehensive"
        ]
    }
    
    # Filter databases based on user tier
    allowed_databases = tier_access.get(user_tier, tier_access["free"])
    accessible_databases = [db for db in databases if db in allowed_databases]
    
    # Auto-detect state for state law searches
    detected_state = _detect_state_in_query(query)
    search_filters = {'state': detected_state} if detected_state else {}
    
    for db_name in accessible_databases:
        if db_name in external_databases and external_databases[db_name]:
            try:
                db_interface = external_databases[db_name]
                
                # Skip authentication check for free services
                if hasattr(db_interface, 'authenticated') and not db_interface.authenticated:
                    if db_name not in ["cornell_law", "openstates", "justia", "google_scholar_legal"]:
                        logger.warning(f"Skipping {db_name} - authentication failed")
                        continue
                
                # Search with enhanced filters
                results = db_interface.search(query, search_filters)
                
                # Add database source info
                for result in results:
                    result['searched_database'] = db_name
                    result['user_tier'] = user_tier
                
                all_results.extend(results)
                logger.info(f"Found {len(results)} results from {db_name}")
                
            except Exception as e:
                logger.error(f"Error searching {db_name}: {e}")
    
    # Enhanced result processing
    all_results = _remove_duplicates_and_rank(all_results, query)
    
    return all_results

def search_free_legal_databases(query: str, user=None) -> List[Dict]:
    """Enhanced free database search with all available APIs"""
    free_databases = [
        "harvard_caselaw", "courtlistener", "federal_register", 
        "cornell_law", "openstates", "justia", "google_scholar_legal"
    ]
    
    # Add congress_gov since we have the API key
    free_databases.append("congress_gov")
    
    return search_external_databases(query, free_databases, user)

def search_legal_databases_comprehensive(query: str, user=None, search_scope: str = "all", 
                                       state: str = None, law_types: List[str] = None) -> List[Dict]:
    """
    MAIN COMPREHENSIVE SEARCH FUNCTION
    This integrates all APIs: traditional legal databases + government APIs + specialized databases
    """
    
    try:
        all_results = []
        
        # 1. Search comprehensive specialized APIs (government data, enforcement, etc.)
        try:
            comprehensive_results = search_comprehensive_legal_databases(query, user, True)
            if comprehensive_results:
                logger.info(f"Comprehensive APIs returned {len(comprehensive_results)} results")
                # Add source type for filtering
                for result in comprehensive_results:
                    result['source_category'] = 'comprehensive_government'
                all_results.extend(comprehensive_results)
        except Exception as e:
            logger.error(f"Comprehensive API search failed: {e}")
        
        # 2. Search traditional legal databases based on scope
        if search_scope == "state_only" and state:
            # State-specific search only
            state_results = search_state_law_databases(query, state, law_types)
            for result in state_results:
                result['source_category'] = 'state_law'
            all_results.extend(state_results)
        
        elif search_scope == "federal_only":
            # Federal databases only
            federal_dbs = ["harvard_caselaw", "courtlistener", "federal_register", "congress_gov"]
            federal_results = search_external_databases(query, federal_dbs, user)
            for result in federal_results:
                result['source_category'] = 'federal_law'
            all_results.extend(federal_results)
        
        else:
            # Comprehensive search (default) - all databases
            all_databases = [
                "harvard_caselaw", "courtlistener", "cornell_law", 
                "openstates", "justia", "google_scholar_legal",
                "federal_register", "congress_gov"
            ]
            
            # Add premium databases if user has access
            user_tier = getattr(user, 'subscription_tier', 'free') if user else 'free'
            if user_tier in ['premium', 'enterprise']:
                all_databases.extend(["lexisnexis", "westlaw"])
            
            traditional_results = search_external_databases(query, all_databases, user)
            for result in traditional_results:
                result['source_category'] = 'traditional_legal'
            all_results.extend(traditional_results)
        
        # 3. Process and rank all results
        all_results = _remove_duplicates_and_rank(all_results, query)
        
        # 4. Apply final filtering by law types if specified
        if law_types:
            all_results = _filter_by_law_types(all_results, law_types)
        
        logger.info(f"Comprehensive legal search returned {len(all_results)} total results")
        return all_results
    
    except Exception as e:
        logger.error(f"Comprehensive legal search failed: {e}")
        # Fallback to basic free search
        return search_free_legal_databases(query, user)

def _filter_by_law_types(results: List[Dict], law_types: List[str]) -> List[Dict]:
    """Filter results by specific law types"""
    
    type_indicators = {
        'cases': ['case', 'opinion', 'court', 'decision', 'ruling'],
        'statutes': ['code', 'statute', 'law', 'usc', 'rcw'],
        'regulations': ['regulation', 'rule', 'cfr', 'administrative'],
        'legislation': ['bill', 'act', 'amendment', 'legislation'],
        'enforcement': ['violation', 'citation', 'enforcement', 'penalty'],
        'data': ['statistics', 'data', 'report', 'analysis']
    }
    
    filtered_results = []
    
    for result in results:
        should_include = False
        
        # Check result content against law type indicators
        result_text = f"{result.get('title', '')} {result.get('description', '')} {result.get('source_database', '')}".lower()
        
        for law_type in law_types:
            if law_type in type_indicators:
                indicators = type_indicators[law_type]
                if any(indicator in result_text for indicator in indicators):
                    should_include = True
                    break
        
        # Also check category and source_database fields
        category = result.get('category', '').lower()
        source_db = result.get('source_database', '').lower()
        
        for law_type in law_types:
            if law_type in category or law_type in source_db:
                should_include = True
                break
        
        if should_include or not law_types:  # Include all if no specific types
            filtered_results.append(result)
    
    return filtered_results

def comprehensive_legal_search(query: str, user=None, include_state_law: bool = True, 
                             target_jurisdiction: str = None) -> Dict:
    """Perform comprehensive search across ALL available legal databases and APIs"""
    
    search_start = datetime.now()
    
    results = {
        'query': query,
        'search_date': search_start.isoformat(),
        'user_tier': getattr(user, 'subscription_tier', 'free') if user else 'free',
        'target_jurisdiction': target_jurisdiction,
        'sources': {},
        'summary': {}
    }
    
    # Detect legal areas and state for intelligent routing
    detected_areas = _detect_legal_areas(query)
    detected_state = target_jurisdiction or _detect_state_in_query(query)
    
    logger.info(f"Comprehensive search - Areas: {detected_areas}, State: {detected_state}")
    
    # 1. Search comprehensive government and specialized APIs
    try:
        comprehensive_results = search_comprehensive_legal_databases(query, user, True)
        results['sources']['comprehensive_apis'] = comprehensive_results
    except Exception as e:
        logger.error(f"Comprehensive API search failed: {e}")
        results['sources']['comprehensive_apis'] = []
    
    # 2. Search traditional legal databases
    try:
        traditional_databases = ["harvard_caselaw", "courtlistener"]
        traditional_results = search_external_databases(query, traditional_databases, user)
        results['sources']['case_law'] = traditional_results
    except Exception as e:
        logger.error(f"Traditional legal search failed: {e}")
        results['sources']['case_law'] = []
    
    # 3. Search federal government databases
    try:
        federal_databases = ["federal_register", "congress_gov"]
        federal_results = search_external_databases(query, federal_databases, user)
        results['sources']['federal_government'] = federal_results
    except Exception as e:
        logger.error(f"Federal search failed: {e}")
        results['sources']['federal_government'] = []
    
    # 4. Search state databases if enabled
    if include_state_law and detected_state:
        try:
            state_databases = ["cornell_law", "openstates", "justia", "state_law_comprehensive"]
            state_results = []
            for db_name in state_databases:
                if db_name in external_databases and external_databases[db_name]:
                    try:
                        db_results = external_databases[db_name].search(query, {'state': detected_state})
                        state_results.extend(db_results)
                    except Exception as e:
                        logger.error(f"State database {db_name} search failed: {e}")
            
            results['sources']['state_law'] = state_results
        except Exception as e:
            logger.error(f"State search failed: {e}")
            results['sources']['state_law'] = []
    
    # Generate summary
    total_results = sum(len(v) for v in results['sources'].values() if isinstance(v, list))
    search_duration = (datetime.now() - search_start).total_seconds()
    
    results['summary'] = {
        'total_results': total_results,
        'search_duration_seconds': search_duration,
        'databases_searched': len([k for k, v in results['sources'].items() if v]),
        'detected_legal_areas': detected_areas,
        'detected_state': detected_state,
        'comprehensive_apis_used': len(results['sources'].get('comprehensive_apis', [])) > 0,
        'government_data_included': any('gov' in str(results['sources'].get('comprehensive_apis', [])))
    }
    
    return results

def _detect_legal_areas(query: str) -> List[str]:
    """Detect legal practice areas from query"""
    from ..config import LEGAL_AREA_KEYWORDS
    
    query_lower = query.lower()
    detected_areas = []
    
    for area, keywords in LEGAL_AREA_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_areas.append(area)
    
    return detected_areas

def get_database_status() -> Dict[str, Dict]:
    """Enhanced database status with comprehensive API integration"""
    status = {}
    
    for name, interface in external_databases.items():
        if interface is None:
            status[name] = {
                "available": False,
                "error": "Not implemented",
                "type": "premium" if name in ["lexisnexis", "westlaw"] else "free"
            }
            continue
            
        try:
            # Test authentication
            auth_success = interface.authenticate({}) if hasattr(interface, 'authenticate') else True
            
            # Test basic functionality
            test_search = interface.search("test query") if hasattr(interface, 'search') else []
            
            status[name] = {
                "available": True,
                "authenticated": auth_success,
                "type": "premium" if name in ["lexisnexis", "westlaw"] else "free",
                "rate_limited": hasattr(interface, 'rate_limiter'),
                "endpoint": getattr(interface, 'api_endpoint', getattr(interface, 'base_url', None)),
                "test_results": len(test_search) if isinstance(test_search, list) else 0,
                "features": _get_database_features(name, interface)
            }
        except Exception as e:
            status[name] = {
                "available": False,
                "error": str(e),
                "type": "premium" if name in ["lexisnexis", "westlaw"] else "free"
            }
    
    # Add comprehensive API status
    try:
        from .comprehensive_legal_apis import get_comprehensive_legal_apis
        comprehensive_apis = get_comprehensive_legal_apis()
        
        for api_name, api_instance in comprehensive_apis.items():
            if api_name != 'comprehensive_hub':
                status[f"comprehensive_{api_name}"] = {
                    "available": True,
                    "type": "free_government_data",
                    "features": ["government_enforcement_data", "real_time_updates", "official_source"]
                }
    except Exception as e:
        logger.error(f"Could not get comprehensive API status: {e}")
    
    return status

def _get_database_features(db_name: str, interface) -> List[str]:
    """Get features available for each database"""
    features = []
    
    if hasattr(interface, 'search_state_code'):
        features.append("state_codes")
    if hasattr(interface, 'search_case_law'):
        features.append("case_law")
    if hasattr(interface, 'search_bills'):
        features.append("legislation")
    if hasattr(interface, 'search_legislators'):
        features.append("legislator_info")
    if hasattr(interface, 'search_federal_code'):
        features.append("federal_codes")
    
    # Database-specific features
    if db_name == "harvard_caselaw":
        features.extend(["full_case_text", "historical_cases", "multiple_jurisdictions"])
    elif db_name == "courtlistener":
        features.extend(["federal_cases", "state_cases", "docket_info"])
    elif db_name == "cornell_law":
        features.extend(["usc", "cfr", "state_codes", "legal_encyclopedia"])
    elif db_name == "openstates":
        features.extend(["current_bills", "legislator_tracking", "voting_records"])
    elif db_name == "justia":
        features.extend(["free_full_text", "all_states", "multiple_code_types"])
    elif db_name == "congress_gov":
        features.extend(["official_federal_legislation", "bill_tracking", "legislative_history"])
    elif db_name == "federal_register":
        features.extend(["federal_regulations", "proposed_rules", "agency_documents"])
    
    return features

def get_available_jurisdictions() -> Dict[str, List[str]]:
    """Get list of available jurisdictions for frontend dropdowns"""
    return {
        'states': [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
            'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
            'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
            'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
            'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
            'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
            'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
        ],
        'federal_courts': [
            'Supreme Court', 'Federal Circuit', 'District Courts', 'Courts of Appeals'
        ],
        'search_types': [
            'comprehensive', 'statutes', 'cases', 'legislation', 'regulations', 'enforcement', 'data'
        ],
        'available_databases': {
            'free': [db for db in external_databases.keys() if external_databases[db] is not None],
            'premium': ['lexisnexis', 'westlaw'] + [db for db in external_databases.keys() if external_databases[db] is not None],
            'government_apis': ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi', 'congress', 'federal_register'],
            'comprehensive': ['all_traditional_plus_government_data']
        }
    }

# Main export functions for integration
__all__ = [
    'search_external_databases',
    'search_free_legal_databases', 
    'search_free_legal_databases_enhanced',
    'search_state_law_databases',
    'search_legal_databases_comprehensive',
    'comprehensive_legal_search',
    'get_database_status',
    'get_available_jurisdictions',
    'external_databases',
    'HarvardCaselawInterface',
    'CourtListenerInterface',
    'FederalRegisterInterface',
    'CongressInterface',
    'CornellLegalAPI',
    'OpenStatesAPI',
    'JustiaLegalAPI',
    'GoogleScholarLegalAPI',
    'StateLawInterface'
]
