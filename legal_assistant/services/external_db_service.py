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
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Rate limiting helper
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
        
        # Check if we need to wait
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

class LexisNexisInterface(LegalDatabaseInterface):
    """Interface for LexisNexis legal database"""
    
    def __init__(self):
        self.api_key = os.environ.get("LEXISNEXIS_API_KEY")
        self.api_endpoint = os.environ.get("LEXISNEXIS_API_ENDPOINT", "https://api.lexisnexis.com/v1")
        self.authenticated = False
        self.rate_limiter = RateLimiter(1000, 50)  # Premium service, higher limits
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with LexisNexis"""
        try:
            if self.api_key:
                self.authenticated = True
                logger.info("LexisNexis authenticated successfully")
                return True
            logger.warning("LexisNexis API key not found")
            return False
        except Exception as e:
            logger.error(f"LexisNexis authentication failed: {e}")
            return False
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search LexisNexis database"""
        if not self.authenticated:
            logger.warning("Not authenticated with LexisNexis")
            return []
        
        self.rate_limiter.wait_if_needed()
        
        try:
            # Placeholder for actual API call
            # In production, this would make real API requests
            return [{
                'title': f'LexisNexis Result for: {query}',
                'source': 'LexisNexis',
                'preview': 'This would contain actual search results',
                'source_database': 'lexisnexis',
                'url': f'{self.api_endpoint}/search?q={query}',
                'date': datetime.now().isoformat(),
                'type': 'premium_legal_content'
            }]
        except Exception as e:
            logger.error(f"LexisNexis search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from LexisNexis"""
        if not self.authenticated:
            return {}
        
        self.rate_limiter.wait_if_needed()
        
        return {
            'id': document_id,
            'content': 'Document content would be here',
            'source': 'LexisNexis',
            'retrieved_at': datetime.now().isoformat()
        }

class WestlawInterface(LegalDatabaseInterface):
    """Interface for Westlaw legal database"""
    
    def __init__(self):
        self.api_key = os.environ.get("WESTLAW_API_KEY")
        self.api_endpoint = os.environ.get("WESTLAW_API_ENDPOINT", "https://api.westlaw.com/v1")
        self.authenticated = False
        self.rate_limiter = RateLimiter(1000, 50)  # Premium service, higher limits
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with Westlaw"""
        try:
            if self.api_key:
                self.authenticated = True
                logger.info("Westlaw authenticated successfully")
                return True
            logger.warning("Westlaw API key not found")
            return False
        except Exception as e:
            logger.error(f"Westlaw authentication failed: {e}")
            return False
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Westlaw database"""
        if not self.authenticated:
            logger.warning("Not authenticated with Westlaw")
            return []
        
        self.rate_limiter.wait_if_needed()
        
        try:
            # Placeholder for actual API call
            return [{
                'title': f'Westlaw Result for: {query}',
                'source': 'Westlaw',
                'preview': 'This would contain actual search results',
                'source_database': 'westlaw',
                'url': f'{self.api_endpoint}/search?q={query}',
                'date': datetime.now().isoformat(),
                'type': 'premium_legal_content'
            }]
        except Exception as e:
            logger.error(f"Westlaw search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from Westlaw"""
        if not self.authenticated:
            return {}
        
        self.rate_limiter.wait_if_needed()
        
        return {
            'id': document_id,
            'content': 'Document content would be here',
            'source': 'Westlaw',
            'retrieved_at': datetime.now().isoformat()
        }

class HarvardCaselawInterface(LegalDatabaseInterface):
    """Harvard Caselaw Access Project - Completely Free"""
    
    def __init__(self):
        self.api_endpoint = "https://api.case.law/v1"
        self.authenticated = True  # No auth needed for basic access
        self.rate_limiter = RateLimiter(500, 20)  # Be respectful
    
    def authenticate(self, credentials: Dict) -> bool:
        """No authentication needed for basic access"""
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Harvard Caselaw"""
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {
                "search": query,
                "page_size": 10
            }
            
            if filters:
                if "jurisdiction" in filters:
                    params["jurisdiction"] = filters["jurisdiction"]
                if "after_date" in filters:
                    params["decision_date_min"] = filters["after_date"]
                if "before_date" in filters:
                    params["decision_date_max"] = filters["before_date"]
            
            response = requests.get(
                f"{self.api_endpoint}/cases/",
                params=params,
                timeout=10,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                data = response.json()
                results = []
                for case in data.get('results', []):
                    results.append({
                        'title': case.get('name', ''),
                        'court': case.get('court', {}).get('name', '') if case.get('court') else '',
                        'date': case.get('decision_date', ''),
                        'citation': case.get('citations', [{}])[0].get('cite', '') if case.get('citations') else '',
                        'url': case.get('frontend_url', ''),
                        'preview': case.get('preview', [{}])[0].get('text', '')[:200] + '...' if case.get('preview') else '',
                        'source_database': 'harvard_caselaw',
                        'id': case.get('id', ''),
                        'jurisdiction': case.get('jurisdiction', {}).get('name', '') if case.get('jurisdiction') else ''
                    })
                
                logger.info(f"Harvard Caselaw found {len(results)} results")
                return results
            else:
                logger.error(f"Harvard search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Harvard Caselaw: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get full case text"""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.api_endpoint}/cases/{document_id}/",
                params={"full_case": "true"},
                timeout=15,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                return response.json()
            else:
                logger.error(f"Failed to get Harvard case {document_id}: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting case from Harvard: {e}")
            return {}

class CourtListenerInterface(LegalDatabaseInterface):
    """CourtListener - Free Federal and State Court Data"""
    
    def __init__(self):
        self.api_endpoint = "https://www.courtlistener.com/api/rest/v4"
        self.api_key = os.environ.get("COURTLISTENER_API_KEY", "")  # Optional, increases rate limits
        self.authenticated = True
        self.rate_limiter = RateLimiter(100, 10)  # Conservative for free tier
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search CourtListener"""
        self.rate_limiter.wait_if_needed()
        
        try:
            headers = {'User-Agent': 'LegalAssistant/1.0'}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            params = {
                "q": query,
                "format": "json",
                "order_by": "score desc"
            }
            
            if filters:
                if "court" in filters:
                    params["court"] = filters["court"]
                if "after_date" in filters:
                    params["filed_after"] = filters["after_date"]
            
            # Search opinions first
            response = requests.get(
                f"{self.api_endpoint}/search/",
                params={**params, "type": "o"},  # opinions
                headers=headers,
                timeout=10
            )
            
            results = []
            
            if response.ok:
                data = response.json()
                opinions = data.get('results', [])
                
                for opinion in opinions:
                    results.append({
                        'title': opinion.get('caseName', 'Unknown Case'),
                        'court': opinion.get('court', ''),
                        'date': opinion.get('dateFiled', ''),
                        'snippet': opinion.get('text', '')[:200] + '...' if opinion.get('text') else '',
                        'url': f"https://www.courtlistener.com{opinion.get('absolute_url', '')}" if opinion.get('absolute_url') else '',
                        'source_database': 'courtlistener',
                        'id': opinion.get('id', ''),
                        'type': 'opinion',
                        'docket_number': opinion.get('docketNumber', ''),
                        'citation': opinion.get('citation', '')
                    })
                
                logger.info(f"CourtListener found {len(results)} results")
                return results
            else:
                logger.error(f"CourtListener search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching CourtListener: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get opinion text"""
        self.rate_limiter.wait_if_needed()
        
        try:
            headers = {'User-Agent': 'LegalAssistant/1.0'}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            response = requests.get(
                f"{self.api_endpoint}/opinions/{document_id}/",
                headers=headers,
                timeout=10
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
        self.api_key = os.environ.get("CONGRESS_API_KEY")
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

# Dictionary of available external databases
external_databases = {
    "lexisnexis": LexisNexisInterface(),
    "westlaw": WestlawInterface(),
    "harvard_caselaw": HarvardCaselawInterface(),
    "courtlistener": CourtListenerInterface(),
    "federal_register": FederalRegisterInterface(),
    "congress_gov": CongressInterface()
}

def search_external_databases(query: str, databases: List[str], 
                            user_tier: str = "free") -> List[Dict]:
    """Search specified external databases based on user tier"""
    all_results = []
    
    # Define what databases each tier can access
    tier_access = {
        "free": ["harvard_caselaw", "courtlistener", "federal_register"],
        "basic": ["harvard_caselaw", "courtlistener", "federal_register", "congress_gov"],
        "premium": ["lexisnexis", "westlaw", "harvard_caselaw", "courtlistener", 
                   "federal_register", "congress_gov"]
    }
    
    # Filter databases based on user tier
    allowed_databases = tier_access.get(user_tier, tier_access["free"])
    accessible_databases = [db for db in databases if db in allowed_databases]
    
    for db_name in accessible_databases:
        if db_name in external_databases:
            try:
                db_interface = external_databases[db_name]
                
                # Authenticate if needed
                if hasattr(db_interface, 'authenticated') and not db_interface.authenticated:
                    db_interface.authenticate({})
                
                # Skip if authentication failed
                if hasattr(db_interface, 'authenticated') and not db_interface.authenticated:
                    logger.warning(f"Skipping {db_name} - authentication failed")
                    continue
                
                # Search
                results = db_interface.search(query)
                all_results.extend(results)
                logger.info(f"Found {len(results)} results from {db_name}")
                
            except Exception as e:
                logger.error(f"Error searching {db_name}: {e}")
    
    return all_results

def search_free_legal_databases(query: str, user_tier: str = "free") -> List[Dict]:
    """Search all available free legal databases"""
    free_databases = ["harvard_caselaw", "courtlistener", "federal_register"]
    
    # Add congress_gov if API key is available
    if os.environ.get("CONGRESS_API_KEY"):
        free_databases.append("congress_gov")
    
    return search_external_databases(query, free_databases, user_tier)

def get_database_status() -> Dict[str, Dict]:
    """Get status of all configured databases"""
    status = {}
    
    for name, interface in external_databases.items():
        try:
            # Test authentication
            auth_success = interface.authenticate({})
            
            status[name] = {
                "available": True,
                "authenticated": auth_success,
                "type": "premium" if name in ["lexisnexis", "westlaw"] else "free",
                "rate_limited": hasattr(interface, 'rate_limiter'),
                "endpoint": getattr(interface, 'api_endpoint', None)
            }
        except Exception as e:
            status[name] = {
                "available": False,
                "error": str(e),
                "type": "premium" if name in ["lexisnexis", "westlaw"] else "free"
            }
    
    return status
