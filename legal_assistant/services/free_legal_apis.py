"""Free and affordable legal database integrations for SMBs/NGOs"""
import requests
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
import os

from ..config import (
    COURTLISTENER_API_KEY, COURTLISTENER_API_ENDPOINT,
    FEDERAL_REGISTER_API_ENDPOINT, CONGRESS_API_KEY, CONGRESS_API_ENDPOINT,
    SEC_EDGAR_API_ENDPOINT, EPA_API_ENDPOINT, WORLD_BANK_API_ENDPOINT,
    SBA_API_ENDPOINT, USPTO_API_ENDPOINT, STATE_APIS,
    API_RATE_LIMITS
)

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, api_name: str):
        self.api_name = api_name
        self.calls = []
        self.limits = API_RATE_LIMITS.get(api_name, {"requests_per_hour": 100, "burst": 10})
    
    def can_call(self) -> bool:
        now = time.time()
        # Remove calls older than 1 hour
        self.calls = [t for t in self.calls if now - t < 3600]
        
        # Check burst limit (last minute)
        recent_calls = [t for t in self.calls if now - t < 60]
        if len(recent_calls) >= self.limits["burst"]:
            return False
        
        # Check hourly limit
        if len(self.calls) >= self.limits["requests_per_hour"]:
            return False
        
        return True
    
    def record_call(self):
        self.calls.append(time.time())

# Rate limiters for each API
rate_limiters = {}

def get_rate_limiter(api_name: str) -> RateLimiter:
    if api_name not in rate_limiters:
        rate_limiters[api_name] = RateLimiter(api_name)
    return rate_limiters[api_name]

class EnhancedCourtListenerAPI:
    """Enhanced CourtListener integration with all endpoints"""
    
    def __init__(self):
        self.base_url = COURTLISTENER_API_ENDPOINT
        self.api_key = COURTLISTENER_API_KEY
        self.rate_limiter = get_rate_limiter("COURTLISTENER")
    
    def search_opinions(self, query: str, jurisdiction: str = None) -> List[Dict]:
        """Search court opinions"""
        if not self.rate_limiter.can_call():
            logger.warning("CourtListener rate limit reached")
            return []
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            params = {
                "q": query,
                "format": "json",
                "order_by": "score desc"
            }
            
            if jurisdiction:
                params["court"] = jurisdiction
            
            response = requests.get(
                f"{self.base_url}search/",
                params=params,
                headers=headers,
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                data = response.json()
                results = []
                
                for item in data.get('results', [])[:10]:
                    results.append({
                        'title': item.get('caseName', ''),
                        'court': item.get('court', ''),
                        'date': item.get('dateFiled', ''),
                        'docket': item.get('docketNumber', ''),
                        'citation': self._format_citation(item),
                        'preview': item.get('snippet', ''),
                        'url': f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                        'source_database': 'courtlistener_free'
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"CourtListener search failed: {e}")
        
        return []
    
    def get_docket_info(self, docket_id: str) -> Dict:
        """Get detailed docket information"""
        if not self.rate_limiter.can_call():
            return {}
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            response = requests.get(
                f"{self.base_url}dockets/{docket_id}/",
                headers=headers,
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get docket info: {e}")
        
        return {}
    
    def _format_citation(self, case_data: Dict) -> str:
        """Format case citation"""
        citations = case_data.get('citation', [])
        if citations:
            return citations[0]
        return f"{case_data.get('docketNumber', 'No docket')} ({case_data.get('court', '')})"

class FederalRegisterAPI:
    """Federal Register API for regulations and notices"""
    
    def __init__(self):
        self.base_url = FEDERAL_REGISTER_API_ENDPOINT
        self.rate_limiter = get_rate_limiter("FEDERAL_REGISTER")
    
    def search_documents(self, query: str, document_type: str = None, 
                        agencies: List[str] = None) -> List[Dict]:
        """Search Federal Register documents"""
        if not self.rate_limiter.can_call():
            return []
        
        try:
            params = {
                "conditions[term]": query,
                "per_page": 20,
                "order": "relevance"
            }
            
            if document_type:
                params["conditions[type][]"] = document_type
            
            if agencies:
                params["conditions[agencies][]"] = agencies
            
            response = requests.get(
                f"{self.base_url}documents.json",
                params=params,
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                data = response.json()
                results = []
                
                for doc in data.get('results', []):
                    results.append({
                        'title': doc.get('title', ''),
                        'type': doc.get('type', ''),
                        'agencies': ', '.join(doc.get('agencies', [])),
                        'publication_date': doc.get('publication_date', ''),
                        'document_number': doc.get('document_number', ''),
                        'summary': doc.get('abstract', ''),
                        'url': doc.get('html_url', ''),
                        'pdf_url': doc.get('pdf_url', ''),
                        'source_database': 'federal_register'
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"Federal Register search failed: {e}")
        
        return []
    
    def get_agency_rules(self, agency: str, days_back: int = 30) -> List[Dict]:
        """Get recent rules from specific agency"""
        if not self.rate_limiter.can_call():
            return []
        
        try:
            date_gte = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                "conditions[agencies][]": agency,
                "conditions[type][]": "rule",
                "conditions[publication_date][gte]": date_gte,
                "per_page": 50
            }
            
            response = requests.get(
                f"{self.base_url}documents.json",
                params=params,
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                return response.json().get('results', [])
            
        except Exception as e:
            logger.error(f"Failed to get agency rules: {e}")
        
        return []

class CongressAPI:
    """Congress.gov API for bills and legislative information"""
    
    def __init__(self):
        self.base_url = CONGRESS_API_ENDPOINT
        self.api_key = CONGRESS_API_KEY
        self.rate_limiter = get_rate_limiter("CONGRESS")
    
    def search_bills(self, query: str, congress: int = None) -> List[Dict]:
        """Search congressional bills"""
        if not self.api_key or not self.rate_limiter.can_call():
            return []
        
        try:
            headers = {"X-API-Key": self.api_key}
            
            # If no congress specified, use current
            if not congress:
                congress = self._get_current_congress()
            
            params = {
                "q": query,
                "format": "json",
                "limit": 20
            }
            
            response = requests.get(
                f"{self.base_url}bill/{congress}",
                params=params,
                headers=headers,
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                data = response.json()
                results = []
                
                for bill in data.get('bills', []):
                    results.append({
                        'bill_id': bill.get('number', ''),
                        'title': bill.get('title', ''),
                        'sponsor': bill.get('sponsor', {}).get('name', ''),
                        'introduced_date': bill.get('introducedDate', ''),
                        'latest_action': bill.get('latestAction', {}).get('text', ''),
                        'status': self._determine_bill_status(bill),
                        'url': bill.get('url', ''),
                        'source_database': 'congress_gov'
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"Congress API search failed: {e}")
        
        return []
    
    def get_bill_details(self, congress: int, bill_type: str, bill_number: int) -> Dict:
        """Get detailed bill information"""
        if not self.api_key or not self.rate_limiter.can_call():
            return {}
        
        try:
            headers = {"X-API-Key": self.api_key}
            
            response = requests.get(
                f"{self.base_url}bill/{congress}/{bill_type}/{bill_number}",
                headers=headers,
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get bill details: {e}")
        
        return {}
    
    def _get_current_congress(self) -> int:
        """Calculate current Congress number"""
        current_year = datetime.now().year
        # Congress changes every 2 years, starting from 1789
        # 118th Congress: 2023-2024
        return 118 + ((current_year - 2023) // 2)
    
    def _determine_bill_status(self, bill_data: Dict) -> str:
        """Determine bill status from data"""
        latest_action = bill_data.get('latestAction', {}).get('text', '').lower()
        
        if 'became public law' in latest_action:
            return 'Enacted'
        elif 'passed' in latest_action:
            return 'Passed'
        elif 'committee' in latest_action:
            return 'In Committee'
        else:
            return 'Introduced'

class SECEdgarAPI:
    """SEC EDGAR API for corporate filings"""
    
    def __init__(self):
        self.base_url = SEC_EDGAR_API_ENDPOINT
        self.rate_limiter = get_rate_limiter("SEC_EDGAR")
    
    def search_company_filings(self, company_name: str, filing_type: str = None) -> List[Dict]:
        """Search SEC filings"""
        if not self.rate_limiter.can_call():
            return []
        
        try:
            # First, get CIK (Central Index Key) for company
            cik = self._get_company_cik(company_name)
            if not cik:
                return []
            
            # Get filings
            params = {"action": "getcompany", "CIK": cik, "output": "json"}
            
            if filing_type:
                params["type"] = filing_type
            
            response = requests.get(
                f"{self.base_url}submissions/CIK{cik:010d}.json",
                timeout=10
            )
            
            self.rate_limiter.record_call()
            
            if response.ok:
                data = response.json()
                results = []
                
                recent_filings = data.get('filings', {}).get('recent', {})
                
                for i in range(min(10, len(recent_filings.get('form', [])))):
                    results.append({
                        'form_type': recent_filings['form'][i],
                        'filing_date': recent_filings['filingDate'][i],
                        'document': recent_filings['primaryDocument'][i],
                        'description': recent_filings.get('primaryDocDescription', [''])[i],
                        'source_database': 'sec_edgar'
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"SEC EDGAR search failed: {e}")
        
        return []
    
    def _get_company_cik(self, company_name: str) -> Optional[str]:
        """Get company CIK from name"""
        # This would need implementation to search company tickers
        # For now, return None
        return None

class EPAEnvironmentalAPI:
    """EPA Environmental data API"""
    
    def __init__(self):
        self.base_url = EPA_API_ENDPOINT
        self.rate_limiter = get_rate_limiter("EPA")
    
    def search_violations(self, facility_name: str = None, state: str = None) -> List[Dict]:
        """Search EPA violations"""
        if not self.rate_limiter.can_call():
            return []
        
        try:
            # EPA's ECHO API endpoint
            echo_url = "https://echo.epa.gov/echo/efr_rest_services.get_facility_info"
            
            params = {"output": "json", "responseset": "2"}
            
            if facility_name:
                params["p_fn"] = facility_name
            if state:
                params["p_st"] = state
            
            response = requests.get(echo_url, params=params, timeout=10)
            self.rate_limiter.record_call()
            
            if response.ok:
                # Parse response and format results
                # Implementation depends on EPA response format
                return []
            
        except Exception as e:
            logger.error(f"EPA search failed: {e}")
        
        return []

class SmallBusinessAPI:
    """SBA and USPTO APIs for small business resources"""
    
    def __init__(self):
        self.sba_url = SBA_API_ENDPOINT
        self.uspto_url = USPTO_API_ENDPOINT
        self.rate_limiter = get_rate_limiter("SBA")
    
    def search_sba_resources(self, topic: str) -> List[Dict]:
        """Search SBA resources"""
        # SBA API implementation
        # Most SBA resources are web-based, not API-based
        return [{
            'title': f'SBA Resources for {topic}',
            'url': f'https://www.sba.gov/search/{topic}',
            'source_database': 'sba',
            'description': 'Visit SBA website for small business resources'
        }]
    
    def search_patents(self, query: str) -> List[Dict]:
        """Search USPTO patents"""
        # USPTO has complex API requirements
        # Simplified placeholder
        return [{
            'title': f'Patent search for: {query}',
            'url': f'https://patents.google.com/?q={query}',
            'source_database': 'uspto',
            'description': 'Use Google Patents for comprehensive search'
        }]

class StateGovernmentAPIs:
    """Interface for state government APIs"""
    
    def __init__(self):
        self.state_apis = STATE_APIS
    
    def search_state_data(self, state: str, query: str) -> List[Dict]:
        """Search state government data"""
        state_lower = state.lower()
        
        if state_lower not in self.state_apis:
            return []
        
        # Each state has different API structure
        # This is a placeholder for state-specific implementations
        return [{
            'title': f'{state} State Data Search',
            'query': query,
            'api_url': self.state_apis[state_lower],
            'source_database': f'{state_lower}_state',
            'note': 'State API integration requires state-specific configuration'
        }]

# Unified free legal search interface
class FreeLegalDatabaseHub:
    """Central hub for all free legal databases"""
    
    def __init__(self):
        self.courtlistener = EnhancedCourtListenerAPI()
        self.federal_register = FederalRegisterAPI()
        self.congress = CongressAPI()
        self.sec = SECEdgarAPI()
        self.epa = EPAEnvironmentalAPI()
        self.small_business = SmallBusinessAPI()
        self.state_apis = StateGovernmentAPIs()
    
    def search_all_free_sources(self, query: str, source_types: List[str] = None) -> Dict:
        """Search across all free legal sources"""
        
        if not source_types:
            source_types = ["cases", "regulations", "legislation", "business"]
        
        results = {
            "query": query,
            "search_date": datetime.now().isoformat(),
            "sources": {}
        }
        
        # Search case law
        if "cases" in source_types:
            try:
                cases = self.courtlistener.search_opinions(query)
                results["sources"]["case_law"] = cases
            except Exception as e:
                logger.error(f"Case law search failed: {e}")
                results["sources"]["case_law"] = []
        
        # Search regulations
        if "regulations" in source_types:
            try:
                regulations = self.federal_register.search_documents(query)
                results["sources"]["regulations"] = regulations
            except Exception as e:
                logger.error(f"Regulation search failed: {e}")
                results["sources"]["regulations"] = []
        
        # Search legislation
        if "legislation" in source_types:
            try:
                bills = self.congress.search_bills(query)
                results["sources"]["legislation"] = bills
            except Exception as e:
                logger.error(f"Legislation search failed: {e}")
                results["sources"]["legislation"] = []
        
        # Search business resources
        if "business" in source_types:
            try:
                sba = self.small_business.search_sba_resources(query)
                results["sources"]["small_business"] = sba
            except Exception as e:
                logger.error(f"Business search failed: {e}")
                results["sources"]["small_business"] = []
        
        # Count total results
        total = sum(len(v) for v in results["sources"].values() if isinstance(v, list))
        results["total_results"] = total
        
        return results

# Global instance
free_legal_hub = FreeLegalDatabaseHub()
