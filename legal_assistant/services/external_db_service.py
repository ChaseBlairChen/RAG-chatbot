import requests
import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import os

logger = logging.getLogger(__name__)

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
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with LexisNexis"""
        try:
            # Placeholder for actual authentication
            if self.api_key:
                self.authenticated = True
                return True
            return False
        except Exception as e:
            logger.error(f"LexisNexis authentication failed: {e}")
            return False
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search LexisNexis database"""
        if not self.authenticated:
            logger.warning("Not authenticated with LexisNexis")
            return []
        
        try:
            # Placeholder for actual API call
            # In production, this would make real API requests
            return [{
                'title': f'LexisNexis Result for: {query}',
                'source': 'LexisNexis',
                'preview': 'This would contain actual search results',
                'source_database': 'lexisnexis'
            }]
        except Exception as e:
            logger.error(f"LexisNexis search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from LexisNexis"""
        if not self.authenticated:
            return {}
        
        # Placeholder for actual implementation
        return {
            'id': document_id,
            'content': 'Document content would be here',
            'source': 'LexisNexis'
        }


class WestlawInterface(LegalDatabaseInterface):
    """Interface for Westlaw legal database"""
    
    def __init__(self):
        self.api_key = os.environ.get("WESTLAW_API_KEY")
        self.api_endpoint = os.environ.get("WESTLAW_API_ENDPOINT", "https://api.westlaw.com/v1")
        self.authenticated = False
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with Westlaw"""
        try:
            if self.api_key:
                self.authenticated = True
                return True
            return False
        except Exception as e:
            logger.error(f"Westlaw authentication failed: {e}")
            return False
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Westlaw database"""
        if not self.authenticated:
            logger.warning("Not authenticated with Westlaw")
            return []
        
        try:
            # Placeholder for actual API call
            return [{
                'title': f'Westlaw Result for: {query}',
                'source': 'Westlaw',
                'preview': 'This would contain actual search results',
                'source_database': 'westlaw'
            }]
        except Exception as e:
            logger.error(f"Westlaw search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from Westlaw"""
        if not self.authenticated:
            return {}
        
        return {
            'id': document_id,
            'content': 'Document content would be here',
            'source': 'Westlaw'
        }


class HarvardCaselawInterface(LegalDatabaseInterface):
    """Harvard Caselaw Access Project - Completely Free"""
    
    def __init__(self):
        self.api_endpoint = "https://api.case.law/v1"
        self.authenticated = True  # No auth needed for basic access
    
    def authenticate(self, credentials: Dict) -> bool:
        # No authentication needed for basic access
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Harvard Caselaw"""
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
            
            response = requests.get(
                f"{self.api_endpoint}/cases/",
                params=params,
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                results = []
                for case in data.get('results', []):
                    results.append({
                        'title': case.get('name', ''),
                        'court': case.get('court', {}).get('name', ''),
                        'date': case.get('decision_date', ''),
                        'citation': case.get('citations', [{}])[0].get('cite', '') if case.get('citations') else '',
                        'url': case.get('frontend_url', ''),
                        'preview': case.get('preview', ''),
                        'source_database': 'harvard_caselaw'
                    })
                return results
            else:
                logger.error(f"Harvard search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Harvard Caselaw: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get full case text"""
        try:
            response = requests.get(
                f"{self.api_endpoint}/cases/{document_id}/",
                params={"full_case": "true"},
                timeout=10
            )
            
            if response.ok:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting case from Harvard: {e}")
            return {}


class CourtListenerInterface(LegalDatabaseInterface):
    """CourtListener - Free Federal and State Court Data"""
    
    def __init__(self):
        self.api_endpoint = "https://www.courtlistener.com/api/rest/v3"
        self.api_key = os.environ.get("COURTLISTENER_API_KEY", "")  # Optional, increases rate limits
        self.authenticated = True
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search CourtListener"""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            params = {
                "q": query,
                "format": "json",
                "order_by": "score desc"
            }
            
            response = requests.get(
                f"{self.api_endpoint}/search/",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                results = []
                for item in data.get('results', []):
                    results.append({
                        'title': item.get('caseName', ''),
                        'court': item.get('court', ''),
                        'date': item.get('dateFiled', ''),
                        'docket': item.get('docketNumber', ''),
                        'snippet': item.get('snippet', ''),
                        'url': f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                        'source_database': 'courtlistener'
                    })
                return results
            else:
                logger.error(f"CourtListener search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching CourtListener: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get opinion text"""
        try:
            headers = {}
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
                return {}
                
        except Exception as e:
            logger.error(f"Error getting opinion from CourtListener: {e}")
            return {}


class JustiaInterface(LegalDatabaseInterface):
    """Justia - Free Legal Information (Web Scraping)"""
    
    def __init__(self):
        self.base_url = "https://law.justia.com"
        self.authenticated = True
    
    def authenticate(self, credentials: Dict) -> bool:
        return True
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Justia using their search page"""
        try:
            # Note: Justia doesn't have an API, so this is a simplified example
            # In production, you'd want to use BeautifulSoup or similar for web scraping
            
            search_url = f"{self.base_url}/search"
            params = {"q": query}
            
            # This is a placeholder - actual implementation would scrape the HTML
            results = [{
                'title': f"Search results for: {query}",
                'description': "Visit Justia website for full results",
                'url': f"{search_url}?q={query}",
                'source_database': 'justia',
                'note': 'Justia requires web scraping for full integration'
            }]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Justia: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict:
        return {"note": "Justia requires web scraping for document retrieval"}


# Dictionary of available external databases
external_databases = {
    "lexisnexis": LexisNexisInterface(),
    "westlaw": WestlawInterface(),
    "harvard_caselaw": HarvardCaselawInterface(),
    "courtlistener": CourtListenerInterface(),
    "justia": JustiaInterface()
}


def search_external_databases(query: str, databases: List[str], user: 'User') -> List[Dict]:
    """Search specified external databases"""
    all_results = []
    
    for db_name in databases:
        if db_name in external_databases:
            try:
                db_interface = external_databases[db_name]
                
                # Authenticate if needed
                if hasattr(db_interface, 'authenticated') and not db_interface.authenticated:
                    db_interface.authenticate({})
                
                # Search
                results = db_interface.search(query)
                all_results.extend(results)
                logger.info(f"Found {len(results)} results from {db_name}")
                
            except Exception as e:
                logger.error(f"Error searching {db_name}: {e}")
    
    return all_results


def search_free_legal_databases(query: str, user: 'User') -> List[Dict]:
    """Search all free legal databases"""
    free_databases = ["harvard_caselaw", "courtlistener", "justia"]
    all_results = []
    
    for db_name in free_databases:
        if db_name in external_databases:
            try:
                results = external_databases[db_name].search(query)
                all_results.extend(results)
                logger.info(f"Found {len(results)} results from {db_name}")
            except Exception as e:
                logger.error(f"Error searching {db_name}: {e}")
    
    return all_results
