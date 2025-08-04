# legal_assistant/services/state_law_apis.py
"""Free state law and legal database integrations"""
import requests
import logging
import re
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin

logger = logging.getLogger(__name__)

class RateLimiter:
    """Enhanced rate limiter for multiple APIs"""
    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        # Check burst limit
        if len(self.calls) >= self.burst:
            sleep_time = 60 / self.requests_per_minute
            time.sleep(sleep_time)
        
        self.calls.append(now)

class CornellLegalAPI:
    """Legal Information Institute (Cornell Law) - Free comprehensive legal database"""
    
    def __init__(self):
        self.base_url = "https://www.law.cornell.edu"
        self.rate_limiter = RateLimiter(30, 5)  # Be respectful
        
    def search_state_code(self, state: str, query: str) -> List[Dict]:
        """Search state legal codes"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # Cornell organizes by state
            state_codes = {
                'california': 'https://www.law.cornell.edu/california/code',
                'new_york': 'https://www.law.cornell.edu/newyork/code',
                'texas': 'https://www.law.cornell.edu/texas/code',
                'florida': 'https://www.law.cornell.edu/florida/code',
                'washington': 'https://www.law.cornell.edu/washington/code',
                # Add more states as needed
            }
            
            state_key = state.lower().replace(' ', '_')
            if state_key not in state_codes:
                return self._search_generic_cornell(query, state)
            
            # Search specific state code
            search_url = f"{state_codes[state_key]}/search"
            params = {'q': query, 'source': 'state_code'}
            
            response = requests.get(search_url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                # Parse Cornell's response (would need HTML parsing)
                return self._parse_cornell_results(response.text, state)
            
        except Exception as e:
            logger.error(f"Cornell Law search failed: {e}")
        
        return []
    
    def search_federal_code(self, query: str, code_type: str = "usc") -> List[Dict]:
        """Search federal codes (USC, CFR)"""
        self.rate_limiter.wait_if_needed()
        
        try:
            if code_type == "usc":
                search_url = f"{self.base_url}/uscode/search"
            elif code_type == "cfr":
                search_url = f"{self.base_url}/cfr/search"
            else:
                search_url = f"{self.base_url}/search"
            
            params = {'q': query, 'source': code_type}
            
            response = requests.get(search_url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                return self._parse_cornell_federal_results(response.text, code_type)
                
        except Exception as e:
            logger.error(f"Cornell federal search failed: {e}")
        
        return []
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Unified search method"""
        state = filters.get('state', 'Washington') if filters else 'Washington'
        return self.search_state_code(state, query)
    
    def _search_generic_cornell(self, query: str, state: str) -> List[Dict]:
        """Generic Cornell search when state-specific not available"""
        try:
            search_url = f"{self.base_url}/search"
            params = {'q': f"{state} {query}", 'source': 'all'}
            
            response = requests.get(search_url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                return [{
                    'title': f'Cornell Law - {state} Legal Resources',
                    'url': f"{self.base_url}/search?q={quote(f'{state} {query}')}",
                    'source_database': 'cornell_law',
                    'state': state,
                    'description': f'Comprehensive legal resources for {state}',
                    'sections': ['State Code', 'Case Law', 'Regulations'],
                    'access_note': 'Full text available at Cornell Law School'
                }]
                
        except Exception as e:
            logger.error(f"Generic Cornell search failed: {e}")
        
        return []
    
    def _parse_cornell_results(self, html: str, state: str) -> List[Dict]:
        """Parse Cornell Law search results"""
        results = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find search result elements (adjust selectors as needed)
            result_elements = soup.find_all('div', class_='search-result')
            
            for element in result_elements[:10]:
                title_elem = element.find('h3') or element.find('a')
                if title_elem:
                    title = title_elem.get_text().strip()
                    url = title_elem.get('href', '')
                    if url and not url.startswith('http'):
                        url = urljoin(self.base_url, url)
                    
                    description_elem = element.find('p') or element.find('div', class_='description')
                    description = description_elem.get_text().strip() if description_elem else ''
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description[:200] + '...' if len(description) > 200 else description,
                        'source_database': 'cornell_law',
                        'state': state,
                        'type': 'state_code'
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing Cornell results: {e}")
        
        return results
    
    def _parse_cornell_federal_results(self, html: str, code_type: str) -> List[Dict]:
        """Parse Cornell federal law results"""
        # Similar parsing logic for federal codes
        return [{
            'title': f'Cornell Law - Federal {code_type.upper()} Search Results',
            'url': f"{self.base_url}/{code_type}/",
            'source_database': 'cornell_law',
            'type': f'federal_{code_type}',
            'description': f'Federal {code_type.upper()} search results from Cornell Law'
        }]

class OpenStatesAPI:
    """OpenStates API - Free state legislative data"""
    
    def __init__(self):
        self.base_url = "https://v3.openstates.org"
        self.api_key = None  # Free tier doesn't require key
        self.rate_limiter = RateLimiter(60, 10)
        
        # State jurisdiction mapping
        self.state_jurisdictions = {
            'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
            'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
            'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
            'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
            'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
            'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
            'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
            'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
            'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
            'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
            'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
            'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
            'wisconsin': 'wi', 'wyoming': 'wy'
        }
    
    def search_bills(self, state: str, query: str, session: str = None) -> List[Dict]:
        """Search state bills"""
        self.rate_limiter.wait_if_needed()
        
        try:
            jurisdiction = self._get_jurisdiction(state)
            if not jurisdiction:
                return []
            
            params = {
                'jurisdiction': jurisdiction,
                'q': query,
                'format': 'json'
            }
            
            if session:
                params['session'] = session
            
            response = requests.get(
                f"{self.base_url}/bills",
                params=params,
                timeout=10,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                data = response.json()
                results = []
                
                for bill in data.get('results', [])[:20]:
                    results.append({
                        'bill_id': bill.get('identifier', ''),
                        'title': bill.get('title', ''),
                        'state': state,
                        'session': bill.get('session', ''),
                        'subject': ', '.join(bill.get('subject', [])),
                        'status': self._get_latest_action(bill),
                        'sponsors': self._get_sponsors(bill),
                        'url': bill.get('openstates_url', ''),
                        'source_database': 'openstates',
                        'updated_at': bill.get('updated_at', ''),
                        'chamber': bill.get('from_organization', '')
                    })
                
                logger.info(f"OpenStates found {len(results)} bills for {state}")
                return results
                
        except Exception as e:
            logger.error(f"OpenStates bill search failed: {e}")
        
        return []
    
    def get_bill_details(self, bill_id: str, jurisdiction: str) -> Dict:
        """Get detailed bill information"""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.base_url}/bills/{jurisdiction}/{bill_id}",
                timeout=10,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                return response.json()
                
        except Exception as e:
            logger.error(f"Failed to get bill details: {e}")
        
        return {}
    
    def search_legislators(self, state: str, name: str = None) -> List[Dict]:
        """Search state legislators"""
        self.rate_limiter.wait_if_needed()
        
        try:
            jurisdiction = self._get_jurisdiction(state)
            if not jurisdiction:
                return []
            
            params = {'jurisdiction': jurisdiction}
            if name:
                params['name'] = name
            
            response = requests.get(
                f"{self.base_url}/people",
                params=params,
                timeout=10,
                headers={'User-Agent': 'LegalAssistant/1.0'}
            )
            
            if response.ok:
                data = response.json()
                results = []
                
                for person in data.get('results', [])[:10]:
                    results.append({
                        'name': person.get('name', ''),
                        'party': person.get('party', [{}])[0].get('name', '') if person.get('party') else '',
                        'chamber': person.get('current_role', {}).get('org_classification', ''),
                        'district': person.get('current_role', {}).get('district', ''),
                        'email': person.get('email', ''),
                        'url': person.get('openstates_url', ''),
                        'source_database': 'openstates',
                        'state': state
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"OpenStates legislator search failed: {e}")
        
        return []
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Unified search method"""
        state = filters.get('state', 'Washington') if filters else 'Washington'
        return self.search_bills(state, query)
    
    def _get_jurisdiction(self, state: str) -> Optional[str]:
        """Get OpenStates jurisdiction code for state"""
        return self.state_jurisdictions.get(state.lower())
    
    def _get_latest_action(self, bill: Dict) -> str:
        """Extract latest action from bill data"""
        actions = bill.get('actions', [])
        if actions:
            latest = max(actions, key=lambda x: x.get('date', ''))
            return latest.get('description', 'Unknown action')
        return 'No actions recorded'
    
    def _get_sponsors(self, bill: Dict) -> List[str]:
        """Extract sponsors from bill data"""
        sponsors = []
        for sponsorship in bill.get('sponsorships', []):
            name = sponsorship.get('name', '')
            if name:
                sponsors.append(name)
        return sponsors

class JustiaLegalAPI:
    """Justia Free Law Database - Comprehensive free legal search"""
    
    def __init__(self):
        self.base_url = "https://law.justia.com"
        self.rate_limiter = RateLimiter(30, 5)  # Be respectful of free service
    
    def search_state_law(self, state: str, query: str, law_type: str = "codes") -> List[Dict]:
        """Search Justia state law database"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # Justia URL structure: /codes/[state]/
            state_slug = state.lower().replace(' ', '-')
            
            search_urls = {
                'codes': f"{self.base_url}/codes/{state_slug}/",
                'cases': f"{self.base_url}/cases/{state_slug}/",
                'regulations': f"{self.base_url}/regulations/{state_slug}/"
            }
            
            base_url = search_urls.get(law_type, search_urls['codes'])
            
            # Perform search (Justia uses Google Custom Search)
            search_url = f"{base_url}search"
            params = {
                'q': query,
                'cx': 'justia_custom_search_id',  # Would need to get actual ID
                'num': 10
            }
            
            # For now, create structured search URLs
            results = []
            
            # Common state code patterns
            code_types = {
                'criminal': f"{base_url}criminal/",
                'civil': f"{base_url}civil/",
                'business': f"{base_url}business/",
                'family': f"{base_url}family/",
                'property': f"{base_url}property/",
                'administrative': f"{base_url}administrative/"
            }
            
            # Determine relevant code types based on query
            relevant_types = self._determine_relevant_codes(query)
            
            for code_type in relevant_types:
                if code_type in code_types:
                    results.append({
                        'title': f'{state} {code_type.title()} Code - {query}',
                        'url': f"{code_types[code_type]}?search={quote(query)}",
                        'source_database': 'justia',
                        'state': state,
                        'code_type': code_type,
                        'description': f'Search {state} {code_type} law for: {query}',
                        'access': 'Free full text'
                    })
            
            logger.info(f"Justia found {len(results)} relevant code sections for {state}")
            return results
            
        except Exception as e:
            logger.error(f"Justia search failed: {e}")
        
        return []
    
    def search_case_law(self, state: str, query: str, court_level: str = "all") -> List[Dict]:
        """Search Justia case law"""
        self.rate_limiter.wait_if_needed()
        
        try:
            state_slug = state.lower().replace(' ', '-')
            
            # Justia case law structure
            court_urls = {
                'supreme': f"{self.base_url}/cases/{state_slug}/supreme-court/",
                'appellate': f"{self.base_url}/cases/{state_slug}/court-of-appeals/",
                'all': f"{self.base_url}/cases/{state_slug}/"
            }
            
            base_url = court_urls.get(court_level, court_urls['all'])
            
            # Create search result
            return [{
                'title': f'{state} Case Law Search - {query}',
                'url': f"{base_url}?search={quote(query)}",
                'source_database': 'justia_cases',
                'state': state,
                'court_level': court_level,
                'description': f'Search {state} case law for: {query}',
                'search_query': query,
                'note': 'Click to access full search results on Justia'
            }]
            
        except Exception as e:
            logger.error(f"Justia case law search failed: {e}")
        
        return []
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Unified search method"""
        state = filters.get('state', 'Washington') if filters else 'Washington'
        return self.search_state_law(state, query)
    
    def _determine_relevant_codes(self, query: str) -> List[str]:
        """Determine which code sections are relevant to query"""
        query_lower = query.lower()
        
        code_keywords = {
            'criminal': ['crime', 'criminal', 'felony', 'misdemeanor', 'arrest', 'sentence', 'bail'],
            'civil': ['lawsuit', 'tort', 'negligence', 'contract', 'damages', 'liability'],
            'business': ['corporation', 'llc', 'partnership', 'securities', 'commercial'],
            'family': ['divorce', 'custody', 'child support', 'marriage', 'adoption'],
            'property': ['real estate', 'property', 'deed', 'zoning', 'landlord', 'tenant'],
            'administrative': ['agency', 'regulation', 'administrative', 'permit', 'license']
        }
        
        relevant = []
        for code_type, keywords in code_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant.append(code_type)
        
        # Default to civil if no specific match
        if not relevant:
            relevant = ['civil', 'criminal']
        
        return relevant

class GoogleScholarLegalAPI:
    """Google Scholar Legal Opinions - Free case law search"""
    
    def __init__(self):
        self.base_url = "https://scholar.google.com"
        self.rate_limiter = RateLimiter(10, 2)  # Very conservative for Google
    
    def search_case_law(self, query: str, state: str = None, court: str = None) -> List[Dict]:
        """Search Google Scholar for case law"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # Enhance query for legal search
            enhanced_query = f"{query} case law"
            if state:
                enhanced_query += f" {state}"
            if court:
                enhanced_query += f" {court}"
            
            # Google Scholar Legal search URL
            search_url = f"{self.base_url}/scholar"
            params = {
                'hl': 'en',
                'as_sdt': '2006',  # Legal documents and patents
                'q': enhanced_query
            }
            
            # Note: Google Scholar requires careful scraping
            # In production, you'd need to handle CAPTCHAs and rate limiting
            
            # For now, return structured search URL
            return [{
                'title': f'Google Scholar Legal Search - {query}',
                'url': f"{search_url}?{requests.compat.urlencode(params)}",
                'source_database': 'google_scholar_legal',
                'query': enhanced_query,
                'state': state,
                'court': court,
                'description': f'Search legal opinions and case law for: {query}',
                'access_note': 'Free access to case law and legal opinions',
                'search_tips': [
                    'Use specific case names for exact matches',
                    'Include jurisdiction (state/federal) for targeted results',
                    'Use legal terminology for better precision'
                ]
            }]
            
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
        
        return []
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Unified search method"""
        state = filters.get('state') if filters else None
        return self.search_case_law(query, state)

class LegalInformationAPI:
    """Aggregate multiple free legal APIs"""
    
    def __init__(self):
        self.cornell = CornellLegalAPI()
        self.openstates = OpenStatesAPI()
        self.justia = JustiaLegalAPI()
        self.google_scholar = GoogleScholarLegalAPI()
    
    def comprehensive_state_search(self, state: str, query: str) -> Dict:
        """Search all available state law sources"""
        results = {
            'state': state,
            'query': query,
            'search_date': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Search state codes
        try:
            cornell_codes = self.cornell.search_state_code(state, query)
            justia_codes = self.justia.search_state_law(state, query, 'codes')
            results['sources']['state_codes'] = cornell_codes + justia_codes
        except Exception as e:
            logger.error(f"State code search failed: {e}")
            results['sources']['state_codes'] = []
        
        # Search state case law
        try:
            justia_cases = self.justia.search_case_law(state, query)
            scholar_cases = self.google_scholar.search_case_law(query, state)
            results['sources']['case_law'] = justia_cases + scholar_cases
        except Exception as e:
            logger.error(f"Case law search failed: {e}")
            results['sources']['case_law'] = []
        
        # Search state legislation
        try:
            openstates_bills = self.openstates.search_bills(state, query)
            results['sources']['legislation'] = openstates_bills
        except Exception as e:
            logger.error(f"Legislation search failed: {e}")
            results['sources']['legislation'] = []
        
        # Search state legislators (for authorship/sponsorship info)
        try:
            legislators = self.openstates.search_legislators(state)
            results['sources']['legislators'] = legislators
        except Exception as e:
            logger.error(f"Legislator search failed: {e}")
            results['sources']['legislators'] = []
        
        # Calculate totals
        total_results = sum(len(v) for v in results['sources'].values() if isinstance(v, list))
        results['total_results'] = total_results
        
        # Add search recommendations
        results['recommendations'] = self._generate_search_recommendations(state, query, results)
        
        return results
    
    def search_federal_law(self, query: str, law_type: str = "all") -> Dict:
        """Search federal law sources"""
        results = {
            'query': query,
            'law_type': law_type,
            'search_date': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Search USC (United States Code)
        if law_type in ['all', 'usc', 'statutes']:
            try:
                usc_results = self.cornell.search_federal_code(query, 'usc')
                results['sources']['usc'] = usc_results
            except Exception as e:
                logger.error(f"USC search failed: {e}")
                results['sources']['usc'] = []
        
        # Search CFR (Code of Federal Regulations)
        if law_type in ['all', 'cfr', 'regulations']:
            try:
                cfr_results = self.cornell.search_federal_code(query, 'cfr')
                results['sources']['cfr'] = cfr_results
            except Exception as e:
                logger.error(f"CFR search failed: {e}")
                results['sources']['cfr'] = []
        
        # Search federal case law
        if law_type in ['all', 'cases']:
            try:
                federal_cases = self.google_scholar.search_case_law(query, court="federal")
                results['sources']['federal_cases'] = federal_cases
            except Exception as e:
                logger.error(f"Federal case search failed: {e}")
                results['sources']['federal_cases'] = []
        
        total_results = sum(len(v) for v in results['sources'].values() if isinstance(v, list))
        results['total_results'] = total_results
        
        return results
    
    def _generate_search_recommendations(self, state: str, query: str, results: Dict) -> List[str]:
        """Generate search improvement recommendations"""
        recommendations = []
        
        total_results = results.get('total_results', 0)
        
        if total_results == 0:
            recommendations.extend([
                f"Try broader search terms for {state} law",
                "Search federal law if state law doesn't cover this topic",
                "Consider searching related legal concepts",
                "Check if this is covered by municipal/local law instead"
            ])
        elif total_results < 5:
            recommendations.extend([
                "Try adding synonyms or related legal terms",
                f"Search neighboring states for similar laws",
                "Consider federal law alternatives"
            ])
        else:
            recommendations.extend([
                "Review state code sections for exact requirements",
                "Check case law for interpretations and precedents",
                "Look at recent legislation for updates"
            ])
        
        return recommendations

class StateLawSearchService:
    """Main service class for state law searches"""
    
    def __init__(self):
        self.legal_api = LegalInformationAPI()
    
    def search_state_specific(self, state: str, query: str, search_type: str = "comprehensive") -> Dict:
        """Main entry point for state-specific legal searches"""
        
        # Detect query type for targeted search
        query_indicators = {
            'statutes': ['statute', 'code section', 'RCW', 'USC', 'law'],
            'cases': ['case', 'v.', 'versus', 'court', 'decision', 'ruling'],
            'regulations': ['regulation', 'rule', 'CFR', 'administrative'],
            'bills': ['bill', 'HB', 'SB', 'legislation', 'sponsor']
        }
        
        detected_types = []
        query_lower = query.lower()
        for search_type_key, indicators in query_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_types.append(search_type_key)
        
        if not detected_types:
            detected_types = ['statutes', 'cases']  # Default
        
        logger.info(f"Detected search types for '{query}': {detected_types}")
        
        if search_type == "comprehensive":
            return self.legal_api.comprehensive_state_search(state, query)
        else:
            # Targeted search based on detected type
            results = {'state': state, 'query': query, 'sources': {}}
            
            for search_type_detected in detected_types:
                if search_type_detected == 'statutes':
                    results['sources']['statutes'] = self.legal_api.cornell.search_state_code(state, query)
                elif search_type_detected == 'cases':
                    results['sources']['cases'] = self.legal_api.justia.search_case_law(state, query)
                elif search_type_detected == 'bills':
                    results['sources']['bills'] = self.legal_api.openstates.search_bills(state, query)
            
            return results
    
    def search_multi_state(self, states: List[str], query: str) -> Dict:
        """Search multiple states for comparison"""
        results = {
            'query': query,
            'states': states,
            'search_date': datetime.now().isoformat(),
            'state_results': {}
        }
        
        for state in states:
            try:
                state_result = self.search_state_specific(state, query, "comprehensive")
                results['state_results'][state] = state_result
            except Exception as e:
                logger.error(f"Multi-state search failed for {state}: {e}")
                results['state_results'][state] = {'error': str(e)}
        
        # Add comparison analysis
        results['comparison'] = self._analyze_state_differences(results['state_results'])
        
        return results
    
    def _analyze_state_differences(self, state_results: Dict) -> Dict:
        """Analyze differences between state laws"""
        analysis = {
            'similarities': [],
            'differences': [],
            'unique_provisions': {},
            'common_patterns': []
        }
        
        # Simple analysis - count result types by state
        for state, result in state_results.items():
            if 'error' not in result:
                source_counts = {k: len(v) for k, v in result.get('sources', {}).items() if isinstance(v, list)}
                analysis['unique_provisions'][state] = source_counts
        
        return analysis

# Global instance
state_law_service = StateLawSearchService()
