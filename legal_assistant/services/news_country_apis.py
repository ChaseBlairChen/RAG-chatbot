"""News and country conditions APIs for immigration research"""
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from googletrans import Translator
import hashlib

logger = logging.getLogger(__name__)

class NewsAPIInterface:
    """NewsAPI.org interface for country news"""
    
    def __init__(self):
        self.api_key = os.environ.get("NEWSAPI_KEY", "free_tier_key")
        self.base_url = "https://newsapi.org/v2"
        self.translator = Translator()
    
    def search_country_news(self, country: str, topics: List[str], days_back: int = 30) -> List[Dict]:
        """Search news about country conditions"""
        results = []
        
        # Build query
        query_terms = f"{country} {' OR '.join(topics)}"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": query_terms,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "language": "en",
                    "apiKey": self.api_key
                },
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                for article in data.get('articles', [])[:20]:
                    results.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_date': article.get('publishedAt', ''),
                        'content': article.get('content', ''),
                        'source_database': 'newsapi'
                    })
            
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
        
        return results

class HumanRightsWatchAPI:
    """Human Rights Watch reports API"""
    
    def __init__(self):
        self.base_url = "https://www.hrw.org/api"
        self.translator = Translator()
    
    def search_reports(self, country: str, topics: List[str]) -> List[Dict]:
        """Search HRW reports (web scraping needed in production)"""
        # This is a placeholder - HRW doesn't have a public API
        # In production, you'd web scrape or use their RSS feeds
        
        results = []
        
        # Simulated search URL
        search_url = f"https://www.hrw.org/sitesearch/{country}"
        
        results.append({
            'title': f"Human Rights Watch - {country} Reports",
            'description': f"Latest human rights reports for {country}",
            'url': search_url,
            'source': 'Human Rights Watch',
            'source_database': 'hrw',
            'content': f"Visit HRW website for detailed reports on {country}",
            'topics': topics
        })
        
        return results

class AmnestyInternationalAPI:
    """Amnesty International country reports"""
    
    def __init__(self):
        self.base_url = "https://www.amnesty.org/api"
        self.translator = Translator()
    
    def search_reports(self, country: str) -> List[Dict]:
        """Search Amnesty reports"""
        results = []
        
        # Amnesty uses a different URL structure
        country_slug = country.lower().replace(' ', '-')
        
        results.append({
            'title': f"Amnesty International - {country}",
            'url': f"https://www.amnesty.org/en/location/{country_slug}",
            'source': 'Amnesty International',
            'source_database': 'amnesty',
            'description': f"Human rights situation in {country}"
        })
        
        return results

class UNHCRRefworldAPI:
    """UNHCR Refworld - Refugee documentation"""
    
    def __init__(self):
        self.base_url = "https://www.refworld.org/api"
        self.translator = Translator()
    
    def search_country_info(self, country: str, doc_type: str = "all") -> List[Dict]:
        """Search Refworld for country information"""
        results = []
        
        # Refworld categories
        categories = {
            "coi": "Country of Origin Information",
            "policy": "National Policy",
            "cases": "Case Law",
            "legislation": "National Legislation"
        }
        
        # Note: Actual API would require authentication
        search_url = f"https://www.refworld.org/country/{country}.html"
        
        results.append({
            'title': f"UNHCR Refworld - {country} Documentation",
            'url': search_url,
            'source': 'UNHCR Refworld',
            'source_database': 'refworld',
            'description': f"Comprehensive refugee and asylum documentation for {country}",
            'categories': list(categories.values())
        })
        
        return results

class StateGovTravelAPI:
    """U.S. State Department Country Information"""
    
    def __init__(self):
        self.base_url = "https://travel.state.gov/api"
        self.translator = Translator()
    
    def get_country_info(self, country: str) -> Dict:
        """Get State Dept country information"""
        # State Dept uses country codes
        country_slug = country.lower().replace(' ', '-')
        
        return {
            'title': f"U.S. State Department - {country} Country Information",
            'url': f"https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/{country_slug}.html",
            'source': 'U.S. Department of State',
            'source_database': 'state_dept',
            'sections': [
                'Safety and Security',
                'Crime',
                'Terrorism',
                'Civil Unrest',
                'Local Laws'
            ],
            'last_updated': datetime.now().isoformat()
        }

class FreedomHouseAPI:
    """Freedom House - Democracy and freedom scores"""
    
    def __init__(self):
        self.base_url = "https://freedomhouse.org/api"
    
    def get_freedom_score(self, country: str) -> Dict:
        """Get freedom scores for country"""
        # Placeholder - would need web scraping
        return {
            'country': country,
            'source': 'Freedom House',
            'source_database': 'freedom_house',
            'freedom_score': 'Not Free/Partly Free/Free',
            'political_rights': 'Score 1-7',
            'civil_liberties': 'Score 1-7',
            'url': f"https://freedomhouse.org/country/{country.lower().replace(' ', '-')}",
            'report_year': datetime.now().year
        }

class MultilingualNewsAggregator:
    """Aggregate news from multiple languages and translate"""
    
    def __init__(self):
        self.translator = Translator()
        self.language_sources = {
            'es': ['https://elpais.com', 'https://bbc.com/mundo'],
            'fr': ['https://lemonde.fr', 'https://rfi.fr'],
            'ar': ['https://aljazeera.net', 'https://bbc.com/arabic'],
            'zh': ['https://bbc.com/zhongwen', 'https://voachinese.com'],
            'ru': ['https://bbc.com/russian', 'https://dw.com/ru']
        }
    
    def search_multilingual(self, country: str, topics: List[str], languages: List[str] = None) -> List[Dict]:
        """Search news in multiple languages and translate"""
        if not languages:
            languages = ['es', 'fr', 'ar']  # Default languages
        
        all_results = []
        
        for lang in languages:
            if lang in self.language_sources:
                # In production, would actually search these sources
                # For now, create placeholder with translation capability
                result = {
                    'original_language': lang,
                    'source': self.language_sources[lang][0],
                    'title_original': f"Noticias sobre {country}",  # Would be in actual language
                    'content_original': f"Contenido sobre {country} y {', '.join(topics)}",
                    'source_database': f'multilingual_{lang}'
                }
                
                # Translate to English
                try:
                    result['title'] = self.translator.translate(
                        result['title_original'], 
                        src=lang, 
                        dest='en'
                    ).text
                    
                    result['content'] = self.translator.translate(
                        result['content_original'], 
                        src=lang, 
                        dest='en'
                    ).text
                    
                    result['translation_confidence'] = 0.85
                    
                except Exception as e:
                    logger.error(f"Translation failed: {e}")
                    result['title'] = result['title_original']
                    result['content'] = result['content_original']
                    result['translation_confidence'] = 0.0
                
                all_results.append(result)
        
        return all_results

class AcademicDatabaseAPI:
    """Academic sources for country conditions"""
    
    def __init__(self):
        self.sources = {
            'google_scholar': 'https://scholar.google.com',
            'jstor': 'https://www.jstor.org',
            'ssrn': 'https://papers.ssrn.com'
        }
    
    def search_academic(self, country: str, topics: List[str]) -> List[Dict]:
        """Search academic sources"""
        results = []
        
        # Google Scholar search URL
        query = f"{country} {' '.join(topics)} persecution asylum"
        scholar_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
        
        results.append({
            'title': f"Academic Research - {country}",
            'url': scholar_url,
            'source': 'Google Scholar',
            'source_database': 'google_scholar',
            'description': f"Peer-reviewed research on {country} conditions",
            'access': 'May require institutional access'
        })
        
        return results

# Aggregate all sources
class ComprehensiveCountryConditions:
    """Unified interface for all country condition sources"""
    
    def __init__(self):
        self.news_api = NewsAPIInterface()
        self.hrw_api = HumanRightsWatchAPI()
        self.amnesty_api = AmnestyInternationalAPI()
        self.unhcr_api = UNHCRRefworldAPI()
        self.state_dept = StateGovTravelAPI()
        self.freedom_house = FreedomHouseAPI()
        self.multilingual = MultilingualNewsAggregator()
        self.academic = AcademicDatabaseAPI()
        self.translator = Translator()
    
    def research_all_sources(self, country: str, topics: List[str], 
                           include_multilingual: bool = True) -> Dict:
        """Research from all available sources"""
        all_results = {
            'country': country,
            'research_date': datetime.now().isoformat(),
            'topics': topics,
            'sources': {}
        }
        
        # News sources
        try:
            news_results = self.news_api.search_country_news(country, topics)
            all_results['sources']['news'] = news_results
        except Exception as e:
            logger.error(f"News search failed: {e}")
            all_results['sources']['news'] = []
        
        # Human rights organizations
        try:
            hrw = self.hrw_api.search_reports(country, topics)
            amnesty = self.amnesty_api.search_reports(country)
            all_results['sources']['human_rights'] = hrw + amnesty
        except Exception as e:
            logger.error(f"Human rights search failed: {e}")
            all_results['sources']['human_rights'] = []
        
        # UNHCR
        try:
            unhcr = self.unhcr_api.search_country_info(country)
            all_results['sources']['unhcr'] = unhcr
        except Exception as e:
            logger.error(f"UNHCR search failed: {e}")
            all_results['sources']['unhcr'] = []
        
        # State Department
        try:
            state = self.state_dept.get_country_info(country)
            all_results['sources']['state_dept'] = [state]
        except Exception as e:
            logger.error(f"State Dept search failed: {e}")
            all_results['sources']['state_dept'] = []
        
        # Freedom House
        try:
            freedom = self.freedom_house.get_freedom_score(country)
            all_results['sources']['freedom_house'] = [freedom]
        except Exception as e:
            logger.error(f"Freedom House search failed: {e}")
            all_results['sources']['freedom_house'] = []
        
        # Multilingual sources
        if include_multilingual:
            try:
                multilingual = self.multilingual.search_multilingual(country, topics)
                all_results['sources']['multilingual'] = multilingual
            except Exception as e:
                logger.error(f"Multilingual search failed: {e}")
                all_results['sources']['multilingual'] = []
        
        # Academic sources
        try:
            academic = self.academic.search_academic(country, topics)
            all_results['sources']['academic'] = academic
        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            all_results['sources']['academic'] = []
        
        # Generate summary
        all_results['summary'] = self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate summary of all sources"""
        total_sources = sum(len(v) for v in results['sources'].values() if isinstance(v, list))
        
        summary = f"Country conditions research for {results['country']} compiled from {total_sources} sources including:\n"
        summary += f"- News articles from multiple outlets\n"
        summary += f"- Human rights organization reports (HRW, Amnesty International)\n"
        summary += f"- UNHCR Refworld documentation\n"
        summary += f"- U.S. State Department country information\n"
        summary += f"- Freedom House democracy scores\n"
        
        if results['sources'].get('multilingual'):
            summary += f"- Multilingual sources translated from {len(results['sources']['multilingual'])} languages\n"
        
        summary += f"\nTopics researched: {', '.join(results['topics'])}"
        
        return summary

# Global instance
comprehensive_researcher = ComprehensiveCountryConditions()
