# legal_assistant/services/comprehensive_legal_apis.py
"""Comprehensive integration of all available free legal APIs"""
import requests
import logging
import re
import time
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from bs4 import BeautifulSoup
import os

logger = logging.getLogger(__name__)

# API Keys from your registrations
CONGRESS_API_KEY = "7J5Bfj6i0F3tg4VZleZ4SyQmVbG0QyIM9tPMQA2M"
DATA_GOV_API_KEY = "yZAV2yQIyyVzDYCHCw39CUBx98HDQQmHjd9wojRe"

class RateLimiter:
    def __init__(self, requests_per_hour: int = 100, requests_per_minute: int = 10):
        self.requests_per_hour = requests_per_hour
        self.requests_per_minute = requests_per_minute
        self.hourly_calls = []
        self.minute_calls = []
    
    def wait_if_needed(self):
        now = time.time()
        
        # Clean old calls
        self.hourly_calls = [t for t in self.hourly_calls if now - t < 3600]
        self.minute_calls = [t for t in self.minute_calls if now - t < 60]
        
        # Check limits
        if len(self.minute_calls) >= self.requests_per_minute:
            time.sleep(1)
        elif len(self.hourly_calls) >= self.requests_per_hour:
            time.sleep(10)
        
        self.hourly_calls.append(now)
        self.minute_calls.append(now)

class EnvironmentalLawAPI:
    """EPA and environmental law APIs"""
    
    def __init__(self):
        self.epa_base = "https://www.epa.gov/enviro/envirofacts-data-service-api"
        self.air_quality_base = "https://www.airnowapi.org/aq"
        self.water_quality_base = "https://www.waterqualitydata.us"
        self.rate_limiter = RateLimiter(200, 20)
    
    def search_environmental_violations(self, facility_name: str = None, state: str = None, 
                                     violation_type: str = None) -> List[Dict]:
        """Search EPA environmental violations"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # EPA Enforcement and Compliance History Online (ECHO)
            echo_url = "https://echo.epa.gov/tools/web-services/facility-search"
            
            params = {
                "output": "json",
                "responseset": "1"
            }
            
            if facility_name:
                params["p_fn"] = facility_name
            if state:
                params["p_st"] = state
            if violation_type:
                params["p_act"] = violation_type  # CAA, CWA, RCRA, etc.
            
            response = requests.get(echo_url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                data = response.json()
                results = []
                
                for facility in data.get('Results', {}).get('Facilities', [])[:20]:
                    results.append({
                        'facility_name': facility.get('FacName', ''),
                        'location': f"{facility.get('FacCity', '')}, {facility.get('FacState', '')}",
                        'violations': facility.get('CurrViol', ''),
                        'enforcement_actions': facility.get('Enforcements', ''),
                        'program': facility.get('Programs', ''),
                        'inspection_date': facility.get('LastInsp', ''),
                        'compliance_status': facility.get('ComplianceStatus', ''),
                        'source_database': 'epa_echo',
                        'url': f"https://echo.epa.gov/detailed-facility-report?fid={facility.get('RegistryID', '')}"
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"EPA ECHO search failed: {e}")
        
        return []
    
    def search_air_quality_data(self, state: str, county: str = None) -> List[Dict]:
        """Search air quality data"""
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {
                "format": "json",
                "api_key": DATA_GOV_API_KEY
            }
            
            if state:
                params["state"] = state
            if county:
                params["county"] = county
            
            response = requests.get(f"{self.air_quality_base}/forecast/state", 
                                  params=params, timeout=10)
            
            if response.ok:
                data = response.json()
                return [{
                    'title': f'Air Quality Data - {state}',
                    'data': data,
                    'source_database': 'epa_air_quality',
                    'type': 'environmental_data'
                }]
                
        except Exception as e:
            logger.error(f"Air quality search failed: {e}")
        
        return []

class ImmigrationLawAPI:
    """Immigration law and data APIs"""
    
    def __init__(self):
        self.uscis_base = "https://egov.uscis.gov"
        self.state_dept_base = "https://travel.state.gov"
        self.rate_limiter = RateLimiter(100, 10)
    
    def check_case_status(self, receipt_number: str) -> Dict:
        """Check USCIS case status"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # USCIS case status check
            url = f"{self.uscis_base}/casestatus/mycasestatus.do"
            data = {"appReceiptNum": receipt_number}
            
            response = requests.post(url, data=data, timeout=10,
                                   headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                # Parse HTML response for case status
                soup = BeautifulSoup(response.text, 'html.parser')
                status_element = soup.find('div', class_='current-status-sec')
                
                if status_element:
                    return {
                        'receipt_number': receipt_number,
                        'status': status_element.get_text().strip(),
                        'source_database': 'uscis_case_status',
                        'checked_date': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"USCIS case status check failed: {e}")
        
        return {}
    
    def get_visa_bulletin_data(self) -> Dict:
        """Get current visa bulletin data"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # State Department visa bulletin
            url = f"{self.state_dept_base}/content/travel/en/legal/visa-law0/visa-bulletin.html"
            
            response = requests.get(url, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                return {
                    'title': 'Current Visa Bulletin',
                    'url': url,
                    'source_database': 'state_dept_visa_bulletin',
                    'last_updated': datetime.now().isoformat(),
                    'description': 'Current priority dates and visa availability'
                }
                
        except Exception as e:
            logger.error(f"Visa bulletin fetch failed: {e}")
        
        return {}

class HousingLawAPI:
    """Housing and homelessness law APIs"""
    
    def __init__(self):
        self.hud_base = "https://www.huduser.gov/hudapi/public"
        self.census_base = "https://api.census.gov/data"
        self.rate_limiter = RateLimiter(150, 15)
    
    def search_fair_market_rents(self, state: str, county: str = None) -> List[Dict]:
        """Search HUD Fair Market Rents"""
        self.rate_limiter.wait_if_needed()
        
        try:
            url = f"{self.hud_base}/fmr/data/{state}"
            if county:
                url += f"/{county}"
            
            params = {"format": "json"}
            
            response = requests.get(url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                data = response.json()
                results = []
                
                for fmr in data.get('data', [])[:10]:
                    results.append({
                        'area_name': fmr.get('area_name', ''),
                        'state': state,
                        'fmr_0br': fmr.get('fmr_0br', ''),
                        'fmr_1br': fmr.get('fmr_1br', ''),
                        'fmr_2br': fmr.get('fmr_2br', ''),
                        'fmr_3br': fmr.get('fmr_3br', ''),
                        'fmr_4br': fmr.get('fmr_4br', ''),
                        'year': fmr.get('year', ''),
                        'source_database': 'hud_fair_market_rents'
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"HUD FMR search failed: {e}")
        
        return []
    
    def search_housing_data(self, state: str, data_type: str = "demographics") -> List[Dict]:
        """Search Census housing data"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # American Community Survey housing data
            year = datetime.now().year - 1  # Use most recent complete year
            dataset = f"{year}/acs/acs1"
            
            # Housing-related variables
            housing_vars = {
                'demographics': ['B25001_001E', 'B25002_001E', 'B25003_001E'],  # Total housing units
                'costs': ['B25070_001E', 'B25091_001E'],  # Housing costs
                'occupancy': ['B25004_001E', 'B25014_001E']  # Vacancy status
            }
            
            variables = housing_vars.get(data_type, housing_vars['demographics'])
            
            params = {
                "get": ",".join(variables + ["NAME"]),
                "for": f"state:{self._get_state_fips(state)}",
                "key": DATA_GOV_API_KEY
            }
            
            response = requests.get(f"{self.census_base}/{dataset}", 
                                  params=params, timeout=10)
            
            if response.ok:
                data = response.json()
                if len(data) > 1:  # First row is headers
                    return [{
                        'state': state,
                        'housing_data': dict(zip(data[0], data[1])),
                        'data_type': data_type,
                        'year': year,
                        'source_database': 'census_housing'
                    }]
                    
        except Exception as e:
            logger.error(f"Census housing search failed: {e}")
        
        return []
    
    def _get_state_fips(self, state_name: str) -> str:
        """Get FIPS code for state"""
        state_fips = {
            'alabama': '01', 'alaska': '02', 'arizona': '04', 'arkansas': '05',
            'california': '06', 'colorado': '08', 'connecticut': '09', 'delaware': '10',
            'florida': '12', 'georgia': '13', 'hawaii': '15', 'idaho': '16',
            'illinois': '17', 'indiana': '18', 'iowa': '19', 'kansas': '20',
            'kentucky': '21', 'louisiana': '22', 'maine': '23', 'maryland': '24',
            'massachusetts': '25', 'michigan': '26', 'minnesota': '27', 'mississippi': '28',
            'missouri': '29', 'montana': '30', 'nebraska': '31', 'nevada': '32',
            'new hampshire': '33', 'new jersey': '34', 'new mexico': '35', 'new york': '36',
            'north carolina': '37', 'north dakota': '38', 'ohio': '39', 'oklahoma': '40',
            'oregon': '41', 'pennsylvania': '42', 'rhode island': '44', 'south carolina': '45',
            'south dakota': '46', 'tennessee': '47', 'texas': '48', 'utah': '49',
            'vermont': '50', 'virginia': '51', 'washington': '53', 'west virginia': '54',
            'wisconsin': '55', 'wyoming': '56'
        }
        return state_fips.get(state_name.lower(), '06')  # Default to California

class BusinessLawAPI:
    """Corporate and business law APIs"""
    
    def __init__(self):
        self.sec_base = "https://data.sec.gov/api"
        self.sba_base = "https://api.sba.gov"
        self.rate_limiter = RateLimiter(100, 10)
    
    def search_sec_filings(self, company_name: str, filing_type: str = None) -> List[Dict]:
        """Search SEC EDGAR filings"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # First get company CIK
            cik = self._get_company_cik(company_name)
            if not cik:
                return []
            
            # Get company filings
            url = f"https://data.sec.gov/submissions/CIK{cik:0>10}.json"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'LegalAssistant legal-research@example.com',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'data.sec.gov'
            })
            
            if response.ok:
                data = response.json()
                results = []
                
                recent_filings = data.get('filings', {}).get('recent', {})
                forms = recent_filings.get('form', [])
                filing_dates = recent_filings.get('filingDate', [])
                accession_numbers = recent_filings.get('accessionNumber', [])
                
                for i in range(min(10, len(forms))):
                    if not filing_type or forms[i] == filing_type:
                        results.append({
                            'company': company_name,
                            'form_type': forms[i],
                            'filing_date': filing_dates[i],
                            'accession_number': accession_numbers[i],
                            'url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_numbers[i].replace('-', '')}/{accession_numbers[i]}-index.htm",
                            'source_database': 'sec_edgar'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"SEC search failed: {e}")
        
        return []
    
    def _get_company_cik(self, company_name: str) -> Optional[int]:
        """Get company CIK from name"""
        try:
            # Search company tickers file
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'LegalAssistant legal-research@example.com'
            })
            
            if response.ok:
                data = response.json()
                company_lower = company_name.lower()
                
                for company_info in data.values():
                    if company_lower in company_info.get('title', '').lower():
                        return company_info.get('cik_str')
                        
        except Exception as e:
            logger.error(f"CIK lookup failed: {e}")
        
        return None

class LaborEmploymentAPI:
    """Department of Labor and employment law APIs"""
    
    def __init__(self):
        self.dol_base = "https://developer.dol.gov/health-and-safety"
        self.osha_base = "https://www.osha.gov/api"
        self.bls_base = "https://api.bls.gov/publicAPI/v2"
        self.rate_limiter = RateLimiter(150, 15)
    
    def search_osha_violations(self, company_name: str = None, state: str = None) -> List[Dict]:
        """Search OSHA violations and citations"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # OSHA enforcement data
            url = f"{self.dol_base}/consumerComplaintSearch"
            
            params = {
                "format": "json",
                "top": "50"
            }
            
            if company_name:
                params["estab_name"] = company_name
            if state:
                params["state"] = state
            
            response = requests.get(url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                data = response.json()
                results = []
                
                for violation in data.get('dataset', [])[:20]:
                    results.append({
                        'company': violation.get('estab_name', ''),
                        'violation_type': violation.get('violation_type', ''),
                        'citation_date': violation.get('issuance_date', ''),
                        'penalty': violation.get('current_penalty', ''),
                        'standard_violated': violation.get('standard', ''),
                        'description': violation.get('description', ''),
                        'state': violation.get('state', ''),
                        'source_database': 'dol_osha'
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"OSHA search failed: {e}")
        
        return []
    
    def search_wage_hour_violations(self, company_name: str = None, state: str = None) -> List[Dict]:
        """Search DOL Wage and Hour violations"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # DOL WHD (Wage and Hour Division) data
            url = "https://enforcedata.dol.gov/views/data_catalogs.php"
            
            # This would require scraping or finding the proper API endpoint
            # For now, return a structured search URL
            return [{
                'title': f'DOL Wage & Hour Violations Search - {company_name or state or "All"}',
                'url': f"https://enforcedata.dol.gov/views/data_summary.php",
                'company': company_name,
                'state': state,
                'source_database': 'dol_wage_hour',
                'description': 'Department of Labor wage and hour violation data'
            }]
            
        except Exception as e:
            logger.error(f"DOL search failed: {e}")
        
        return []

class IntellectualPropertyAPI:
    """USPTO patent and trademark APIs"""
    
    def __init__(self):
        self.uspto_base = "https://developer.uspto.gov/api-catalog"
        self.patent_base = "https://patents.uspto.gov/api"
        self.trademark_base = "https://tsdrapi.uspto.gov"
        self.rate_limiter = RateLimiter(200, 20)
    
    def search_patents(self, query: str, inventor: str = None) -> List[Dict]:
        """Search USPTO patents"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # Use Google Patents as primary source (free and comprehensive)
            google_patents_url = "https://patents.google.com/xhr/query"
            
            params = {
                "url": quote(f"q={query}"),
                "exp": ""
            }
            
            if inventor:
                params["url"] = quote(f"q={query} inventor:{inventor}")
            
            # For now, provide structured search URLs since Google Patents API is complex
            results = [{
                'title': f'Patent Search: {query}',
                'url': f"https://patents.google.com/?q={quote(query)}",
                'search_query': query,
                'inventor': inventor,
                'source_database': 'google_patents',
                'description': f'Search patents for: {query}',
                'access': 'Free full-text patent search'
            }]
            
            # Also provide USPTO search URL
            results.append({
                'title': f'USPTO Patent Search: {query}',
                'url': f"https://ppubs.uspto.gov/pubwebapp/static/pages/ppubsbasic.html?term={quote(query)}",
                'source_database': 'uspto_patents',
                'description': 'Official USPTO patent database',
                'access': 'Free official patent search'
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Patent search failed: {e}")
        
        return []
    
    def search_trademarks(self, mark: str, owner: str = None) -> List[Dict]:
        """Search USPTO trademarks"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # USPTO Trademark Electronic Search System (TESS)
            tess_url = f"https://tmsearch.uspto.gov/bin/gate.exe?f=searchss&state=4806:1e7v2n.1.1"
            
            return [{
                'title': f'Trademark Search: {mark}',
                'url': f"https://tmsearch.uspto.gov/search/search-information",
                'mark': mark,
                'owner': owner,
                'source_database': 'uspto_trademarks',
                'description': f'Search trademarks for: {mark}',
                'access': 'Free official trademark search'
            }]
            
        except Exception as e:
            logger.error(f"Trademark search failed: {e}")
        
        return []

class HealthcareLawAPI:
    """FDA and healthcare law APIs"""
    
    def __init__(self):
        self.fda_base = "https://api.fda.gov"
        self.cms_base = "https://data.cms.gov/api"
        self.rate_limiter = RateLimiter(120, 12)
    
    def search_fda_enforcement(self, product_type: str = "drug", 
                             date_range: int = 30) -> List[Dict]:
        """Search FDA enforcement actions"""
        self.rate_limiter.wait_if_needed()
        
        try:
            endpoint_map = {
                'drug': 'drug/enforcement',
                'device': 'device/enforcement', 
                'food': 'food/enforcement'
            }
            
            endpoint = endpoint_map.get(product_type, 'drug/enforcement')
            url = f"{self.fda_base}/{endpoint}.json"
            
            # Search recent enforcement actions
            params = {
                "search": f"report_date:[{(datetime.now() - timedelta(days=date_range)).strftime('%Y%m%d')} TO {datetime.now().strftime('%Y%m%d')}]",
                "limit": 20
            }
            
            response = requests.get(url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                data = response.json()
                results = []
                
                for enforcement in data.get('results', []):
                    results.append({
                        'product_description': enforcement.get('product_description', ''),
                        'reason_for_recall': enforcement.get('reason_for_recall', ''),
                        'classification': enforcement.get('classification', ''),
                        'company': enforcement.get('recalling_firm', ''),
                        'report_date': enforcement.get('report_date', ''),
                        'status': enforcement.get('status', ''),
                        'source_database': f'fda_{product_type}_enforcement'
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"FDA enforcement search failed: {e}")
        
        return []

class CriminalJusticeAPI:
    """FBI and criminal justice data APIs"""
    
    def __init__(self):
        self.fbi_base = "https://api.usa.gov/crime/fbi/cde"
        self.bjs_base = "https://bjs.ojp.gov/api"
        self.rate_limiter = RateLimiter(100, 10)
    
    def search_crime_data(self, state: str = None, offense_type: str = None) -> List[Dict]:
        """Search FBI crime data"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # FBI Crime Data Explorer API
            url = f"{self.fbi_base}/estimate/state"
            
            params = {
                "API_KEY": DATA_GOV_API_KEY,
                "format": "json"
            }
            
            if state:
                params["state_abbr"] = self._get_state_abbr(state)
            if offense_type:
                params["offense"] = offense_type
            
            response = requests.get(url, params=params, timeout=10,
                                  headers={'User-Agent': 'LegalAssistant/1.0'})
            
            if response.ok:
                data = response.json()
                return [{
                    'title': f'FBI Crime Statistics - {state or "National"}',
                    'data': data.get('results', []),
                    'state': state,
                    'offense_type': offense_type,
                    'source_database': 'fbi_crime_data'
                }]
                
        except Exception as e:
            logger.error(f"FBI crime data search failed: {e}")
        
        return []
    
    def _get_state_abbr(self, state_name: str) -> str:
        """Get state abbreviation"""
        state_abbrs = {
            'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
            'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
            'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
            'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
            'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
            'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
            'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
            'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
            'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
            'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
            'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
            'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
            'wisconsin': 'WI', 'wyoming': 'WY'
        }
        return state_abbrs.get(state_name.lower(), 'CA')

class InternationalLawAPI:
    """International law and treaty APIs"""
    
    def __init__(self):
        self.un_base = "https://treaties.un.org/api"
        self.worldbank_base = "https://api.worldbank.org/v2"
        self.rate_limiter = RateLimiter(100, 10)
    
    def search_treaties(self, topic: str, country: str = None) -> List[Dict]:
        """Search UN treaty collection"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # UN Treaty Collection search
            params = {
                "format": "json",
                "query": topic
            }
            
            if country:
                params["participant"] = country
            
            # Note: UN Treaty API may require different authentication
            # For now, provide search URL
            return [{
                'title': f'UN Treaty Collection - {topic}',
                'url': f"https://treaties.un.org/pages/AdvanceSearch.aspx?tab=UNTS&clang=_en",
                'topic': topic,
                'country': country,
                'source_database': 'un_treaties',
                'description': f'Search international treaties on: {topic}'
            }]
            
        except Exception as e:
            logger.error(f"UN treaty search failed: {e}")
        
        return []

class ComprehensiveLegalHub:
    """Unified hub for all free legal APIs"""
    
    def __init__(self):
        self.environmental = EnvironmentalLawAPI()
        self.immigration = ImmigrationLawAPI()
        self.housing = HousingLawAPI()
        self.business = BusinessLawAPI()
        self.labor = LaborEmploymentAPI()
        self.ip = IntellectualPropertyAPI()
        self.international = InternationalLawAPI()
        self.criminal = CriminalJusticeAPI()
    
    def intelligent_search(self, query: str, detected_state: str = None) -> Dict:
        """Intelligently route searches based on query content"""
        
        # Detect legal areas from query
        legal_areas = self._detect_legal_areas(query)
        detected_state = detected_state or self._detect_state_from_query(query)
        
        logger.info(f"Detected legal areas: {legal_areas}, State: {detected_state}")
        
        results = {
            'query': query,
            'detected_state': detected_state,
            'legal_areas': legal_areas,
            'search_date': datetime.now().isoformat(),
            'results_by_area': {}
        }
        
        # Search relevant APIs based on detected areas
        for area in legal_areas:
            try:
                if area == 'environmental':
                    env_results = self.environmental.search_environmental_violations(
                        state=detected_state
                    )
                    results['results_by_area']['environmental'] = env_results
                
                elif area == 'immigration':
                    # Check if query contains receipt number
                    receipt_match = re.search(r'[A-Z]{3}\d{10}', query)
                    if receipt_match:
                        case_status = self.immigration.check_case_status(receipt_match.group())
                        results['results_by_area']['immigration'] = [case_status] if case_status else []
                    else:
                        visa_data = self.immigration.get_visa_bulletin_data()
                        results['results_by_area']['immigration'] = [visa_data] if visa_data else []
                
                elif area == 'housing':
                    housing_results = self.housing.search_fair_market_rents(
                        state=detected_state or 'CA'
                    )
                    results['results_by_area']['housing'] = housing_results
                
                elif area == 'business':
                    # Extract company name if present
                    company_match = re.search(r'\b([A-Z][a-zA-Z\s&]+(?:Inc\.|Corp\.|LLC|Co\.))\b', query)
                    if company_match:
                        company_name = company_match.group(1)
                        sec_results = self.business.search_sec_filings(company_name)
                        results['results_by_area']['business'] = sec_results
                
                elif area == 'labor':
                    # Extract company name for OSHA search
                    company_match = re.search(r'\b([A-Z][a-zA-Z\s&]+(?:Inc\.|Corp\.|LLC|Co\.))\b', query)
                    company_name = company_match.group(1) if company_match else None
                    
                    osha_results = self.labor.search_osha_violations(
                        company_name=company_name,
                        state=detected_state
                    )
                    results['results_by_area']['labor'] = osha_results
                
                elif area == 'intellectual_property':
                    patent_results = self.ip.search_patents(query)
                    trademark_results = self.ip.search_trademarks(query)
                    results['results_by_area']['intellectual_property'] = patent_results + trademark_results
                
                elif area == 'criminal':
                    crime_results = self.criminal.search_crime_data(
                        state=detected_state
                    )
                    results['results_by_area']['criminal'] = crime_results
                
                elif area == 'international':
                    treaty_results = self.international.search_treaties(query)
                    results['results_by_area']['international'] = treaty_results
                    
            except Exception as e:
                logger.error(f"Search failed for area {area}: {e}")
                results['results_by_area'][area] = []
        
        # Calculate total results
        total_results = sum(
            len(v) for v in results['results_by_area'].values() 
            if isinstance(v, list)
        )
        results['total_results'] = total_results
        
        return results
    
    def _detect_legal_areas(self, query: str) -> List[str]:
        """Detect legal practice areas from query"""
        query_lower = query.lower()
        areas = []
        
        area_keywords = {
            'environmental': [
                'environmental', 'epa', 'pollution', 'clean air', 'clean water', 
                'hazardous waste', 'superfund', 'emissions', 'toxic', 'contamination'
            ],
            'immigration': [
                'immigration', 'visa', 'green card', 'asylum', 'deportation', 
                'uscis', 'naturalization', 'refugee', 'border', 'customs'
            ],
            'housing': [
                'housing', 'rental', 'landlord', 'tenant', 'eviction', 'fair housing',
                'discrimination', 'section 8', 'rent control', 'homelessness'
            ],
            'business': [
                'corporate', 'business', 'sec filing', 'securities', 'merger',
                'acquisition', 'ipo', 'quarterly report', 'annual report', '10-k', '10-q'
            ],
            'labor': [
                'employment', 'labor', 'osha', 'workplace safety', 'wage', 'overtime',
                'discrimination', 'harassment', 'workers compensation', 'union'
            ],
            'intellectual_property': [
                'patent', 'trademark', 'copyright', 'intellectual property', 'trade secret',
                'infringement', 'licensing', 'invention', 'brand'
            ],
            'criminal': [
                'criminal', 'crime', 'arrest', 'prosecution', 'sentencing', 'prison',
                'felony', 'misdemeanor', 'fbi', 'investigation'
            ],
            'international': [
                'international', 'treaty', 'foreign', 'diplomatic', 'trade agreement',
                'human rights', 'war crimes', 'international court'
            ]
        }
        
        for area, keywords in area_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                areas.append(area)
        
        # Default to general if no specific area detected
        if not areas:
            areas = ['general']
        
        return areas
    
    def _detect_state_from_query(self, query: str) -> Optional[str]:
        """Detect state from query text"""
        states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
            'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
            'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
            'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
            'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
            'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
            'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
        ]
        
        # Also check for state abbreviations and codes
        state_indicators = {
            'Washington': ['WA', 'Wash.', 'RCW', 'WAC'],
            'California': ['CA', 'Cal.', 'Calif.'],
            'New York': ['NY', 'N.Y.'],
            'Texas': ['TX', 'Tex.'],
            'Florida': ['FL', 'Fla.']
        }
        
        # Check for state codes first (more specific)
        for state, indicators in state_indicators.items():
            if any(indicator in query for indicator in indicators):
                return state
        
        # Check for full state names
        for state in states:
            if state.lower() in query.lower():
                return state
        
        return None

# Global instance
comprehensive_legal_hub = ComprehensiveLegalHub()

# Integration function for your external_db_service.py
def get_comprehensive_legal_apis() -> Dict:
    """Get all comprehensive legal APIs for integration"""
    return {
        'environmental_law': EnvironmentalLawAPI(),
        'immigration_law': ImmigrationLawAPI(),
        'housing_law': HousingLawAPI(),
        'business_law': BusinessLawAPI(),
        'labor_employment': LaborEmploymentAPI(),
        'intellectual_property': IntellectualPropertyAPI(),
        'healthcare_law': HealthcareLawAPI(),
        'criminal_justice': CriminalJusticeAPI(),
        'international_law': InternationalLawAPI(),
        'comprehensive_hub': comprehensive_legal_hub
    }

# Main search function to integrate with your existing system
def search_comprehensive_legal_databases(query: str, user=None, auto_detect_areas: bool = True) -> List[Dict]:
    """
    Main search function that intelligently searches all relevant legal APIs
    ADD THIS to your processors/query_processor.py external search section
    """
    
    try:
        if auto_detect_areas:
            # Use intelligent routing
            hub_results = comprehensive_legal_hub.intelligent_search(query)
            
            # Flatten results for consistent API response
            flattened_results = []
            for area, area_results in hub_results.get('results_by_area', {}).items():
                if isinstance(area_results, list):
                    for result in area_results:
                        result['legal_area'] = area
                        result['detected_state'] = hub_results.get('detected_state')
                        flattened_results.append(result)
            
            return flattened_results
        else:
            # Manual search across all APIs
            all_results = []
            detected_state = comprehensive_legal_hub._detect_state_from_query(query)
            
            # Search each API category
            apis = get_comprehensive_legal_apis()
            
            for api_name, api_instance in apis.items():
                if api_name == 'comprehensive_hub':
                    continue
                    
                try:
                    # Call appropriate search method based on API type
                    if hasattr(api_instance, 'search_environmental_violations'):
                        results = api_instance.search_environmental_violations(state=detected_state)
                    elif hasattr(api_instance, 'search_sec_filings'):
                        # Extract company name from query
                        company_match = re.search(r'\b([A-Z][a-zA-Z\s&]+(?:Inc\.|Corp\.|LLC))\b', query)
                        if company_match:
                            results = api_instance.search_sec_filings(company_match.group(1))
                        else:
                            results = []
                    elif hasattr(api_instance, 'search_osha_violations'):
                        results = api_instance.search_osha_violations(state=detected_state)
                    elif hasattr(api_instance, 'search_patents'):
                        results = api_instance.search_patents(query)
                    else:
                        results = []
                    
                    for result in results:
                        result['api_source'] = api_name
                        result['detected_state'] = detected_state
                    
                    all_results.extend(results)
                    
                except Exception as e:
                    logger.error(f"API {api_name} search failed: {e}")
            
            return all_results
            
    except Exception as e:
        logger.error(f"Comprehensive legal search failed: {e}")
        return []


# ADD THIS OPTIMIZED FUNCTION TO YOUR comprehensive_legal_apis.py

def search_comprehensive_legal_databases_optimized(query: str, user=None, max_results: int = 10) -> List[Dict]:
    """
    OPTIMIZED search function that limits API calls and results for faster responses
    REPLACE the existing search_comprehensive_legal_databases function with this
    """
    
    try:
        # Detect legal areas for smart routing
        legal_areas = comprehensive_legal_hub._detect_legal_areas(query)
        detected_state = comprehensive_legal_hub._detect_state_from_query(query)
        
        logger.info(f"OPTIMIZED: Detected areas: {legal_areas[:2]}, State: {detected_state}")  # Limit logging
        
        # Use smart routing to limit API calls
        if 'environmental' in legal_areas:
            # For environmental queries, prioritize fast government APIs
            logger.info("🌿 FAST Environmental search - limiting to key APIs")
            
            # Only search the most relevant and fastest APIs
            env_results = []
            
            # 1. Quick federal law search (fast)
            try:
                from ..services.external_db_service import search_external_databases
                fed_results = search_external_databases(query, ["congress_gov", "federal_register"], user)
                env_results.extend(fed_results[:3])  # Limit to top 3
                logger.info(f"Federal env results: {len(fed_results[:3])}")
            except Exception as e:
                logger.error(f"Federal env search failed: {e}")
            
            # 2. Only search EPA if we have few results (conditional)
            if len(env_results) < 2:
                try:
                    epa_results = comprehensive_legal_hub.environmental.search_environmental_violations(
                        state=detected_state
                    )
                    for result in epa_results[:3]:  # Limit EPA results
                        result['legal_area'] = 'environmental'
                        result['detected_state'] = detected_state
                    env_results.extend(epa_results[:3])
                    logger.info(f"EPA results: {len(epa_results[:3])}")
                except Exception as e:
                    logger.error(f"EPA search failed: {e}")
            
            return env_results[:max_results]
        
        elif 'immigration' in legal_areas:
            # For immigration, check for receipt number first
            logger.info("🗽 FAST Immigration search")
            
            receipt_match = re.search(r'[A-Z]{3}\d{10}', query)
            if receipt_match:
                # Direct USCIS case status lookup
                try:
                    case_status = comprehensive_legal_hub.immigration.check_case_status(receipt_match.group())
                    if case_status:
                        case_status['legal_area'] = 'immigration'
                        return [case_status]
                except Exception as e:
                    logger.error(f"USCIS status check failed: {e}")
            
            # Otherwise, quick federal immigration law search
            try:
                from ..services.external_db_service import search_external_databases
                imm_results = search_external_databases(query, ["congress_gov", "federal_register"], user)
                return imm_results[:max_results]
            except Exception as e:
                logger.error(f"Immigration search failed: {e}")
                return []
        
        else:
            # For general legal queries, use only fast APIs
            logger.info("⚖️ FAST General legal search")
            
            try:
                from ..services.external_db_service import search_external_databases
                # Use only the fastest, most reliable APIs
                fast_apis = ["congress_gov", "justia", "cornell_law"]
                results = search_external_databases(query, fast_apis, user)
                
                for result in results:
                    result['legal_area'] = 'general_legal'
                    result['detected_state'] = detected_state
                
                return results[:max_results]
            except Exception as e:
                logger.error(f"General legal search failed: {e}")
                return []
            
    except Exception as e:
        logger.error(f"OPTIMIZED search failed: {e}")
        return []

# ADD THIS TO ComprehensiveLegalHub class
def fast_environmental_search(self, query: str, state: str = None) -> List[Dict]:
    """Fast environmental search with limited API calls"""
    results = []
    
    try:
        # Only search EPA violations if we have a specific state
        if state:
            env_violations = self.environmental.search_environmental_violations(
                state=state, violation_type="CAA"  # Focus on Clean Air Act
            )
            results.extend(env_violations[:3])  # Limit to top 3
        
        return results
    except Exception as e:
        logger.error(f"Fast environmental search failed: {e}")
        return []

def fast_business_search(self, query: str, company_name: str = None) -> List[Dict]:
    """Fast business search with limited scope"""
    results = []
    
    try:
        if company_name:
            sec_results = self.business.search_sec_filings(company_name)
            results.extend(sec_results[:2])  # Limit to top 2
        
        return results
    except Exception as e:
        logger.error(f"Fast business search failed: {e}")
        return []
