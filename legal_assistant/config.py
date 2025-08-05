"""Configuration and environment variables - Enhanced for comprehensive legal research with government APIs"""
import os
from typing import List
from cryptography.fernet import Fernet

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# Database Paths
DEFAULT_CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
USER_CONTAINERS_PATH = os.path.abspath(os.path.join(os.getcwd(), "user-containers"))

# File Processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
LEGAL_EXTENSIONS = {'.pdf', '.txt', '.docx', '.rtf'}

EXTERNAL_API_TIMEOUT = 3  # Maximum 3 seconds per API call
MAX_CONCURRENT_APIS = 3   # Only call 3 APIs concurrently, not 8
ENABLE_API_CACHING = True # Cache API results
API_CACHE_TTL = 300       # Cache for 5 minutes

# SEARCH OPTIMIZATION (update these)
MIN_RELEVANCE_SCORE = 0.6  # up from 0.1 to filter noise
DEFAULT_SEARCH_K = 5       # down from 15 to reduce processing
ENHANCED_SEARCH_K = 8      # down from 20 to reduce processing

# DISABLE PROBLEMATIC FEATURES (add these)
DISABLE_EXTERNAL_SEARCH_ON_TIMEOUT = True
EXTERNAL_SEARCH_TIMEOUT = 5  # Maximum 5 seconds for ALL external searches
SKIP_FAILED_APIS = True      # Don't retry failed APIs

# Existing Premium Database Configuration
LEXISNEXIS_API_KEY = os.environ.get("LEXISNEXIS_API_KEY")
LEXISNEXIS_API_ENDPOINT = os.environ.get("LEXISNEXIS_API_ENDPOINT")
WESTLAW_API_KEY = os.environ.get("WESTLAW_API_KEY")
WESTLAW_API_ENDPOINT = os.environ.get("WESTLAW_API_ENDPOINT")

# === YOUR PROVIDED API KEYS ===
CONGRESS_API_KEY = "7J5Bfj6i0F3tg4VZleZ4SyQmVbG0QyIM9tPMQA2M"
DATA_GOV_API_KEY = "yZAV2yQIyyVzDYCHCw39CUBx98HDQQmHjd9wojRe"

# === COMPREHENSIVE FREE LEGAL APIS ===

# Core Legal Research APIs
CASELAW_ACCESS_PROJECT_API = "https://api.case.law/v1"
COURTLISTENER_API_KEY = os.environ.get("COURTLISTENER_API_KEY", "")
COURTLISTENER_API_ENDPOINT = "https://www.courtlistener.com/api/rest/v4"
JUSTIA_API_ENDPOINT = "https://law.justia.com"
OPENSTATES_API_ENDPOINT = "https://v3.openstates.org"
CORNELL_LAW_ENDPOINT = "https://www.law.cornell.edu"
GOOGLE_SCHOLAR_LEGAL_ENDPOINT = "https://scholar.google.com"

# Harvard Caselaw Access Project - FREE (No API key required)
HARVARD_LEGAL_API_ENDPOINT = "https://api.case.law/v1"

# Google Scholar - Web scraping (be respectful of rate limits)
GOOGLE_SCHOLAR_ENABLED = os.environ.get("GOOGLE_SCHOLAR_ENABLED", "true").lower() == "true"

# Federal Government APIs (using your Congress API key)
CONGRESS_API_ENDPOINT = "https://api.congress.gov/v3"
FEDERAL_REGISTER_API_ENDPOINT = "https://www.federalregister.gov/api/v1"
GOVINFO_API_ENDPOINT = "https://api.govinfo.gov"
LOC_API_ENDPOINT = "https://www.loc.gov/apis"

# Environmental Law APIs (using your Data.gov API key)
EPA_ENVIROFACTS_API = "https://www.epa.gov/enviro/envirofacts-data-service-api"
EPA_AIR_QUALITY_API = "https://www.airnowapi.org/aq"
EPA_WATER_QUALITY_API = "https://www.waterqualitydata.us"
NOAA_CLIMATE_API = "https://www.ncdc.noaa.gov/cdo-web/webservices/v2"
USGS_WATER_API = "https://waterservices.usgs.gov"
NASA_ENVIRONMENTAL_API = "https://api.nasa.gov"
EIA_ENERGY_API = "https://www.eia.gov/opendata"

# API Key assignments for environmental APIs
EPA_API_KEY = DATA_GOV_API_KEY
NOAA_API_KEY = DATA_GOV_API_KEY  
NASA_API_KEY = DATA_GOV_API_KEY
EIA_API_KEY = DATA_GOV_API_KEY
AIRNOW_API_KEY = DATA_GOV_API_KEY

# Additional government APIs
REGULATIONS_GOV_API = "https://api.regulations.gov/v4"
AIRNOW_API = "https://www.airnowapi.org"
FTC_API = "https://www.ftc.gov/developer"

# === FREE & AFFORDABLE LEGAL DATABASES (HIGH PRIORITY) ===

# CourtListener - FREE comprehensive legal database
COURTLISTENER_API_ENDPOINT = "https://www.courtlistener.com/api/rest/v4/"

# Harvard Caselaw Access Project - FREE
HARVARD_LEGAL_API_KEY = os.environ.get("HARVARD_LEGAL_API_KEY")
HARVARD_LEGAL_API_ENDPOINT = os.environ.get("HARVARD_LEGAL_API_ENDPOINT", "https://api.case.law/v1")

# Justia - FREE legal database
JUSTIA_API_ENDPOINT = "https://law.justia.com/api/"

# === GOVERNMENT & REGULATORY APIS (FREE) ===

# Federal Register API - FREE
FEDERAL_REGISTER_API_ENDPOINT = "https://www.federalregister.gov/api/v1/"

# Congress.gov API - FREE
CONGRESS_API_ENDPOINT = "https://api.congress.gov/v3/"

# SEC EDGAR API - FREE
SEC_EDGAR_API_ENDPOINT = "https://data.sec.gov/api/"

# EPA Environmental Data - FREE
EPA_API_ENDPOINT = "https://www.epa.gov/enviro/web-services"

# === ENVIRONMENTAL & CLIMATE LAW SPECIFIC ===

# Climate Policy Initiative API
CLIMATE_POLICY_API_ENDPOINT = "https://climatepolicyinitiative.org/api/"

# Environmental Law Institute (if they have an API)
ELI_API_KEY = os.environ.get("ELI_API_KEY")
ELI_API_ENDPOINT = os.environ.get("ELI_API_ENDPOINT")

# Carbon Disclosure Project API
CDP_API_KEY = os.environ.get("CDP_API_KEY")
CDP_API_ENDPOINT = "https://data.cdp.net/api/"

# === INTERNATIONAL & NGO RESOURCES ===

# UN Treaty Collection
UN_TREATY_API_ENDPOINT = "https://treaties.un.org/api/"

# World Bank Open Data API - FREE
WORLD_BANK_API_ENDPOINT = "https://api.worldbank.org/v2/"

# === SMB-FOCUSED RESOURCES ===

# SBA (Small Business Administration) API - FREE
SBA_API_ENDPOINT = "https://api.sba.gov/"

# USPTO (Patent/Trademark) API - FREE
USPTO_API_ENDPOINT = "https://developer.uspto.gov/api-catalog"

# Business.gov API (if available)
BUSINESS_GOV_API_ENDPOINT = os.environ.get("BUSINESS_GOV_API_ENDPOINT")

# === STATE & LOCAL GOVERNMENT ===

# Many states have open data portals - configure as needed
STATE_APIS = {
    "california": "https://data.ca.gov/api/",
    "new_york": "https://data.ny.gov/api/",
    "texas": "https://data.texas.gov/api/",
    # Add more states as needed
}

# Immigration Law APIs
USCIS_CASE_STATUS_API = "https://egov.uscis.gov/casestatus"
STATE_DEPT_VISA_API = "https://travel.state.gov"
ICE_DETENTION_API = "https://www.ice.gov/detain/detention-facilities"
CBP_BORDER_WAIT_API = "https://bwt.cbp.gov"
EOIR_COURT_API = "https://www.justice.gov/eoir"

# Housing & Homelessness APIs (using your Data.gov API key)
HUD_PUBLIC_HOUSING_API = "https://www.huduser.gov/hudapi/public"
HUD_FAIR_MARKET_RENT_API = "https://www.huduser.gov/hudapi/public/fmr"
CENSUS_HOUSING_API = "https://api.census.gov/data"
NYC_HOUSING_DATA_API = "https://data.cityofnewyork.us"

# Corporate & Business Law APIs
SEC_EDGAR_API = "https://data.sec.gov/api"
SEC_INVESTMENT_ADVISER_API = "https://adviserinfo.sec.gov"
DELAWARE_CORP_API = "https://corp.delaware.gov"
CALIFORNIA_SOS_API = "https://businesssearch.sos.ca.gov"
SAM_GOV_API = "https://sam.gov"

# Labor & Employment APIs (using your Data.gov API key)
DOL_OSHA_API = "https://developer.dol.gov/health-and-safety"
DOL_WAGE_HOUR_API = "https://enforcedata.dol.gov"
BLS_API = "https://api.bls.gov/publicAPI/v2"
EEOC_DATA_API = "https://www.eeoc.gov/statistics"

# Intellectual Property APIs
USPTO_PATENT_API = "https://patents.uspto.gov/api"
USPTO_TRADEMARK_API = "https://tsdrapi.uspto.gov"
GOOGLE_PATENTS_ENDPOINT = "https://patents.google.com"

# Healthcare Law APIs
FDA_DRUG_API = "https://api.fda.gov/drug"
FDA_DEVICE_API = "https://api.fda.gov/device"
FDA_FOOD_API = "https://api.fda.gov/food"
CMS_DATA_API = "https://data.cms.gov/api"

# Criminal Justice APIs (using your Data.gov API key)
FBI_CRIME_DATA_API = "https://api.usa.gov/crime/fbi/cde"
BJS_API = "https://bjs.ojp.gov/api"
NCVS_API = "https://crime-data-explorer.app.cloud.gov/api"

# International Law APIs
UN_TREATY_API = "https://treaties.un.org/api"
WORLD_BANK_LEGAL_API = "https://datahelpdesk.worldbank.org/knowledgebase"
ICJ_API = "https://www.icj-cij.org"
EUR_LEX_API = "https://eur-lex.europa.eu"

# Immigration Features
SUPPORTED_LANGUAGES = ["en", "es", "zh", "ar", "fr", "hi", "pt", "ru", "bn", "ur"]
AUTO_TRANSLATION_ENABLED = False  # Set to True when translation API is configured
RFE_STANDARD_DEADLINE_DAYS = 87
ASYLUM_ONE_YEAR_DEADLINE_DAYS = 365

# Security for Immigration Data
ENCRYPTION_KEY = os.environ.get("IMMIGRATION_ENCRYPTION_KEY", Fernet.generate_key())
DATA_RETENTION_DAYS = 2555  # 7 years per immigration law requirements
HIPAA_COMPLIANT_STORAGE = True

# Batch Processing
MAX_BATCH_SIZE = 50
BATCH_PROCESSING_TIMEOUT = 300  # 5 minutes

# Mobile Optimization
MOBILE_CHUNK_SIZE = 500  # Smaller chunks for mobile
LOW_BANDWIDTH_MODE = True

# Model Names
EMBEDDING_MODELS = [
    "nlpaueb/legal-bert-base-uncased",
    "law-ai/InCaseLawBERT", 
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1",
    "microsoft/DialoGPT-medium",
    "sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2"
]

FAST_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
]

# AI Models
AI_MODELS = [
    "moonshotai/kimi-k2:free",
    "deepseek/deepseek-chat",
    "microsoft/phi-3-mini-128k-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "openchat/openchat-7b:free"
]

# Chunk Sizes - OPTIMIZED
DEFAULT_CHUNK_SIZE = 1000     # Reduced from 1500
LEGISLATIVE_CHUNK_SIZE = 1500 # Reduced from 2000
DEFAULT_CHUNK_OVERLAP = 200   # Reduced from 300
LEGISLATIVE_CHUNK_OVERLAP = 300  # Reduced from 500

# Search Settings - IMPROVED
DEFAULT_SEARCH_K = 15  # Increased from 10
ENHANCED_SEARCH_K = 20  # Increased from 12
COMPREHENSIVE_SEARCH_K = 30  # Increased from 20
MIN_RELEVANCE_SCORE = 0.1  # CHANGED from 0.3 to be more permissive

# Confidence Score Weights - REBALANCED
CONFIDENCE_WEIGHTS = {
    "relevance": 0.5,     # Increased weight on relevance
    "document_count": 0.2, # Decreased weight on count
    "consistency": 0.2,
    "completeness": 0.1
}

# Add new search configuration
SEARCH_CONFIG = {
    "rerank_enabled": True,
    "hybrid_search_enabled": False,  # CHANGED to False to disable hybrid search
    "keyword_weight": 0.3,
    "semantic_weight": 0.7,
    "max_results_to_rerank": 50,
    "query_expansion_enabled": True,
    "min_score_threshold": 0.1,  # CHANGED from 0.3 to be more permissive
    "boost_exact_matches": True,
    "boost_factor": 1.5
}

# === ENHANCED API TIER CONFIGURATION ===
# Updated to include all new APIs
API_TIERS = {
    "free": [
        # Core legal research
        "CASELAW_ACCESS_PROJECT", "COURTLISTENER", "JUSTIA", "OPENSTATES", 
        "CORNELL_LAW", "GOOGLE_SCHOLAR_LEGAL",
        
        # Federal government  
        "CONGRESS", "FEDERAL_REGISTER", "GOVINFO", "LOC",
        
        # Environmental
        "EPA_ENVIROFACTS", "EPA_AIR_QUALITY", "EPA_WATER_QUALITY", "NOAA_CLIMATE",
        "USGS_WATER", "NASA_ENVIRONMENTAL", "EIA_ENERGY", "AIRNOW", "REGULATIONS_GOV",
        
        # Immigration
        "USCIS_CASE_STATUS", "STATE_DEPT_VISA", "ICE_DETENTION", "CBP_BORDER_WAIT",
        
        # Housing
        "HUD_PUBLIC_HOUSING", "HUD_FAIR_MARKET_RENT", "CENSUS_HOUSING",
        
        # Business
        "SEC_EDGAR", "SEC_INVESTMENT_ADVISER", "DELAWARE_CORP", "SAM_GOV", "FTC",
        
        # Labor
        "DOL_OSHA", "DOL_WAGE_HOUR", "BLS", "EEOC_DATA",
        
        # Intellectual Property
        "USPTO_PATENT", "USPTO_TRADEMARK", "GOOGLE_PATENTS",
        
        # Healthcare
        "FDA_DRUG", "FDA_DEVICE", "FDA_FOOD", "CMS_DATA",
        
        # Criminal Justice
        "FBI_CRIME_DATA", "BJS", "NCVS",
        
        # International
        "UN_TREATY", "WORLD_BANK_LEGAL", "ICJ", "EUR_LEX"
    ],
    "basic": [
        "free", "CLIMATE_POLICY", "CDP", "UN_TREATY"
    ],
    "premium": [
        "basic", "LEXISNEXIS", "WESTLAW", "ELI"
    ]
}

# === ENHANCED RATE LIMITING CONFIGURATION ===
API_RATE_LIMITS = {
    "COURTLISTENER": {"requests_per_hour": 100, "burst": 10},
    "HARVARD_LEGAL": {"requests_per_hour": 500, "burst": 20},
    "FEDERAL_REGISTER": {"requests_per_hour": 1000, "burst": 50},
    "CONGRESS": {"requests_per_hour": 5000, "burst": 100},
    "SEC_EDGAR": {"requests_per_hour": 500, "burst": 25},
    "EPA": {"requests_per_hour": 200, "burst": 10},
    "GOOGLE_SCHOLAR": {"requests_per_hour": 20, "burst": 2},  # Be very conservative
    
    # Core legal APIs
    "CASELAW_ACCESS_PROJECT": {"requests_per_hour": 1000, "burst": 100},
    "OPENSTATES": {"requests_per_hour": 100, "burst": 10},
    "CORNELL_LAW": {"requests_per_hour": 200, "burst": 5},
    "JUSTIA": {"requests_per_hour": 150, "burst": 5},
    "GOOGLE_SCHOLAR_LEGAL": {"requests_per_hour": 10, "burst": 2},
    
    # Government APIs
    "REGULATIONS_GOV": {"requests_per_hour": 250, "burst": 25},
    "AIRNOW": {"requests_per_hour": 500, "burst": 50},
    "FTC": {"requests_per_hour": 100, "burst": 10},
    
    # Environmental APIs
    "EPA_ENVIROFACTS": {"requests_per_hour": 300, "burst": 30},
    "EPA_AIR_QUALITY": {"requests_per_hour": 500, "burst": 50},
    "NOAA_CLIMATE": {"requests_per_hour": 1000, "burst": 100},
    
    # Immigration APIs
    "USCIS_CASE_STATUS": {"requests_per_hour": 50, "burst": 5},  # Be conservative
    "STATE_DEPT_VISA": {"requests_per_hour": 100, "burst": 10},
    
    # Housing APIs
    "HUD_PUBLIC_HOUSING": {"requests_per_hour": 200, "burst": 20},
    "CENSUS_HOUSING": {"requests_per_hour": 500, "burst": 50},
    
    # Business APIs
    "SEC_EDGAR": {"requests_per_hour": 300, "burst": 30},
    
    # Labor APIs
    "DOL_OSHA": {"requests_per_hour": 150, "burst": 15},
    "BLS": {"requests_per_hour": 500, "burst": 50},
    
    # IP APIs
    "USPTO_PATENT": {"requests_per_hour": 100, "burst": 10},
    "GOOGLE_PATENTS": {"requests_per_hour": 20, "burst": 2},
    
    # Healthcare APIs
    "FDA_DRUG": {"requests_per_hour": 200, "burst": 20},
    "CMS_DATA": {"requests_per_hour": 300, "burst": 30},
    
    # Criminal Justice APIs
    "FBI_CRIME_DATA": {"requests_per_hour": 200, "burst": 20},
    
    # International APIs
    "UN_TREATY": {"requests_per_hour": 100, "burst": 10},
    "WORLD_BANK_LEGAL": {"requests_per_hour": 500, "burst": 50}
}

# === INTELLIGENT LEGAL AREA DETECTION ===

# Keywords for automatically detecting legal practice areas
LEGAL_AREA_KEYWORDS = {
    'environmental': [
        'environmental', 'epa', 'pollution', 'clean air act', 'clean water act',
        'superfund', 'cercla', 'rcra', 'toxic substances', 'hazardous waste',
        'emissions', 'air quality', 'water quality', 'contamination', 'cleanup',
        'environmental impact', 'nepa', 'endangered species act'
    ],
    
    'immigration': [
        'immigration', 'visa', 'green card', 'asylum', 'refugee', 'deportation',
        'removal', 'uscis', 'ice', 'cbp', 'naturalization', 'citizenship',
        'work permit', 'ead', 'priority date', 'visa bulletin', 'adjustment of status',
        'consular processing', 'inadmissibility', 'waiver'
    ],
    
    'housing': [
        'housing', 'rental', 'landlord', 'tenant', 'eviction', 'fair housing',
        'discrimination', 'section 8', 'rent control', 'homelessness', 'affordable housing',
        'public housing', 'housing authority', 'fair market rent', 'housing voucher',
        'tenant rights', 'landlord obligations', 'habitability'
    ],
    
    'business': [
        'corporate', 'business', 'sec filing', 'securities', 'merger', 'acquisition',
        'ipo', 'public offering', 'quarterly report', 'annual report', '10-k', '10-q',
        'proxy statement', 'insider trading', 'corporate governance', 'fiduciary duty',
        'shareholder', 'board of directors', 'llc', 'corporation', 'partnership'
    ],
    
    'labor': [
        'employment', 'labor', 'osha', 'workplace safety', 'wage', 'overtime',
        'discrimination', 'harassment', 'workers compensation', 'union', 'collective bargaining',
        'wrongful termination', 'at-will employment', 'flsa', 'family medical leave',
        'disability accommodation', 'equal employment', 'workplace injury'
    ],
    
    'intellectual_property': [
        'patent', 'trademark', 'copyright', 'intellectual property', 'trade secret',
        'infringement', 'licensing', 'invention', 'brand', 'trade dress',
        'prior art', 'patent prosecution', 'trademark registration', 'fair use',
        'dmca', 'patent troll', 'royalty', 'ip portfolio'
    ],
    
    'healthcare': [
        'healthcare', 'medical', 'fda', 'drug approval', 'medical device',
        'hipaa', 'patient privacy', 'medicare', 'medicaid', 'cms',
        'clinical trial', 'medical malpractice', 'informed consent',
        'healthcare fraud', 'pharmaceutical', 'biotech', 'medical records'
    ],
    
    'criminal': [
        'criminal', 'crime', 'arrest', 'prosecution', 'sentencing', 'prison',
        'felony', 'misdemeanor', 'fbi', 'investigation', 'search warrant',
        'miranda rights', 'fourth amendment', 'fifth amendment', 'due process',
        'criminal procedure', 'bail', 'plea bargain', 'criminal defense'
    ],
    
    'international': [
        'international', 'treaty', 'foreign', 'diplomatic', 'trade agreement',
        'human rights', 'war crimes', 'international court', 'extradition',
        'international arbitration', 'foreign investment', 'international trade',
        'sanctions', 'export control', 'foreign corrupt practices'
    ],
    
    'constitutional': [
        'constitutional', 'constitution', 'first amendment', 'second amendment',
        'fourth amendment', 'fifth amendment', 'fourteenth amendment',
        'free speech', 'religion', 'due process', 'equal protection',
        'commerce clause', 'supremacy clause', 'bill of rights'
    ],
    
    'family': [
        'family law', 'divorce', 'custody', 'child support', 'alimony',
        'spousal support', 'adoption', 'guardianship', 'domestic violence',
        'restraining order', 'prenuptial agreement', 'marriage', 'paternity'
    ],
    
    'tax': [
        'tax', 'taxation', 'irs', 'income tax', 'estate tax', 'gift tax',
        'tax evasion', 'tax fraud', 'tax deduction', 'tax credit',
        'tax audit', 'tax court', 'tax lien', 'tax levy'
    ]
}

# === STATE DETECTION ENHANCEMENT ===

# Enhanced state detection patterns
STATE_DETECTION_PATTERNS = {
    'washington': ['WA', 'Wash.', 'Washington', 'RCW', 'WAC', 'Revised Code of Washington'],
    'california': ['CA', 'Cal.', 'Calif.', 'California', 'Cal. Code', 'California Code'],
    'new_york': ['NY', 'N.Y.', 'New York', 'N.Y. Law', 'New York Law'],
    'texas': ['TX', 'Tex.', 'Texas', 'Tex. Code', 'Texas Code'],
    'florida': ['FL', 'Fla.', 'Florida', 'Fla. Stat.', 'Florida Statutes'],
    'illinois': ['IL', 'Ill.', 'Illinois', 'ILCS', 'Illinois Compiled Statutes'],
    'pennsylvania': ['PA', 'Pa.', 'Pennsylvania', 'Pa. Code', 'Pennsylvania Code'],
    'ohio': ['OH', 'Ohio', 'Ohio Rev. Code', 'Ohio Revised Code'],
    'georgia': ['GA', 'Ga.', 'Georgia', 'O.C.G.A.'],
    'north_carolina': ['NC', 'N.C.', 'North Carolina', 'N.C. Gen. Stat.'],
    'michigan': ['MI', 'Mich.', 'Michigan', 'Mich. Comp. Laws'],
    'new_jersey': ['NJ', 'N.J.', 'New Jersey', 'N.J. Stat.'],
    'virginia': ['VA', 'Va.', 'Virginia', 'Va. Code'],
    'massachusetts': ['MA', 'Mass.', 'Massachusetts', 'Mass. Gen. Laws'],
    'indiana': ['IN', 'Ind.', 'Indiana', 'Ind. Code'],
    'arizona': ['AZ', 'Ariz.', 'Arizona', 'Ariz. Rev. Stat.'],
    'tennessee': ['TN', 'Tenn.', 'Tennessee', 'Tenn. Code'],
    'missouri': ['MO', 'Mo.', 'Missouri', 'Mo. Rev. Stat.'],
    'maryland': ['MD', 'Md.', 'Maryland', 'Md. Code'],
    'wisconsin': ['WI', 'Wis.', 'Wisconsin', 'Wis. Stat.'],
    'colorado': ['CO', 'Colo.', 'Colorado', 'Colo. Rev. Stat.'],
    'minnesota': ['MN', 'Minn.', 'Minnesota', 'Minn. Stat.'],
    'south_carolina': ['SC', 'S.C.', 'South Carolina', 'S.C. Code'],
    'alabama': ['AL', 'Ala.', 'Alabama', 'Ala. Code'],
    'louisiana': ['LA', 'La.', 'Louisiana', 'La. Rev. Stat.'],
    'kentucky': ['KY', 'Ky.', 'Kentucky', 'Ky. Rev. Stat.'],
    'oregon': ['OR', 'Or.', 'Oregon', 'Or. Rev. Stat.'],
    'oklahoma': ['OK', 'Okla.', 'Oklahoma', 'Okla. Stat.'],
    'connecticut': ['CT', 'Conn.', 'Connecticut', 'Conn. Gen. Stat.'],
    'utah': ['UT', 'Utah', 'Utah Code'],
    'iowa': ['IA', 'Iowa', 'Iowa Code'],
    'nevada': ['NV', 'Nev.', 'Nevada', 'Nev. Rev. Stat.'],
    'arkansas': ['AR', 'Ark.', 'Arkansas', 'Ark. Code'],
    'mississippi': ['MS', 'Miss.', 'Mississippi', 'Miss. Code'],
    'kansas': ['KS', 'Kan.', 'Kansas', 'Kan. Stat.'],
    'new_mexico': ['NM', 'N.M.', 'New Mexico', 'N.M. Stat.'],
    'nebraska': ['NE', 'Neb.', 'Nebraska', 'Neb. Rev. Stat.'],
    'west_virginia': ['WV', 'W.Va.', 'West Virginia', 'W.Va. Code'],
    'idaho': ['ID', 'Idaho', 'Idaho Code'],
    'hawaii': ['HI', 'Hawaii', 'Haw. Rev. Stat.'],
    'new_hampshire': ['NH', 'N.H.', 'New Hampshire', 'N.H. Rev. Stat.'],
    'maine': ['ME', 'Me.', 'Maine', 'Me. Rev. Stat.'],
    'montana': ['MT', 'Mont.', 'Montana', 'Mont. Code'],
    'rhode_island': ['RI', 'R.I.', 'Rhode Island', 'R.I. Gen. Laws'],
    'delaware': ['DE', 'Del.', 'Delaware', 'Del. Code'],
    'south_dakota': ['SD', 'S.D.', 'South Dakota', 'S.D. Codified Laws'],
    'north_dakota': ['ND', 'N.D.', 'North Dakota', 'N.D. Cent. Code'],
    'alaska': ['AK', 'Alaska', 'Alaska Stat.'],
    'vermont': ['VT', 'Vt.', 'Vermont', 'Vt. Stat.'],
    'wyoming': ['WY', 'Wyo.', 'Wyoming', 'Wyo. Stat.']
}

# === COMPREHENSIVE DATABASE CAPABILITIES ===

# Map each API to its capabilities
COMPREHENSIVE_API_CAPABILITIES = {
    # Core Legal Research
    'caselaw_access_project': {
        'content_types': ['cases', 'court_opinions'],
        'coverage': ['federal', 'all_states', 'historical'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['citation_search', 'full_text_search', 'court_filtering']
    },
    
    'courtlistener': {
        'content_types': ['cases', 'dockets', 'oral_arguments', 'judges'],
        'coverage': ['federal', 'state'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['real_time_updates', 'docket_tracking', 'judge_info']
    },
    
    'justia': {
        'content_types': ['cases', 'statutes', 'regulations', 'legal_news'],
        'coverage': ['federal', 'all_states'],
        'authority_level': 'high',
        'full_text': True,
        'search_features': ['comprehensive_coverage', 'free_access', 'organized_by_jurisdiction']
    },
    
    'cornell_law': {
        'content_types': ['statutes', 'regulations', 'constitution', 'legal_encyclopedia'],
        'coverage': ['federal', 'major_states'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['academic_quality', 'well_organized', 'authoritative']
    },
    
    'openstates': {
        'content_types': ['bills', 'legislators', 'committees', 'votes'],
        'coverage': ['all_states'],
        'authority_level': 'high',
        'full_text': True,
        'search_features': ['current_legislation', 'legislator_tracking', 'voting_records']
    },
    
    # Federal Government
    'congress_gov': {
        'content_types': ['bills', 'laws', 'congressional_records', 'committee_reports'],
        'coverage': ['federal'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['official_source', 'legislative_history', 'bill_tracking']
    },
    
    'federal_register': {
        'content_types': ['regulations', 'proposed_rules', 'notices', 'presidential_documents'],
        'coverage': ['federal'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['official_regulations', 'rulemaking_process', 'agency_documents']
    },
    
    # Environmental
    'epa_envirofacts': {
        'content_types': ['enforcement_actions', 'facility_data', 'violations', 'permits'],
        'coverage': ['national'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['facility_search', 'violation_tracking', 'enforcement_history']
    },
    
    'epa_air_quality': {
        'content_types': ['air_quality_data', 'monitoring_data', 'forecasts'],
        'coverage': ['national', 'local'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['real_time_data', 'geographic_filtering', 'historical_trends']
    },
    
    # Immigration
    'uscis_case_status': {
        'content_types': ['case_status', 'processing_times', 'form_status'],
        'coverage': ['national'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['receipt_number_lookup', 'real_time_status', 'processing_updates']
    },
    
    'state_dept_visa': {
        'content_types': ['visa_bulletin', 'priority_dates', 'country_quotas'],
        'coverage': ['international'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['monthly_updates', 'category_tracking', 'country_specific']
    },
    
    # Housing
    'hud_fair_market_rent': {
        'content_types': ['rent_data', 'housing_costs', 'market_analysis'],
        'coverage': ['national', 'local'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['geographic_filtering', 'annual_updates', 'bedroom_categories']
    },
    
    'census_housing': {
        'content_types': ['demographic_data', 'housing_statistics', 'economic_data'],
        'coverage': ['national', 'state', 'local'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['detailed_demographics', 'time_series', 'geographic_granularity']
    },
    
    # Business
    'sec_edgar': {
        'content_types': ['corporate_filings', 'financial_reports', 'proxy_statements'],
        'coverage': ['public_companies'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['company_search', 'filing_type_filter', 'date_filtering']
    },
    
    # Labor
    'dol_osha': {
        'content_types': ['safety_violations', 'citations', 'inspection_data'],
        'coverage': ['national'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['company_search', 'violation_type', 'penalty_amounts']
    },
    
    'bls_data': {
        'content_types': ['employment_statistics', 'wage_data', 'labor_trends'],
        'coverage': ['national', 'state', 'local'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['time_series', 'demographic_breakdown', 'industry_analysis']
    },
    
    # Intellectual Property
    'uspto_patents': {
        'content_types': ['patents', 'patent_applications', 'patent_prosecution'],
        'coverage': ['national'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['inventor_search', 'assignee_search', 'classification_search']
    },
    
    'uspto_trademarks': {
        'content_types': ['trademarks', 'trademark_applications', 'trademark_status'],
        'coverage': ['national'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['mark_search', 'owner_search', 'status_tracking']
    },
    
    # Healthcare
    'fda_enforcement': {
        'content_types': ['drug_recalls', 'device_recalls', 'food_recalls', 'enforcement_actions'],
        'coverage': ['national'],
        'authority_level': 'very_high',
        'full_text': True,
        'search_features': ['product_search', 'company_search', 'recall_classification']
    },
    
    # Criminal Justice
    'fbi_crime_data': {
        'content_types': ['crime_statistics', 'arrest_data', 'victimization_data'],
        'coverage': ['national', 'state', 'local'],
        'authority_level': 'very_high',
        'full_text': False,
        'search_features': ['geographic_filtering', 'crime_type', 'time_series']
    }
}

# === SMART SEARCH ROUTING CONFIGURATION ===

# Configuration for automatically routing searches to appropriate APIs
SMART_SEARCH_ROUTING = {
    'environmental_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['environmental'],
        'primary_apis': ['epa_envirofacts', 'epa_air_quality', 'federal_register'],
        'secondary_apis': ['congress_gov', 'state_law_comprehensive'],
        'search_strategy': 'enforcement_first'  # Start with enforcement data
    },
    
    'immigration_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['immigration'],
        'primary_apis': ['uscis_case_status', 'state_dept_visa', 'federal_register'],
        'secondary_apis': ['courtlistener', 'congress_gov'],
        'search_strategy': 'status_first'  # Check case status if receipt number found
    },
    
    'housing_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['housing'],
        'primary_apis': ['hud_fair_market_rent', 'census_housing', 'state_law_comprehensive'],
        'secondary_apis': ['courtlistener', 'justia'],
        'search_strategy': 'data_and_law'  # Combine data with legal authorities
    },
    
    'business_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['business'],
        'primary_apis': ['sec_edgar', 'state_law_comprehensive', 'justia'],
        'secondary_apis': ['courtlistener', 'federal_register'],
        'search_strategy': 'filings_and_law'  # Corporate filings plus legal framework
    },
    
    'labor_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['labor'],
        'primary_apis': ['dol_osha', 'bls_data', 'state_law_comprehensive'],
        'secondary_apis': ['courtlistener', 'federal_register'],
        'search_strategy': 'violations_and_standards'  # OSHA data plus legal standards
    },
    
    'intellectual_property': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['intellectual_property'],
        'primary_apis': ['uspto_patents', 'uspto_trademarks', 'courtlistener'],
        'secondary_apis': ['federal_register', 'justia'],
        'search_strategy': 'ip_comprehensive'  # Patents, trademarks, and case law
    },
    
    'healthcare_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['healthcare'],
        'primary_apis': ['fda_enforcement', 'cms_data', 'federal_register'],
        'secondary_apis': ['courtlistener', 'state_law_comprehensive'],
        'search_strategy': 'regulatory_focus'  # FDA/CMS regulations plus case law
    },
    
    'criminal_law': {
        'trigger_keywords': LEGAL_AREA_KEYWORDS['criminal'],
        'primary_apis': ['fbi_crime_data', 'state_law_comprehensive', 'courtlistener'],
        'secondary_apis': ['justia', 'congress_gov'],
        'search_strategy': 'statute_and_data'  # Criminal codes plus crime statistics
    }
}

# === QUERY ENHANCEMENT CONFIGURATION ===

# Patterns to enhance queries for better API results
QUERY_ENHANCEMENT_PATTERNS = {
    'environmental': {
        'add_terms': ['violation', 'compliance', 'enforcement', 'regulation'],
        'expand_acronyms': {
            'CAA': 'Clean Air Act',
            'CWA': 'Clean Water Act', 
            'RCRA': 'Resource Conservation and Recovery Act',
            'CERCLA': 'Comprehensive Environmental Response Compensation and Liability Act'
        }
    },
    
    'immigration': {
        'add_terms': ['uscis', 'status', 'processing', 'form'],
        'expand_acronyms': {
            'EAD': 'Employment Authorization Document',
            'AOS': 'Adjustment of Status',
            'PD': 'Priority Date',
            'RFE': 'Request for Evidence'
        }
    },
    
    'business': {
        'add_terms': ['filing', 'sec', 'corporate', 'compliance'],
        'expand_acronyms': {
            'IPO': 'Initial Public Offering',
            'M&A': 'Mergers and Acquisitions',
            'SOX': 'Sarbanes-Oxley Act'
        }
    },
    
    'labor': {
        'add_terms': ['osha', 'violation', 'workplace', 'safety'],
        'expand_acronyms': {
            'FLSA': 'Fair Labor Standards Act',
            'FMLA': 'Family and Medical Leave Act',
            'ADA': 'Americans with Disabilities Act'
        }
    }
}

# === API RESPONSE STANDARDIZATION ===

# Standard fields for normalizing responses across different APIs
STANDARD_RESPONSE_FIELDS = {
    'title': ['title', 'name', 'case_name', 'facility_name', 'company_name'],
    'description': ['description', 'summary', 'abstract', 'snippet'],
    'date': ['date', 'filing_date', 'decision_date', 'report_date', 'updated_date'],
    'url': ['url', 'link', 'web_url', 'detail_url'],
    'authority': ['court', 'agency', 'jurisdiction', 'issuing_authority'],
    'location': ['state', 'jurisdiction', 'county', 'city'],
    'category': ['type', 'category', 'classification', 'form_type']
}

# === ENHANCED FEATURE FLAGS ===
class FeatureFlags:
    AI_ENABLED: bool = bool(OPENROUTER_API_KEY)
    AIOHTTP_AVAILABLE: bool = False
    OPEN_SOURCE_NLP_AVAILABLE: bool = False
    PYMUPDF_AVAILABLE: bool = False
    PDFPLUMBER_AVAILABLE: bool = False
    UNSTRUCTURED_AVAILABLE: bool = False
    OCR_AVAILABLE: bool = False
    HYBRID_SEARCH_AVAILABLE: bool = False  # CHANGED to False to disable hybrid search
    
    # Core legal research APIs
    CASELAW_ACCESS_PROJECT_AVAILABLE: bool = True
    COURTLISTENER_AVAILABLE: bool = True
    JUSTIA_AVAILABLE: bool = True
    CORNELL_LAW_AVAILABLE: bool = True
    OPENSTATES_AVAILABLE: bool = True
    GOOGLE_SCHOLAR_LEGAL_AVAILABLE: bool = True
    
    # Federal government APIs (with your API keys)
    CONGRESS_GOV_AVAILABLE: bool = bool(CONGRESS_API_KEY)
    FEDERAL_REGISTER_AVAILABLE: bool = True
    DATA_GOV_AVAILABLE: bool = bool(DATA_GOV_API_KEY)
    
    # Environmental APIs
    EPA_APIS_AVAILABLE: bool = bool(DATA_GOV_API_KEY)
    ENVIRONMENTAL_RESEARCH_ENABLED: bool = True
    
    # Immigration APIs
    IMMIGRATION_APIS_AVAILABLE: bool = True
    IMMIGRATION_CASE_TRACKING_ENABLED: bool = True
    
    # Housing APIs
    HOUSING_APIS_AVAILABLE: bool = bool(DATA_GOV_API_KEY)
    HOUSING_DATA_RESEARCH_ENABLED: bool = True
    
    # Business APIs
    BUSINESS_APIS_AVAILABLE: bool = True
    SEC_RESEARCH_ENABLED: bool = True
    
    # Labor APIs
    LABOR_APIS_AVAILABLE: bool = bool(DATA_GOV_API_KEY)
    OSHA_RESEARCH_ENABLED: bool = True
    
    # IP APIs
    IP_APIS_AVAILABLE: bool = True
    PATENT_RESEARCH_ENABLED: bool = True
    
    # Healthcare APIs
    HEALTHCARE_APIS_AVAILABLE: bool = True
    FDA_RESEARCH_ENABLED: bool = True
    
    # Criminal Justice APIs
    CRIMINAL_JUSTICE_APIS_AVAILABLE: bool = bool(DATA_GOV_API_KEY)
    CRIME_DATA_RESEARCH_ENABLED: bool = True
    
    # International APIs
    INTERNATIONAL_APIS_AVAILABLE: bool = True
    
    # Smart features
    INTELLIGENT_AREA_DETECTION: bool = True
    AUTO_STATE_DETECTION: bool = True
    MULTI_API_SEARCH: bool = True
    COMPREHENSIVE_LEGAL_RESEARCH: bool = True
    STATE_LAW_RESEARCH_ENABLED: bool = True
    GOVERNMENT_DATA_RESEARCH: bool = True

def initialize_comprehensive_features():
    """Initialize all comprehensive legal research features with API testing"""
    
    print("🚀 Initializing Comprehensive Legal Research System...")
    
    # Test API connectivity with your provided keys
    api_test_results = {}
    
    # Test Congress.gov API
    try:
        import requests
        congress_test = requests.get(
            f"{CONGRESS_API_ENDPOINT}/bill",
            headers={"X-API-Key": CONGRESS_API_KEY},
            params={"format": "json", "limit": 1},
            timeout=5
        )
        api_test_results['congress_gov'] = congress_test.status_code == 200
        print(f"Congress.gov API: {'✅ Working' if api_test_results['congress_gov'] else '❌ Failed'}")
    except Exception as e:
        api_test_results['congress_gov'] = False
        print(f"Congress.gov API: ❌ Connection failed - {e}")
    
    # Test Data.gov APIs
    try:
        # Test with Census API (uses Data.gov key)
        census_test = requests.get(
            "https://api.census.gov/data/2022/acs/acs1",
            params={"get": "NAME", "for": "state:06", "key": DATA_GOV_API_KEY},
            timeout=5
        )
        api_test_results['data_gov'] = census_test.status_code == 200
        print(f"Data.gov APIs: {'✅ Working' if api_test_results['data_gov'] else '❌ Failed'}")
    except Exception as e:
        api_test_results['data_gov'] = False
        print(f"Data.gov APIs: ❌ Connection failed - {e}")
    
    # Test free APIs (no keys required)
    free_apis = {
        'Harvard Caselaw': 'https://api.case.law/v1/cases/',
        'CourtListener': 'https://www.courtlistener.com/api/rest/v4/search/',
        'Justia': 'https://law.justia.com/',
        'Cornell Law': 'https://www.law.cornell.edu/',
        'OpenStates': 'https://v3.openstates.org/jurisdictions'
    }
    
    working_free_apis = 0
    for api_name, url in free_apis.items():
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'LegalAssistant/1.0'})
            is_working = response.status_code in [200, 201]
            api_test_results[api_name.lower().replace(' ', '_')] = is_working
            if is_working:
                working_free_apis += 1
            print(f"{api_name}: {'✅ Available' if is_working else '❌ Unavailable'}")
        except Exception as e:
            api_test_results[api_name.lower().replace(' ', '_')] = False
            print(f"{api_name}: ❌ Connection failed")
    
    # Update feature flags based on test results
    FeatureFlags.CONGRESS_GOV_AVAILABLE = api_test_results.get('congress_gov', False)
    FeatureFlags.DATA_GOV_AVAILABLE = api_test_results.get('data_gov', False)
    
    # Environmental features depend on Data.gov key
    FeatureFlags.EPA_APIS_AVAILABLE = api_test_results.get('data_gov', False)
    FeatureFlags.ENVIRONMENTAL_RESEARCH_ENABLED = FeatureFlags.EPA_APIS_AVAILABLE
    
    # Housing features depend on Data.gov key
    FeatureFlags.HOUSING_APIS_AVAILABLE = api_test_results.get('data_gov', False)
    FeatureFlags.HOUSING_DATA_RESEARCH_ENABLED = FeatureFlags.HOUSING_APIS_AVAILABLE
    
    # Labor features depend on Data.gov key
    FeatureFlags.LABOR_APIS_AVAILABLE = api_test_results.get('data_gov', False)
    FeatureFlags.OSHA_RESEARCH_ENABLED = FeatureFlags.LABOR_APIS_AVAILABLE
    
    # Criminal justice features depend on Data.gov key
    FeatureFlags.CRIMINAL_JUSTICE_APIS_AVAILABLE = api_test_results.get('data_gov', False)
    FeatureFlags.CRIME_DATA_RESEARCH_ENABLED = FeatureFlags.CRIMINAL_JUSTICE_APIS_AVAILABLE
    
    # Overall comprehensive research capability
    total_working_apis = sum(api_test_results.values()) + working_free_apis
    FeatureFlags.COMPREHENSIVE_LEGAL_RESEARCH = total_working_apis >= 5
    
    # Print summary
    print(f"\n📊 API Connectivity Summary:")
    print(f"   Working APIs: {total_working_apis}")
    print(f"   Congress.gov: {'✅' if FeatureFlags.CONGRESS_GOV_AVAILABLE else '❌'}")
    print(f"   Data.gov: {'✅' if FeatureFlags.DATA_GOV_AVAILABLE else '❌'}")
    print(f"   Free Legal APIs: {working_free_apis}/5")
    print(f"   Environmental Research: {'✅' if FeatureFlags.ENVIRONMENTAL_RESEARCH_ENABLED else '❌'}")
    print(f"   Immigration Tracking: {'✅' if FeatureFlags.IMMIGRATION_CASE_TRACKING_ENABLED else '❌'}")
    print(f"   Housing Data: {'✅' if FeatureFlags.HOUSING_DATA_RESEARCH_ENABLED else '❌'}")
    print(f"   Business Research: {'✅' if FeatureFlags.SEC_RESEARCH_ENABLED else '❌'}")
    print(f"   Labor/OSHA Research: {'✅' if FeatureFlags.OSHA_RESEARCH_ENABLED else '❌'}")
    print(f"   Crime Data: {'✅' if FeatureFlags.CRIME_DATA_RESEARCH_ENABLED else '❌'}")
    print(f"\n🎯 Comprehensive Legal Research: {'✅ ENABLED' if FeatureFlags.COMPREHENSIVE_LEGAL_RESEARCH else '❌ LIMITED'}")
    
    if FeatureFlags.COMPREHENSIVE_LEGAL_RESEARCH:
        print("🔥 Your system now has access to comprehensive legal research across:")
        print("   • Case law from all jurisdictions")
        print("   • Federal and state statutes") 
        print("   • Current legislation tracking")
        print("   • Environmental compliance data")
        print("   • Immigration case tracking")
        print("   • Corporate filings and business law")
        print("   • Labor violations and safety data")
        print("   • Patent and trademark searches")
        print("   • Healthcare regulations and recalls")
        print("   • Criminal justice statistics")
        print("   • Housing and fair market rent data")
    
    return api_test_results

def initialize_feature_flags():
    """Initialize feature flags by checking for available dependencies"""
    
    # Check OCR
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        FeatureFlags.OCR_AVAILABLE = True
        print("✅ OCR support available - can process scanned PDFs")
    except ImportError:
        FeatureFlags.OCR_AVAILABLE = False
        print("⚠️ OCR not available - install pytesseract and pdf2image")

    # Hybrid search is now disabled by default
    FeatureFlags.HYBRID_SEARCH_AVAILABLE = False
    print("⚠️ Hybrid search disabled for better performance")
    
    # Check existing features
    try:
        import aiohttp
        FeatureFlags.AIOHTTP_AVAILABLE = True
        print("✅ Async HTTP support available")
    except ImportError:
        FeatureFlags.AIOHTTP_AVAILABLE = False
        print("⚠️ Async HTTP not available - install aiohttp")
    
    try:
        import spacy
        import nltk
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = True
        print("✅ Open source NLP available")
    except ImportError:
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = False
        print("⚠️ Open source NLP not available - install spacy and nltk")
    
    try:
        import fitz  # PyMuPDF
        FeatureFlags.PYMUPDF_AVAILABLE = True
        print("✅ PyMuPDF available for PDF processing")
    except ImportError:
        FeatureFlags.PYMUPDF_AVAILABLE = False
        print("⚠️ PyMuPDF not available - install PyMuPDF")
    
    try:
        import pdfplumber
        FeatureFlags.PDFPLUMBER_AVAILABLE = True
        print("✅ PDFPlumber available for advanced PDF parsing")
    except ImportError:
        FeatureFlags.PDFPLUMBER_AVAILABLE = False
        print("⚠️ PDFPlumber not available - install pdfplumber")
    
    try:
        from unstructured.partition.auto import partition
        FeatureFlags.UNSTRUCTURED_AVAILABLE = True
        print("✅ Unstructured available for document parsing")
    except ImportError:
        FeatureFlags.UNSTRUCTURED_AVAILABLE = False
        print("⚠️ Unstructured not available - install unstructured[all-docs]")
    
    # Update AI enabled flag
    FeatureFlags.AI_ENABLED = bool(FeatureFlags.AIOHTTP_AVAILABLE and OPENROUTER_API_KEY)
    
    print(f"Document processing status: PyMuPDF={FeatureFlags.PYMUPDF_AVAILABLE}, pdfplumber={FeatureFlags.PDFPLUMBER_AVAILABLE}, Unstructured={FeatureFlags.UNSTRUCTURED_AVAILABLE}")

    # Check API availability
    print("🔍 Checking API availability...")
    if FeatureFlags.COURTLISTENER_AVAILABLE:
        print("✅ CourtListener API available (free legal database)")
    
    # Harvard Caselaw is always available (free, no key required)
    FeatureFlags.HARVARD_LEGAL_AVAILABLE = True
    print("✅ Harvard Caselaw Access Project available (free)")
    
    if CONGRESS_API_KEY:
        FeatureFlags.CONGRESS_AVAILABLE = True
        print("✅ Congress.gov API configured")
    else:
        FeatureFlags.CONGRESS_AVAILABLE = False
    
    if DATA_GOV_API_KEY:
        print("✅ Environmental APIs available (EPA, Federal Register)")

    # Now run the comprehensive feature initialization
    return initialize_comprehensive_features()

# === ADDITIONAL UTILITY FUNCTIONS ===

def detect_legal_area(query_text: str) -> List[str]:
    """Automatically detect legal practice areas from query text"""
    detected_areas = []
    query_lower = query_text.lower()
    
    for area, keywords in LEGAL_AREA_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                if area not in detected_areas:
                    detected_areas.append(area)
                break
    
    return detected_areas

def detect_state_jurisdiction(query_text: str) -> List[str]:
    """Automatically detect state jurisdictions from query text"""
    detected_states = []
    query_upper = query_text.upper()
    
    for state, patterns in STATE_DETECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.upper() in query_upper:
                if state not in detected_states:
                    detected_states.append(state)
                break
    
    return detected_states

def get_relevant_apis(query_text: str, legal_areas: List[str] = None, states: List[str] = None) -> List[str]:
    """Return most relevant APIs based on query content and detected legal areas"""
    
    # If no legal areas provided, try to detect them
    if not legal_areas:
        legal_areas = detect_legal_area(query_text)
    
    relevant_apis = set()
    
    # Map legal areas to specific APIs
    for area in legal_areas:
        if area == 'environmental':
            relevant_apis.update(['EPA_ENVIROFACTS', 'EPA_AIR_QUALITY', 'NOAA_CLIMATE', 'EIA_ENERGY', 'AIRNOW', 'REGULATIONS_GOV'])
        elif area == 'immigration':
            relevant_apis.update(['USCIS_CASE_STATUS', 'STATE_DEPT_VISA', 'EOIR_COURT', 'ICE_DETENTION'])
        elif area == 'housing':
            relevant_apis.update(['HUD_PUBLIC_HOUSING', 'HUD_FAIR_MARKET_RENT', 'CENSUS_HOUSING'])
        elif area == 'business':
            relevant_apis.update(['SEC_EDGAR', 'SEC_INVESTMENT_ADVISER', 'DELAWARE_CORP', 'FTC'])
        elif area == 'labor':
            relevant_apis.update(['DOL_OSHA', 'DOL_WAGE_HOUR', 'BLS', 'EEOC_DATA'])
        elif area == 'intellectual_property':
            relevant_apis.update(['USPTO_PATENT', 'USPTO_TRADEMARK', 'GOOGLE_PATENTS'])
        elif area == 'healthcare':
            relevant_apis.update(['FDA_DRUG', 'FDA_DEVICE', 'FDA_FOOD', 'CMS_DATA'])
        elif area == 'criminal':
            relevant_apis.update(['FBI_CRIME_DATA', 'BJS', 'NCVS'])
        elif area == 'constitutional':
            relevant_apis.update(['COURTLISTENER', 'CASELAW_ACCESS_PROJECT', 'JUSTIA'])
        elif area in SMART_SEARCH_ROUTING:
            # Fallback to smart routing config
            routing_config = SMART_SEARCH_ROUTING[area]
            relevant_apis.update(routing_config['primary_apis'])
            relevant_apis.update(routing_config['secondary_apis'])
    
    # Add state-specific APIs if states detected
    if states:
        relevant_apis.add('state_law_comprehensive')
    
    # If no specific areas detected, use general legal research APIs
    if not relevant_apis:
        relevant_apis.update(['COURTLISTENER', 'CASELAW_ACCESS_PROJECT', 'CONGRESS', 'FEDERAL_REGISTER', 'JUSTIA'])
    
    return list(relevant_apis)

def enhance_query(query_text: str, legal_areas: List[str]) -> str:
    """Enhance query with legal-specific terms and expansions"""
    enhanced_query = query_text
    
    for area in legal_areas:
        if area in QUERY_ENHANCEMENT_PATTERNS:
            patterns = QUERY_ENHANCEMENT_PATTERNS[area]
            
            # Expand acronyms
            for acronym, expansion in patterns.get('expand_acronyms', {}).items():
                if acronym in enhanced_query:
                    enhanced_query = enhanced_query.replace(acronym, f"{acronym} {expansion}")
    
    return enhanced_query

# Initialize feature flags when module is imported
initialize_feature_flags()

