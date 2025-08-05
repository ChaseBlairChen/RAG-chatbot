"""Configuration and environment variables - Enhanced for comprehensive legal research with government APIs"""
import os
from typing import List
from cryptography.fernet import Fernet

# CRITICAL FIXES - Add missing constants
APP_REFERER = os.environ.get("APP_REFERER", "http://localhost:3000")
APP_TITLE = os.environ.get("APP_TITLE", "Legal Assistant API")

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

# External API Keys (set your own)
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY", "")
DATA_GOV_API_KEY = os.environ.get("DATA_GOV_API_KEY", "")

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

# Federal Government APIs
CONGRESS_API_ENDPOINT = "https://api.congress.gov/v3"
FEDERAL_REGISTER_API_ENDPOINT = "https://www.federalregister.gov/api/v1"
GOVINFO_API_ENDPOINT = "https://api.govinfo.gov"
LOC_API_ENDPOINT = "https://www.loc.gov/apis"

# Environmental Law APIs
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

# Immigration Law APIs
USCIS_CASE_STATUS_API = "https://egov.uscis.gov/casestatus"
STATE_DEPT_VISA_API = "https://travel.state.gov"
ICE_DETENTION_API = "https://www.ice.gov/detain/detention-facilities"
CBP_BORDER_WAIT_API = "https://bwt.cbp.gov"
EOIR_COURT_API = "https://www.justice.gov/eoir"

# Housing & Homelessness APIs
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

# Labor & Employment APIs
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

# Criminal Justice APIs
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
    """Simplified feature initialization without network testing"""
    
    print("🚀 Initializing Legal Assistant...")
    
    # Test local dependencies only (no network calls)
    working_features = 0
    
    # Check if we have API keys configured
    if CONGRESS_API_KEY:
        print("✅ Congress API key available")
        working_features += 1
        FeatureFlags.CONGRESS_GOV_AVAILABLE = True
    else:
        print("⚠️ Congress API key not set")
        FeatureFlags.CONGRESS_GOV_AVAILABLE = False
    
    if DATA_GOV_API_KEY:
        print("✅ Data.gov API key available") 
        working_features += 1
        FeatureFlags.DATA_GOV_AVAILABLE = True
    else:
        print("⚠️ Data.gov API key not set")
        FeatureFlags.DATA_GOV_AVAILABLE = False
    
    # Update feature flags based on available keys
    FeatureFlags.EPA_APIS_AVAILABLE = FeatureFlags.DATA_GOV_AVAILABLE
    FeatureFlags.ENVIRONMENTAL_RESEARCH_ENABLED = FeatureFlags.EPA_APIS_AVAILABLE
    FeatureFlags.HOUSING_APIS_AVAILABLE = FeatureFlags.DATA_GOV_AVAILABLE
    FeatureFlags.HOUSING_DATA_RESEARCH_ENABLED = FeatureFlags.HOUSING_APIS_AVAILABLE
    FeatureFlags.LABOR_APIS_AVAILABLE = FeatureFlags.DATA_GOV_AVAILABLE
    FeatureFlags.OSHA_RESEARCH_ENABLED = FeatureFlags.LABOR_APIS_AVAILABLE
    FeatureFlags.CRIMINAL_JUSTICE_APIS_AVAILABLE = FeatureFlags.DATA_GOV_AVAILABLE
    FeatureFlags.CRIME_DATA_RESEARCH_ENABLED = FeatureFlags.CRIMINAL_JUSTICE_APIS_AVAILABLE
    
    # Overall comprehensive research capability
    FeatureFlags.COMPREHENSIVE_LEGAL_RESEARCH = working_features >= 1
    
    print(f"📊 Feature Summary: {working_features} API keys configured")
    print(f"🎯 Comprehensive Research: {'ENABLED' if FeatureFlags.COMPREHENSIVE_LEGAL_RESEARCH else 'LIMITED'}")
    
    if FeatureFlags.COMPREHENSIVE_LEGAL_RESEARCH:
        print("🔥 Enhanced legal research features available:")
        print("   • Federal legislation tracking")
        print("   • Environmental compliance data") 
        print("   • Immigration case tracking")
        print("   • Corporate filings and business law")
        print("   • Labor violations and safety data")
        print("   • Healthcare regulations and recalls")
        print("   • Criminal justice statistics")
        print("   • Housing and fair market rent data")
    
    return {"working_apis": working_features}

def initialize_feature_flags():
    """Initialize feature flags by checking for available dependencies"""
    
    print("🔍 Checking dependencies...")
    
    # Check aiohttp
    try:
        import aiohttp
        FeatureFlags.AIOHTTP_AVAILABLE = True
        print("✅ aiohttp available - AI features enabled")
    except ImportError:
        FeatureFlags.AIOHTTP_AVAILABLE = False
        print("⚠️ aiohttp not available - install with: pip install aiohttp")

    # Check OCR
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        FeatureFlags.OCR_AVAILABLE = True
        print("✅ OCR support available - can process scanned PDFs")
    except ImportError:
        FeatureFlags.OCR_AVAILABLE = False
        print("⚠️ OCR not available - install pytesseract and pdf2image for scanned PDFs")

    # Hybrid search is disabled by default for performance
    FeatureFlags.HYBRID_SEARCH_AVAILABLE = False
    print("⚠️ Hybrid search disabled for better performance")
    
    # Check open source NLP
    try:
        import spacy
        import nltk
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = True
        print("✅ Open source NLP available")
    except ImportError:
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = False
        print("⚠️ Open source NLP not available - install spacy and nltk")
    
    # Check PDF processors
    try:
        import fitz  # PyMuPDF
        FeatureFlags.PYMUPDF_AVAILABLE = True
        print("✅ PyMuPDF available for PDF processing")
    except ImportError:
        FeatureFlags.PYMUPDF_AVAILABLE = False
        print("⚠️ PyMuPDF not available - install PyMuPDF for better PDF support")
    
    try:
        import pdfplumber
        FeatureFlags.PDFPLUMBER_AVAILABLE = True
        print("✅ PDFPlumber available for advanced PDF parsing")
    except ImportError:
        FeatureFlags.PDFPLUMBER_AVAILABLE = False
        print("⚠️ PDFPlumber not available - install pdfplumber for table extraction")
    
    try:
        from unstructured.partition.auto import partition
        FeatureFlags.UNSTRUCTURED_AVAILABLE = True
        print("✅ Unstructured available for document parsing")
    except ImportError:
        FeatureFlags.UNSTRUCTURED_AVAILABLE = False
        print("⚠️ Unstructured not available - install unstructured[all-docs] for advanced processing")
    
    # Update AI enabled flag
    FeatureFlags.AI_ENABLED = bool(FeatureFlags.AIOHTTP_AVAILABLE and OPENROUTER_API_KEY)
    
    print(f"📊 Document processing: PyMuPDF={FeatureFlags.PYMUPDF_AVAILABLE}, "
          f"pdfplumber={FeatureFlags.PDFPLUMBER_AVAILABLE}, "
          f"Unstructured={FeatureFlags.UNSTRUCTURED_AVAILABLE}")

    print(f"🤖 AI Status: {'ENABLED' if FeatureFlags.AI_ENABLED else 'DISABLED - Set OPENAI_API_KEY to enable'}")

    # Now run the comprehensive feature initialization (without network calls)
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

# Initialize feature flags when module is imported
initialize_feature_flags()
