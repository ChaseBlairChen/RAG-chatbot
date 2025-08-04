# legal_assistant/processors/query_processor.py - ENHANCED WITH COMPREHENSIVE APIs
"""Query processing logic - Enhanced with comprehensive legal API integration and government data access"""
import re
import logging
import traceback
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from ..models import QueryResponse, ComprehensiveAnalysisRequest, AnalysisType
from ..config import FeatureFlags, OPENROUTER_API_KEY, MIN_RELEVANCE_SCORE
from ..services import (
    ComprehensiveAnalysisProcessor,
    combined_search,
    calculate_confidence_score,
    call_openrouter_api
)
from ..services.external_db_service import (
    search_free_legal_databases, 
    search_free_legal_databases_enhanced,
    search_legal_databases_comprehensive
)
from ..storage.managers import add_to_conversation, get_conversation_context
from ..utils import (
    parse_multiple_questions,
    extract_bill_information,
    extract_universal_information,
    format_context_for_llm
)

# Import comprehensive legal APIs
try:
    from ..services.comprehensive_legal_apis import (
        comprehensive_legal_hub, 
        search_comprehensive_legal_databases
    )
    COMPREHENSIVE_APIS_AVAILABLE = True
except ImportError:
    logger.warning("Comprehensive legal APIs not available")
    COMPREHENSIVE_APIS_AVAILABLE = False
    comprehensive_legal_hub = None

logger = logging.getLogger(__name__)

# Safe import with fallback
try:
    from ..services.news_country_apis import comprehensive_researcher
except ImportError:
    logger.warning("News country APIs not available")
    comprehensive_researcher = None

try:
    from ..utils.immigration_helpers import get_form_info
except ImportError:
    logger.warning("Immigration helpers not available")
    def get_form_info(form_number):
        return f"Form {form_number} information not available"

def extract_statutory_information(context_text: str, statute_citation: str) -> dict:
    """Extract specific information from statutory text"""
    extracted_info = {
        "requirements": [],
        "durations": [],
        "numbers": [],
        "procedures": []
    }
    
    try:
        # Look for duration patterns
        duration_patterns = [
            r"minimum of (\d+) (?:minutes?|hours?)",
            r"at least (\d+) (?:minutes?|hours?)",
            r"(\d+)-(?:minute|hour) (?:minimum|maximum)",
            r"(?:shall|must) be (\d+) (?:minutes?|hours?)"
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, context_text, re.IGNORECASE)
            for match in matches:
                extracted_info["durations"].append(f"{match} minutes/hours")
        
        # Look for numerical requirements
        number_patterns = [
            r"(?:maximum|minimum) of (\d+) (?:participants?|people|individuals?)",
            r"at least (\d+) (?:speakers?|facilitators?|members?)",
            r"no more than (\d+) (?:participants?|attendees?)"
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, context_text, re.IGNORECASE)
            for match in matches:
                extracted_info["numbers"].append(match)
        
        return extracted_info
        
    except Exception as e:
        logger.error(f"Error extracting statutory information: {e}")
        return extracted_info

def format_legal_citation(result: Dict) -> str:
    """Format a legal database result into a proper citation"""
    source_db = result.get('source_database', '').lower()
    
    # Harvard Library Caselaw Access Project format
    if 'harvard' in source_db or 'caselaw' in source_db:
        case_name = result.get('title', 'Unknown Case')
        court = result.get('court', '')
        date = result.get('date', '')
        citation = result.get('citation', '')
        
        if citation:
            # Standard format: Case Name, Citation (Court Year)
            year = date.split('-')[0] if date else ''
            return f"{case_name}, {citation} ({court} {year})"
        else:
            # Fallback format without citation
            return f"{case_name} ({court} {date})"
    
    # CourtListener format
    elif 'courtlistener' in source_db:
        case_name = result.get('title', 'Unknown Case')
        docket = result.get('docket_number', '')
        court = result.get('court', '')
        date = result.get('date', '')
        
        if docket:
            return f"{case_name}, No. {docket} ({court} {date})"
        else:
            return f"{case_name} ({court} {date})"
    
    # Government database formats
    elif 'epa' in source_db:
        facility = result.get('facility_name', result.get('title', 'EPA Action'))
        location = result.get('location', '')
        violation = result.get('violation_type', '')
        date = result.get('date', result.get('citation_date', ''))
        return f"EPA Enforcement: {facility} {location} - {violation} ({date})"
    
    elif 'sec' in source_db:
        company = result.get('company', result.get('title', 'SEC Filing'))
        form_type = result.get('form_type', 'Filing')
        date = result.get('filing_date', result.get('date', ''))
        return f"SEC {form_type}: {company} ({date})"
    
    elif 'osha' in source_db or 'dol' in source_db:
        company = result.get('company', result.get('title', 'DOL Action'))
        violation = result.get('violation_type', 'Violation')
        date = result.get('citation_date', result.get('date', ''))
        return f"OSHA Citation: {company} - {violation} ({date})"
    
    elif 'uscis' in source_db:
        receipt = result.get('receipt_number', '')
        status = result.get('status', result.get('title', 'Case Status'))
        date = result.get('checked_date', result.get('date', ''))
        return f"USCIS Case {receipt}: {status} ({date})"
    
    elif 'fda' in source_db:
        product = result.get('product_description', result.get('title', 'FDA Action'))
        company = result.get('company', result.get('recalling_firm', ''))
        date = result.get('report_date', result.get('date', ''))
        return f"FDA Action: {company} - {product} ({date})"
    
    # Google Scholar format
    elif 'scholar' in source_db or 'google' in source_db:
        case_name = result.get('title', 'Unknown Case')
        citation = result.get('citation', '')
        year = result.get('date', '').split('-')[0] if result.get('date') else ''
        
        if citation:
            return f"{case_name}, {citation} ({year})"
        else:
            return f"{case_name} ({year})"
    
    # Generic format for other sources
    else:
        title = result.get('title', 'Unknown')
        source = result.get('source_database', 'Unknown Source').replace('_', ' ').title()
        date = result.get('date', '')
        
        if date:
            return f"{source}: {title} ({date})"
        else:
            return f"{source}: {title}"

def detect_statutory_question(question: str) -> bool:
    """Detect if this is a statutory/regulatory question requiring detailed extraction"""
    statutory_indicators = [
        # Federal Laws and Regulations
        r'\bUSC\s+\d+', r'\bU\.S\.C\.\s*§?\s*\d+', r'\bCFR\s+\d+', r'\bC\.F\.R\.\s*§?\s*\d+',
        r'\bFed\.\s*R\.\s*Civ\.\s*P\.', r'\bFed\.\s*R\.\s*Crim\.\s*P\.', r'\bFed\.\s*R\.\s*Evid\.',
        
        # Washington State
        r'\bRCW\s+\d+\.\d+\.\d+', r'\bWAC\s+\d+', r'\bWash\.\s*Rev\.\s*Code\s*§?\s*\d+',
        
        # California
        r'\bCal\.\s*(?:Bus\.\s*&\s*Prof\.|Civ\.|Comm\.|Corp\.|Educ\.|Fam\.|Gov\.|Health\s*&\s*Safety|Ins\.|Lab\.|Penal|Prob\.|Pub\.\s*Util\.|Rev\.\s*&\s*Tax\.|Veh\.|Welf\.\s*&\s*Inst\.)\s*Code\s*§?\s*\d+',
        
        # Generic statutory terms
        r'\bstatute[s]?', r'\bregulation[s]?', r'\bcode\s+section[s]?',
        r'\bminimum\s+standards?', r'\brequirements?', r'\bmust\s+meet',
        r'\bstandards?.*regulat', r'\bcomposition.*conduct',
        r'\bpolicies.*record[- ]keeping', r'\bmandatory\s+(?:standards?|requirements?)',
        r'\bshall\s+(?:meet|comply|maintain)', r'\bmust\s+(?:include|contain|provide)',
        r'\brequired\s+(?:by\s+law|under|pursuant\s+to)',
        r'\bregulatory\s+(?:standards?|requirements?|compliance)',
        r'\bstatutory\s+(?:mandate|requirement|provision)',
        r'\blegal\s+(?:standards?|requirements?|obligations?)',
        r'\bcompliance\s+with.*(?:statute|regulation|code)',
        r'\badministrative\s+(?:rule[s]?|regulation[s]?|code)',
    ]
    
    for pattern in statutory_indicators:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False

def detect_legal_search_intent(question: str) -> bool:
    """Detect if this question would benefit from external legal database search"""
    legal_search_indicators = [
        # Case law queries
        r'\bcase\s*law\b', r'\bcases?\b', r'\bprecedent\b', r'\bruling\b',
        r'\bdecision\b', r'\bcourt\s*opinion\b', r'\bjudgment\b',
        
        # Specific legal queries
        r'\bmiranda\b', r'\bconstitutional\b', r'\bamendment\b',
        r'\bsupreme\s*court\b', r'\bappellate\b', r'\bdistrict\s*court\b',
        
        # Legal research terms
        r'\blegal\s*research\b', r'\bfind\s*cases?\b', r'\blook\s*up\s*law\b',
        r'\bsearch\s*(?:for\s*)?(?:cases?|law|precedent)\b',
        
        # Government/enforcement queries
        r'\bviolation\b', r'\benforcement\b', r'\bcitation\b', r'\bpenalty\b',
        r'\bcompliance\b', r'\binspection\b', r'\brecall\b', r'\bwarning\b',
        
        # Legal concepts - expanded
        r'\bliability\b', r'\bnegligence\b', r'\bcontract\s*law\b', r'\btort\b',
        r'\bcriminal\s*law\b', r'\bcivil\s*law\b', r'\bduty\s*of\s*care\b'
    ]
    
    for pattern in legal_search_indicators:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False

def detect_government_data_need(question: str) -> bool:
    """Detect if question needs government enforcement/statistical data"""
    government_indicators = [
        # Enforcement queries
        r'\bviolation\b', r'\benforcement\b', r'\bcitation\b', r'\bpenalty\b',
        r'\bfine\b', r'\binspection\b', r'\bcompliance\b',
        
        # Agency-specific
        r'\bepa\b', r'\bosha\b', r'\bsec\b', r'\bfda\b', r'\buscis\b',
        r'\bfbi\b', r'\bdol\b', r'\bhud\b',
        
        # Data queries
        r'\bstatistics\b', r'\bdata\b', r'\brates?\b', r'\btrends?\b',
        r'\bnumbers?\b', r'\bfigures?\b', r'\breport\b',
        
        # Status queries
        r'\bstatus\b', r'\bprocessing\s*time\b', r'\bcase\s*status\b',
        r'\bcurrent\b', r'\brecent\b', r'\blatest\b'
    ]
    
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in government_indicators)

def detect_immigration_query(question: str) -> bool:
    """Detect if this is an immigration-related query"""
    immigration_indicators = [
        # Case types
        r'\basylum\b', r'\brefugee\b', r'\bgreen\s*card\b', r'\bvisa\b',
        r'\bimmigration\b', r'\bnaturalization\b', r'\bcitizenship\b',
        
        # Forms
        r'\bI-\d{3}\b', r'\bN-\d{3}\b', r'\bUSCIS\b',
        
        # Procedures
        r'\bcredible\s*fear\b', r'\bremoval\b', r'\bdeportation\b',
        r'\bwork\s*permit\b', r'\bEAD\b', r'\bpriority\s*date\b',
        
        # Country conditions
        r'\bcountry\s*conditions?\b', r'\bpersecution\b', r'\bhuman\s*rights\b',
        
        # Status questions
        r'\bcase\s*status\b', r'\bprocessing\s*time\b', r'\binterview\b',
        
        # Receipt numbers
        r'\b[A-Z]{3}\d{10}\b'  # USCIS receipt number format
    ]
    
    for pattern in immigration_indicators:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False

def detect_query_type(question: str) -> List[str]:
    """Enhanced query type detection for comprehensive API routing"""
    query_types = []
    question_lower = question.lower()
    
    # Environmental law patterns
    environmental_patterns = [
        r'\bepa\b', r'\benvironmental\b', r'\bpollution\b', r'\bclimate\b',
        r'\bemissions\b', r'\bclean\s+air\b', r'\bclean\s+water\b',
        r'\bwetlands\b', r'\bendangered\s+species\b', r'\bnepa\b',
        r'\bcercla\b', r'\brcra\b', r'\btsca\b', r'\bsuperfund\b',
        r'\bhazardous\s+waste\b', r'\bair\s+quality\b', r'\bwater\s+quality\b'
    ]
    
    # Business law patterns
    business_patterns = [
        r'\bsec\b', r'\bsecurities\b', r'\bcorporate\b', r'\bfilings?\b',
        r'\bipo\b', r'\bmerger\b', r'\bacquisition\b', r'\b10-k\b', r'\b10-q\b',
        r'\bsba\b', r'\bsmall\s+business\b', r'\bstartup\b', r'\bincorporat',
        r'\bllc\b', r'\bcorporation\b', r'\bpatent\b', r'\btrademark\b'
    ]
    
    # Labor law patterns
    labor_patterns = [
        r'\bosha\b', r'\bworkplace\s+safety\b', r'\blabor\b', r'\bemployment\b',
        r'\bwage\b', r'\bovertime\b', r'\bdiscrimination\b', r'\bharassment\b',
        r'\bworkers?\s+comp\b', r'\bunion\b', r'\bcollective\s+bargaining\b'
    ]
    
    # Healthcare patterns
    healthcare_patterns = [
        r'\bfda\b', r'\bdrug\b', r'\bmedical\s+device\b', r'\brecall\b',
        r'\bhipaa\b', r'\bpatient\s+privacy\b', r'\bmedicare\b', r'\bmedicaid\b',
        r'\bhealthcare\b', r'\bmedical\b', r'\bpharmaceutical\b'
    ]
    
    # Criminal justice patterns
    criminal_patterns = [
        r'\bcriminal\b', r'\bcrime\b', r'\bfbi\b', r'\barrest\b',
        r'\bprosecution\b', r'\bsentencing\b', r'\bprison\b', r'\bfelony\b',
        r'\bmisdemeanor\b', r'\bviolent\s+crime\b', r'\bproperty\s+crime\b'
    ]
    
    # Housing patterns
    housing_patterns = [
        r'\bhousing\b', r'\brental\b', r'\blandlord\b', r'\btenant\b',
        r'\beviction\b', r'\bfair\s+housing\b', r'\bhud\b', r'\bsection\s+8\b',
        r'\brent\s+control\b', r'\bhomelessness\b', r'\baffordable\s+housing\b'
    ]
    
    # Check each pattern type
    if any(re.search(pattern, question_lower) for pattern in environmental_patterns):
        query_types.append('environmental')
    
    if any(re.search(pattern, question_lower) for pattern in business_patterns):
        query_types.append('business')
    
    if any(re.search(pattern, question_lower) for pattern in labor_patterns):
        query_types.append('labor')
    
    if any(re.search(pattern, question_lower) for pattern in healthcare_patterns):
        query_types.append('healthcare')
    
    if any(re.search(pattern, question_lower) for pattern in criminal_patterns):
        query_types.append('criminal')
    
    if any(re.search(pattern, question_lower) for pattern in housing_patterns):
        query_types.append('housing')
    
    # Existing checks
    if detect_statutory_question(question):
        query_types.append('statutory')
    
    if detect_legal_search_intent(question):
        query_types.append('case_law')
    
    if detect_immigration_query(question):
        query_types.append('immigration')
    
    if detect_government_data_need(question):
        query_types.append('government_data')
    
    return query_types

def should_search_comprehensive_databases(question: str, search_scope: str) -> bool:
    """Determine if comprehensive government databases should be searched"""
    
    # Don't search external if user explicitly wants only their documents
    if search_scope == "user_only":
        return False
    
    # Check for government data needs
    if detect_government_data_need(question):
        return True
    
    # Check for legal areas that benefit from comprehensive search
    query_types = detect_query_type(question)
    comprehensive_areas = ['environmental', 'business', 'labor', 'healthcare', 'criminal', 'immigration', 'housing']
    
    if any(area in query_types for area in comprehensive_areas):
        return True
    
    # Check for enforcement/compliance questions
    enforcement_keywords = ['violation', 'enforcement', 'compliance', 'citation', 'penalty', 'recall']
    if any(keyword in question.lower() for keyword in enforcement_keywords):
        return True
    
    return detect_legal_search_intent(question) or detect_statutory_question(question)

def enhanced_external_search_with_comprehensive_apis(question: str, query_types: List[str], 
                                                    search_external: bool) -> Tuple[str, List[Dict]]:
    """Enhanced external search using comprehensive legal APIs and government databases"""
    
    if not search_external:
        return None, []
    
    external_context = None
    external_source_info = []
    
    try:
        logger.info("🔍 Searching comprehensive legal databases and government APIs...")
        
        # 1. Search comprehensive APIs if available (government data, enforcement, etc.)
        if COMPREHENSIVE_APIS_AVAILABLE:
            try:
                comprehensive_results = search_comprehensive_legal_databases(
                    query=question, 
                    auto_detect_areas=True
                )
                
                if comprehensive_results:
                    logger.info(f"🏛️ Found {len(comprehensive_results)} results from government APIs")
                    
                    # Format comprehensive results with proper citations
                    comp_context, comp_source_info = format_comprehensive_results_with_citations(
                        comprehensive_results
                    )
                    
                    if comp_context:
                        external_context = comp_context
                        external_source_info.extend(comp_source_info)
                        
            except Exception as e:
                logger.error(f"Comprehensive API search failed: {e}")
        
        # 2. Search traditional legal databases for legal context
        try:
            # Use enhanced search with detected query types
            traditional_results = search_free_legal_databases_enhanced(
                question, None, query_types
            )
            
            if traditional_results:
                logger.info(f"📚 Found {len(traditional_results)} results from traditional legal databases")
                
                # Format traditional results
                traditional_context, traditional_source_info = format_external_results_with_citations(
                    traditional_results
                )
                
                # Combine contexts
                if external_context and traditional_context:
                    external_context = f"{external_context}\n\n{traditional_context}"
                    external_source_info.extend(traditional_source_info)
                elif traditional_context and not external_context:
                    external_context = traditional_context
                    external_source_info = traditional_source_info
                    
        except Exception as e:
            logger.error(f"Traditional legal database search failed: {e}")
        
        # 3. Fallback to basic search if nothing found
        if not external_context:
            logger.info("🔄 Falling back to basic legal database search...")
            try:
                basic_results = search_free_legal_databases(question, None)
                if basic_results:
                    external_context, external_source_info = format_external_results_with_citations(
                        basic_results
                    )
            except Exception as e:
                logger.error(f"Basic external search also failed: {e}")
    
    except Exception as e:
        logger.error(f"Enhanced external search failed completely: {e}")
    
    return external_context, external_source_info

def format_comprehensive_results_with_citations(comprehensive_results: List[Dict]) -> Tuple[str, List[Dict]]:
    """Format comprehensive API results with proper citations and government data emphasis"""
    
    if not comprehensive_results:
        return "", []
    
    # Group results by legal area and source type
    results_by_category = {}
    source_info = []
    
    for idx, result in enumerate(comprehensive_results):
        legal_area = result.get('legal_area', result.get('enforcement_area', 'general'))
        source_db = result.get('source_database', 'unknown')
        detected_state = result.get('detected_state', 'Federal')
        
        if legal_area not in results_by_category:
            results_by_category[legal_area] = []
        
        results_by_category[legal_area].append(result)
        
        # Create source info entry
        source_info.append({
            'file_name': format_legal_citation(result),
            'page': None,
            'relevance': result.get('relevance_score', 0.9),  # Government data gets high relevance
            'source_type': 'comprehensive_government_database',
            'database': source_db,
            'legal_area': legal_area,
            'jurisdiction': detected_state,
            'url': result.get('url', ''),
            'date': result.get('date', ''),
            'authority_level': 'very_high' if any(gov in source_db for gov in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi']) else 'high'
        })
    
    # Format the context text
    formatted_text = "\n\n## COMPREHENSIVE LEGAL & GOVERNMENT DATABASE RESULTS:\n"
    formatted_text += "(Official government enforcement data, regulatory information, and authoritative legal sources)\n\n"
    
    for legal_area, area_results in results_by_category.items():
        if not area_results:
            continue
            
        # Add section header for this legal area
        area_title = legal_area.replace('_', ' ').title()
        if area_title.lower() != 'general':
            formatted_text += f"### {area_title} - Official Data & Legal Framework\n"
        
        for idx, result in enumerate(area_results, 1):
            # Create proper citation
            citation = format_legal_citation(result)
            
            formatted_text += f"#### {idx}. {citation}\n"
            
            # Add database attribution with authority indicator
            source_db = result.get('source_database', 'Unknown')
            if any(gov in source_db for gov in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi', 'congress', 'federal_register']):
                authority_indicator = "🏛️"
                authority_text = "Official U.S. Government Database"
            elif any(academic in source_db for academic in ['harvard', 'cornell']):
                authority_indicator = "🎓"
                authority_text = "Academic Legal Institution"
            else:
                authority_indicator = "📊"
                authority_text = "Legal Database"
            
            formatted_text += f"**Source:** {authority_indicator} {authority_text} - {format_database_name(source_db)}\n"
            
            # Add jurisdiction info
            jurisdiction = result.get('detected_state') or result.get('state', 'Federal')
            formatted_text += f"**Jurisdiction:** {jurisdiction}\n"
            
            # Add date if available
            date = (result.get('date') or result.get('filing_date') or 
                   result.get('report_date') or result.get('citation_date'))
            if date:
                formatted_text += f"**Date:** {date}\n"
            
            # Add government-specific data
            if 'violation_type' in result:
                formatted_text += f"**Violation Type:** {result['violation_type']}\n"
            if 'penalty' in result and result['penalty']:
                formatted_text += f"**Penalty:** {result['penalty']}\n"
            if 'status' in result and result['status']:
                formatted_text += f"**Status:** {result['status']}\n"
            if 'classification' in result:
                formatted_text += f"**Classification:** {result['classification']}\n"
            if 'company' in result and result['company']:
                formatted_text += f"**Company/Facility:** {result['company']}\n"
            
            # Add description/summary
            description = (result.get('description') or result.get('summary') or 
                         result.get('reason_for_recall') or result.get('snippet', ''))
            
            if description:
                # Clean and truncate description
                clean_description = re.sub(r'\s+', ' ', description).strip()
                if len(clean_description) > 400:
                    clean_description = clean_description[:397] + "..."
                formatted_text += f"**Details:** {clean_description}\n"
            
            # Add URL for full details
            if result.get('url'):
                formatted_text += f"**Full Information:** [View Official Source]({result['url']})\n"
            
            formatted_text += "\n"
    
    return formatted_text, source_info

def format_database_name(source_db: str) -> str:
    """Format database name for display"""
    database_names = {
        'epa_echo': 'EPA Enforcement & Compliance History Online',
        'epa_air_quality': 'EPA Air Quality System',
        'epa_water_quality': 'EPA Water Quality Portal',
        'sec_edgar': 'SEC EDGAR Corporate Filings Database',
        'dol_osha': 'Department of Labor OSHA Enforcement Database',
        'dol_wage_hour': 'DOL Wage & Hour Division Enforcement',
        'fda_drug_enforcement': 'FDA Drug Enforcement Reports',
        'fda_device_enforcement': 'FDA Medical Device Safety Reports', 
        'fda_food_enforcement': 'FDA Food Safety & Recalls',
        'uscis_case_status': 'USCIS Case Status System',
        'state_dept_visa_bulletin': 'State Department Visa Bulletin',
        'hud_fair_market_rents': 'HUD Fair Market Rent Database',
        'census_housing': 'U.S. Census Bureau Housing Statistics',
        'fbi_crime_data': 'FBI Crime Data Explorer',
        'uspto_patents': 'USPTO Patent Database',
        'uspto_trademarks': 'USPTO Trademark Database',
        'google_patents': 'Google Patents Search',
        'bls_data': 'Bureau of Labor Statistics',
        'congress_gov': 'Congress.gov Official Legislative Database',
        'openstates': 'OpenStates Legislative Tracking',
        'justia': 'Justia Free Law Database',
        'cornell_law': 'Cornell Law School Legal Information Institute',
        'harvard_caselaw': 'Harvard Law School Caselaw Access Project',
        'courtlistener': 'CourtListener Federal & State Court Database',
        'federal_register': 'Federal Register - Official U.S. Government Regulations'
    }
    
    return database_names.get(source_db, source_db.replace('_', ' ').title())

def format_external_results_with_citations(external_results: List[Dict]) -> Tuple[str, List[Dict]]:
    """Format external results with proper citations and return both formatted text and source info"""
    if not external_results:
        return "", []
    
    formatted_text = "\n\n## LEGAL DATABASE RESULTS:\n"
    formatted_text += "(Results from authoritative legal databases and academic institutions)\n\n"
    
    source_info = []
    
    for idx, result in enumerate(external_results[:5], 1):  # Limit to top 5 results
        # Create proper citation
        citation = format_legal_citation(result)
        
        formatted_text += f"### {idx}. {citation}\n"
        
        # Add database attribution
        source_db = result.get('source_database', 'Unknown')
        formatted_text += f"**Database:** {format_database_name(source_db)}\n"
        
        # Add court and jurisdiction info
        court = result.get('court', '')
        jurisdiction = result.get('jurisdiction', '')
        if court:
            formatted_text += f"**Court:** {court}\n"
        if jurisdiction:
            formatted_text += f"**Jurisdiction:** {jurisdiction}\n"
        
        # Add date decided
        date = result.get('date', '')
        if date:
            formatted_text += f"**Date:** {date}\n"
        
        # Add relevant excerpt/preview
        preview = result.get('preview') or result.get('snippet', '') or result.get('description', '')
        if preview:
            # Clean up the preview text
            preview = re.sub(r'\s+', ' ', preview).strip()
            if len(preview) > 500:
                preview = preview[:497] + "..."
            formatted_text += f"**Relevant Excerpt:** {preview}\n"
        
        # Add URL for full text
        if result.get('url'):
            formatted_text += f"**Full Text:** [View Source]({result['url']})\n"
        
        formatted_text += "\n"
        
        # Create source info entry with proper citation
        source_info.append({
            'file_name': citation,  # Use the formatted citation as the file name
            'page': None,
            'relevance': result.get('relevance_score', 0.8),
            'source_type': 'external_legal_database',
            'database': source_db,
            'url': result.get('url', ''),
            'court': court,
            'date': date
        })
    
    return formatted_text, source_info

def create_comprehensive_legal_prompt(context_text: str, question: str, conversation_context: str,
                                    sources_searched: list, retrieval_method: str, 
                                    document_id: str = None, instruction: str = "balanced", 
                                    external_context: str = None, detected_areas: List[str] = None) -> str:
    """Create prompt for comprehensive legal research questions with government data"""
    
    # Combine contexts
    if external_context:
        full_context = f"{context_text}\n\n{external_context}"
    else:
        full_context = context_text
    
    legal_areas_text = ", ".join(detected_areas) if detected_areas else "general legal research"
    
    return f"""You are a comprehensive legal research assistant with access to official government databases, enforcement data, and authoritative legal sources.

DETECTED LEGAL AREAS: {legal_areas_text}

SOURCE HIERARCHY & AUTHORITY:
- **HIGHEST AUTHORITY** (Official Government Sources): 
  🏛️ Federal enforcement databases (EPA, SEC, DOL, FDA, USCIS, FBI)
  🏛️ Official legislative sources (Congress.gov, Federal Register)
  🏛️ Government statistical and regulatory data

- **VERY HIGH AUTHORITY** (Legal Authorities):
  🎓 Federal and state statutes from authoritative academic sources
  🎓 Court decisions from Harvard Caselaw Access Project and CourtListener
  📚 Federal regulations and administrative guidance

- **HIGH AUTHORITY** (Reliable Legal Sources):
  📊 Professional legal databases (Justia, Cornell Law)
  📊 State legislative tracking (OpenStates)
  📊 Legal academic resources

COMPREHENSIVE RESEARCH APPROACH:
1. **Government Data First**: When available, lead with official enforcement actions, violations, statistics, and regulatory compliance data
2. **Legal Framework**: Provide the controlling legal authorities (statutes, regulations, case law) that create the obligations
3. **Practical Examples**: Use real enforcement actions and government data to illustrate legal requirements
4. **Current Status**: Include recent developments, enforcement trends, or regulatory updates
5. **Multi-Jurisdictional**: Address both federal and state law when relevant
6. **Compliance Guidance**: Explain practical steps for legal compliance based on enforcement patterns

ENHANCED CITATION & AUTHORITY REQUIREMENTS:
- Use official database names with authority indicators (🏛️ for government, 🎓 for academic, 📊 for legal databases)
- Include jurisdiction information (Federal/State) and enforcement agency when applicable
- Provide specific details (case numbers, violation dates, penalty amounts, filing information)
- Link to official sources when available
- Emphasize government enforcement data as the most current and practical guidance

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}
{f"DOCUMENT FILTER: Specific document {document_id}" if document_id else "DOCUMENT SCOPE: All available sources including government databases"}

RESPONSE STYLE: {instruction}

CONVERSATION HISTORY:
{conversation_context}

COMPREHENSIVE LEGAL CONTEXT (Government databases, enforcement data, legal authorities):
{full_context}

USER QUESTION:
{question}

RESPONSE INSTRUCTIONS:
- **Lead with government data** when available (enforcement actions, violations, official statistics, regulatory guidance)
- **Provide specific, actionable information** from official sources with concrete examples
- **Include enforcement patterns and compliance guidance** from government databases
- **Explain the underlying legal framework** using primary legal authorities
- **Note jurisdictional differences** and enforcement variations by agency/state
- **Include practical compliance steps** based on real enforcement examples
- **Cite official sources** with authority indicators for maximum credibility

LEGAL RESEARCH FRAMEWORK:
- When government databases provide enforcement or violation data, lead with that concrete information
- Follow with the underlying legal authorities (statutes, regulations, case law) that create the legal obligations
- Include practical guidance on compliance based on enforcement patterns and agency guidance
- Note recent enforcement trends, policy changes, or regulatory updates
- Provide actionable next steps for legal compliance or further research

GOVERNMENT DATA INTEGRATION:
- EPA data shows actual environmental violations and enforcement priorities
- SEC data reveals corporate compliance issues and enforcement actions
- DOL/OSHA data demonstrates workplace safety violations and penalties
- FDA data tracks product recalls and safety enforcement
- USCIS data provides real-time immigration case processing information
- FBI data shows crime statistics and enforcement patterns

RESPONSE:"""

def create_statutory_prompt(context_text: str, question: str, conversation_context: str, 
                          sources_searched: list, retrieval_method: str, document_id: str = None,
                          external_context: str = None) -> str:
    """Create an enhanced prompt specifically for statutory analysis"""
    
    # Add external context if available
    if external_context:
        context_text = f"{context_text}\n\n{external_context}"
    
    return f"""You are a legal research assistant specializing in statutory analysis. Your job is to extract COMPLETE, SPECIFIC information from legal documents.

STRICT SOURCE REQUIREMENTS:
- Answer ONLY based on the retrieved documents provided in the context
- Do NOT use general legal knowledge, training data, assumptions, or inferences beyond what's explicitly stated
- If information is not in the provided documents, state: "This information is not available in the provided documents"
- When citing external database results, use the full citation format provided

🔴 CRITICAL: You MUST extract actual numbers, durations, and specific requirements. NEVER use placeholders like "[duration not specified]" or "[requirements not listed]".

🔥 CRITICAL FAILURE PREVENTION:
- If you write "[duration not specified]" you have FAILED at your job
- If you write "[duties not specified]" you have FAILED at your job  
- If you write "[requirements not listed]" you have FAILED at your job
- READ EVERY WORD of the context before claiming information is missing
- The human is counting on you to find the actual requirements

MANDATORY EXTRACTION RULES:
1. 📖 READ EVERY WORD of the provided context before claiming anything is missing
2. 🔢 EXTRACT ALL NUMBERS: durations (60 minutes), quantities (25 people), percentages, dollar amounts
3. 📝 QUOTE EXACT LANGUAGE: Use quotation marks for statutory text
4. 📋 LIST ALL REQUIREMENTS: Number each requirement found (1., 2., 3., etc.)
5. 🎯 BE SPECIFIC: Include section numbers, subsection letters, paragraph numbers
6. ⚠️ ONLY claim information is "missing" after thorough analysis of ALL provided text

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}
{f"DOCUMENT FILTER: Specific document {document_id}" if document_id else "DOCUMENT SCOPE: All available documents"}

RESPONSE FORMAT REQUIRED FOR STATUTORY QUESTIONS:

## SPECIFIC REQUIREMENTS FOUND:
[List each requirement with exact quotes and citations]

## NUMERICAL STANDARDS:
[Extract ALL numbers, durations, thresholds, limits]

## PROCEDURAL RULES:
[Detail composition, conduct, attendance, record-keeping rules]

## ROLES AND RESPONSIBILITIES:
[Define each party's specific duties with citations]

## ENFORCEMENT DATA (if available):
[Include any government enforcement examples, violations, or compliance guidance]

## INFORMATION NOT FOUND:
[Only list what is genuinely absent after thorough review]

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT TO ANALYZE WORD-BY-WORD:
{context_text}

USER QUESTION REQUIRING COMPLETE EXTRACTION:
{question}

RESPONSE:"""

def create_regular_prompt(context_text: str, question: str, conversation_context: str, 
                         sources_searched: list, retrieval_method: str, document_id: str = None, 
                         instruction: str = "balanced", external_context: str = None) -> str:
    """Create the regular prompt for non-statutory questions"""
    
    # Add external context if available
    if external_context:
        context_text = f"{context_text}\n\n{external_context}"
    
    return f"""You are a legal research assistant with access to comprehensive legal databases and government enforcement data.

SOURCE HIERARCHY:
- **PRIMARY**: Information from the retrieved documents provided in the context
- **GOVERNMENT DATA**: Official enforcement actions, violations, and regulatory guidance (highest authority)
- **LEGAL AUTHORITIES**: Statutes, regulations, and case law from authoritative sources
- **SECONDARY**: General legal knowledge ONLY when documents are unavailable

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}
{f"DOCUMENT FILTER: Specific document {document_id}" if document_id else "DOCUMENT SCOPE: All available documents"}

RESPONSE STYLE: {instruction}

CONVERSATION HISTORY:
{conversation_context}

COMPREHENSIVE LEGAL CONTEXT:
{context_text}

USER QUESTION:
{question}

RESPONSE APPROACH:
- **FIRST**: Check for official government data (enforcement actions, violations, compliance guidance)
- **SECOND**: Provide legal framework from authoritative sources (statutes, regulations, case law)
- **THIRD**: Include practical compliance guidance based on enforcement patterns
- **FOURTH**: Note jurisdictional variations and recent developments
- **ALWAYS**: Cite sources with authority indicators (🏛️ government, 🎓 academic, 📊 legal databases)

RESPONSE:"""

def create_immigration_prompt(context_text: str, question: str, conversation_context: str,
                            sources_searched: list, retrieval_method: str, document_id: str = None) -> str:
    """Create prompt specifically for immigration queries"""
    
    return f"""You are an immigration legal assistant with access to official USCIS data, visa bulletins, and immigration law databases.

IMMIGRATION-SPECIFIC AUTHORITY SOURCES:
🏛️ **Official Government**: USCIS case status, State Dept visa bulletins, immigration court data
🎓 **Legal Authorities**: Immigration statutes, regulations, and federal court decisions
📊 **Professional Resources**: Immigration law databases and practice guides

IMPORTANT IMMIGRATION CONTEXT:
- Always note that immigration law is complex and changes frequently
- Processing times and requirements vary by service center and case type
- Country conditions can change rapidly affecting asylum claims
- Each case is unique and requires individual assessment

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}

When answering immigration questions:
1. **Lead with official data** - USCIS processing times, case status, visa bulletin information
2. **Be specific about forms and procedures** - Include form numbers, deadlines, and requirements
3. **Cite authoritative sources** - USCIS, State Dept, federal courts, and immigration statutes
4. **Include country conditions when relevant** - For asylum/refugee claims
5. **Note time-sensitive information** - Priority dates, filing deadlines, age-out issues
6. **Highlight critical warnings** - Bars to admission, unlawful presence, criminal issues

CONVERSATION HISTORY:
{conversation_context}

IMMIGRATION CONTEXT (including official government data):
{context_text}

USER QUESTION:
{question}

RESPONSE INSTRUCTIONS:
- Provide practical, actionable guidance based on official sources
- Include specific form numbers, deadlines, and current processing information
- Note any recent policy changes or enforcement priorities
- Always include disclaimer about consulting an immigration attorney

RESPONSE:"""

def consolidate_sources(source_info: List[Dict]) -> List[Dict]:
    """Consolidate multiple chunks from the same document into meaningful source entries"""
    from collections import defaultdict
    
    # Group sources by document
    document_groups = defaultdict(list)
    
    for source in source_info:
        key = (source['file_name'], source['source_type'])
        document_groups[key].append(source)
    
    consolidated_sources = []
    
    for (file_name, source_type), sources in document_groups.items():
        if source_type in ['external_legal_database', 'comprehensive_government_database']:
            # External sources should not be consolidated - each is unique
            consolidated_sources.extend(sources)
        else:
            # For internal documents, consolidate chunks
            relevance_scores = [s['relevance'] for s in sources]
            max_relevance = max(relevance_scores)
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            
            # Get unique pages if available
            pages = [s['page'] for s in sources if s.get('page') is not None]
            unique_pages = sorted(set(pages)) if pages else []
            
            consolidated_source = {
                'file_name': file_name,
                'source_type': source_type,
                'relevance': max_relevance,
                'avg_relevance': avg_relevance,
                'num_chunks': len(sources),
                'pages': unique_pages,
                'authority_level': sources[0].get('authority_level', 'medium')
            }
            
            consolidated_sources.append(consolidated_source)
    
    # Sort by authority level first, then relevance
    authority_order = {'very_high': 4, 'high': 3, 'medium_high': 2, 'medium': 1, 'low': 0}
    consolidated_sources.sort(
        key=lambda x: (authority_order.get(x.get('authority_level', 'medium'), 1), x.get('relevance', 0)), 
        reverse=True
    )
    
    return consolidated_sources

def format_source_display(source: Dict) -> str:
    """Format a source for display in the response with authority indicators"""
    source_type = source['source_type'].replace('_', ' ').title()
    authority_level = source.get('authority_level', 'medium')
    
    # Authority indicators
    if authority_level == 'very_high':
        authority_icon = "🏛️"
    elif authority_level == 'high':
        authority_icon = "🎓"
    else:
        authority_icon = "📊"
    
    if source_type in ['External Legal Database', 'Comprehensive Government Database']:
        # External sources with authority indicators
        display = f"- {authority_icon} {source['file_name']}"
        
        if source.get('database'):
            db_name = format_database_name(source['database'])
            display += f" [{db_name}]"
        
        if source.get('url'):
            display += f" - [Full Text]({source['url']})"
        
        return display
    else:
        # Internal document format
        display = f"- {authority_icon} {source['file_name']}"
        
        # Add page information if available
        if source.get('pages'):
            if len(source['pages']) == 1:
                display += f" (Page {source['pages'][0]})"
            elif len(source['pages']) > 1:
                if len(source['pages']) <= 3:
                    display += f" (Pages {', '.join(map(str, source['pages']))})"
                else:
                    display += f" (Pages {source['pages'][0]}-{source['pages'][-1]})"
        
        # Add relevance score with authority context
        relevance = source['relevance']
        if relevance <= 1.0:
            relevance_pct = relevance * 100
        else:
            relevance_pct = relevance
            
        if relevance_pct >= 80:
            relevance_label = "Excellent match"
        elif relevance_pct >= 60:
            relevance_label = "Good match"
        elif relevance_pct >= 40:
            relevance_label = "Fair match"
        else:
            relevance_label = "Partial match"
        
        display += f" ({relevance_label}: {relevance_pct:.0f}/100, {authority_level.replace('_', ' ').title()} Authority)"
        
        return display

def process_query(question: str, session_id: str, user_id: Optional[str], search_scope: str, 
                 response_style: str = "balanced", use_enhanced_rag: bool = True, 
                 document_id: str = None, search_external: bool = None) -> QueryResponse:
    """Main query processing function with comprehensive legal API integration"""
    try:
        logger.info(f"Processing query - Question: '{question}', User: {user_id}, Scope: {search_scope}, Enhanced: {use_enhanced_rag}, Document: {document_id}")
        
        # Enhanced query type detection
        query_types = detect_query_type(question)
        logger.info(f"Detected query types: {query_types}")
        
        # Check for specific question types
        is_statutory = detect_statutory_question(question)
        is_immigration = 'immigration' in query_types
        needs_government_data = detect_government_data_need(question)
        
        if is_statutory:
            logger.info("🏛️ Detected statutory/regulatory question - using enhanced extraction")
        if is_immigration:
            logger.info("🗽 Detected immigration-related query")
        if needs_government_data:
            logger.info("📊 Detected need for government enforcement/statistical data")
        
        # Handle country conditions for immigration
        country_context = ""
        if is_immigration and comprehensive_researcher:
            country_conditions_match = re.search(
                r'(?:country\s*conditions?|human\s*rights?|persecution|asylum\s*claim)\s*(?:for|in|about)?\s*([A-Z][a-zA-Z\s]+)',
                question, re.IGNORECASE
            )
            
            if country_conditions_match:
                country = country_conditions_match.group(1).strip()
                logger.info(f"📍 Country conditions query detected for: {country}")
                
                try:
                    country_results = comprehensive_researcher.research_all_sources(
                        country=country,
                        topics=['persecution', 'human_rights', 'government', 'violence'],
                        include_multilingual=True
                    )
                    
                    country_context = f"""
COMPREHENSIVE COUNTRY CONDITIONS RESEARCH FOR {country.upper()}:
{country_results.get('summary', '')}
"""
                except Exception as e:
                    logger.error(f"Country conditions research failed: {e}")
        
        # Handle immigration form queries
        form_context = ""
        if is_immigration:
            form_match = re.search(r'\b(I-\d{3}|N-\d{3})\b', question, re.IGNORECASE)
            if form_match:
                form_number = form_match.group(1).upper()
                logger.info(f"📋 Form-specific query for: {form_number}")
                form_context = f"\nFORM {form_number} INFORMATION:\n{get_form_info(form_number)}\n"
        
        # Determine comprehensive search strategy
        if search_external is None:
            search_external = should_search_comprehensive_databases(question, search_scope)
        
        if search_external:
            logger.info("📚 Will search comprehensive legal databases and government APIs")
        
        # Handle comprehensive analysis requests
        if any(phrase in question.lower() for phrase in ["comprehensive analysis", "complete analysis", "full analysis"]):
            logger.info("Detected comprehensive analysis request")
            
            try:
                comp_request = ComprehensiveAnalysisRequest(
                    document_id=document_id,
                    analysis_types=[AnalysisType.COMPREHENSIVE],
                    user_id=user_id or "default_user",
                    session_id=session_id,
                    response_style=response_style
                )
                
                processor = ComprehensiveAnalysisProcessor()
                comp_result = processor.process_comprehensive_analysis(comp_request)
                
                formatted_response = f"""# Comprehensive Legal Document Analysis

## Document Summary
{comp_result.document_summary or 'No summary available'}

## Key Clauses Analysis
{comp_result.key_clauses or 'No clauses analysis available'}

## Risk Assessment
{comp_result.risk_assessment or 'No risk assessment available'}

## Timeline & Deadlines
{comp_result.timeline_deadlines or 'No timeline information available'}

## Party Obligations
{comp_result.party_obligations or 'No obligations analysis available'}

## Missing Clauses Analysis
{comp_result.missing_clauses or 'No missing clauses analysis available'}

---
**Analysis Confidence:** {comp_result.overall_confidence:.1%}
**Processing Time:** {comp_result.processing_time:.2f} seconds
"""
                
                add_to_conversation(session_id, "user", question)
                add_to_conversation(session_id, "assistant", formatted_response)
                
                return QueryResponse(
                    response=formatted_response,
                    error=None,
                    context_found=True,
                    sources=comp_result.sources_by_section.get('summary', []),
                    session_id=session_id,
                    confidence_score=comp_result.overall_confidence,
                    expand_available=False,
                    sources_searched=["comprehensive_analysis"],
                    retrieval_method=comp_result.retrieval_method
                )
                
            except Exception as e:
                logger.error(f"Comprehensive analysis failed: {e}")
        
        # Parse questions and get conversation context
        questions = parse_multiple_questions(question) if use_enhanced_rag else [question]
        combined_query = " ".join(questions)
        conversation_context = get_conversation_context(session_id)
        
        # Adjust search parameters based on question type
        search_k = 20 if is_statutory else 15 if needs_government_data else 10
        
        # Search internal documents
        retrieved_results, sources_searched, retrieval_method = combined_search(
            combined_query, 
            user_id, 
            search_scope, 
            conversation_context,
            use_enhanced=use_enhanced_rag,
            k=search_k,
            document_id=document_id
        )
        
        # Search comprehensive external databases
        external_context = None
        external_source_info = []
        
        if search_external:
            external_context, external_source_info = enhanced_external_search_with_comprehensive_apis(
                question, query_types, search_external
            )
            
            if external_context:
                sources_searched.append("comprehensive_legal_databases")
        
        # Check if we have any results
        if not retrieved_results and not external_context:
            return QueryResponse(
                response="I couldn't find any relevant information to answer your question in the searched sources.",
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.1,
                sources_searched=sources_searched,
                retrieval_method=retrieval_method
            )
        
        # Format context for LLM
        max_context_length = 8000 if is_statutory else 6000 if needs_government_data else 4000
        context_text, source_info = format_context_for_llm(retrieved_results, max_length=max_context_length)
        
        # Add metadata to source_info
        for i, (doc, score) in enumerate(retrieved_results[:len(source_info)]):
            if hasattr(doc, 'metadata'):
                source_info[i]['metadata'] = doc.metadata
                if 'chunk_index' in doc.metadata:
                    source_info[i]['chunk_index'] = doc.metadata['chunk_index']
        
        # Combine source info from internal and external sources
        all_source_info = source_info + external_source_info
        
        # Enhanced information extraction
        bill_match = re.search(r"(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)", question, re.IGNORECASE)
        statute_match = re.search(r"(RCW|USC|CFR|WAC)\s+(\d+\.\d+\.\d+|\d+)", question, re.IGNORECASE)
        extracted_info = {}

        if bill_match:
            bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
            logger.info(f"Searching for bill: {bill_number}")
            extracted_info = extract_bill_information(context_text, bill_number)
        elif statute_match:
            statute_citation = f"{statute_match.group(1)} {statute_match.group(2)}"
            logger.info(f"🏛️ Searching for statute: {statute_citation}")
            extracted_info = extract_statutory_information(context_text, statute_citation)
        else:
            extracted_info = extract_universal_information(context_text, question)

        # Add extracted information to context
        if extracted_info:
            enhancement = "\n\nKEY INFORMATION EXTRACTED:\n"
            for key, value in extracted_info.items():
                if value:
                    if isinstance(value, list):
                        enhancement += f"- {key.replace('_', ' ').title()}: {', '.join(value[:5])}\n"
                    else:
                        enhancement += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            if enhancement.strip() != "KEY INFORMATION EXTRACTED:":
                context_text += enhancement
        
        # Add immigration-specific context
        if country_context:
            context_text = country_context + "\n\n" + context_text
        if form_context:
            context_text = form_context + "\n\n" + context_text
        
        # Style instructions
        style_instructions = {
            "concise": "Please provide a concise answer (1-2 sentences) based on the context.",
            "balanced": "Please provide a balanced answer (2-3 paragraphs) based on the context.",
            "detailed": "Please provide a detailed answer with explanations based on the context."
        }
        
        instruction = style_instructions.get(response_style, style_instructions["balanced"])
        
        # Choose the appropriate prompt based on question type and available data
        detected_areas = [area for area in query_types if area in ['environmental', 'business', 'labor', 'healthcare', 'criminal', 'immigration', 'housing']]
        
        if external_context and (needs_government_data or detected_areas):
            # Use comprehensive prompt when we have government data
            prompt = create_comprehensive_legal_prompt(
                context_text, question, conversation_context,
                sources_searched, retrieval_method, document_id,
                instruction, external_context, detected_areas
            )
        elif is_immigration:
            prompt = create_immigration_prompt(
                context_text, question, conversation_context,
                sources_searched, retrieval_method, document_id
            )
        elif is_statutory:
            prompt = create_statutory_prompt(
                context_text, question, conversation_context, 
                sources_searched, retrieval_method, document_id,
                external_context
            )
        else:
            prompt = create_regular_prompt(
                context_text, question, conversation_context, 
                sources_searched, retrieval_method, document_id, 
                instruction, external_context
            )
        
        # Generate AI response
        if FeatureFlags.AI_ENABLED and OPENROUTER_API_KEY:
            response_text = call_openrouter_api(prompt, OPENROUTER_API_KEY)
        else:
            response_text = f"Based on the retrieved documents and databases:\n\n{context_text}"
            if external_context:
                response_text += f"\n\n{external_context}"
        
        # Add context-specific notices
        if is_immigration:
            response_text += "\n\n**Immigration Law Notice**: This information is general guidance only. Immigration law is complex and changes frequently. Please consult with an immigration attorney for advice specific to your situation."
        
        if needs_government_data and external_source_info:
            response_text += "\n\n**Government Data Notice**: This response includes official government enforcement data and regulatory information. Always verify current requirements with the relevant agencies."
        
        # Filter and consolidate sources
        relevant_sources = [s for s in all_source_info if s['relevance'] >= MIN_RELEVANCE_SCORE]
        consolidated_sources = consolidate_sources(relevant_sources)
        
        # Add source citations to response
        if consolidated_sources:
            response_text += "\n\n**SOURCES:**"
            response_text += "\n*🏛️ = Official Government, 🎓 = Academic/Authoritative, 📊 = Legal Database*"
            for source in consolidated_sources:
                response_text += "\n" + format_source_display(source)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(retrieved_results, len(response_text))
        
        # Add to conversation
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", response_text, all_source_info)
        
        return QueryResponse(
            response=response_text,
            error=None,
            context_found=True,
            sources=all_source_info,
            session_id=session_id,
            confidence_score=float(confidence_score),
            sources_searched=sources_searched,
            expand_available=len(questions) > 1 if use_enhanced_rag else False,
            retrieval_method=retrieval_method
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        return QueryResponse(
            response=None,
            error=str(e),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="error"
        )
