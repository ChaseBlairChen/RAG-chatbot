"""Query processing logic - Enhanced for better statutory analysis with external database integration and proper citation formatting"""
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
from ..services.external_db_service import search_free_legal_databases, search_free_legal_databases_enhanced
from ..storage.managers import add_to_conversation, get_conversation_context
from ..utils import (
    parse_multiple_questions,
    extract_bill_information,
    extract_universal_information,
    format_context_for_llm
)

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
        case_name = result.get('title', 'Unknown Case')
        court = result.get('court', '')
        date = result.get('date', '')
        citation = result.get('citation', '')
        
        parts = [case_name]
        if citation:
            parts.append(citation)
        if court or date:
            parts.append(f"({court} {date})".strip())
        
        return ", ".join(parts)


def detect_statutory_question(question: str) -> bool:
    """Detect if this is a statutory/regulatory question requiring detailed extraction"""
    statutory_indicators = [
        # Federal Laws and Regulations
        r'\bUSC\s+\d+',                    # US Code
        r'\bU\.S\.C\.\s*§?\s*\d+',         # U.S.C. § 123
        r'\bCFR\s+\d+',                    # Code of Federal Regulations
        r'\bC\.F\.R\.\s*§?\s*\d+',         # C.F.R. § 123
        r'\bFed\.\s*R\.\s*Civ\.\s*P\.',    # Federal Rules of Civil Procedure
        r'\bFed\.\s*R\.\s*Crim\.\s*P\.',   # Federal Rules of Criminal Procedure
        r'\bFed\.\s*R\.\s*Evid\.',         # Federal Rules of Evidence
        
        # Washington State
        r'\bRCW\s+\d+\.\d+\.\d+',          # Washington Revised Code
        r'\bWAC\s+\d+',                    # Washington Administrative Code
        r'\bWash\.\s*Rev\.\s*Code\s*§?\s*\d+', # Washington Revised Code alternate format
        
        # California
        r'\bCal\.\s*(?:Bus\.\s*&\s*Prof\.|Civ\.|Comm\.|Corp\.|Educ\.|Fam\.|Gov\.|Health\s*&\s*Safety|Ins\.|Lab\.|Penal|Prob\.|Pub\.\s*Util\.|Rev\.\s*&\s*Tax\.|Veh\.|Welf\.\s*&\s*Inst\.)\s*Code\s*§?\s*\d+',
        r'\bCalifornia\s+(?:Business|Civil|Commercial|Corporation|Education|Family|Government|Health|Insurance|Labor|Penal|Probate|Public\s+Utilities|Revenue|Vehicle|Welfare)\s+Code\s*§?\s*\d+',
        r'\bCal\.\s*Code\s*Regs\.\s*tit\.\s*\d+',  # California Code of Regulations
        
        # Texas
        r'\bTex\.\s*(?:Agric\.|Alco\.|Bus\.\s*&\s*Com\.|Civ\.\s*Prac\.\s*&\s*Rem\.|Code\s*Crim\.\s*Proc\.|Educ\.|Elec\.|Fam\.|Gov\.|Health\s*&\s*Safety|Hum\.\s*Res\.|Ins\.|Lab\.|Loc\.\s*Gov\.|Nat\.\s*Res\.|Occ\.|Parks\s*&\s*Wild\.|Penal|Prop\.|Tax|Transp\.|Util\.|Water)\s*Code\s*(?:Ann\.)?\s*§?\s*\d+',
        r'\bTexas\s+(?:Agriculture|Alcoholic\s+Beverage|Business|Civil\s+Practice|Criminal\s+Procedure|Education|Election|Family|Government|Health|Human\s+Resources|Insurance|Labor|Local\s+Government|Natural\s+Resources|Occupations|Parks|Penal|Property|Tax|Transportation|Utilities|Water)\s+Code\s*§?\s*\d+',
        r'\bTex\.\s*Admin\.\s*Code\s*tit\.\s*\d+',  # Texas Administrative Code
        
        # New York
        r'\bN\.Y\.\s*(?:Agric\.\s*&\s*Mkts\.|Arts\s*&\s*Cult\.\s*Aff\.|Bank\.|Bus\.\s*Corp\.|Civ\.\s*Prac\.\s*L\.\s*&\s*R\.|Civ\.\s*Rights|Civ\.\s*Serv\.|Com\.|Correct\.|County|Crim\.\s*Proc\.|Dom\.\s*Rel\.|Econ\.\s*Dev\.|Educ\.|Elec\.|Empl\.|Energy|Envtl\.\s*Conserv\.|Est\.\s*Powers\s*&\s*Trusts|Exec\.|Fam\.\s*Ct\.\s*Act|Gen\.\s*Bus\.|Gen\.\s*City|Gen\.\s*Constr\.|Gen\.\s*Mun\.|Gen\.\s*Oblig\.|High\.|Indian|Ins\.|Jud\.|Lab\.|Lien|Local\s*Fin\.|Ment\.\s*Hyg\.|Mil\.|Multi-Dwell\.|Multi-Mun\.|Nav\.|Not-for-Profit\s*Corp\.|Parks\s*Rec\.\s*&\s*Hist\.\s*Preserv\.|Penal|Pers\.\s*Prop\.|Priv\.\s*Hous\.\s*Fin\.|Pub\.\s*Auth\.|Pub\.\s*Health|Pub\.\s*Hous\.|Pub\.\s*Off\.|Pub\.\s*Serv\.|Racing\s*Pari-Mut\.\s*Wag\.\s*&\s*Breed\.|Real\s*Prop\.|Real\s*Prop\.\s*Actions\s*&\s*Proc\.|Real\s*Prop\.\s*Tax|Relig\.\s*Corp\.|Retire\.\s*&\s*Soc\.\s*Sec\.|Rural\s*Elec\.\s*Coop\.|Second\s*Class\s*Cities|Soc\.\s*Serv\.|State|State\s*Fin\.|Surr\.\s*Ct\.\s*Proc\.\s*Act|Tax|Town|Transp\.|Transp\.\s*Corp\.|U\.C\.C\.|Unconsol\.|Veh\.\s*&\s*Traf\.|Vill\.|Vol\.\s*Fire\s*Benefit|Workers\'\s*Comp\.)\s*(?:Law)?\s*§?\s*\d+',
        r'\bNew\s+York\s+(?:Criminal\s+Procedure|Penal|Civil\s+Practice|Family\s+Court\s+Act|General\s+Business|Vehicle\s+and\s+Traffic)\s+Law\s*§?\s*\d+',
        r'\bN\.Y\.C\.R\.R\.\s*tit\.\s*\d+',         # New York Codes, Rules and Regulations
        
        # Additional state patterns (condensed for brevity)
        r'\bFla\.\s*Stat\.\s*(?:Ann\.)?\s*§?\s*\d+', # Florida Statutes
        r'\b\d+\s*ILCS\s*\d+', # Illinois Compiled Statutes
        r'\b\d+\s*Pa\.\s*(?:C\.S\.|Cons\.\s*Stat\.)\s*§?\s*\d+', # Pennsylvania
        r'\bOhio\s*Rev\.\s*Code\s*(?:Ann\.)?\s*§?\s*\d+', # Ohio
        r'\bO\.C\.G\.A\.\s*§?\s*\d+',              # Georgia
        r'\bN\.C\.\s*Gen\.\s*Stat\.\s*§?\s*\d+',   # North Carolina
        r'\bMich\.\s*Comp\.\s*Laws\s*(?:Ann\.)?\s*§?\s*\d+', # Michigan
        r'\bN\.J\.\s*Stat\.\s*(?:Ann\.)?\s*§?\s*\d+', # New Jersey
        r'\bVa\.\s*Code\s*(?:Ann\.)?\s*§?\s*\d+',   # Virginia
        
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
        
        # Citations and references
        r'\bv\.\s+\w+\b', r'\bversus\b', r'\bcite\b', r'\bcitation\b',
        
        # Jurisdictional queries
        r'\bfederal\s*(?:law|court)\b', r'\bstate\s*(?:law|court)\b',
        r'\b\w+\s*(?:circuit|district)\b',
        
        # Legal concepts - expanded for negligence and torts
        r'\bliability\b', r'\bnegligence\b', r'\bcontract\s*law\b',
        r'\btort\b', r'\bcriminal\s*law\b', r'\bcivil\s*law\b',
        r'\bduty\s*of\s*care\b', r'\bbreach\b', r'\bcausation\b',
        r'\bproximate\s*cause\b', r'\bdamages\b', r'\bstandard\s*of\s*care\b',
        r'\breasonable\s*person\b', r'\bforeseeability\b', r'\bcomparative\s*negligence\b',
        r'\bcontributory\s*negligence\b', r'\bassumption\s*of\s*risk\b',
        r'\bres\s*ipsa\s*loquitur\b', r'\bvicarious\s*liability\b',
        r'\bintentional\s*tort\b', r'\bstrict\s*liability\b',
        r'\bproducts\s*liability\b', r'\bmalpractice\b', r'\bpremises\s*liability\b'
    ]
    
    for pattern in legal_search_indicators:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False


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
        r'\bcase\s*status\b', r'\bprocessing\s*time\b', r'\binterview\b'
    ]
    
    for pattern in immigration_indicators:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False


def detect_query_type(question: str) -> List[str]:
    """Detect the type of legal query for targeted searching"""
    query_types = []
    
    # Environmental law patterns
    environmental_patterns = [
        r'\bepa\b', r'\benvironmental\b', r'\bpollution\b', r'\bclimate\b',
        r'\bemissions\b', r'\bclean\s+air\b', r'\bclean\s+water\b',
        r'\bwetlands\b', r'\bendangered\s+species\b', r'\bnepa\b',
        r'\bcercla\b', r'\brcra\b', r'\btsca\b'
    ]
    
    # Business law patterns
    business_patterns = [
        r'\bsba\b', r'\bsmall\s+business\b', r'\bstartup\b', r'\bincorporat',
        r'\bllc\b', r'\bcorporation\b', r'\bsec\b', r'\bsecurities\b',
        r'\bpatent\b', r'\btrademark\b', r'\bcopyright\b', r'\bcontract\b'
    ]
    
    # Regulatory patterns
    regulatory_patterns = [
        r'\bregulation\b', r'\brule\b', r'\bcfr\b', r'\bfederal\s+register\b',
        r'\bagency\b', r'\badministrative\b'
    ]
    
    # Check each pattern type
    question_lower = question.lower()
    
    if any(re.search(pattern, question_lower) for pattern in environmental_patterns):
        query_types.append('environmental')
    
    if any(re.search(pattern, question_lower) for pattern in business_patterns):
        query_types.append('business')
    
    if any(re.search(pattern, question_lower) for pattern in regulatory_patterns):
        query_types.append('regulatory')
    
    # Existing checks
    if detect_statutory_question(question):
        query_types.append('statutory')
    
    if detect_legal_search_intent(question):
        query_types.append('case_law')
    
    if detect_immigration_query(question):
        query_types.append('immigration')
    
    return query_types


def should_search_external_databases(question: str, search_scope: str) -> bool:
    """Determine if external databases should be searched"""
    # Don't search external if user explicitly wants only their documents
    if search_scope == "user_only":
        return False
    
    # Check if this is a legal question that would benefit from external search
    return detect_legal_search_intent(question) or detect_statutory_question(question)


def format_external_results_with_citations(external_results: List[Dict]) -> Tuple[str, List[Dict]]:
    """Format external results with proper citations and return both formatted text and source info"""
    if not external_results:
        return "", []
    
    formatted_text = "\n\n## EXTERNAL LEGAL DATABASE RESULTS:\n"
    formatted_text += "(These results are from public legal databases including Harvard Law School Library Caselaw Access Project, CourtListener, and Google Scholar)\n\n"
    
    source_info = []
    
    for idx, result in enumerate(external_results[:5], 1):  # Limit to top 5 results
        # Create proper citation
        citation = format_legal_citation(result)
        
        formatted_text += f"### {idx}. {citation}\n"
        
        # Add database attribution
        source_db = result.get('source_database', 'Unknown')
        if 'harvard' in source_db.lower():
            formatted_text += f"**Database:** Harvard Law School Library - Caselaw Access Project\n"
        elif 'courtlistener' in source_db.lower():
            formatted_text += f"**Database:** Free Law Project - CourtListener\n"
        elif 'scholar' in source_db.lower() or 'google' in source_db.lower():
            formatted_text += f"**Database:** Google Scholar Legal Opinions and Journals\n"
        else:
            formatted_text += f"**Database:** {source_db}\n"
        
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
            formatted_text += f"**Date Decided:** {date}\n"
        
        # Add relevant excerpt/preview
        preview = result.get('preview') or result.get('snippet', '')
        if preview:
            # Clean up the preview text
            preview = re.sub(r'\s+', ' ', preview).strip()
            if len(preview) > 500:
                preview = preview[:497] + "..."
            formatted_text += f"**Relevant Excerpt:** {preview}\n"
        
        # Add URL for full text
        if result.get('url'):
            formatted_text += f"**Full Text Available:** {result['url']}\n"
        
        formatted_text += "\n"
        
        # Create source info entry with proper citation
        source_info.append({
            'file_name': citation,  # Use the formatted citation as the file name
            'page': None,
            'relevance': 0.8,  # Fixed relevance for external results
            'source_type': 'external_database',
            'database': source_db,
            'url': result.get('url', ''),
            'court': court,
            'date': date
        })
    
    return formatted_text, source_info


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
- When citing external database results, use the full citation format provided (e.g., "Smith v. Jones, 123 F.3d 456 (9th Cir. 2023)")

🔴 CRITICAL: You MUST extract actual numbers, durations, and specific requirements. NEVER use placeholders like "[duration not specified]" or "[requirements not listed]".

🔥 CRITICAL FAILURE PREVENTION:
- If you write "[duration not specified]" you have FAILED at your job
- If you write "[duties not specified]" you have FAILED at your job  
- If you write "[requirements not listed]" you have FAILED at your job
- READ EVERY WORD of the context before claiming information is missing
- The human is counting on you to find the actual requirements
- Your job is to be a thorough legal researcher, not a lazy summarizer

MANDATORY EXTRACTION RULES:
1. 📖 READ EVERY WORD of the provided context before claiming anything is missing
2. 🔢 EXTRACT ALL NUMBERS: durations (60 minutes), quantities (25 people), percentages, dollar amounts
3. 📝 QUOTE EXACT LANGUAGE: Use quotation marks for statutory text
4. 📋 LIST ALL REQUIREMENTS: Number each requirement found (1., 2., 3., etc.)
5. 🎯 BE SPECIFIC: Include section numbers, subsection letters, paragraph numbers
6. ⚠️ ONLY claim information is "missing" after thorough analysis of ALL provided text

INSTRUCTIONS FOR THOROUGH ANALYSIS:
1. **READ CAREFULLY**: Scan the entire context for information that answers the user's question
2. **EXTRACT COMPLETELY**: When extracting requirements, include FULL details (e.g., "60 minutes" not just "minimum of")
3. **QUOTE VERBATIM**: For statutory standards, use exact quotes: `"[Exact Text]" (Source)`
4. **ENUMERATE EXPLICITLY**: Present listed requirements as numbered points with full quotes
5. **CITE SOURCES**: Reference the document name or case citation for each fact
6. **BE COMPLETE**: Explicitly note missing standards: "Documents lack full subsection [X]"
7. **USE DECISIVE PHRASING**: State facts directly ("The statute requires...") - NEVER "documents indicate"

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

## INFORMATION NOT FOUND:
[Only list what is genuinely absent after thorough review]

EXAMPLES OF WHAT YOU MUST DO:

❌ WRONG: "The victim impact panel should be a minimum of [duration not specified]."
✅ RIGHT: "The statute requires victim impact panels to be 'a minimum of sixty (60) minutes in duration' (RCW 10.01.230(3)(a))."

❌ WRONG: "Specific details on composition are not available"
✅ RIGHT: "Panel composition requirements include: (1) 'at least two victim impact speakers' (RCW 10.01.230(2)(a)), (2) 'one trained facilitator' (RCW 10.01.230(2)(b)), (3) 'maximum of twenty-five participants per session' (RCW 10.01.230(2)(c))."

❌ WRONG: "The facilitator's role is not specified"
✅ RIGHT: "The designated facilitator must: (1) 'maintain order during presentations' (RCW 10.01.230(4)(a)), (2) 'ensure compliance with attendance policies' (RCW 10.01.230(4)(b)), (3) 'submit quarterly reports to the Traffic Safety Commission' (RCW 10.01.230(4)(c))."

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT TO ANALYZE WORD-BY-WORD:
{context_text}

USER QUESTION REQUIRING COMPLETE EXTRACTION:
{question}

MANDATORY STEPS FOR YOUR RESPONSE:
1. 🔍 Scan the ENTIRE context for specific numbers, requirements, and procedures - READ EVERY WORD
2. 📊 Extract ALL quantitative information (minutes, hours, numbers of people, etc.) - NO PLACEHOLDERS ALLOWED
3. 📜 Quote the exact statutory language for each requirement - FULL QUOTES, NOT SUMMARIES
4. 🏷️ Provide specific citations (section, subsection, paragraph) - EXACT REFERENCES
5. 📝 Organize information clearly with headers and numbered lists
6. ⚠️ Only claim information is missing if it's truly not in the provided text AFTER reading every word twice

🚨 BEFORE YOU RESPOND: Ask yourself these questions:
- Did I read every single word of the provided context?
- Did I look for numbers, durations, quantities in ALL sections?
- Am I using any lazy placeholders like "[not specified]"?
- Can I quote the exact text that supports each claim I'm making?

BEGIN THOROUGH STATUTORY ANALYSIS:

ADDITIONAL GUIDANCE:
- After fully answering based solely on the provided documents, if relevant key legal principles under Washington state law, any other U.S. state law, or U.S. federal law are not found in the sources, you may add a clearly labeled general legal principles disclaimer.
- This disclaimer must clearly state it is NOT based on the provided documents but represents general background knowledge of applicable Washington state, other state, and federal law.
- Do NOT use this disclaimer to answer the user's question directly; it serves only as supplementary context.
- This disclaimer must explicitly state that these principles are not found in the provided documents but are usually relevant legal background.
- Format this disclaimer distinctly at the end of the response under a heading such as "GENERAL LEGAL PRINCIPLES DISCLAIMER."

RESPONSE:"""


def create_regular_prompt(context_text: str, question: str, conversation_context: str, 
                         sources_searched: list, retrieval_method: str, document_id: str = None, 
                         instruction: str = "balanced", external_context: str = None) -> str:
    """Create the regular prompt for non-statutory questions"""
    
    # Add external context if available
    if external_context:
        context_text = f"{context_text}\n\n{external_context}"
    
    return f"""You are a legal research assistant. Provide thorough, accurate responses based on the provided documents.

STRICT SOURCE REQUIREMENTS:
- Answer ONLY based on the retrieved documents provided in the context
- Do NOT use general legal knowledge, training data, assumptions, or inferences beyond what's explicitly stated
- If information is not in the provided documents, state: "This information is not available in the provided documents"
- When citing external database results, use the full citation format provided (e.g., "Smith v. Jones, 123 F.3d 456 (9th Cir. 2023)")

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}
{f"DOCUMENT FILTER: Specific document {document_id}" if document_id else "DOCUMENT SCOPE: All available documents"}

HALLUCINATION CHECK - Before responding, verify:
1. Is each claim supported by the retrieved documents?
2. Am I adding information not present in the sources?
3. If uncertain, default to "information not available"

INSTRUCTIONS FOR THOROUGH ANALYSIS:
1. **READ CAREFULLY**: Scan the entire context for information that answers the user's question
2. **EXTRACT COMPLETELY**: When extracting requirements, include FULL details (e.g., "60 minutes" not just "minimum of")
3. **QUOTE VERBATIM**: For statutory standards, use exact quotes: `"[Exact Text]" (Source)`
4. **ENUMERATE EXPLICITLY**: Present listed requirements as numbered points with full quotes
5. **CITE SOURCES**: Reference the document name or case citation for each fact
6. **BE COMPLETE**: Explicitly note missing standards: "Documents lack full subsection [X]"
7. **USE DECISIVE PHRASING**: State facts directly ("The statute requires...") - NEVER "documents indicate"

RESPONSE STYLE: {instruction}

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT (ANALYZE THOROUGHLY):
{context_text}

USER QUESTION:
{question}

RESPONSE APPROACH:
- **FIRST**: Identify what specific information the user is asking for. Do not reference any statute, case law, or principle unless it appears verbatim in the context.
- **SECOND**: Search the context thoroughly for that information  
- **THIRD**: Present any information found clearly and completely. At the end of your response, list all facts provided and their source documents for verification.
- **FOURTH**: Note what information is not available (if any)
- **ALWAYS**: Cite the source document or case for each fact provided

ADDITIONAL GUIDANCE:
- After fully answering based solely on the provided documents, if relevant key legal principles under Washington state law, any other U.S. state law, or U.S. federal law are not found in the sources, you may add a clearly labeled general legal principles disclaimer.
- This disclaimer must clearly state it is NOT based on the provided documents but represents general background knowledge of applicable Washington state, other state, and federal law.
- Do NOT use this disclaimer to answer the user's question directly; it serves only as supplementary context.
- This disclaimer must explicitly state that these principles are not found in the provided documents but are usually relevant legal background.
- Format this disclaimer distinctly at the end of the response under a heading such as "GENERAL LEGAL PRINCIPLES DISCLAIMER."

RESPONSE:"""


def create_immigration_prompt(context_text: str, question: str, conversation_context: str,
                            sources_searched: list, retrieval_method: str, document_id: str = None) -> str:
    """Create prompt specifically for immigration queries"""
    
    return f"""You are an immigration legal assistant with expertise in U.S. immigration law, asylum claims, and country conditions research.

IMPORTANT IMMIGRATION CONTEXT:
- Always note that immigration law is complex and changes frequently
- Processing times and requirements vary by service center and case type
- Country conditions can change rapidly
- Each case is unique and requires individual assessment

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}
{f"DOCUMENT FILTER: Specific document {document_id}" if document_id else "DOCUMENT SCOPE: All available documents"}

When answering immigration questions:
1. **Be specific about forms and procedures** - Include form numbers, deadlines, and requirements
2. **Cite authoritative sources** - USCIS, State Dept, UNHCR, and court decisions
3. **Include relevant country conditions** - When applicable to asylum/refugee claims  
4. **Note time-sensitive information** - Priority dates, filing deadlines, age-out issues
5. **Highlight critical warnings** - Bars to admission, unlawful presence, criminal issues
6. **Suggest evidence needed** - Documents, testimony, expert reports
7. **Mention related forms/procedures** - Related applications that may be needed

For country conditions:
- Summarize current human rights situation
- Note specific risks for particular groups
- Include recent changes or trends
- Cite multiple sources for credibility

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT (including country research if applicable):
{context_text}

USER QUESTION:
{question}

RESPONSE INSTRUCTIONS:
- Provide practical, actionable guidance
- Include specific form numbers and deadlines
- Note any recent policy changes
- Warn about common pitfalls
- Suggest next steps
- If asylum-related, address credibility and corroboration
- Always include disclaimer about consulting an attorney

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
        if source_type == 'external_database':
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
            
            # Get chunk information if available
            chunk_indices = []
            for s in sources:
                if 'chunk_index' in s:
                    chunk_indices.append(s['chunk_index'])
                elif 'metadata' in s and 'chunk_index' in s['metadata']:
                    chunk_indices.append(s['metadata']['chunk_index'])
            
            consolidated_source = {
                'file_name': file_name,
                'source_type': source_type,
                'relevance': max_relevance,
                'avg_relevance': avg_relevance,
                'num_chunks': len(sources),
                'pages': unique_pages,
                'chunk_indices': sorted(set(chunk_indices)) if chunk_indices else [],
                'chunk_details': sources  # Keep original chunks for reference
            }
            
            consolidated_sources.append(consolidated_source)
    
    # Sort by relevance
    consolidated_sources.sort(key=lambda x: x.get('relevance', 0), reverse=True)
    
    return consolidated_sources


def format_source_display(source: Dict) -> str:
    """Format a source for display in the response"""
    source_type = source['source_type'].replace('_', ' ').title()
    
    if source_type == 'External Database':
        # Use the citation format for external sources
        display = f"- {source['file_name']}"
        if source.get('database'):
            db_name = source['database']
            if 'harvard' in db_name.lower():
                display += " [Harvard Law School Library - Caselaw Access Project]"
            elif 'courtlistener' in db_name.lower():
                display += " [Free Law Project - CourtListener]"
            elif 'scholar' in db_name.lower() or 'google' in db_name.lower():
                display += " [Google Scholar]"
            else:
                display += f" [{db_name}]"
        if source.get('url'):
            display += f" - [Full Text]({source['url']})"
        return display
    else:
        # Internal document format
        display = f"- {source['file_name']}"
        
        # Add page information if available
        if source.get('pages'):
            if len(source['pages']) == 1:
                display += f" (Page {source['pages'][0]})"
            elif len(source['pages']) > 1:
                if len(source['pages']) <= 3:
                    display += f" (Pages {', '.join(map(str, source['pages']))})"
                else:
                    display += f" (Pages {source['pages'][0]}-{source['pages'][-1]})"
        
        # Add chunk information if no page info
        elif source.get('num_chunks', 1) > 1:
            display += f" ({source['num_chunks']} relevant sections)"
        
        # Add relevance score with interpretation
        # Check if relevance is already in 0-100 range or needs conversion
        relevance = source['relevance']
        if relevance <= 1.0:
            # Convert from 0-1 to 0-100 scale
            relevance_pct = relevance * 100
        else:
            # Already in percentage scale
            relevance_pct = relevance
            
        if relevance_pct >= 80:
            relevance_label = "Excellent match"
        elif relevance_pct >= 60:
            relevance_label = "Good match"
        elif relevance_pct >= 40:
            relevance_label = "Fair match"
        else:
            relevance_label = "Partial match"
        
        if source.get('avg_relevance') and source['num_chunks'] > 1:
            avg = source['avg_relevance']
            if avg <= 1.0:
                avg_pct = avg * 100
            else:
                avg_pct = avg
            display += f" ({relevance_label}: {relevance_pct:.0f}/100, Avg: {avg_pct:.0f}/100)"
        else:
            display += f" ({relevance_label}: {relevance_pct:.0f}/100)"
        
        return display


def process_query(question: str, session_id: str, user_id: Optional[str], search_scope: str, 
                 response_style: str = "balanced", use_enhanced_rag: bool = True, 
                 document_id: str = None, search_external: bool = None) -> QueryResponse:
    """Main query processing function with enhanced statutory handling and external database integration"""
    try:
        logger.info(f"Processing query - Question: '{question}', User: {user_id}, Scope: {search_scope}, Enhanced: {use_enhanced_rag}, Document: {document_id}")
        
        # Check if this is a statutory question
        is_statutory = detect_statutory_question(question)
        if is_statutory:
            logger.info("🏛️ Detected statutory/regulatory question - using enhanced extraction")
        
        # Detect query types
        query_types = detect_query_type(question)
        logger.info(f"Detected query types: {query_types}")
        
        # Check if this is an immigration query
        is_immigration = 'immigration' in query_types
        if is_immigration:
            logger.info("🗽 Detected immigration-related query")
        
        # Check for country conditions specifically
        country_conditions_match = re.search(
            r'(?:country\s*conditions?|human\s*rights?|persecution|asylum\s*claim)\s*(?:for|in|about)?\s*([A-Z][a-zA-Z\s]+)',
            question, re.IGNORECASE
        )
        
        country_context = ""
        if country_conditions_match and comprehensive_researcher:
            country = country_conditions_match.group(1).strip()
            logger.info(f"📍 Country conditions query detected for: {country}")
            
            try:
                # Get comprehensive country research
                country_results = comprehensive_researcher.research_all_sources(
                    country=country,
                    topics=['persecution', 'human_rights', 'government', 'violence'],
                    include_multilingual=True
                )
                
                # Format the results for the LLM
                country_context = f"""
COMPREHENSIVE COUNTRY CONDITIONS RESEARCH FOR {country.upper()}:

Sources consulted:
- News articles from multiple outlets (including translated foreign press)
- Human Rights Watch reports
- Amnesty International documentation  
- UNHCR Refworld
- U.S. State Department country information
- Freedom House democracy scores
- Academic research

{country_results.get('summary', '')}

Key findings by topic:
"""
                
                for source_type, sources in country_results.get('sources', {}).items():
                    if sources and isinstance(sources, list):
                        country_context += f"\n{source_type.upper()}:\n"
                        for source in sources[:3]:
                            if isinstance(source, dict):
                                country_context += f"- {source.get('title', '')}\n"
            except Exception as e:
                logger.error(f"Country conditions research failed: {e}")
        
        # Detect form-specific queries
        form_match = re.search(r'\b(I-\d{3}|N-\d{3})\b', question, re.IGNORECASE)
        form_context = ""
        if form_match:
            form_number = form_match.group(1).upper()
            logger.info(f"📋 Form-specific query for: {form_number}")
            
            # Add form-specific context
            form_context = f"\nFORM {form_number} INFORMATION:\n{get_form_info(form_number)}\n"
        
        # Determine if we should search external databases
        if search_external is None:
            search_external = should_search_external_databases(question, search_scope) or is_immigration
        
        if search_external:
            logger.info("📚 Will search external legal databases for this query")
        
        # Determine which free databases to search
        source_types = []
        if 'case_law' in query_types:
            source_types.append('cases')
        if 'regulatory' in query_types or 'environmental' in query_types:
            source_types.append('regulations')
        if 'statutory' in query_types:
            source_types.append('legislation')
        if 'business' in query_types:
            source_types.append('business')
        
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

**Sources:** {len(comp_result.sources_by_section.get('summary', []))} document sections analyzed
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
        
        questions = parse_multiple_questions(question) if use_enhanced_rag else [question]
        combined_query = " ".join(questions)
        
        conversation_context = get_conversation_context(session_id)
        
        # For statutory questions, get more context
        search_k = 20 if is_statutory else 10
        
        retrieved_results, sources_searched, retrieval_method = combined_search(
            combined_query, 
            user_id, 
            search_scope, 
            conversation_context,
            use_enhanced=use_enhanced_rag,
            k=search_k,
            document_id=document_id
        )
        
        # Initialize source_info
        source_info = []
        
        # Search external databases if appropriate
        external_context = None
        external_source_info = []
        
        if search_external:
            try:
                logger.info("🔍 Searching external legal databases...")
                
                # Enhance the query for better legal database results
                enhanced_query = question
                
                # Add legal context based on the type of question
                if "negligence" in question.lower():
                    enhanced_query = f"{question} negligence tort duty care breach causation damages"
                elif "first amendment" in question.lower():
                    enhanced_query = f"{question} constitutional law free speech religion press"
                elif "miranda" in question.lower():
                    enhanced_query = f"{question} criminal procedure fifth amendment"
                elif "fourth amendment" in question.lower():
                    enhanced_query = f"{question} search seizure warrant constitutional"
                elif "case" in question.lower() or "precedent" in question.lower():
                    enhanced_query = f"{question} supreme court appellate decision"
                else:
                    # Generic enhancement for legal queries
                    enhanced_query = f"{question} legal case law court decision"
                
                logger.info(f"Enhanced query for external search: {enhanced_query}")
                external_results = search_free_legal_databases(enhanced_query, None)
                
                if external_results:
                    logger.info(f"📚 Found {len(external_results)} results from external databases")
                    sources_searched.append("external_legal_databases")
                    
                    # Format external results with proper citations
                    external_context, external_source_info = format_external_results_with_citations(external_results)
                        
            except Exception as e:
                logger.error(f"External database search failed: {e}")
                # Continue with regular search results even if external search fails
        
        # Search free databases if appropriate
        if source_types and search_external:
            try:
                logger.info(f"🆓 Searching free legal databases for: {source_types}")
                free_results = search_free_legal_databases_enhanced(question, None, source_types)
                
                if free_results:
                    logger.info(f"📚 Found {len(free_results)} results from free databases")
                    sources_searched.append("free_legal_databases")
                    
                    # Format free results
                    free_context = "\n\n## FREE LEGAL DATABASE RESULTS:\n"
                    for idx, result in enumerate(free_results[:10], 1):
                        free_context += f"\n### {idx}. {result.get('title', 'No title')}\n"
                        
                        if result.get('source_database'):
                            free_context += f"**Source:** {result['source_database'].replace('_', ' ').title()}\n"
                        
                        if result.get('date') or result.get('publication_date'):
                            free_context += f"**Date:** {result.get('date') or result.get('publication_date')}\n"
                        
                        if result.get('court'):
                            free_context += f"**Court:** {result['court']}\n"
                        
                        if result.get('agencies'):
                            free_context += f"**Agencies:** {result['agencies']}\n"
                        
                        if result.get('summary') or result.get('preview'):
                            preview = result.get('summary') or result.get('preview', '')
                            free_context += f"**Summary:** {preview[:300]}...\n"
                        
                        if result.get('url'):
                            free_context += f"**Link:** {result['url']}\n"
                    
                    # Add to context
                    if external_context:
                        external_context += free_context
                    else:
                        external_context = free_context
                        
            except Exception as e:
                logger.error(f"Free database search failed: {e}")
        
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
        
        # Format context for LLM - more context for statutory questions
        max_context_length = 8000 if is_statutory else 3000
        context_text, source_info = format_context_for_llm(retrieved_results, max_length=max_context_length)
        
        # Add metadata to source_info for better consolidation
        for i, (doc, score) in enumerate(retrieved_results[:len(source_info)]):
            if hasattr(doc, 'metadata'):
                source_info[i]['metadata'] = doc.metadata
                if 'chunk_index' in doc.metadata:
                    source_info[i]['chunk_index'] = doc.metadata['chunk_index']
        
        # Combine source info from both internal and external sources
        all_source_info = source_info + external_source_info
        
        # Enhanced information extraction
        bill_match = re.search(r"(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)", question, re.IGNORECASE)
        statute_match = re.search(r"(RCW|USC|CFR|WAC)\s+(\d+\.\d+\.\d+|\d+)", question, re.IGNORECASE)
        extracted_info = {}

        if bill_match:
            # Bill-specific extraction
            bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
            logger.info(f"Searching for bill: {bill_number}")
            
            # Search specifically for chunks containing this bill
            bill_specific_results = []
            for doc, score in retrieved_results:
                if hasattr(doc, 'metadata') and 'contains_bills' in doc.metadata and bill_number in doc.metadata['contains_bills']:
                    bill_specific_results.append((doc, score))
                    logger.info(f"Found {bill_number} in chunk {doc.metadata.get('chunk_index', 'unknown')} with score {score}")
            
            # If we found bill-specific chunks, prioritize them
            if bill_specific_results:
                logger.info(f"Using {len(bill_specific_results)} bill-specific chunks for {bill_number}")
                # Use the bill-specific chunks with boosted relevance
                boosted_results = [(doc, min(score + 0.3, 1.0)) for doc, score in bill_specific_results]
                other_results = [r for r in retrieved_results if r not in bill_specific_results]
                retrieved_results = boosted_results + other_results
                retrieved_results = retrieved_results[:search_k]  # Limit to original k value
            
            extracted_info = extract_bill_information(context_text, bill_number)
        elif statute_match:
            # Statute-specific extraction
            statute_citation = f"{statute_match.group(1)} {statute_match.group(2)}"
            logger.info(f"🏛️ Searching for statute: {statute_citation}")
            extracted_info = extract_statutory_information(context_text, statute_citation)
        else:
            # Universal extraction for any document type
            extracted_info = extract_universal_information(context_text, question)

        # Add extracted information to context to make it more visible to AI
        if extracted_info:
            enhancement = "\n\nKEY INFORMATION FOUND:\n"
            for key, value in extracted_info.items():
                if value:  # Only add if there's actual content
                    if isinstance(value, list):
                        enhancement += f"- {key.replace('_', ' ').title()}: {', '.join(value[:5])}\n"
                    else:
                        enhancement += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            if enhancement.strip() != "KEY INFORMATION FOUND:":
                context_text += enhancement
        
        # Add immigration-specific context
        if country_context:
            context_text = country_context + "\n\n" + context_text
        if form_context:
            context_text = form_context + "\n\n" + context_text
        
        style_instructions = {
            "concise": "Please provide a concise answer (1-2 sentences) based on the context.",
            "balanced": "Please provide a balanced answer (2-3 paragraphs) based on the context.",
            "detailed": "Please provide a detailed answer with explanations based on the context."
        }
        
        instruction = style_instructions.get(response_style, style_instructions["balanced"])
        
        # Choose the appropriate prompt based on question type
        if is_immigration:
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
        
        if FeatureFlags.AI_ENABLED and OPENROUTER_API_KEY:
            response_text = call_openrouter_api(prompt, OPENROUTER_API_KEY)
        else:
            response_text = f"Based on the retrieved documents:\n\n{context_text}\n\nPlease review this information to answer your question."
            if external_context:
                response_text += f"\n\n{external_context}"
        
        # Add immigration-specific tips if relevant
        if is_immigration:
            response_text += "\n\n**Immigration Law Notice**: This information is general guidance only. Immigration law is complex and changes frequently. Please consult with an immigration attorney for advice specific to your situation."
        
        relevant_sources = [s for s in all_source_info if s['relevance'] >= MIN_RELEVANCE_SCORE]
        
        # Consolidate duplicate sources
        consolidated_sources = consolidate_sources(relevant_sources)
        
        if consolidated_sources:
            response_text += "\n\n**SOURCES:**"
            response_text += "\n*Relevance scores range from 0-100, where higher scores indicate better matches to your query.*"
            for source in consolidated_sources:
                response_text += "\n" + format_source_display(source)
        
        confidence_score = calculate_confidence_score(retrieved_results, len(response_text))
        
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
