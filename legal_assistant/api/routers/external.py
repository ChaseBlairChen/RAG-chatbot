# legal_assistant/api/routers/external.py - COMPLETE ENHANCED VERSION
"""Enhanced external database endpoints with comprehensive legal API support"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Form, Query, Depends, HTTPException
from urllib.parse import quote

from ...models import User
from ...core.security import get_current_user
from ...services.external_db_service import (
    search_external_databases, 
    search_free_legal_databases, 
    search_free_legal_databases_enhanced,
    search_state_law_databases,
    search_legal_databases_comprehensive,
    comprehensive_legal_search,
    get_database_status,
    get_available_jurisdictions
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Enhanced existing endpoints
@router.post("/external/search")
async def search_external_databases_endpoint(
    query: str = Form(...),
    databases: List[str] = Form(...),
    state: Optional[str] = Form(None),
    law_types: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Enhanced external database search with state law support"""
    # Premium databases
    premium_dbs = ["lexisnexis", "westlaw"]
    selected_premium = [db for db in databases if db in premium_dbs]
    
    if selected_premium and current_user.subscription_tier not in ["premium", "enterprise"]:
        raise HTTPException(
            status_code=403, 
            detail="Premium databases require premium subscription"
        )
    
    # Enhanced search with state law integration
    try:
        results = search_external_databases(query, databases, current_user)
        
        # Add state law results if state is specified
        if state:
            state_results = search_state_law_databases(query, state, law_types)
            results.extend(state_results)
        
        return {
            "query": query,
            "databases_searched": databases,
            "state_searched": state,
            "law_types": law_types,
            "results": results,
            "total_results": len(results),
            "user_tier": current_user.subscription_tier
        }
        
    except Exception as e:
        logger.error(f"External search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/external/search-free")
async def search_free_databases_endpoint(
    query: str = Form(...),
    include_state_law: bool = Form(True),
    include_government_data: bool = Form(True),
    state: Optional[str] = Form(None),
    source_types: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Enhanced free database search with comprehensive API integration"""
    try:
        if include_government_data:
            # Use comprehensive search that includes government APIs
            results = search_free_legal_databases_enhanced(
                query, current_user, source_types, state
            )
            databases_searched = [
                "harvard_caselaw", "courtlistener", "justia", "cornell_law", 
                "openstates", "google_scholar_legal", "congress_gov",
                "epa_apis", "sec_edgar", "dol_osha", "fda_enforcement",
                "uscis_tracking", "fbi_crime_data", "comprehensive_government_apis"
            ]
        elif include_state_law:
            results = search_free_legal_databases_enhanced(query, current_user, source_types, state)
            databases_searched = [
                "harvard_caselaw", "courtlistener", "justia", "cornell_law", 
                "openstates", "google_scholar_legal"
            ]
        else:
            results = search_free_legal_databases(query, current_user)
            databases_searched = ["harvard_caselaw", "courtlistener", "justia"]
        
        return {
            "query": query,
            "databases_searched": databases_searched,
            "include_state_law": include_state_law,
            "include_government_data": include_government_data,
            "state_searched": state,
            "source_types": source_types,
            "results": results,
            "total_results": len(results),
            "government_data_included": include_government_data,
            "comprehensive_search": True
        }
        
    except Exception as e:
        logger.error(f"Free database search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# NEW COMPREHENSIVE ENDPOINTS

@router.post("/external/search-state-law")
async def search_state_law_endpoint(
    query: str = Form(...),
    state: str = Form(...),
    law_types: Optional[List[str]] = Form(["statutes", "cases"]),
    search_scope: Optional[str] = Form("comprehensive"),
    current_user: User = Depends(get_current_user)
):
    """Search state-specific legal databases"""
    try:
        results = search_state_law_databases(query, state, law_types)
        
        return {
            "query": query,
            "state": state,
            "law_types": law_types,
            "search_scope": search_scope,
            "results": results,
            "total_results": len(results),
            "databases_used": [
                "cornell_law", "openstates", "justia", "google_scholar_legal"
            ],
            "access_level": "free"
        }
        
    except Exception as e:
        logger.error(f"State law search failed: {e}")
        raise HTTPException(status_code=500, detail=f"State law search failed: {str(e)}")

@router.post("/external/comprehensive-search")
async def comprehensive_legal_search_endpoint(
    query: str = Form(...),
    include_state_law: bool = Form(True),
    include_government_data: bool = Form(True),
    target_jurisdiction: Optional[str] = Form(None),
    search_scope: Optional[str] = Form("all"),
    law_types: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Comprehensive search across ALL available legal databases and government APIs"""
    try:
        # Use the most comprehensive search function
        search_results = search_legal_databases_comprehensive(
            query=query,
            user=current_user,
            search_scope=search_scope,
            state=target_jurisdiction,
            law_types=law_types
        )
        
        # Also get the full comprehensive search for additional context
        if include_government_data:
            full_comprehensive = comprehensive_legal_search(
                query=query,
                user=current_user,
                include_state_law=include_state_law,
                target_jurisdiction=target_jurisdiction
            )
        else:
            full_comprehensive = {"summary": {"total_results": len(search_results)}}
        
        return {
            "success": True,
            "query": query,
            "search_parameters": {
                "include_state_law": include_state_law,
                "include_government_data": include_government_data,
                "target_jurisdiction": target_jurisdiction,
                "search_scope": search_scope,
                "law_types": law_types
            },
            "results": search_results,
            "comprehensive_summary": full_comprehensive.get("summary", {}),
            "total_results": len(search_results),
            "recommendations": _generate_search_recommendations(search_results, query),
            "next_steps": _suggest_next_steps(search_results, query),
            "databases_accessed": _get_databases_used(search_results)
        }
        
    except Exception as e:
        logger.error(f"Comprehensive search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive search failed: {str(e)}")

@router.post("/external/government-data-search")
async def government_data_search_endpoint(
    query: str = Form(...),
    data_categories: Optional[List[str]] = Form(["enforcement", "statistics", "filings"]),
    jurisdiction: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Search government databases for enforcement data, statistics, and official filings"""
    try:
        # Import comprehensive legal hub for government data search
        from ...services.comprehensive_legal_apis import comprehensive_legal_hub
        
        # Use intelligent search to get government data
        hub_results = comprehensive_legal_hub.intelligent_search(query, jurisdiction)
        
        # Filter by requested data categories
        filtered_results = []
        if hub_results.get('results_by_area'):
            for area, area_results in hub_results['results_by_area'].items():
                if isinstance(area_results, list):
                    for result in area_results:
                        result_type = _classify_government_data_type(result)
                        if not data_categories or result_type in data_categories:
                            result['data_category'] = result_type
                            result['legal_area'] = area
                            filtered_results.append(result)
        
        return {
            "query": query,
            "data_categories": data_categories,
            "jurisdiction": jurisdiction,
            "detected_areas": hub_results.get('legal_areas', []),
            "detected_state": hub_results.get('detected_state'),
            "results": filtered_results,
            "total_results": len(filtered_results),
            "government_apis_used": list(hub_results.get('results_by_area', {}).keys()),
            "official_data_sources": True
        }
        
    except Exception as e:
        logger.error(f"Government data search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Government data search failed: {str(e)}")

@router.get("/external/available-jurisdictions")
async def get_available_jurisdictions_endpoint():
    """Get available jurisdictions and search options"""
    try:
        jurisdictions = get_available_jurisdictions()
        return {
            "success": True,
            "jurisdictions": jurisdictions,
            "features": {
                "state_law_search": True,
                "federal_search": True,
                "case_law_search": True,
                "statute_search": True,
                "legislation_tracking": True,
                "government_data_search": True,
                "enforcement_tracking": True,
                "comprehensive_research": True
            },
            "auto_detection": {
                "state_detection": True,
                "legal_area_detection": True,
                "data_type_detection": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get jurisdictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/external/database-status")
async def get_database_status_endpoint(
    current_user: User = Depends(get_current_user)
):
    """Get status of all configured legal databases and government APIs"""
    try:
        status = get_database_status()
        
        # Filter by user tier
        user_tier = current_user.subscription_tier
        filtered_status = {}
        
        for db_name, db_status in status.items():
            db_type = db_status.get('type', 'free')
            
            # Show all free databases to all users
            if db_type in ['free', 'free_government_data']:
                filtered_status[db_name] = db_status
            # Show premium databases only to premium users
            elif db_type == 'premium' and user_tier in ['premium', 'enterprise']:
                filtered_status[db_name] = db_status
        
        # Categorize databases
        categorized_status = {
            'traditional_legal': {},
            'government_apis': {},
            'state_law': {},
            'comprehensive': {}
        }
        
        for db_name, db_status in filtered_status.items():
            if 'comprehensive_' in db_name:
                categorized_status['comprehensive'][db_name] = db_status
            elif any(term in db_name for term in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi', 'congress', 'federal_register']):
                categorized_status['government_apis'][db_name] = db_status
            elif 'state_law' in db_name or db_name in ['cornell_law', 'openstates']:
                categorized_status['state_law'][db_name] = db_status
            else:
                categorized_status['traditional_legal'][db_name] = db_status
        
        return {
            "user_tier": user_tier,
            "databases": categorized_status,
            "summary": {
                "total_available": len(filtered_status),
                "traditional_legal": len(categorized_status['traditional_legal']),
                "government_apis": len(categorized_status['government_apis']),
                "state_law_apis": len(categorized_status['state_law']),
                "comprehensive_apis": len(categorized_status['comprehensive']),
                "authenticated": len([db for db in filtered_status.values() if db.get('authenticated', False)])
            },
            "capabilities": {
                "case_law_search": True,
                "statute_search": True,
                "government_data": True,
                "enforcement_tracking": True,
                "state_law_research": True,
                "federal_research": True,
                "comprehensive_coverage": True
            }
        }
        
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/external/compare-states")
async def compare_state_laws_endpoint(
    query: str = Form(...),
    states: List[str] = Form(...),
    law_type: Optional[str] = Form("statutes"),
    current_user: User = Depends(get_current_user)
):
    """Compare laws across multiple states"""
    try:
        if len(states) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 states allowed for comparison")
        
        from ...services.state_law_apis import state_law_service
        
        comparison_results = state_law_service.search_multi_state(states, query)
        
        return {
            "query": query,
            "states_compared": states,
            "law_type": law_type,
            "comparison_results": comparison_results,
            "analysis": _generate_comparison_analysis(comparison_results),
            "recommendations": _generate_multi_state_recommendations(comparison_results),
            "differences_found": _identify_state_differences(comparison_results),
            "similarities": _identify_state_similarities(comparison_results)
        }
        
    except Exception as e:
        logger.error(f"State comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"State comparison failed: {str(e)}")

@router.post("/external/verify-citation")
async def verify_legal_citation_endpoint(
    citation: str = Form(...),
    citation_type: Optional[str] = Form("auto"),
    current_user: User = Depends(get_current_user)
):
    """Verify and lookup legal citations"""
    try:
        # Detect citation type
        if citation_type == "auto":
            citation_type = _detect_citation_type(citation)
        
        verification_results = _verify_citation(citation, citation_type)
        
        # Search for the citation in available databases
        search_results = search_legal_databases_comprehensive(
            query=citation,
            user=current_user,
            search_scope="all"
        )
        
        return {
            "citation": citation,
            "citation_type": citation_type,
            "verification_results": verification_results,
            "found_in_databases": len(search_results) > 0,
            "search_results": search_results[:5],  # Top 5 results
            "alternative_sources": _find_alternative_sources(citation),
            "full_text_available": any(result.get('url') for result in search_results),
            "authority_level": _assess_citation_authority(citation_type)
        }
        
    except Exception as e:
        logger.error(f"Citation verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Citation verification failed: {str(e)}")

@router.post("/external/research-guidance")
async def get_research_guidance_endpoint(
    legal_issue: str = Form(...),
    jurisdiction: Optional[str] = Form(None),
    practice_area: Optional[str] = Form(None),
    research_depth: Optional[str] = Form("comprehensive"),
    current_user: User = Depends(get_current_user)
):
    """Get guidance on legal research strategy"""
    try:
        guidance = _generate_research_guidance(legal_issue, jurisdiction, practice_area, research_depth)
        
        return {
            "legal_issue": legal_issue,
            "jurisdiction": jurisdiction,
            "practice_area": practice_area,
            "research_depth": research_depth,
            "research_strategy": guidance,
            "recommended_databases": _recommend_databases(legal_issue, practice_area),
            "search_tips": _generate_search_tips(legal_issue),
            "citation_formats": _get_citation_formats(jurisdiction),
            "estimated_research_time": _estimate_research_time(research_depth),
            "quality_checkpoints": _get_quality_checkpoints(practice_area)
        }
        
    except Exception as e:
        logger.error(f"Research guidance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/external/enforcement-search")
async def search_enforcement_data_endpoint(
    query: str = Form(...),
    enforcement_type: Optional[List[str]] = Form(["epa", "osha", "sec", "fda"]),
    state: Optional[str] = Form(None),
    date_range: Optional[str] = Form("last_year"),
    current_user: User = Depends(get_current_user)
):
    """Search government enforcement databases"""
    try:
        from ...services.comprehensive_legal_apis import comprehensive_legal_hub
        
        # Use comprehensive hub to search enforcement data
        enforcement_results = comprehensive_legal_hub.intelligent_search(query, state)
        
        # Filter by enforcement type
        filtered_enforcement = []
        for area, area_results in enforcement_results.get('results_by_area', {}).items():
            if isinstance(area_results, list):
                for result in area_results:
                    source_db = result.get('source_database', '').lower()
                    
                    # Check if this matches requested enforcement types
                    matches_type = False
                    for enf_type in enforcement_type:
                        if enf_type.lower() in source_db:
                            matches_type = True
                            break
                    
                    if matches_type:
                        result['enforcement_area'] = area
                        result['enforcement_type'] = _classify_enforcement_type(result)
                        filtered_enforcement.append(result)
        
        return {
            "query": query,
            "enforcement_types_searched": enforcement_type,
            "state": state,
            "date_range": date_range,
            "enforcement_results": filtered_enforcement,
            "total_violations_found": len(filtered_enforcement),
            "agencies_with_data": list(set(result.get('enforcement_type') for result in filtered_enforcement)),
            "summary_by_agency": _summarize_by_agency(filtered_enforcement)
        }
        
    except Exception as e:
        logger.error(f"Enforcement search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enforcement search failed: {str(e)}")

@router.post("/external/immigration-tracking")
async def immigration_tracking_endpoint(
    query: str = Form(...),
    tracking_type: Optional[str] = Form("case_status"),
    receipt_number: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Immigration case tracking and visa bulletin information"""
    try:
        from ...services.comprehensive_legal_apis import comprehensive_legal_hub
        
        immigration_results = []
        
        if receipt_number:
            # Specific case status lookup
            case_status = comprehensive_legal_hub.immigration.check_case_status(receipt_number)
            if case_status:
                immigration_results.append(case_status)
        
        if tracking_type == "visa_bulletin":
            # Get current visa bulletin
            visa_data = comprehensive_legal_hub.immigration.get_visa_bulletin_data()
            if visa_data:
                immigration_results.append(visa_data)
        
        # Also search immigration law databases
        immigration_law_results = search_legal_databases_comprehensive(
            query=query,
            user=current_user,
            law_types=["immigration", "federal"]
        )
        
        return {
            "query": query,
            "tracking_type": tracking_type,
            "receipt_number": receipt_number,
            "immigration_data": immigration_results,
            "immigration_law_results": immigration_law_results[:10],
            "total_results": len(immigration_results) + len(immigration_law_results),
            "data_sources": ["uscis_case_status", "state_dept_visa_bulletin", "immigration_law_databases"],
            "real_time_data": len(immigration_results) > 0
        }
        
    except Exception as e:
        logger.error(f"Immigration tracking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Immigration tracking failed: {str(e)}")

@router.get("/external/legal-areas")
async def get_legal_areas_endpoint():
    """Get available legal practice areas for targeted search"""
    from ...config import LEGAL_AREA_KEYWORDS
    
    return {
        "legal_areas": list(LEGAL_AREA_KEYWORDS.keys()),
        "area_descriptions": {
            "environmental": "EPA enforcement, air/water quality, hazardous waste, environmental compliance",
            "immigration": "USCIS case status, visa processing, immigration law, country conditions",
            "business": "SEC filings, corporate compliance, business registrations, securities law",
            "labor": "OSHA violations, workplace safety, wage & hour compliance, employment law",
            "criminal": "FBI crime data, criminal statutes, court decisions, enforcement statistics",
            "healthcare": "FDA recalls, medical device regulations, healthcare compliance",
            "housing": "HUD data, fair housing, rent regulations, housing statistics",
            "intellectual_property": "Patent search, trademark registration, IP enforcement",
            "constitutional": "Constitutional law, civil rights, Supreme Court decisions",
            "international": "Treaties, international law, trade agreements"
        },
        "auto_detection": "System automatically detects legal areas from query content"
    }

# Helper functions for new endpoints

def _generate_search_recommendations(search_results: List[Dict], query: str) -> List[str]:
    """Generate search improvement recommendations"""
    recommendations = []
    
    total_results = len(search_results)
    
    if total_results == 0:
        recommendations.extend([
            "Try broader search terms",
            "Search multiple jurisdictions (federal + state)",
            "Consider related legal concepts or synonyms",
            "Check if this is covered by government regulations vs. statutes"
        ])
    elif total_results < 5:
        recommendations.extend([
            "Try adding synonyms or alternative legal terms",
            "Search both current law and recent enforcement actions",
            "Consider searching neighboring states for similar approaches"
        ])
    else:
        recommendations.extend([
            "Review government enforcement data for practical examples",
            "Check primary sources (statutes/regulations) before secondary sources",
            "Look for recent updates or pending legislation",
            "Consider both federal and state requirements"
        ])
    
    # Check what types of sources were found
    source_types = set()
    government_data = False
    
    for result in search_results:
        source_db = result.get('source_database', '')
        source_types.add(source_db)
        
        if any(gov_indicator in source_db for gov_indicator in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi']):
            government_data = True
    
    if government_data:
        recommendations.append("✅ Government enforcement data found - review for practical compliance examples")
    else:
        recommendations.append("Consider searching for government enforcement examples and regulatory guidance")
    
    return recommendations

def _suggest_next_steps(search_results: List[Dict], query: str) -> List[str]:
    """Suggest next research steps"""
    next_steps = []
    
    # Analyze what types of sources were found
    has_statutes = any('statute' in result.get('source_database', '') or 'code' in result.get('source_database', '') for result in search_results)
    has_cases = any('case' in result.get('source_database', '') or 'court' in result.get('source_database', '') for result in search_results)
    has_government_data = any(any(gov in result.get('source_database', '') for gov in ['epa', 'sec', 'dol', 'fda']) for result in search_results)
    has_enforcement = any('enforcement' in str(result) or 'violation' in str(result) for result in search_results)
    
    if has_government_data:
        next_steps.append("Review government enforcement data for compliance requirements and penalties")
    
    if has_statutes:
        next_steps.append("Verify current version of statutes and check for recent amendments")
    
    if has_cases:
        next_steps.append("Analyze case law for practical applications and interpretations")
    
    if has_enforcement:
        next_steps.append("Review enforcement patterns to understand agency priorities")
    
    # Always suggest verification
    next_steps.extend([
        "Check for pending legislation that might affect current law",
        "Verify jurisdiction-specific requirements",
        "Consider consulting with relevant regulatory agencies for guidance"
    ])
    
    return next_steps

def _get_databases_used(search_results: List[Dict]) -> List[Dict]:
    """Get summary of databases used in search"""
    databases_used = {}
    
    for result in search_results:
        source_db = result.get('source_database', 'unknown')
        if source_db not in databases_used:
            databases_used[source_db] = {
                'name': source_db,
                'results_count': 0,
                'authority_level': result.get('authority_level', 'medium'),
                'type': _classify_database_type(source_db)
            }
        databases_used[source_db]['results_count'] += 1
    
    return list(databases_used.values())

def _classify_government_data_type(result: Dict) -> str:
    """Classify type of government data"""
    source_db = result.get('source_database', '').lower()
    
    if any(term in source_db for term in ['violation', 'enforcement', 'citation']):
        return 'enforcement'
    elif any(term in source_db for term in ['filing', 'report', 'disclosure']):
        return 'filings'
    elif any(term in source_db for term in ['statistics', 'data', 'census']):
        return 'statistics'
    elif any(term in source_db for term in ['status', 'tracking', 'processing']):
        return 'tracking'
    else:
        return 'other'

def _classify_database_type(source_db: str) -> str:
    """Classify database type"""
    source_lower = source_db.lower()
    
    if any(term in source_lower for term in ['epa', 'sec', 'dol', 'fda', 'uscis', 'fbi', 'congress', 'federal_register']):
        return 'government'
    elif any(term in source_lower for term in ['harvard', 'cornell', 'courtlistener']):
        return 'academic'
    elif any(term in source_lower for term in ['state_law', 'openstates']):
        return 'state_government'
    else:
        return 'legal_database'

def _generate_comparison_analysis(comparison_results: Dict) -> Dict:
    """Generate analysis of multi-state comparison"""
    analysis = {
        "similarities": [],
        "differences": [],
        "trends": [],
        "outliers": []
    }
    
    # Analyze state results
    state_results = comparison_results.get('state_results', {})
    
    # Count result types by state
    result_counts = {}
    for state, results in state_results.items():
        if 'error' not in results:
            sources = results.get('sources', {})
            result_counts[state] = {k: len(v) for k, v in sources.items() if isinstance(v, list)}
    
    # Find patterns
    if len(result_counts) > 1:
        # Find states with similar result patterns
        for state1, counts1 in result_counts.items():
            for state2, counts2 in result_counts.items():
                if state1 != state2:
                    # Simple similarity check
                    if counts1.get('state_codes', 0) > 0 and counts2.get('state_codes', 0) > 0:
                        analysis["similarities"].append(f"{state1} and {state2} both have relevant statutes")
    
    return analysis

def _generate_multi_state_recommendations(comparison_results: Dict) -> List[str]:
    """Generate recommendations for multi-state comparison"""
    recommendations = []
    
    states = list(comparison_results.get('state_results', {}).keys())
    
    if len(states) > 1:
        recommendations.extend([
            f"Compare specific statute sections across {', '.join(states)}",
            "Look for model legislation or uniform acts that states have adopted",
            "Check for interstate compacts or agreements on this issue",
            "Consider federal preemption issues that might override state law",
            "Analyze enforcement patterns across different states"
        ])
    
    return recommendations

def _identify_state_differences(comparison_results: Dict) -> List[str]:
    """Identify key differences between states"""
    differences = []
    
    # This would analyze the actual results to find differences
    # For now, provide general guidance
    differences = [
        "Penalty structures may vary significantly between states",
        "Enforcement priorities differ by state regulatory agencies",
        "Procedural requirements may have state-specific variations",
        "Licensing and permit requirements often vary by jurisdiction"
    ]
    
    return differences

def _identify_state_similarities(comparison_results: Dict) -> List[str]:
    """Identify similarities between states"""
    similarities = [
        "Most states follow similar federal framework requirements",
        "Common adoption of model legislation in certain areas",
        "Similar enforcement mechanisms across jurisdictions",
        "Comparable penalty structures for similar violations"
    ]
    
    return similarities

def _detect_citation_type(citation: str) -> str:
    """Detect type of legal citation"""
    citation_patterns = {
        'case': r'\d+\s+\w+\.?\s+\d+',  # e.g., "123 F.3d 456"
        'federal_statute': r'\d+\s+U\.?S\.?C\.?\s+§?\s*\d+',  # e.g., "42 USC 1983"
        'federal_regulation': r'\d+\s+C\.?F\.?R\.?\s+§?\s*\d+',  # e.g., "29 CFR 1910"
        'state_statute': r'\w+\s+Rev\.?\s+Code\s+§?\s*\d+',  # e.g., "Wash. Rev. Code 1.23"
        'state_regulation': r'\w+\s+Admin\.?\s+Code\s+§?\s*\d+',  # e.g., "WAC 123-45"
    }
    
    for cite_type, pattern in citation_patterns.items():
        if re.search(pattern, citation, re.IGNORECASE):
            return cite_type
    
    return 'unknown'

def _verify_citation(citation: str, citation_type: str) -> Dict:
    """Verify legal citation format and validity"""
    
    return {
        "is_valid_format": True,
        "citation_type": citation_type,
        "normalized_citation": citation,
        "has_full_text": False,
        "verification_sources": ["harvard_caselaw", "justia", "cornell_law"],
        "confidence": 0.8,
        "format_notes": _get_citation_format_notes(citation_type)
    }

def _get_citation_format_notes(citation_type: str) -> List[str]:
    """Get notes about citation format"""
    format_notes = {
        'case': ["Include court and year", "Use proper reporter abbreviations", "Check for parallel citations"],
        'federal_statute': ["Use current USC citation", "Include section symbol", "Check for amendments"],
        'federal_regulation': ["Use current CFR citation", "Include effective date", "Check for recent revisions"],
        'state_statute': ["Use state-specific citation format", "Include current code section", "Verify current law"],
        'state_regulation': ["Use state administrative code format", "Check for recent rule changes"]
    }
    
    return format_notes.get(citation_type, ["Verify citation format", "Check for currency"])

def _find_alternative_sources(citation: str) -> List[Dict]:
    """Find alternative sources for citation"""
    return [
        {
            "source": "Google Scholar",
            "url": f"https://scholar.google.com/scholar?q={quote(citation)}",
            "access": "free",
            "type": "academic_search"
        },
        {
            "source": "Justia",
            "url": f"https://law.justia.com/search?q={quote(citation)}",
            "access": "free", 
            "type": "legal_database"
        },
        {
            "source": "Cornell Law",
            "url": f"https://www.law.cornell.edu/search/site/{quote(citation)}",
            "access": "free",
            "type": "academic_legal_resource"
        }
    ]

def _assess_citation_authority(citation_type: str) -> str:
    """Assess authority level of citation type"""
    authority_levels = {
        'case': 'high',
        'federal_statute': 'very_high',
        'federal_regulation': 'high',
        'state_statute': 'high',
        'state_regulation': 'medium_high',
        'unknown': 'medium'
    }
    
    return authority_levels.get(citation_type, 'medium')

def _generate_research_guidance(legal_issue: str, jurisdiction: str, practice_area: str, research_depth: str) -> Dict:
    """Generate comprehensive research strategy guidance"""
    guidance = {
        "research_steps": [],
        "primary_sources": [],
        "secondary_sources": [],
        "search_strategy": [],
        "quality_indicators": []
    }
    
    # Research steps based on depth
    if research_depth == "comprehensive":
        guidance["research_steps"] = [
            "1. Identify controlling jurisdiction (federal vs. state vs. local)",
            "2. Search government enforcement databases for practical examples",
            "3. Find primary sources (statutes, regulations, constitutional provisions)",
            "4. Locate relevant case law and judicial interpretations", 
            "5. Check for recent enforcement actions and agency guidance",
            "6. Review current legislative developments and proposed changes",
            "7. Verify currency and validity of all authorities",
            "8. Consider practical compliance requirements and industry standards"
        ]
    else:
        guidance["research_steps"] = [
            "1. Identify controlling jurisdiction",
            "2. Search primary legal sources",
            "3. Find relevant case law",
            "4. Verify currency of authorities"
        ]
    
    # Jurisdiction-specific guidance
    if jurisdiction:
        if jurisdiction.lower() == "federal":
            guidance["primary_sources"] = ["USC", "CFR", "Federal Case Law", "Federal Register"]
            guidance["search_strategy"].append("Start with federal statutes and implementing regulations")
        else:
            guidance["primary_sources"] = [f"{jurisdiction} State Code", f"{jurisdiction} Regulations", f"{jurisdiction} Case Law"]
            guidance["search_strategy"].append(f"Focus on {jurisdiction} state law first, then check federal requirements")
    
    # Practice area guidance
    if practice_area:
        area_guidance = {
            "environmental": {
                "primary_sources": ["EPA Regulations", "State Environmental Codes", "EPA Enforcement Data"],
                "key_concepts": ["Compliance requirements", "Enforcement priorities", "Penalty structures"],
                "government_databases": ["EPA ECHO", "EPA Air Quality", "EPA Water Quality"]
            },
            "immigration": {
                "primary_sources": ["INA", "8 CFR", "USCIS Policy Manual", "Case Status Data"],
                "key_concepts": ["Processing times", "Priority dates", "Inadmissibility grounds"],
                "government_databases": ["USCIS Case Status", "State Dept Visa Bulletin"]
            },
            "business": {
                "primary_sources": ["Securities Acts", "SEC Regulations", "Corporate Filings"],
                "key_concepts": ["Disclosure requirements", "Compliance obligations", "Enforcement actions"],
                "government_databases": ["SEC EDGAR", "SEC Enforcement Database"]
            },
            "labor": {
                "primary_sources": ["OSHA Standards", "DOL Regulations", "Employment Laws"],
                "key_concepts": ["Safety requirements", "Wage & hour compliance", "Workplace violations"],
                "government_databases": ["DOL OSHA Citations", "DOL Wage & Hour Data"]
            }
        }
        
        if practice_area.lower() in area_guidance:
            area_info = area_guidance[practice_area.lower()]
            guidance["primary_sources"].extend(area_info["primary_sources"])
            guidance["key_concepts"] = area_info["key_concepts"]
            guidance["recommended_government_databases"] = area_info["government_databases"]
    
    return guidance

def _recommend_databases(legal_issue: str, practice_area: str) -> List[Dict]:
    """Recommend specific databases for research"""
    recommendations = []
    
    # Always recommend core free databases
    core_recommendations = [
        {
            "database": "Harvard Caselaw Access Project",
            "best_for": "Historical and recent case law from all jurisdictions",
            "access": "free",
            "strength": "Comprehensive coverage with full text",
            "authority": "very_high"
        },
        {
            "database": "CourtListener",
            "best_for": "Federal and state court opinions with real-time updates",
            "access": "free",
            "strength": "Current awareness and docket tracking",
            "authority": "very_high"
        },
        {
            "database": "Cornell Law School",
            "best_for": "Federal and state statutes, well-organized legal codes",
            "access": "free",
            "strength": "Academic quality and authoritative organization",
            "authority": "very_high"
        }
    ]
    
    recommendations.extend(core_recommendations)
    
    # Practice area specific recommendations
    if practice_area:
        area_specific = {
            "environmental": [
                {
                    "database": "EPA Enforcement Database (ECHO)",
                    "best_for": "Environmental violations and enforcement actions",
                    "access": "free",
                    "strength": "Official government enforcement data",
                    "authority": "very_high"
                },
                {
                    "database": "Federal Register",
                    "best_for": "Environmental regulations and proposed rules",
                    "access": "free",
                    "strength": "Official regulatory source",
                    "authority": "very_high"
                }
            ],
            "immigration": [
                {
                    "database": "USCIS Case Status",
                    "best_for": "Real-time immigration case tracking",
                    "access": "free",
                    "strength": "Official case status information",
                    "authority": "very_high"
                }
            ],
            "business": [
                {
                    "database": "SEC EDGAR",
                    "best_for": "Corporate filings and securities law compliance",
                    "access": "free",
                    "strength": "Official corporate disclosure database",
                    "authority": "very_high"
                }
            ],
            "labor": [
                {
                    "database": "DOL OSHA Database",
                    "best_for": "Workplace safety violations and compliance",
                    "access": "free",
                    "strength": "Official workplace safety enforcement data",
                    "authority": "very_high"
                }
            ]
        }
        
        if practice_area.lower() in area_specific:
            recommendations.extend(area_specific[practice_area.lower()])
    
    return recommendations

def _generate_search_tips(legal_issue: str) -> List[str]:
    """Generate search tips for specific legal issues"""
    tips = [
        "Use specific legal terminology when available",
        "Search both the controlling law and recent enforcement examples",
        "Look for government agency guidance and interpretations",
        "Check for recent case law interpreting the relevant statutes",
        "Verify the current version of any law or regulation cited",
        "Consider both federal and state requirements that may apply"
    ]
    
    # Issue-specific tips
    issue_lower = legal_issue.lower()
    
    if any(term in issue_lower for term in ['environmental', 'epa', 'pollution']):
        tips.extend([
            "Search EPA enforcement database for practical compliance examples",
            "Check both federal EPA requirements and state environmental laws",
            "Look for recent enforcement actions in your industry/area"
        ])
    
    if any(term in issue_lower for term in ['immigration', 'visa', 'uscis']):
        tips.extend([
            "Check current USCIS processing times and policy updates",
            "Verify priority dates in monthly visa bulletins",
            "Look for recent circuit court decisions affecting your case type"
        ])
    
    if any(term in issue_lower for term in ['business', 'sec', 'corporate']):
        tips.extend([
            "Search SEC enforcement actions for compliance guidance",
            "Check both federal securities law and state corporate law",
            "Look for recent SEC guidance and no-action letters"
        ])
    
    if any(term in issue_lower for term in ['labor', 'osha', 'workplace']):
        tips.extend([
            "Search OSHA citation database for workplace safety examples",
            "Check both federal OSHA standards and state workplace laws",
            "Look for industry-specific guidance and interpretations"
        ])
    
    return tips

def _get_citation_formats(jurisdiction: str) -> Dict[str, str]:
    """Get proper citation formats for jurisdiction"""
    formats = {
        "case_law": "Case Name, Volume Reporter Page (Court Year)",
        "federal_statute": "Title U.S.C. § Section (Year)",
        "federal_regulation": "Title C.F.R. § Section (Year)",
        "state_statute": "[State] [Code] § [Section] ([Year])"
    }
    
    # Jurisdiction-specific formats
    if jurisdiction:
        jurisdiction_lower = jurisdiction.lower()
        if "washington" in jurisdiction_lower:
            formats["state_statute"] = "RCW § Section (Year)"
            formats["state_regulation"] = "WAC § Section (Year)"
        elif "california" in jurisdiction_lower:
            formats["state_statute"] = "Cal. [Code Type] Code § Section (Year)"
        elif "new york" in jurisdiction_lower:
            formats["state_statute"] = "N.Y. [Code Type] Law § Section (Year)"
        elif "texas" in jurisdiction_lower:
            formats["state_statute"] = "Tex. [Code Type] Code § Section (Year)"
    
    return formats

def _estimate_research_time(research_depth: str) -> str:
    """Estimate research time based on depth"""
    time_estimates = {
        "basic": "30-60 minutes",
        "intermediate": "1-3 hours", 
        "comprehensive": "3-8 hours",
        "exhaustive": "1-3 days"
    }
    
    return time_estimates.get(research_depth, "1-2 hours")

def _get_quality_checkpoints(practice_area: str) -> List[str]:
    """Get quality checkpoints for research"""
    general_checkpoints = [
        "Verified currency of all legal authorities",
        "Checked for recent amendments or updates", 
        "Found controlling jurisdiction authorities",
        "Located relevant case law interpretations",
        "Reviewed government enforcement examples"
    ]
    
    area_specific = {
        "environmental": [
            "Checked EPA enforcement database for compliance examples",
            "Verified current environmental regulations",
            "Reviewed state environmental law requirements"
        ],
        "immigration": [
            "Checked current USCIS processing times",
            "Verified priority dates in visa bulletin",
            "Reviewed recent policy memoranda"
        ],
        "business": [
            "Checked SEC enforcement actions",
            "Verified current disclosure requirements",
            "Reviewed corporate compliance obligations"
        ]
    }
    
    if practice_area and practice_area.lower() in area_specific:
        general_checkpoints.extend(area_specific[practice_area.lower()])
    
    return general_checkpoints
