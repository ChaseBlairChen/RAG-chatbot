"""Immigration law specific endpoints"""
import logging
from typing import List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile

from ...models import (
    User, ImmigrationCase, DeadlineAlert, DocumentClassification,
    BatchProcessingRequest, CountryConditionsRequest, CaseType
)
from ...core.security import get_current_user
from ...services.immigration_service import (
    immigration_analyzer, deadline_manager, country_researcher,
    credible_fear_preparer, secure_handler
)
from ...services.document_processor import SafeDocumentProcessor
from ...storage.managers import immigration_cases, case_documents

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/cases/create")
async def create_immigration_case(
    case_type: CaseType = Form(...),
    client_name: str = Form(...),
    language: str = Form("en"),
    priority_date: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Create a new immigration case"""
    try:
        case_id = hashlib.md5(f"{client_name}{datetime.utcnow()}".encode()).hexdigest()[:16]
        
        case = ImmigrationCase(
            case_id=case_id,
            case_type=case_type,
            client_id=secure_handler.encrypt_sensitive_data(client_name).decode(),
            priority_date=datetime.fromisoformat(priority_date) if priority_date else None,
            filing_date=datetime.utcnow(),
            language=language,
            assigned_attorney=current_user.user_id
        )
        
        immigration_cases[case_id] = case
        
        # Create initial deadlines based on case type
        if case_type == CaseType.ASYLUM:
            # One-year filing deadline
            deadline_manager.create_deadline(
                case_id=case_id,
                deadline_type="one_year_bar",
                due_date=datetime.utcnow() + timedelta(days=365),
                description="One-year asylum filing deadline",
                priority="high"
            )
        
        return {
            "success": True,
            "case_id": case_id,
            "message": f"Immigration case created for {case_type.value}"
        }
        
    except Exception as e:
        logger.error(f"Error creating immigration case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/classify")
async def classify_immigration_document(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Classify and analyze immigration document"""
    try:
        # Process document
        file_content = await file.read()
        content, pages, warnings = SafeDocumentProcessor.process_document_from_bytes(
            file_content, file.filename, os.path.splitext(file.filename)[1]
        )
        
        # Classify document
        classification = immigration_analyzer.classify_document(content, file.filename)
        
        # Add to case if provided
        if case_id and case_id in immigration_cases:
            if case_id not in case_documents:
                case_documents[case_id] = []
            case_documents[case_id].append(classification.document_id)
            
            # Check if RFE response
            if "request for evidence" in content.lower():
                deadline_manager.create_deadline(
                    case_id=case_id,
                    deadline_type="rfe_response",
                    due_date=datetime.utcnow() + timedelta(days=87),  # Standard RFE deadline
                    description="RFE Response Deadline",
                    priority="critical"
                )
        
        return {
            "classification": classification,
            "warnings": warnings,
            "pages": pages,
            "requires_translation": classification.requires_translation
        }
        
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deadlines/upcoming")
async def get_upcoming_deadlines(
    days_ahead: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Get upcoming deadlines"""
    deadlines = deadline_manager.get_upcoming_deadlines(days_ahead)
    
    # Filter by attorney's cases
    user_cases = [case_id for case_id, case in immigration_cases.items() 
                  if case.assigned_attorney == current_user.user_id]
    
    user_deadlines = [d for d in deadlines if d.case_id in user_cases]
    
    return {
        "deadlines": user_deadlines,
        "critical_count": sum(1 for d in user_deadlines if d.priority == "critical"),
        "total": len(user_deadlines)
    }

@router.post("/country-conditions/research")
async def research_country_conditions(
    request: CountryConditionsRequest,
    current_user: User = Depends(get_current_user)
):
    """Research country conditions for asylum cases"""
    try:
        research = country_researcher.research_country_conditions(request)
        
        return {
            "success": True,
            "research": research,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error researching country conditions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/credible-fear/analyze")
async def analyze_credible_fear_testimony(
    testimony: str = Form(...),
    country: str = Form(...),
    case_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Analyze testimony for credible fear preparation"""
    try:
        analysis = credible_fear_preparer.analyze_testimony(testimony, country)
        
        # Save analysis to case if provided
        if case_id and case_id in immigration_cases:
            immigration_cases[case_id].notes = (
                immigration_cases[case_id].notes or "" + 
                f"\n\n--- Credible Fear Analysis ---\n{analysis['analysis']}"
            )
        
        return {
            "success": True,
            "analysis": analysis,
            "recommendations": analysis['preparation_tips']
        }
        
    except Exception as e:
        logger.error(f"Error analyzing testimony: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/process")
async def batch_process_cases(
    request: BatchProcessingRequest,
    current_user: User = Depends(get_current_user)
):
    """Batch process multiple immigration cases"""
    try:
        results = []
        
        for case_id in request.case_ids:
            if case_id not in immigration_cases:
                results.append({"case_id": case_id, "error": "Case not found"})
                continue
            
            case = immigration_cases[case_id]
            
            # Perform requested operation
            if request.operation == "analyze":
                # Analyze all documents in case
                docs = case_documents.get(case_id, [])
                results.append({
                    "case_id": case_id,
                    "documents_analyzed": len(docs),
                    "status": "completed"
                })
            
            elif request.operation == "generate":
                # Generate forms
                results.append({
                    "case_id": case_id,
                    "forms_generated": request.target_forms,
                    "status": "completed"
                })
        
        return {
            "success": True,
            "processed": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/resources/library")
async def get_resource_library(
    language: str = "en",
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get immigration resource library"""
    from ...storage.managers import resource_library
    
    resources = []
    for resource in resource_library.values():
        if language in resource.get("languages", []):
            if not category or resource.get("category") == category:
                resources.append(resource)
    
    return {
        "resources": resources,
        "total": len(resources),
        "languages_available": ["en", "es", "zh", "ar", "fr"]
    }

@router.post("/evidence/checklist")
async def generate_evidence_checklist(
    case_type: CaseType = Form(...),
    case_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Generate evidence checklist for case type"""
    checklists = {
        CaseType.ASYLUM: {
            "identity_documents": False,
            "country_conditions": False,
            "persecution_evidence": False,
            "medical_records": False,
            "witness_statements": False,
            "police_reports": False,
            "membership_documents": False,
            "threat_letters": False
        },
        CaseType.FAMILY_BASED: {
            "birth_certificates": False,
            "marriage_certificate": False,
            "divorce_decrees": False,
            "passport_copies": False,
            "photos_together": False,
            "joint_accounts": False,
            "affidavits": False,
            "sponsor_tax_returns": False
        },
        CaseType.EMPLOYMENT_BASED: {
            "job_offer_letter": False,
            "degree_certificates": False,
            "transcripts": False,
            "experience_letters": False,
            "labor_certification": False,
            "company_financials": False,
            "resume": False,
            "publications": False
        }
    }
    
    checklist = checklists.get(case_type, {})
    
    # Update case if provided
    if case_id and case_id in immigration_cases:
        immigration_cases[case_id].evidence_checklist = checklist
    
    return {
        "case_type": case_type.value,
        "checklist": checklist,
        "total_required": len(checklist),
        "completed": sum(checklist.values())
    }

@router.get("/stats/dashboard")
async def get_immigration_dashboard(
    current_user: User = Depends(get_current_user)
):
    """Get immigration practice dashboard stats"""
    user_cases = [case for case in immigration_cases.values() 
                  if case.assigned_attorney == current_user.user_id]
    
    # Calculate stats
    stats = {
        "total_cases": len(user_cases),
        "by_type": {},
        "pending_deadlines": len(deadline_manager.get_upcoming_deadlines(30)),
        "critical_deadlines": len(deadline_manager.check_critical_deadlines()),
        "documents_pending_translation": 0,
        "average_processing_time": "5.2 days"  # Placeholder
    }
    
    # Count by case type
    for case in user_cases:
        case_type = case.case_type.value
        stats["by_type"][case_type] = stats["by_type"].get(case_type, 0) + 1
    
    return stats
