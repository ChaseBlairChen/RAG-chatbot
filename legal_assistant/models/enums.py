"""Enumeration types for the legal assistant application"""
from enum import Enum

class AnalysisType(str, Enum):
    """Types of analysis that can be performed on legal documents"""
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "document_summary"
    CLAUSES = "key_clauses"
    RISKS = "risk_assessment"
    TIMELINE = "timeline_deadlines"
    OBLIGATIONS = "party_obligations"
    MISSING_CLAUSES = "missing_clauses"
    # Immigration-specific
    IMMIGRATION_FORM = "immigration_form_analysis"
    COUNTRY_CONDITIONS = "country_conditions"
    EVIDENCE_CHECKLIST = "evidence_checklist"
    CONSISTENCY_CHECK = "consistency_check"
    CREDIBLE_FEAR = "credible_fear_prep"

class DocumentCategory(str, Enum):
    """Document categories for immigration law"""
    USCIS_FORM = "uscis_form"
    EVIDENCE = "evidence"
    IDENTITY = "identity_document"
    FINANCIAL = "financial_document"
    LEGAL = "legal_document"
    MEDICAL = "medical_record"
    COUNTRY_EVIDENCE = "country_evidence"
    CORRESPONDENCE = "correspondence"
    TRANSLATION = "translation"

class ImmigrationFormType(str, Enum):
    """Common USCIS form types"""
    I_130 = "I-130"  # Family-based petition
    I_485 = "I-485"  # Adjustment of status
    I_765 = "I-765"  # Work authorization
    I_131 = "I-131"  # Travel document
    I_589 = "I-589"  # Asylum application
    I_129 = "I-129"  # Nonimmigrant worker
    I_140 = "I-140"  # Employment-based petition
    N_400 = "N-400"  # Naturalization
    I_751 = "I-751"  # Remove conditions
    I_90 = "I-90"   # Green card replacement

class CaseType(str, Enum):
    """Immigration case types"""
    ASYLUM = "asylum"
    FAMILY_BASED = "family_based"
    EMPLOYMENT_BASED = "employment_based"
    REMOVAL_DEFENSE = "removal_defense"
    NATURALIZATION = "naturalization"
    HUMANITARIAN = "humanitarian"
    INVESTOR = "investor"
