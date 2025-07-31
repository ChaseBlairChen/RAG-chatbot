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
