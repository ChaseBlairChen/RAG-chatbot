"""Immigration-specific services"""
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
from cryptography.fernet import Fernet

from ..models import (
    ImmigrationCase, DeadlineAlert, DocumentClassification, 
    DocumentCategory, ImmigrationFormType, CaseType,
    CountryConditionsRequest
)
from ..storage.managers import immigration_cases, deadline_alerts
from .rag_service import combined_search
from .ai_service import call_openrouter_api
from ..utils.formatting import format_context_for_llm

logger = logging.getLogger(__name__)

class ImmigrationDocumentAnalyzer:
    """Analyzes and classifies immigration documents"""
    
    FORM_PATTERNS = {
        ImmigrationFormType.I_130: r"I-130|Petition for Alien Relative",
        ImmigrationFormType.I_485: r"I-485|Application to Register Permanent Residence",
        ImmigrationFormType.I_765: r"I-765|Application for Employment Authorization",
        ImmigrationFormType.I_589: r"I-589|Application for Asylum",
        ImmigrationFormType.I_129: r"I-129|Petition for.*Nonimmigrant Worker",
        ImmigrationFormType.N_400: r"N-400|Application for Naturalization",
    }
    
    EVIDENCE_KEYWORDS = {
        DocumentCategory.IDENTITY: ["passport", "birth certificate", "driver license", "national id"],
        DocumentCategory.FINANCIAL: ["bank statement", "tax return", "pay stub", "employment letter"],
        DocumentCategory.MEDICAL: ["medical exam", "vaccination", "medical record", "doctor"],
        DocumentCategory.COUNTRY_EVIDENCE: ["country condition", "news article", "human rights", "persecution"],
    }
    
    def classify_document(self, content: str, filename: str) -> DocumentClassification:
        """Classify immigration document"""
        # Detect form type
        form_type = None
        for form, pattern in self.FORM_PATTERNS.items():
            if re.search(pattern, content, re.IGNORECASE):
                form_type = form
                break
        
        # Detect category
        category = DocumentCategory.USCIS_FORM if form_type else DocumentCategory.EVIDENCE
        
        # Check for evidence types
        content_lower = content.lower()
        for cat, keywords in self.EVIDENCE_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                category = cat
                break
        
        # Detect language (simple check)
        language = self._detect_language(content)
        requires_translation = language != "en"
        
        # Extract key data
        extracted_data = self._extract_form_data(content, form_type) if form_type else {}
        
        return DocumentClassification(
            document_id=hashlib.md5(f"{filename}{datetime.utcnow()}".encode()).hexdigest()[:16],
            category=category,
            form_type=form_type,
            language=language,
            requires_translation=requires_translation,
            extracted_data=extracted_data,
            confidence=0.85
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Check for common Spanish words
        spanish_indicators = ["para", "por", "con", "sin", "sobre", "entre", "hasta"]
        spanish_count = sum(1 for word in spanish_indicators if f" {word} " in text.lower())
        
        if spanish_count >= 3:
            return "es"
        
        # Add more language detection as needed
        return "en"
    
    def _extract_form_data(self, content: str, form_type: ImmigrationFormType) -> Dict:
        """Extract data from immigration forms"""
        data = {}
        
        # Extract common fields
        patterns = {
            "alien_number": r"A[\-\s]?(\d{8,9})",
            "receipt_number": r"[A-Z]{3}\d{10}",
            "priority_date": r"Priority Date:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            "date_of_birth": r"Date of Birth:?\s*(\d{1,2}/\d{1,2}/\d{4})",
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                data[field] = match.group(1) if match.groups() else match.group(0)
        
        return data

class DeadlineManager:
    """Manages immigration deadlines and alerts"""
    
    @staticmethod
    def create_deadline(case_id: str, deadline_type: str, due_date: datetime, 
                       description: str, priority: str = "normal") -> DeadlineAlert:
        """Create a new deadline alert"""
        deadline = DeadlineAlert(
            deadline_id=hashlib.md5(f"{case_id}{deadline_type}{due_date}".encode()).hexdigest()[:16],
            case_id=case_id,
            deadline_type=deadline_type,
            due_date=due_date,
            description=description,
            priority=priority
        )
        
        deadline_alerts[deadline.deadline_id] = deadline
        logger.info(f"Created deadline: {deadline_type} for case {case_id} due {due_date}")
        
        return deadline
    
    @staticmethod
    def get_upcoming_deadlines(days_ahead: int = 30) -> List[DeadlineAlert]:
        """Get deadlines in the next N days"""
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        upcoming = []
        
        for deadline in deadline_alerts.values():
            if not deadline.completed and deadline.due_date <= cutoff_date:
                upcoming.append(deadline)
        
        # Sort by due date and priority
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        upcoming.sort(key=lambda d: (d.due_date, priority_order.get(d.priority, 2)))
        
        return upcoming
    
    @staticmethod
    def check_critical_deadlines() -> List[DeadlineAlert]:
        """Check for critical deadlines within 7 days"""
        critical = []
        cutoff = datetime.utcnow() + timedelta(days=7)
        
        for deadline in deadline_alerts.values():
            if not deadline.completed and deadline.due_date <= cutoff:
                deadline.priority = "critical"
                critical.append(deadline)
        
        return critical

class CountryConditionsResearcher:
    """Research country conditions for asylum cases"""
    
    def research_country_conditions(self, request: CountryConditionsRequest) -> Dict:
        """Research country conditions using RAG"""
        topics_query = " ".join(request.topics)
        query = f"{request.country} {topics_query} {request.date_range or 'recent'}"
        
        # Search for country conditions
        results, sources, method = combined_search(
            query=query,
            user_id=None,
            search_scope="all",
            conversation_context="",
            k=20
        )
        
        if not results:
            return {
                "country": request.country,
                "summary": "No country conditions information found.",
                "topics": {},
                "sources": []
            }
        
        # Format results
        context, source_info = format_context_for_llm(results, max_length=6000)
        
        # Create analysis prompt
        prompt = f"""Analyze the following country conditions information for {request.country}:

{context}

Provide a comprehensive analysis covering:
1. Current human rights situation
2. Government persecution of specific groups
3. Violence and security conditions
4. Treatment of asylum seekers who return

Focus on: {', '.join(request.topics)}

Organize the information by topic and include specific examples with dates."""
        
        # Use longer timeout for research analysis
        analysis = call_openrouter_api(prompt, timeout=120)
        
        return {
            "country": request.country,
            "summary": analysis,
            "topics": self._parse_topics(analysis, request.topics),
            "sources": source_info,
            "research_date": datetime.utcnow().isoformat()
        }
    
    def _parse_topics(self, analysis: str, topics: List[str]) -> Dict[str, str]:
        """Parse analysis into topics"""
        topic_sections = {}
        
        for topic in topics:
            # Find section about this topic
            pattern = rf"(?i){topic}[:\s]*([^#]+?)(?=\n[A-Z]|\n\d\.|\Z)"
            match = re.search(pattern, analysis)
            if match:
                topic_sections[topic] = match.group(1).strip()
        
        return topic_sections

class CredibleFearPreparer:
    """Prepare credible fear interview documentation"""
    
    def analyze_testimony(self, testimony: str, country: str) -> Dict:
        """Analyze testimony for consistency and completeness"""
        prompt = f"""Analyze this asylum testimony for credible fear preparation:

TESTIMONY:
{testimony}

COUNTRY OF ORIGIN: {country}

Provide:
1. Key persecution claims identified
2. Consistency check - any contradictions or gaps
3. Missing information that should be addressed
4. Strengths of the case
5. Potential challenges
6. Recommended additional evidence needed

Be supportive but thorough in identifying areas that need clarification."""
        
        analysis = call_openrouter_api(prompt)
        
        # Extract sections
        return {
            "analysis": analysis,
            "persecution_claims": self._extract_claims(testimony),
            "consistency_score": self._check_consistency(testimony),
            "missing_elements": self._identify_gaps(testimony),
            "preparation_tips": self._generate_tips(analysis)
        }
    
    def _extract_claims(self, testimony: str) -> List[str]:
        """Extract persecution claims"""
        claims = []
        
        # Look for persecution indicators
        indicators = ["threatened", "attacked", "beaten", "arrested", "detained", 
                     "tortured", "discriminated", "targeted", "forced", "fled"]
        
        sentences = testimony.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in indicators):
                claims.append(sentence.strip())
        
        return claims[:10]  # Top 10 claims
    
    def _check_consistency(self, testimony: str) -> float:
        """Basic consistency check"""
        # Check for date consistency
        dates = re.findall(r'\b(\d{4})\b', testimony)
        if dates:
            date_range = max(dates) - min(dates)
            if date_range > 20:  # More than 20 years span might indicate issues
                return 0.7
        
        return 0.85  # Default consistency score
    
    def _identify_gaps(self, testimony: str) -> List[str]:
        """Identify missing elements"""
        gaps = []
        
        required_elements = {
            "specific_dates": r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            "perpetrator_identity": r'(government|police|military|group)',
            "medical_evidence": r'(hospital|doctor|medical|injury)',
            "witness_information": r'(witness|saw|observed)',
        }
        
        for element, pattern in required_elements.items():
            if not re.search(pattern, testimony, re.IGNORECASE):
                gaps.append(f"Missing: {element.replace('_', ' ')}")
        
        return gaps
    
    def _generate_tips(self, analysis: str) -> List[str]:
        """Generate preparation tips"""
        return [
            "Practice telling your story chronologically",
            "Prepare specific dates and locations",
            "Gather any supporting documents",
            "Be prepared to explain any inconsistencies",
            "Focus on facts rather than opinions"
        ]

class SecureDataHandler:
    """Handle sensitive immigration data with encryption"""
    
    def __init__(self):
        # In production, load from secure key management
        self.cipher_suite = Fernet(Fernet.generate_key())
    
    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive information"""
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive information"""
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def sanitize_for_display(self, text: str) -> str:
        """Remove sensitive information for display"""
        # Remove A-numbers
        text = re.sub(r'A\d{8,9}', 'A-XXXXXX', text)
        # Remove SSNs
        text = re.sub(r'\d{3}-\d{2}-\d{4}', 'XXX-XX-XXXX', text)
        # Remove passport numbers
        text = re.sub(r'[A-Z]\d{8}', 'X-XXXXXX', text)
        
        return text

# Initialize global instances
immigration_analyzer = ImmigrationDocumentAnalyzer()
deadline_manager = DeadlineManager()
country_researcher = CountryConditionsResearcher()
credible_fear_preparer = CredibleFearPreparer()
secure_handler = SecureDataHandler()
