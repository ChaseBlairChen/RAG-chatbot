"""Immigration-specific helper functions"""
import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

def calculate_priority_date_wait(category: str, country: str, priority_date: datetime) -> Dict:
    """Calculate estimated wait time based on visa bulletin"""
    # This would integrate with State Dept API in production
    # For now, return estimates
    wait_times = {
        "F1": {"world": 84, "mexico": 276, "philippines": 144, "india": 144, "china": 144},
        "F2A": {"world": 24, "mexico": 24, "philippines": 24, "india": 24, "china": 24},
        "F2B": {"world": 60, "mexico": 264, "philippines": 132, "india": 96, "china": 60},
        "F3": {"world": 144, "mexico": 276, "philippines": 216, "india": 180, "china": 156},
        "F4": {"world": 156, "mexico": 288, "philippines": 240, "india": 180, "china": 168},
    }
    
    country_key = country.lower() if country.lower() in wait_times.get(category, {}) else "world"
    months_wait = wait_times.get(category, {}).get(country_key, 60)
    
    estimated_date = priority_date + timedelta(days=months_wait * 30)
    
    return {
        "category": category,
        "country": country,
        "priority_date": priority_date.isoformat(),
        "estimated_wait_months": months_wait,
        "estimated_current_date": estimated_date.isoformat(),
        "disclaimer": "Estimates based on historical data. Check official visa bulletin."
    }

def validate_alien_number(a_number: str) -> bool:
    """Validate A-Number format"""
    pattern = r'^A?\d{8,9}$'
    return bool(re.match(pattern, a_number.replace("-", "").replace(" ", "")))

def validate_receipt_number(receipt: str) -> Tuple[bool, str]:
    """Validate USCIS receipt number and identify service center"""
    pattern = r'^([A-Z]{3})(\d{10})$'
    match = re.match(pattern, receipt)
    
    if not match:
        return False, ""
    
    service_centers = {
        "EAC": "Vermont Service Center",
        "WAC": "California Service Center",
        "LIN": "Nebraska Service Center",
        "SRC": "Texas Service Center",
        "MSC": "National Benefits Center",
        "IOE": "USCIS Electronic Immigration System"
    }
    
    center_code = match.group(1)
    return True, service_centers.get(center_code, "Unknown Service Center")

def calculate_age_out(child_dob: datetime, priority_date: datetime, category: str) -> Dict:
    """Calculate if child will age out under CSPA"""
    current_age = (datetime.utcnow() - child_dob).days / 365.25
    
    # Simplified CSPA calculation
    if category in ["F2A", "F1"]:
        cspa_protection = True
        protected_age = current_age  # More complex in reality
    else:
        cspa_protection = False
        protected_age = current_age
    
    return {
        "current_age": round(current_age, 1),
        "will_age_out": current_age >= 21,
        "cspa_protected": cspa_protection,
        "protected_age": round(protected_age, 1),
        "safe_filing_deadline": (child_dob + timedelta(days=21*365)).isoformat()
    }

def generate_interview_prep_checklist(case_type: str) -> List[Dict]:
    """Generate interview preparation checklist"""
    base_checklist = [
        {"item": "Valid passport", "required": True},
        {"item": "Interview notice", "required": True},
        {"item": "All original documents", "required": True},
        {"item": "Translations of foreign documents", "required": True},
        {"item": "Photos (passport style)", "required": True, "quantity": 2}
    ]
    
    if case_type == "naturalization":
        base_checklist.extend([
            {"item": "Green card", "required": True},
            {"item": "Tax returns (5 years)", "required": True},
            {"item": "Selective service registration (if applicable)", "required": False}
        ])
    elif case_type == "adjustment":
        base_checklist.extend([
            {"item": "I-94 arrival record", "required": True},
            {"item": "Birth certificate", "required": True},
            {"item": "Medical exam (I-693)", "required": True}
        ])
    
    return base_checklist

def detect_potential_inadmissibility(text: str) -> List[str]:
    """Detect potential inadmissibility issues in text"""
    issues = []
    
    triggers = {
        "criminal": ["arrest", "conviction", "jail", "prison", "charged", "guilty"],
        "immigration_violation": ["deported", "removed", "overstay", "unlawful presence"],
        "misrepresentation": ["false", "lied", "fraud", "misrepresent"],
        "health": ["tuberculosis", "mental disorder", "drug abuse"],
        "security": ["terrorist", "espionage", "genocide", "torture"]
    }
    
    text_lower = text.lower()
    for category, keywords in triggers.items():
        if any(keyword in text_lower for keyword in keywords):
            issues.append(category)
    
    return issues

def format_case_status_update(case_id: str, status: str, details: str = "") -> str:
    """Format case status update for client communication"""
    templates = {
        "received": f"Good news! USCIS has received your case (Case #{case_id}). {details}",
        "approved": f"Congratulations! Your case (#{case_id}) has been APPROVED! {details}",
        "rfe": f"USCIS has requested additional evidence for case #{case_id}. Don't worry, this is common. {details}",
        "denied": f"Unfortunately, case #{case_id} was denied. We can discuss appeal options. {details}",
        "transferred": f"Your case #{case_id} has been transferred to another office. {details}"
    }
    
    return templates.get(status, f"Status update for case #{case_id}: {status}. {details}")

def estimate_processing_time(form_type: str, service_center: str = "average") -> Dict:
    """Estimate processing times by form type"""
    # In production, this would pull from USCIS processing times API
    processing_times = {
        "I-130": {"normal": 8, "premium": None},
        "I-485": {"normal": 12, "premium": None},
        "I-765": {"normal": 3, "premium": None},
        "I-131": {"normal": 4, "premium": None},
        "N-400": {"normal": 9, "premium": None},
        "I-129": {"normal": 4, "premium": 0.5},
        "I-140": {"normal": 6, "premium": 0.5}
    }
    
    times = processing_times.get(form_type, {"normal": 6, "premium": None})
    
    return {
        "form_type": form_type,
        "normal_processing_months": times["normal"],
        "premium_available": times["premium"] is not None,
        "premium_processing_months": times["premium"],
        "last_updated": datetime.utcnow().isoformat(),
        "disclaimer": "Processing times vary by service center and case complexity"
    }
