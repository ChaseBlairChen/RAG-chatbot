# Enhanced Legal Document Analysis with DeepSeek Integration
# Combines safe fact extraction with AI-powered analysis

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
import re
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import io
import aiohttp
import asyncio

# Import your existing document processing code
from document_analysis import (
    SafeDocumentProcessor, 
    NoHallucinationAnalyzer,
    PYMUPDF_AVAILABLE,
    PDFPLUMBER_AVAILABLE
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
safe_analyzer = NoHallucinationAnalyzer()

# OpenRouter configuration for DeepSeek
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

# Analysis prompts for different tools
ANALYSIS_PROMPTS = {
    'summarize': """You are a legal document analyst. Analyze this document and provide:
1. A clear summary of the document's purpose and type
2. The main parties involved (with their roles)
3. Key terms and conditions
4. Important dates and deadlines
5. Financial obligations or amounts
6. Any notable risks or concerns

Document text:
{document_text}

Provide a structured summary in plain English while maintaining legal accuracy.""",

    'extract-clauses': """You are a legal document analyst. Extract and categorize the following types of clauses from this document:
1. Termination clauses
2. Indemnification provisions
3. Liability limitations
4. Governing law and jurisdiction
5. Confidentiality/NDA provisions
6. Payment terms
7. Dispute resolution mechanisms

For each clause found, provide:
- Clause type
- Summary of the provision
- Exact location/section reference if available
- Any unusual or concerning aspects

Document text:
{document_text}""",

    'missing-clauses': """You are a legal document analyst. Review this contract and identify commonly expected clauses that appear to be missing or inadequately addressed:

Consider standard clauses such as:
- Force majeure
- Limitation of liability
- Indemnification
- Dispute resolution/arbitration
- Confidentiality
- Termination provisions
- Assignment restrictions
- Severability
- Entire agreement
- Notice provisions
- Governing law

Document text:
{document_text}

For each missing clause, explain why it's typically important and the risks of its absence.""",

    'risk-flagging': """You are a legal risk analyst. Identify and assess legal risks in this document:

Look for:
1. Unilateral termination rights
2. Broad indemnification requirements
3. Unlimited liability exposure
4. Vague or ambiguous obligations
5. Unfavorable payment terms
6. Lack of protection clauses
7. Unusual warranty provisions
8. Problematic intellectual property terms

Document text:
{document_text}

For each risk, provide:
- Risk description
- Severity (High/Medium/Low)
- Potential impact
- Suggested mitigation""",

    'timeline-extraction': """You are a legal document analyst. Extract all time-related information:

Find and list:
1. Contract start and end dates
2. Payment deadlines
3. Notice periods
4. Renewal dates and terms
5. Termination notice requirements
6. Performance deadlines
7. Warranty periods
8. Any other time-sensitive obligations

Document text:
{document_text}

Present as a chronological timeline with clear labels.""",

    'obligations': """You are a legal document analyst. List all obligations and requirements for each party:

Identify:
1. What each party must do
2. When they must do it
3. Conditions or prerequisites
4. Consequences of non-compliance
5. Reporting or notification requirements

Document text:
{document_text}

Organize by party and priority/timeline."""
}

class EnhancedAnalysisResponse(BaseModel):
    # Compatibility fields
    response: Optional[str] = None
    summary: Optional[str] = None
    factual_summary: Optional[str] = None
    
    # Analysis results
    ai_analysis: Optional[str] = None
    extraction_results: Optional[Dict[str, Any]] = None
    
    # Metadata
    analysis_type: str
    confidence_score: float
    processing_info: Optional[Dict[str, Any]] = None
    verification_status: str
    status: str = "completed"
    success: bool = True
    
    # Additional info
    warnings: List[str] = []
    session_id: str
    timestamp: str
    model_used: str = "deepseek-chat-v3"

# FastAPI app
app = FastAPI(
    title="Enhanced Legal Document Analysis with DeepSeek",
    description="Combines fact extraction with AI-powered legal analysis",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def call_deepseek(prompt: str, temperature: float = 0.3) -> str:
    """Call DeepSeek via OpenRouter API"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",  # Optional but recommended
        "X-Title": "Legal Document Analyzer"       # Optional but recommended
    }
    
    payload = {
        "model": "deepseek/deepseek-chat",  # OpenRouter model name
        "messages": [
            {
                "role": "system",
                "content": "You are an expert legal document analyst. Provide thorough, accurate analysis while clearly marking any uncertainties. Always include relevant disclaimers about seeking professional legal advice."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    return f"AI analysis failed: {response.status} - {error_text}"
    except Exception as e:
        logger.error(f"OpenRouter API exception: {e}")
        return f"AI analysis failed: {str(e)}"

async def analyze_with_deepseek(document_text: str, analysis_type: str) -> Tuple[str, float]:
    """Perform AI analysis using DeepSeek"""
    
    # Get the appropriate prompt
    prompt_template = ANALYSIS_PROMPTS.get(analysis_type, ANALYSIS_PROMPTS['summarize'])
    prompt = prompt_template.format(document_text=document_text[:15000])  # Limit context length
    
    # Call DeepSeek
    ai_response = await call_deepseek(prompt)
    
    # Simple confidence scoring based on response quality
    confidence = 0.85  # Base confidence for DeepSeek
    if "unclear" in ai_response.lower() or "cannot determine" in ai_response.lower():
        confidence = 0.65
    elif len(ai_response) < 100:
        confidence = 0.5
    
    return ai_response, confidence

@app.post("/document-analysis", response_model=EnhancedAnalysisResponse)
async def enhanced_document_analysis(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    analysis_type: Optional[str] = Form("summarize")
):
    """Enhanced analysis combining fact extraction with DeepSeek AI analysis"""
    
    session_id = session_id or str(uuid.uuid4())
    
    try:
        # Log the request
        logger.info(f"Enhanced analysis request: {file.filename}, type: {analysis_type}, session: {session_id}")
        
        # Process document to extract text
        document_text, file_type, processing_info = SafeDocumentProcessor.process_document_safe(file)
        
        # Perform fact extraction (your existing safe extraction)
        extraction_results = safe_analyzer.extract_document_facts(document_text)
        
        # Perform AI analysis with DeepSeek
        ai_analysis, confidence_score = await analyze_with_deepseek(document_text, analysis_type)
        
        # Combine results
        combined_summary = f"""## AI Legal Analysis ({analysis_type})

{ai_analysis}

---

## Verified Facts Extracted from Document

{safe_analyzer.generate_factual_summary(document_text)}

---

**⚠️ DISCLAIMER**: This AI-generated analysis is for informational purposes only and does not constitute legal advice. The extracted facts above are verified from the document, while the analysis section is AI-generated. Always consult with a qualified attorney for legal matters.
"""
        
        # Determine overall status
        successful_extractions = len([k for k, v in extraction_results.items() 
                                    if not (isinstance(v, list) and v and v[0].get('status') == 'failed_to_extract')])
        
        if successful_extractions >= 3 and confidence_score > 0.7:
            verification_status = "high_confidence"
        elif successful_extractions >= 1 or confidence_score > 0.5:
            verification_status = "medium_confidence"
        else:
            verification_status = "low_confidence"
        
        return EnhancedAnalysisResponse(
            response=combined_summary,  # For compatibility
            summary=combined_summary,   # For compatibility
            factual_summary=combined_summary,  # For compatibility
            ai_analysis=ai_analysis,
            extraction_results=extraction_results,
            analysis_type=analysis_type,
            confidence_score=confidence_score,
            processing_info=processing_info,
            verification_status=verification_status,
            status="completed",
            success=True,
            warnings=processing_info.get('warnings', []),
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            model_used="deepseek-chat-v3"
        )
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {type(e).__name__}: {str(e)}")
        
        # Return a failed response with error details
        error_message = f"""## Analysis Failed

**Error**: {type(e).__name__}: {str(e)}

The document could not be analyzed. Please check:
1. The file is a valid PDF, DOCX, or TXT document
2. The file is not corrupted
3. The file size is under 10MB

If the problem persists, please contact support.
"""
        
        return EnhancedAnalysisResponse(
            response=error_message,
            summary=error_message,
            factual_summary=error_message,
            ai_analysis=None,
            extraction_results=None,
            analysis_type=analysis_type,
            confidence_score=0.0,
            processing_info={"error": str(e), "error_type": type(e).__name__},
            verification_status="failed",
            status="failed",
            success=False,
            warnings=[],
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "Enhanced Mode Active",
        "version": "4.0.0",
        "features": [
            "✅ DeepSeek AI Analysis",
            "✅ Fact Extraction",
            "📍 Source Tracking",
            "🔍 Multi-level Analysis",
            "⚠️ Legal Disclaimers"
        ],
        "pdf_processing": {
            "pymupdf_available": PYMUPDF_AVAILABLE,
            "pdfplumber_available": PDFPLUMBER_AVAILABLE
        },
        "ai_model": "deepseek-chat-v3-0324:free",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint with information"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Legal Document Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            .feature { background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 5px 0; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Enhanced Legal Document Analysis API</h1>
            <p><strong>Version 4.0.0</strong> - Powered by DeepSeek AI</p>
            
            <div class="feature">
                <h3>✨ What's New</h3>
                <ul>
                    <li>AI-powered analysis using DeepSeek v3</li>
                    <li>Comprehensive legal document understanding</li>
                    <li>Combined AI insights with fact verification</li>
                    <li>Support for all major analysis types</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>📍 Available Endpoints</h3>
                <div class="endpoint">POST /document-analysis</div>
                <p>Main analysis endpoint - upload document and specify analysis type</p>
                
                <div class="endpoint">GET /health</div>
                <p>Check system status and capabilities</p>
            </div>
            
            <div class="feature">
                <h3>🔧 Analysis Types</h3>
                <ul>
                    <li><code>summarize</code> - Comprehensive document summary</li>
                    <li><code>extract-clauses</code> - Extract key legal clauses</li>
                    <li><code>missing-clauses</code> - Identify missing standard clauses</li>
                    <li><code>risk-flagging</code> - Flag potential legal risks</li>
                    <li><code>timeline-extraction</code> - Extract all dates and deadlines</li>
                    <li><code>obligations</code> - List party obligations</li>
                </ul>
            </div>
            
            <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
                Built with DeepSeek AI for powerful legal document understanding 🎯
            </p>
        </div>
    </body>
    </html>
    """

# Import existing endpoints that still work
from document_analysis import (
    verify_extraction,
    test_upload,
    get_extraction_capabilities
)

app.post("/verify-extraction")(verify_extraction)
app.post("/test-upload")(test_upload)
app.get("/extraction-capabilities")(get_extraction_capabilities)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Enhanced Legal Document Analysis on port {port}")
    logger.info(f"AI Model: DeepSeek Chat v3")
    uvicorn.run(app, host="0.0.0.0", port=port)
