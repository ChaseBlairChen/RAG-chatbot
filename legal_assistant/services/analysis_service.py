"""
Enhanced Analysis Service with AI Agents for Specialized Document Analysis
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json

from ..models import AnalysisType, DocumentCategory
from ..services.ai_service import call_openrouter_api
from ..utils.formatting import format_context_for_llm

logger = logging.getLogger(__name__)

class AnalysisAgent(Enum):
    """Specialized analysis agents"""
    CONTRACT_ANALYST = "contract_analyst"
    IMMIGRATION_SPECIALIST = "immigration_specialist"
    LITIGATION_ANALYST = "litigation_analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_ASSESSOR = "risk_assessor"
    GENERAL_ANALYST = "general_analyst"

@dataclass
class AnalysisTask:
    """Analysis task definition"""
    agent: AnalysisAgent
    document_content: str
    document_type: str
    analysis_focus: List[str]
    user_context: Optional[str] = None
    priority: str = "normal"

@dataclass
class AnalysisResult:
    """Enhanced analysis result"""
    agent: AnalysisAgent
    analysis_type: str
    findings: Dict[str, Any]
    recommendations: List[str]
    risk_level: str
    confidence_score: float
    processing_time: float
    sources_cited: List[str]

class AIAnalysisAgent:
    """Base class for AI analysis agents"""
    
    def __init__(self, agent_type: AnalysisAgent):
        self.agent_type = agent_type
        self.specialized_prompts = self._get_specialized_prompts()
    
    def _get_specialized_prompts(self) -> Dict[str, str]:
        """Get specialized prompts for this agent"""
        return {
            "contract_analyst": """You are an expert contract analyst. Analyze the document for:
1. Key Terms and Conditions
2. Obligations and Rights
3. Risk Factors and Red Flags
4. Missing Clauses
5. Unfavorable Terms
6. Compliance Issues
7. Recommendations for Improvement

Structure your analysis with clear sections and actionable insights.""",
            
            "immigration_specialist": """You are an immigration law specialist. Analyze the document for:
1. Immigration Status Implications
2. Form Requirements and Deadlines
3. Evidence Requirements
4. Potential Issues or Red Flags
5. USCIS Processing Considerations
6. Legal Strategy Recommendations
7. Risk Assessment

Focus on practical immigration law applications.""",
            
            "litigation_analyst": """You are a litigation analyst. Analyze the document for:
1. Legal Claims and Defenses
2. Evidence Strength
3. Procedural Issues
4. Statute of Limitations
5. Jurisdiction and Venue
6. Settlement Considerations
7. Trial Strategy Implications

Provide litigation-focused analysis and recommendations.""",
            
            "compliance_officer": """You are a compliance officer. Analyze the document for:
1. Regulatory Compliance Issues
2. Required Disclosures
3. Reporting Obligations
4. Record-Keeping Requirements
5. Penalty Risks
6. Compliance Recommendations
7. Monitoring Requirements

Focus on regulatory and compliance aspects.""",
            
            "risk_assessor": """You are a legal risk assessor. Analyze the document for:
1. Legal Risks and Liabilities
2. Financial Exposure
3. Reputational Risks
4. Operational Risks
5. Mitigation Strategies
6. Insurance Considerations
7. Risk Monitoring Recommendations

Provide comprehensive risk assessment and mitigation strategies."""
        }
    
    async def analyze_document(self, task: AnalysisTask) -> AnalysisResult:
        """Analyze document using specialized agent"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create specialized prompt
            prompt = self._create_analysis_prompt(task)
            
            # Get AI analysis
            analysis_text = await self._get_ai_analysis(prompt, task.document_content)
            
            # Parse and structure results
            findings = self._parse_analysis_results(analysis_text)
            
            # Generate recommendations
            recommendations = self._extract_recommendations(analysis_text)
            
            # Assess risk level
            risk_level = self._assess_risk_level(findings)
            
            # Calculate confidence
            confidence_score = self._calculate_confidence(findings, task.document_content)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AnalysisResult(
                agent=self.agent_type,
                analysis_type=task.analysis_focus[0] if task.analysis_focus else "general",
                findings=findings,
                recommendations=recommendations,
                risk_level=risk_level,
                confidence_score=confidence_score,
                processing_time=processing_time,
                sources_cited=[]
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {self.agent_type}: {e}")
            return self._create_error_result(task, str(e))
    
    def _create_analysis_prompt(self, task: AnalysisTask) -> str:
        """Create specialized analysis prompt"""
        base_prompt = self.specialized_prompts.get(self.agent_type.value, self.specialized_prompts["general_analyst"])
        
        prompt = f"""
{base_prompt}

DOCUMENT TYPE: {task.document_type}
ANALYSIS FOCUS: {', '.join(task.analysis_focus)}
USER CONTEXT: {task.user_context or 'None provided'}

Please analyze the following document and provide a structured response in JSON format with the following structure:
{{
    "key_findings": {{
        "main_issues": ["list of main issues"],
        "strengths": ["list of strengths"],
        "weaknesses": ["list of weaknesses"],
        "risks": ["list of risks"]
    }},
    "detailed_analysis": {{
        "section_1": "detailed analysis of section 1",
        "section_2": "detailed analysis of section 2"
    }},
    "recommendations": ["list of specific recommendations"],
    "risk_assessment": {{
        "overall_risk": "low/medium/high",
        "risk_factors": ["list of risk factors"],
        "mitigation_strategies": ["list of mitigation strategies"]
    }},
    "next_steps": ["list of recommended next steps"]
}}

DOCUMENT CONTENT:
{task.document_content[:8000]}  # Limit content length
"""
        return prompt
    
    async def _get_ai_analysis(self, prompt: str, document_content: str) -> str:
        """Get AI analysis with enhanced prompting"""
        try:
            response = call_openrouter_api(prompt, timeout=120)
            return response
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._create_fallback_analysis(document_content)
    
    def _parse_analysis_results(self, analysis_text: str) -> Dict[str, Any]:
        """Parse AI analysis results into structured format"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback to text parsing
            return self._parse_text_analysis(analysis_text)
        except Exception as e:
            logger.error(f"Failed to parse analysis results: {e}")
            return {"error": "Failed to parse analysis results", "raw_text": analysis_text}
    
    def _parse_text_analysis(self, text: str) -> Dict[str, Any]:
        """Parse text-based analysis into structured format"""
        findings = {
            "key_findings": {"main_issues": [], "strengths": [], "weaknesses": [], "risks": []},
            "detailed_analysis": {},
            "recommendations": [],
            "risk_assessment": {"overall_risk": "medium", "risk_factors": [], "mitigation_strategies": []},
            "next_steps": []
        }
        
        # Extract recommendations
        rec_pattern = r'(?:recommend|suggest|advise|should|must).*?(?:\.|$)'
        recommendations = re.findall(rec_pattern, text, re.IGNORECASE)
        findings["recommendations"] = [rec.strip() for rec in recommendations[:5]]
        
        # Extract risk indicators
        risk_pattern = r'\b(risk|danger|warning|caution|concern|issue|problem)\b'
        risk_factors = re.findall(risk_pattern, text, re.IGNORECASE)
        findings["risk_assessment"]["risk_factors"] = list(set(risk_factors))
        
        return findings
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract actionable recommendations from analysis"""
        recommendations = []
        
        # Look for recommendation patterns
        patterns = [
            r'(?:recommend|suggest|advise|should|must|consider|implement).*?(?:\.|$)',
            r'(?:action item|next step|priority|urgent).*?(?:\.|$)',
            r'(?:improve|enhance|strengthen|address|resolve).*?(?:\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            recommendations.extend([match.strip() for match in matches])
        
        return list(set(recommendations))[:10]  # Limit to 10 unique recommendations
    
    def _assess_risk_level(self, findings: Dict[str, Any]) -> str:
        """Assess overall risk level based on findings"""
        risk_indicators = 0
        
        # Count risk factors
        if "risk_assessment" in findings:
            risk_factors = findings["risk_assessment"].get("risk_factors", [])
            risk_indicators += len(risk_factors)
        
        # Count issues and weaknesses
        if "key_findings" in findings:
            issues = findings["key_findings"].get("main_issues", [])
            weaknesses = findings["key_findings"].get("weaknesses", [])
            risk_indicators += len(issues) + len(weaknesses)
        
        # Determine risk level
        if risk_indicators >= 5:
            return "high"
        elif risk_indicators >= 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, findings: Dict[str, Any], document_content: str) -> float:
        """Calculate confidence score for analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on structured findings
        if "key_findings" in findings and findings["key_findings"]:
            confidence += 0.2
        
        if "recommendations" in findings and findings["recommendations"]:
            confidence += 0.2
        
        if "risk_assessment" in findings and findings["risk_assessment"]:
            confidence += 0.1
        
        # Decrease confidence for errors
        if "error" in findings:
            confidence -= 0.3
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_fallback_analysis(self, document_content: str) -> str:
        """Create fallback analysis when AI fails"""
        return f"""
Analysis could not be completed due to technical issues.

Document Summary:
- Document length: {len(document_content)} characters
- Document type: General legal document
- Recommendation: Please review manually or try again later

Key areas to review:
1. Document structure and completeness
2. Legal requirements and compliance
3. Potential risks and issues
4. Recommended actions
"""
    
    def _create_error_result(self, task: AnalysisTask, error: str) -> AnalysisResult:
        """Create error result when analysis fails"""
        return AnalysisResult(
            agent=self.agent_type,
            analysis_type="error",
            findings={"error": error},
            recommendations=["Please try again or contact support"],
            risk_level="unknown",
            confidence_score=0.0,
            processing_time=0.0,
            sources_cited=[]
        )

class AnalysisService:
    """Enhanced analysis service with multiple AI agents"""
    
    def __init__(self):
        self.agents = {
            AnalysisAgent.CONTRACT_ANALYST: AIAnalysisAgent(AnalysisAgent.CONTRACT_ANALYST),
            AnalysisAgent.IMMIGRATION_SPECIALIST: AIAnalysisAgent(AnalysisAgent.IMMIGRATION_SPECIALIST),
            AnalysisAgent.LITIGATION_ANALYST: AIAnalysisAgent(AnalysisAgent.LITIGATION_ANALYST),
            AnalysisAgent.COMPLIANCE_OFFICER: AIAnalysisAgent(AnalysisAgent.COMPLIANCE_OFFICER),
            AnalysisAgent.RISK_ASSESSOR: AIAnalysisAgent(AnalysisAgent.RISK_ASSESSOR),
            AnalysisAgent.GENERAL_ANALYST: AIAnalysisAgent(AnalysisAgent.GENERAL_ANALYST)
        }
    
    async def analyze_document_comprehensive(self, document_content: str, document_type: str, 
                                           analysis_types: List[str], user_context: str = None) -> List[AnalysisResult]:
        """Perform comprehensive document analysis using multiple agents"""
        results = []
        
        # Determine which agents to use based on analysis types
        agents_to_use = self._select_agents(analysis_types, document_type)
        
        # Create analysis tasks
        tasks = []
        for agent in agents_to_use:
            task = AnalysisTask(
                agent=agent,
                document_content=document_content,
                document_type=document_type,
                analysis_focus=analysis_types,
                user_context=user_context
            )
            tasks.append(task)
        
        # Run analyses concurrently
        analysis_tasks = [self.agents[task.agent].analyze_document(task) for task in tasks]
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [result for result in results if isinstance(result, AnalysisResult)]
        
        return valid_results
    
    def _select_agents(self, analysis_types: List[str], document_type: str) -> List[AnalysisAgent]:
        """Select appropriate agents based on analysis types and document type"""
        selected_agents = []
        
        # Always include general analyst
        selected_agents.append(AnalysisAgent.GENERAL_ANALYST)
        
        # Select specialized agents based on analysis types
        for analysis_type in analysis_types:
            if "contract" in analysis_type.lower():
                selected_agents.append(AnalysisAgent.CONTRACT_ANALYST)
            elif "immigration" in analysis_type.lower():
                selected_agents.append(AnalysisAgent.IMMIGRATION_SPECIALIST)
            elif "litigation" in analysis_type.lower():
                selected_agents.append(AnalysisAgent.LITIGATION_ANALYST)
            elif "compliance" in analysis_type.lower():
                selected_agents.append(AnalysisAgent.COMPLIANCE_OFFICER)
            elif "risk" in analysis_type.lower():
                selected_agents.append(AnalysisAgent.RISK_ASSESSOR)
        
        # Select based on document type
        if "contract" in document_type.lower():
            selected_agents.append(AnalysisAgent.CONTRACT_ANALYST)
        elif "immigration" in document_type.lower():
            selected_agents.append(AnalysisAgent.IMMIGRATION_SPECIALIST)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in selected_agents:
            if agent not in seen:
                seen.add(agent)
                unique_agents.append(agent)
        
        return unique_agents

# Global instance
analysis_service = AnalysisService()
