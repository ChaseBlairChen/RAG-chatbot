"""Analysis endpoints"""
import logging
from fastapi import APIRouter, Depends, HTTPException

from ...models import User, ComprehensiveAnalysisRequest, StructuredAnalysisResponse, AnalysisType
from ...core.security import get_current_user
from ...services.analysis_service import ComprehensiveAnalysisProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/comprehensive-analysis", response_model=StructuredAnalysisResponse)
async def comprehensive_document_analysis(
    request: ComprehensiveAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Comprehensive document analysis endpoint"""
    logger.info(f"Comprehensive analysis request: user={request.user_id}, doc={request.document_id}, types={request.analysis_types}")
    
    try:
        if request.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Cannot analyze documents for different user")
        
        processor = ComprehensiveAnalysisProcessor()
        result = processor.process_comprehensive_analysis(request)
        
        logger.info(f"Comprehensive analysis completed: confidence={result.overall_confidence:.2f}, time={result.processing_time:.2f}s")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/quick-analysis/{document_id}")
async def quick_document_analysis(
    document_id: str,
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
    current_user: User = Depends(get_current_user)
):
    """Quick analysis endpoint for single documents"""
    try:
        request = ComprehensiveAnalysisRequest(
            document_id=document_id,
            analysis_types=[analysis_type],
            user_id=current_user.user_id,
            response_style="detailed"
        )
        
        processor = ComprehensiveAnalysisProcessor()
        result = processor.process_comprehensive_analysis(request)
        
        return {
            "success": True,
            "analysis": result,
            "message": f"Analysis completed with {result.overall_confidence:.1%} confidence"
        }
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Analysis failed"
        }
