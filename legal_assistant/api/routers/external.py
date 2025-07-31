"""External database endpoints"""
import logging
from typing import List
from fastapi import APIRouter, Form, Depends, HTTPException

from ...models import User
from ...core.security import get_current_user
from ...services.external_db_service import search_external_databases

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/external/search")
async def search_external_databases_endpoint(
    query: str = Form(...),
    databases: List[str] = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Search external legal databases (requires premium subscription)"""
    if current_user.subscription_tier not in ["premium", "enterprise"]:
        raise HTTPException(
            status_code=403, 
            detail="External database access requires premium subscription"
        )
    
    results = search_external_databases(query, databases, current_user)
    
    return {
        "query": query,
        "databases_searched": databases,
        "results": results,
        "total_results": len(results)
    }
