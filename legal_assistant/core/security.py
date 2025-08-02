"""Authentication and authorization"""
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..models import User
from ..storage.managers import user_sessions
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get current user from credentials"""
    if credentials is None:
        default_user_id = "user_demo"
        logger.info(f"No credentials provided, using default user: {default_user_id}")
        
        if default_user_id not in user_sessions:
            from ..services.container_manager import get_container_manager
            container_manager = get_container_manager()
            user_sessions[default_user_id] = User(
                user_id=default_user_id,
                container_id=container_manager.get_container_id(default_user_id),
                subscription_tier="free"
            )
        return user_sessions[default_user_id]
    
    token = credentials.credentials
    logger.info(f"Token received: {token}")
    
    # Handle different token formats
    if token.startswith("user_"):
        # Token is like "user_demo_123456789"
        # Extract the username part (user_demo)
        parts = token.split('_')
        if len(parts) >= 2:
            user_id = f"{parts[0]}_{parts[1]}"  # This will give us "user_demo"
        else:
            user_id = token
    else:
        # For other token formats, just use a fixed user for now
        user_id = "user_demo"
    
    logger.info(f"Extracted user_id: {user_id}")
    
    if user_id not in user_sessions:
        from ..services.container_manager import get_container_manager
        container_manager = get_container_manager()
        user_sessions[user_id] = User(
            user_id=user_id,
            container_id=container_manager.get_container_id(user_id),
            subscription_tier="free"
        )
    
    return user_sessions[user_id]
