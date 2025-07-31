"""Authentication and authorization"""
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..models import User
from ..storage.managers import user_sessions

security = HTTPBearer(auto_error=False)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get current user from credentials"""
    if credentials is None:
        default_user_id = "debug_user"
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
    user_id = f"user_{token[:8]}"
    
    if user_id not in user_sessions:
        from ..services.container_manager import get_container_manager
        container_manager = get_container_manager()
        user_sessions[user_id] = User(
            user_id=user_id,
            container_id=container_manager.get_container_id(user_id),
            subscription_tier="free"
        )
    
    return user_sessions[user_id]
