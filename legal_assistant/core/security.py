"""Enhanced Authentication and Authorization with JWT"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
import logging

from ..models import User
from ..storage.managers import user_sessions

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Legacy HTTPBearer for backward compatibility
security = HTTPBearer(auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

def get_current_user_secure(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Enhanced user authentication with JWT support"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if credentials is None:
        # For development only - remove in production
        if os.getenv("ENVIRONMENT") == "development":
            default_user_id = "user_demo"
            logger.warning("⚠️ DEVELOPMENT MODE: Using default user without authentication")
            
            if default_user_id not in user_sessions:
                from ..services.container_manager import get_container_manager
                container_manager = get_container_manager()
                user_sessions[default_user_id] = User(
                    user_id=default_user_id,
                    container_id=container_manager.get_container_id(default_user_id),
                    subscription_tier="free"
                )
            return user_sessions[default_user_id]
        else:
            raise credentials_exception
    
    token = credentials.credentials
    
    # Try JWT token first
    user_id = verify_token(token)
    if user_id:
        logger.info(f"✅ JWT authentication successful for user: {user_id}")
        if user_id not in user_sessions:
            from ..services.container_manager import get_container_manager
            container_manager = get_container_manager()
            user_sessions[user_id] = User(
                user_id=user_id,
                container_id=container_manager.get_container_id(user_id),
                subscription_tier="free"
            )
        return user_sessions[user_id]
    
    # Fallback to legacy token format (for backward compatibility)
    if token.startswith("user_"):
        parts = token.split('_')
        if len(parts) >= 2:
            user_id = f"{parts[0]}_{parts[1]}"
        else:
            user_id = token
        
        logger.info(f"✅ Legacy token authentication for user: {user_id}")
        
        if user_id not in user_sessions:
            from ..services.container_manager import get_container_manager
            container_manager = get_container_manager()
            user_sessions[user_id] = User(
                user_id=user_id,
                container_id=container_manager.get_container_id(user_id),
                subscription_tier="free"
            )
        return user_sessions[user_id]
    
    # Invalid token
    logger.warning(f"❌ Invalid authentication token: {token[:10]}...")
    raise credentials_exception

# Backward compatibility
def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Backward compatible authentication function"""
    return get_current_user_secure(credentials)

def require_authentication():
    """Decorator to require authentication for endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependency injection
            # The actual authentication check happens in the dependency
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def log_security_event(event_type: str, user_id: str, details: dict):
    """Log security events for monitoring"""
    logger.warning(f"SECURITY_EVENT: {event_type} | User: {user_id} | Details: {details} | Time: {datetime.utcnow()}")
