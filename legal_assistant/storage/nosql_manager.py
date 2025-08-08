#nosql_manager.py
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from typing import Optional
import os


from .nosql_models import (
    UserDocument,
    UploadedFileDocument,
    ProcessingStatusDocument,
    ConversationDocument,
    ImmigrationCaseDocument
)


logger = logging.getLogger(__name__)

class NoSQLManager:
    """
    Manages NoSQL database connections with fallback to in-memory storage.
    Supports both MongoDB and Redis for different use cases.
    """
    
    def __init__(self):
        self.mongodb_client = None
        self.mongodb_available = False
        self.redis_client = None
        self.redis_available = False
        self.database = None
        
        # Fallback to in-memory storage if NoSQL unavailable
        self._fallback_storage = {
            'users': {},
            'uploaded_files': {},
            'processing_status': {},
            'conversations': {},
            'immigration_cases': {}
        }
    
    async def initialize_mongodb(self):
        """Initialize MongoDB connection with security enhancements"""
        try:
            # Get MongoDB connection string from environment
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            database_name = os.getenv("MONGODB_DATABASE", "legal_assistant")
            
            # Security enhancements for production
            if "mongodb://localhost" not in mongodb_url:
                # Production: Add SSL and authentication parameters
                if "?" not in mongodb_url:
                    mongodb_url += "?ssl=true&ssl_cert_reqs=CERT_NONE"
                else:
                    mongodb_url += "&ssl=true&ssl_cert_reqs=CERT_NONE"
                
                # Add connection pooling for better performance
                mongodb_url += "&maxPoolSize=10&minPoolSize=1"
                
                logger.info("🔒 Production MongoDB: SSL/TLS enabled")
            
            # Create async MongoDB client with security options
            self.mongodb_client = AsyncIOMotorClient(
                mongodb_url,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,         # 10 second connection timeout
                socketTimeoutMS=20000,          # 20 second socket timeout
            )
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            
            # Get database
            self.database = self.mongodb_client[database_name]
            
            # Initialize Beanie with document models
            await init_beanie(
                database=self.database,
                document_models=[
                    UserDocument,
                    UploadedFileDocument, 
                    ProcessingStatusDocument,
                    ConversationDocument,
                    ImmigrationCaseDocument
                ]
            )
            
            self.mongodb_available = True
            logger.info(f"✅ MongoDB connected successfully to {database_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB not available: {e}")
            logger.info("Falling back to in-memory storage for development")
            self.mongodb_available = False
    
    async def initialize_redis(self):
        """Initialize Redis connection with security enhancements"""
        try:
            import redis.asyncio as redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            
            # Security enhancements for production
            if "redis://localhost" not in redis_url:
                # Production: Add SSL support
                if "redis://" in redis_url:
                    redis_url = redis_url.replace("redis://", "rediss://")
                
                logger.info("🔒 Production Redis: SSL/TLS enabled")
            
            # Create Redis client with security options
            self.redis_client = redis.from_url(
                redis_url, 
                decode_responses=True,
                socket_timeout=5,      # 5 second socket timeout
                socket_connect_timeout=5,  # 5 second connection timeout
                retry_on_timeout=True,
                health_check_interval=30,  # Health check every 30 seconds
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Set up security configurations
            if "redis://localhost" not in redis_url:
                # Production: Configure key expiration and access controls
                await self.redis_client.config_set("maxmemory-policy", "allkeys-lru")
                await self.redis_client.config_set("timeout", "300")  # 5 minute timeout
            
            self.redis_available = True
            logger.info("✅ Redis connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.redis_available = False
    
    async def initialize(self):
        """Initialize all NoSQL connections"""
        await self.initialize_mongodb()
        await self.initialize_redis()
        
        return {
            'mongodb_available': self.mongodb_available,
            'redis_available': self.redis_available,
            'fallback_mode': not self.mongodb_available
        }
    
    async def close_connections(self):
        """Close all database connections"""
        if self.mongodb_client:
            self.mongodb_client.close()
        if self.redis_client:
            await self.redis_client.close()

# Global NoSQL manager
_nosql_manager = None

async def get_nosql_manager() -> NoSQLManager:
    """Get or create NoSQL manager instance"""
    global _nosql_manager
    if _nosql_manager is None:
        _nosql_manager = NoSQLManager()
        await _nosql_manager.initialize()
    return _nosql_manager
