"""
Metadata store service for caching and retrieving analysis results.
This service provides methods to store, retrieve, and manage analysis results
for surveys, metrics, and insights.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
import redis
import redis.asyncio
import pickle
from datetime import datetime, timedelta

from ..config import settings

logger = logging.getLogger(__name__)

class MetadataStore:
    """Service for storing and retrieving analysis metadata."""
    
    def __init__(self, redis_url: str = None):
        """
        Initialize the metadata store service.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis = redis.from_url(self.redis_url)
        logger.info(f"Initialized metadata store with Redis: {self.redis_url}")
        
        # Create a connection pool for async operations
        self._pool = None
        
    async def _get_redis_pool(self):
        """Get or create the async Redis connection pool."""
        if self._pool is None:
            self._pool = await redis.asyncio.from_url(self.redis_url)
        return self._pool
    
    def _get_key(self, key_type: str, survey_id: int, entity_id: Optional[str] = None) -> str:
        """
        Generate a Redis key for storing data.
        
        Args:
            key_type: Type of analysis (base, metric, cross_metric, etc.)
            survey_id: Survey ID
            entity_id: Optional entity ID (metric ID, question ID, etc.)
            
        Returns:
            Redis key string
        """
        if entity_id:
            return f"{key_type}:{survey_id}:{entity_id}"
        return f"{key_type}:{survey_id}"
    
    async def store_analysis_result(
        self,
        key_type: str,
        survey_id: int,
        result: Any,
        entity_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store an analysis result in the metadata store.
        
        Args:
            key_type: Type of analysis (base, metric, cross_metric, etc.)
            survey_id: Survey ID
            result: Analysis result to store
            entity_id: Optional entity ID (metric ID, question ID, etc.)
            ttl: Time-to-live in seconds
            
        Returns:
            Success status
        """
        try:
            key = self._get_key(key_type, survey_id, entity_id)
            
            # Add timestamp to the result
            if isinstance(result, dict):
                result["timestamp"] = datetime.now().isoformat()
            
            # Serialize the result
            serialized = pickle.dumps(result)
            
            # Determine TTL
            if ttl is None:
                ttl = settings.CACHE_TTL.get(key_type, 3600)  # Default 1 hour
            
            # Store in Redis asynchronously
            redis = await self._get_redis_pool()
            await redis.set(key, serialized, ex=ttl)
            
            logger.info(f"Stored analysis result for {key} with TTL {ttl}s")
            return True
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            return False
    
    async def get_analysis_result(
        self,
        key_type: str,
        survey_id: int,
        entity_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve an analysis result from the metadata store.
        
        Args:
            key_type: Type of analysis (base, metric, cross_metric, etc.)
            survey_id: Survey ID
            entity_id: Optional entity ID (metric ID, question ID, etc.)
            
        Returns:
            Analysis result if found, None otherwise
        """
        try:
            key = self._get_key(key_type, survey_id, entity_id)
            
            # Get data from Redis asynchronously
            redis = await self._get_redis_pool()
            data = await redis.get(key)
            
            if not data:
                logger.info(f"No analysis result found for {key}")
                return None
            
            # Deserialize the result
            result = pickle.loads(data)
            logger.info(f"Retrieved analysis result for {key}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving analysis result: {str(e)}")
            return None
    
    async def delete_analysis_result(
        self,
        key_type: str,
        survey_id: int,
        entity_id: Optional[str] = None
    ) -> bool:
        """
        Delete an analysis result from the metadata store.
        
        Args:
            key_type: Type of analysis (base, metric, cross_metric, etc.)
            survey_id: Survey ID
            entity_id: Optional entity ID (metric ID, question ID, etc.)
            
        Returns:
            Success status
        """
        try:
            key = self._get_key(key_type, survey_id, entity_id)
            
            # Delete key asynchronously
            redis = await self._get_redis_pool()
            await redis.delete(key)
            
            logger.info(f"Deleted analysis result for {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting analysis result: {str(e)}")
            return False
    
    async def delete_survey_data(self, survey_id: int) -> bool:
        """
        Delete all analysis results for a survey.
        
        Args:
            survey_id: Survey ID
            
        Returns:
            Success status
        """
        try:
            # Get all keys for the survey
            pattern = f"*:{survey_id}:*"
            
            # Use async Redis
            redis = await self._get_redis_pool()
            keys = await redis.keys(pattern)
            
            if keys:
                await redis.delete(*keys)
                logger.info(f"Deleted {len(keys)} analysis results for survey {survey_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting survey data: {str(e)}")
            return False
    
    async def is_cache_valid(
        self,
        key_type: str,
        survey_id: int,
        entity_id: Optional[str] = None,
        max_age: Optional[int] = None
    ) -> bool:
        """
        Check if a cached result is still valid.
        
        Args:
            key_type: Type of analysis (base, metric, cross_metric, etc.)
            survey_id: Survey ID
            entity_id: Optional entity ID (metric ID, question ID, etc.)
            max_age: Maximum age in seconds
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            result = await self.get_analysis_result(key_type, survey_id, entity_id)
            
            if not result:
                return False
            
            # Check if result has timestamp
            if not isinstance(result, dict) or "timestamp" not in result:
                return False
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(result["timestamp"])
            
            # Use default TTL if max_age not provided
            if max_age is None:
                max_age = settings.CACHE_TTL.get(key_type, 3600)  # Default 1 hour
            
            # Check if cache is still valid
            age = (datetime.now() - timestamp).total_seconds()
            return age < max_age
        except Exception as e:
            logger.error(f"Error checking cache validity: {str(e)}")
            return False

# Create a singleton instance
metadata_store = MetadataStore() 