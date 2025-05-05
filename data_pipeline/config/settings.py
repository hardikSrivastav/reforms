"""
Configuration settings for the data pipeline.
"""

import os
from typing import Dict, Any, List, Optional
from functools import lru_cache

# Environment variables
ENV = os.environ.get("ENV", "development")
DEBUG = os.environ.get("DEBUG", "1") == "1"

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# Service URLs
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Model settings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-4")

# Cache settings
ANALYSIS_CACHE_TTL = int(os.environ.get("ANALYSIS_CACHE_TTL", 86400))  # 24 hours in seconds
EMBEDDING_CACHE_TTL = int(os.environ.get("EMBEDDING_CACHE_TTL", 604800))  # 7 days in seconds

# Vector database settings
VECTOR_DIMENSION = int(os.environ.get("VECTOR_DIMENSION", 1536))  # For OpenAI embeddings
VECTOR_SIMILARITY = os.environ.get("VECTOR_SIMILARITY", "cosine")

# Collection prefix for Redis and other stores
PREFIX = os.environ.get("PREFIX", "survey_insights")

# Analysis settings
MAX_CLUSTERS = int(os.environ.get("MAX_CLUSTERS", 5))
MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", 3))
ANOMALY_THRESHOLD = float(os.environ.get("ANOMALY_THRESHOLD", 0.8))
CORRELATION_THRESHOLD = float(os.environ.get("CORRELATION_THRESHOLD", 0.3))
TEMPORAL_DRIFT_THRESHOLD = float(os.environ.get("TEMPORAL_DRIFT_THRESHOLD", 0.2))
API_BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

# Collection names for vector database
COLLECTION_NAMES = {
    "response": "survey_responses",
    "question": "survey_questions",
    "metric": "survey_metrics",
    "demographic": "demographic_segments",
    "survey": "surveys"
}

# Mapping of significance levels to p-values
SIGNIFICANCE_LEVELS = {
    "high": 0.01,
    "medium": 0.05,
    "low": 0.1,
    "none": 1.0
}

@lru_cache()
def get_collection_name(level: str, survey_id: Optional[int] = None) -> str:
    """
    Get the collection name for a given level and survey ID.
    
    Args:
        level: The level (response, question, metric, etc.)
        survey_id: Optional survey ID to include in the collection name
        
    Returns:
        The collection name
    """
    base_name = COLLECTION_NAMES.get(level, f"{level}s")
    
    if survey_id is not None:
        return f"{base_name}_{survey_id}"
    
    return base_name

# Vector Database Settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Redis Settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PREFIX = "survey_insights"

# OpenAI API Settings
BASE_ANALYSIS_MODEL = "gpt-3.5-turbo"
METRIC_ANALYSIS_MODEL = "gpt-4"

# Embedding Model Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")  # OpenAI model
EMBEDDING_DIMENSION = 1536  # Dimension for OpenAI ada-002 embeddings

# Collection Names
SURVEY_RESPONSES_COLLECTION = "survey_responses"
QUESTION_EMBEDDINGS_COLLECTION = "question_embeddings"
METRIC_EMBEDDINGS_COLLECTION = "metric_embeddings"

# Analysis Settings
MAX_TOKENS_PER_API_CALL = 8000
DEFAULT_BATCH_SIZE = 100

# Processing Queue Settings
QUEUE_BACKEND = "redis"

# Caching Settings
CACHE_TTL = {
    "base_analysis": 60 * 15,  # 15 minutes
    "metric_analysis": 60 * 60 * 24,  # 24 hours
    "cross_metric_analysis": 60 * 60 * 24 * 7,  # 7 days
}

# Threshold for significant change in response data
RESPONSE_CHANGE_THRESHOLD = 0.05  # 5% change

# OpenAI API Settings
CROSS_METRIC_ANALYSIS_MODEL = "gpt-4"
COMPLETION_MODEL = "gpt-4"  # Default model for AI completion tasks 