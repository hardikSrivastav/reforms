"""
Qdrant client service for vector database operations.
This service manages connections to the Qdrant vector database,
creates collections, and provides methods for vector operations.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import json
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    Range, Match, MatchValue, ScrollRequest
)

from ..config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(self, url: str = None, api_key: str = None):
        """
        Initialize the Qdrant service.
        
        Args:
            url: URL of the Qdrant server
            api_key: API key for Qdrant authentication
        """
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        logger.info(f"Initialized Qdrant client with URL: {self.url}")
    
    async def initialize_collections(self):
        """
        Initialize all required collections if they don't exist.
        """
        # Create survey responses collection
        await self.create_collection_if_not_exists(
            collection_name=settings.SURVEY_RESPONSES_COLLECTION,
            vector_size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
        
        # Create question embeddings collection
        await self.create_collection_if_not_exists(
            collection_name=settings.QUESTION_EMBEDDINGS_COLLECTION,
            vector_size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
        
        # Create metric embeddings collection
        await self.create_collection_if_not_exists(
            collection_name=settings.METRIC_EMBEDDINGS_COLLECTION,
            vector_size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
        
        logger.info("All collections initialized successfully")
    
    async def create_collection_if_not_exists(
        self, 
        collection_name: str, 
        vector_size: int,
        distance: Distance = Distance.COSINE
    ):
        """
        Create a collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vector embeddings
            distance: Distance metric to use (COSINE, DOT, EUCLID)
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            # Add necessary payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="survey_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            if collection_name == settings.SURVEY_RESPONSES_COLLECTION:
                # Add additional indexes for survey responses
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="question_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="response_date",
                    field_schema=models.PayloadSchemaType.DATETIME
                )
            
            logger.info(f"Collection {collection_name} created successfully with indexes")
        else:
            logger.info(f"Collection {collection_name} already exists")
    
    async def upsert_vectors(
        self, 
        collection_name: str, 
        points: List[PointStruct]
    ):
        """
        Add or update vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} vectors to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors to {collection_name}: {str(e)}")
            return False
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter_condition: Optional[Filter] = None,
        limit: int = 10
    ):
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            filter_condition: Filter condition
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_condition,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {str(e)}")
            return []
    
    async def create_survey_filter(self, survey_id: int) -> Filter:
        """
        Create a filter for a specific survey.
        
        Args:
            survey_id: ID of the survey
            
        Returns:
            Filter condition for the survey
        """
        return Filter(
            must=[
                FieldCondition(
                    key="survey_id",
                    match=Match(
                        value=survey_id
                    )
                )
            ]
        )
    
    async def delete_survey_data(self, survey_id: int):
        """
        Delete all data related to a survey.
        
        Args:
            survey_id: ID of the survey
        """
        filter_condition = await self.create_survey_filter(survey_id)
        
        # Delete from all collections
        for collection in [
            settings.SURVEY_RESPONSES_COLLECTION,
            settings.QUESTION_EMBEDDINGS_COLLECTION,
            settings.METRIC_EMBEDDINGS_COLLECTION
        ]:
            try:
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=filter_condition
                    )
                )
                logger.info(f"Deleted survey {survey_id} data from {collection}")
            except Exception as e:
                logger.error(f"Error deleting survey {survey_id} data from {collection}: {str(e)}")

# Create a singleton instance
qdrant_service = QdrantService() 