"""
Semantic search service using vector embeddings.
Provides methods for searching responses, questions, and metrics by semantic similarity.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from ..config import settings
from ..services.qdrant_client import qdrant_service
from .embedding_service import embedding_service

logger = logging.getLogger(__name__)

class SemanticSearchService:
    """Service for semantic search operations using vector embeddings."""
    
    async def search_similar_responses(self, query: str, survey_id: int, limit: int = 10):
        """
        Search for responses semantically similar to the query.
        
        Args:
            query: Search query text
            survey_id: Survey ID to search within
            limit: Maximum number of results
            
        Returns:
            List of similar responses with similarity scores
        """
        # Generate embedding for the query
        query_embedding = await embedding_service.get_embedding(query)
        
        # Create filter for the survey
        filter_condition = await qdrant_service.create_survey_filter(survey_id)
        
        # Search in responses collection
        results = await qdrant_service.search(
            collection_name=settings.SURVEY_RESPONSES_COLLECTION,
            query_vector=query_embedding,
            filter_condition=filter_condition,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "response_id": result.payload.get("response_id"),
                "question_id": result.payload.get("question_id"),
                "answer": result.payload.get("answer"),
                "similarity_score": result.score,
                "response_date": result.payload.get("response_date")
            })
        
        return formatted_results
    
    async def find_related_responses(self, response_id: str, question_id: str, survey_id: int, limit: int = 10):
        """
        Find responses related to a specific response.
        
        Args:
            response_id: ID of the source response
            question_id: Question ID
            survey_id: Survey ID
            limit: Maximum number of results
            
        Returns:
            List of related responses
        """
        # Search for the point to get its vector
        point_id = f"{response_id}_{question_id}"
        point = await qdrant_service.client.retrieve(
            collection_name=settings.SURVEY_RESPONSES_COLLECTION,
            ids=[point_id]
        )
        
        if not point or not point.vectors:
            return []
        
        # Use the vector to find similar responses
        vector = point.vectors.get(point_id).vector
        
        # Create filter that excludes the original response
        filter_condition = await qdrant_service.create_survey_filter(survey_id)
        
        # Search for similar responses
        results = await qdrant_service.search(
            collection_name=settings.SURVEY_RESPONSES_COLLECTION,
            query_vector=vector,
            filter_condition=filter_condition,
            limit=limit + 1  # Get extra to account for the original
        )
        
        # Format and filter out the original
        formatted_results = []
        for result in results:
            result_id = result.payload.get("response_id")
            result_question = result.payload.get("question_id")
            
            if result_id != response_id or result_question != question_id:
                formatted_results.append({
                    "response_id": result_id,
                    "question_id": result_question,
                    "answer": result.payload.get("answer"),
                    "similarity_score": result.score,
                    "response_date": result.payload.get("response_date")
                })
                
                if len(formatted_results) >= limit:
                    break
        
        return formatted_results
    
    async def search_similar_questions(self, query: str, survey_id: int, limit: int = 10):
        """
        Search for questions semantically similar to the query.
        
        Args:
            query: Search query text
            survey_id: Survey ID to search within
            limit: Maximum number of results
            
        Returns:
            List of similar questions with similarity scores
        """
        # Generate embedding for the query
        query_embedding = await embedding_service.get_embedding(query)
        
        # Create filter for the survey
        filter_condition = await qdrant_service.create_survey_filter(survey_id)
        
        # Search in questions collection
        results = await qdrant_service.search(
            collection_name=settings.QUESTION_EMBEDDINGS_COLLECTION,
            query_vector=query_embedding,
            filter_condition=filter_condition,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "question_id": result.payload.get("question_id"),
                "question_text": result.payload.get("question_text"),
                "question_type": result.payload.get("question_type"),
                "similarity_score": result.score
            })
        
        return formatted_results
    
    async def search_similar_metrics(self, query: str, survey_id: int, limit: int = 10):
        """
        Search for metrics semantically similar to the query.
        
        Args:
            query: Search query text
            survey_id: Survey ID to search within
            limit: Maximum number of results
            
        Returns:
            List of similar metrics with similarity scores
        """
        # Generate embedding for the query
        query_embedding = await embedding_service.get_embedding(query)
        
        # Create filter for the survey
        filter_condition = await qdrant_service.create_survey_filter(survey_id)
        
        # Search in metrics collection
        results = await qdrant_service.search(
            collection_name=settings.METRIC_EMBEDDINGS_COLLECTION,
            query_vector=query_embedding,
            filter_condition=filter_condition,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "metric_id": result.payload.get("metric_id"),
                "metric_name": result.payload.get("metric_name"),
                "metric_description": result.payload.get("metric_description"),
                "metric_type": result.payload.get("metric_type"),
                "similarity_score": result.score
            })
        
        return formatted_results

# Create a singleton instance
semantic_search_service = SemanticSearchService() 