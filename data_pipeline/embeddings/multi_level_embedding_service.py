"""
Multi-level embedding service for generating and managing embeddings at different levels of granularity.
This service provides methods to create hierarchical embeddings for survey data,
including question-level, metric-level, and demographic aggregations.
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime

from ..config import settings
from ..services.qdrant_client import qdrant_service, PointStruct
from ..services.metadata_store import metadata_store
from .embedding_service import embedding_service

logger = logging.getLogger(__name__)

class MultiLevelEmbeddingService:
    """
    Service for generating and managing embeddings at multiple levels of abstraction.
    Implements the hierarchical embedding approach described in the survey insights architecture.
    """
    
    def __init__(self):
        """Initialize the multi-level embedding service."""
        self.levels = ["response", "question", "metric", "demographic", "survey"]
        self.cache_ttl = settings.EMBEDDING_CACHE_TTL
        logger.info(f"Initialized multi-level embedding service")
        
        # Create collection names for different levels
        self.collections = {
            "response": settings.SURVEY_RESPONSES_COLLECTION,
            "question": settings.QUESTION_EMBEDDINGS_COLLECTION,
            "metric": settings.METRIC_EMBEDDINGS_COLLECTION,
            "demographic": f"{settings.PREFIX}_demographic_embeddings",
            "survey": f"{settings.PREFIX}_survey_embeddings"
        }
    
    async def generate_aggregate_embeddings(self, survey_id: int) -> Dict[str, Any]:
        """
        Generate aggregate embeddings for a survey at multiple levels.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with results of the embedding generation process
        """
        logger.info(f"Generating aggregate embeddings for survey {survey_id}")
        
        # Check if we already have recent aggregate embeddings
        cache_key = f"aggregate_embeddings:{survey_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached aggregate embeddings for survey {survey_id}")
            return cached_result
        
        # Count points at each level to verify base data exists
        points_by_level = {}
        for level, collection in self.collections.items():
            if level == "response":
                # For responses, we need to check if we have embeddings already
                filter_condition = {
                    "must": [
                        {"key": "survey_id", "match": {"value": survey_id}}
                    ]
                }
                count = await qdrant_service.count_points(collection, filter_condition)
                points_by_level[level] = count
        
        # If we don't have enough response embeddings, return an error
        if points_by_level.get("response", 0) < 10:
            logger.warning(f"Not enough response embeddings for survey {survey_id} to generate aggregates")
            return {
                "survey_id": survey_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": "Not enough response embeddings to generate aggregates"
            }
        
        # Create aggregate embeddings for each level
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "levels": {}
        }
        
        # Process each level in parallel
        tasks = []
        for level in ["question", "metric", "demographic", "survey"]:
            if level != "response":  # Skip response level as we already have those
                tasks.append(self._generate_level_embeddings(survey_id, level))
        
        # Wait for all tasks to complete
        level_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, level in enumerate(["question", "metric", "demographic", "survey"]):
            if isinstance(level_results[i], Exception):
                logger.error(f"Error generating {level} embeddings: {str(level_results[i])}")
                result["levels"][level] = {
                    "status": "error",
                    "error": str(level_results[i])
                }
            else:
                result["levels"][level] = level_results[i]
        
        # Store in cache
        await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
        
        return result
    
    async def _generate_level_embeddings(self, survey_id: int, level: str) -> Dict[str, Any]:
        """
        Generate embeddings for a specific level.
        
        Args:
            survey_id: The ID of the survey
            level: The level to generate embeddings for
            
        Returns:
            Dictionary with results for this level
        """
        logger.info(f"Generating {level} embeddings for survey {survey_id}")
        
        if level == "question":
            return await self._generate_question_level_embeddings(survey_id)
        elif level == "metric":
            return await self._generate_metric_level_embeddings(survey_id)
        elif level == "demographic":
            return await self._generate_demographic_level_embeddings(survey_id)
        elif level == "survey":
            return await self._generate_survey_level_embedding(survey_id)
        else:
            raise ValueError(f"Unknown level: {level}")
    
    async def _generate_question_level_embeddings(self, survey_id: int) -> Dict[str, Any]:
        """
        Generate aggregated embeddings for each question based on all responses.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with results for question-level embeddings
        """
        # Get all questions for this survey
        filter_condition = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}}
            ]
        }
        
        questions = await qdrant_service.scroll(
            collection_name=self.collections["question"],
            filter_condition=filter_condition,
            limit=100  # Assuming a survey doesn't have more than 100 questions
        )
        
        question_ids = [q.payload.get("question_id") for q in questions.points if q.payload.get("question_id")]
        
        if not question_ids:
            return {
                "status": "error",
                "error": "No questions found for this survey"
            }
        
        # Process each question
        question_results = []
        for question_id in question_ids:
            # Get all responses for this question
            response_filter = {
                "must": [
                    {"key": "survey_id", "match": {"value": survey_id}},
                    {"key": "question_id", "match": {"value": question_id}}
                ]
            }
            
            responses = await qdrant_service.scroll(
                collection_name=self.collections["response"],
                filter_condition=response_filter,
                limit=1000  # Get up to 1000 responses per question
            )
            
            if not responses.points:
                continue
                
            # Extract vectors and calculate mean embedding
            vectors = [p.vector for p in responses.points if p.vector]
            if not vectors:
                continue
                
            # Calculate mean embedding
            mean_vector = np.mean(vectors, axis=0).tolist()
            
            # Create payload with metadata
            payload = {
                "survey_id": survey_id,
                "question_id": question_id,
                "embedding_type": "question_aggregate",
                "response_count": len(vectors),
                "timestamp": datetime.now().isoformat()
            }
            
            # Get question text if available
            if responses.points and "question_text" in responses.points[0].payload:
                payload["question_text"] = responses.points[0].payload["question_text"]
            
            # Create point for vector database
            point = PointStruct(
                id=f"agg_{survey_id}_{question_id}",
                vector=mean_vector,
                payload=payload
            )
            
            # Upsert to vector database
            result = await qdrant_service.upsert_vectors(
                collection_name=self.collections["question"],
                points=[point]
            )
            
            question_results.append({
                "question_id": question_id,
                "response_count": len(vectors),
                "status": "success"
            })
        
        return {
            "status": "success",
            "processed_questions": len(question_results),
            "question_results": question_results
        }
    
    async def _generate_metric_level_embeddings(self, survey_id: int) -> Dict[str, Any]:
        """
        Generate aggregated embeddings for each metric based on relevant questions.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with results for metric-level embeddings
        """
        # Get all metrics for this survey
        filter_condition = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}}
            ]
        }
        
        metrics = await qdrant_service.scroll(
            collection_name=self.collections["metric"],
            filter_condition=filter_condition,
            limit=100  # Assuming a survey doesn't have more than 100 metrics
        )
        
        metric_ids = [m.payload.get("metric_id") for m in metrics.points if m.payload.get("metric_id")]
        
        if not metric_ids:
            return {
                "status": "error",
                "error": "No metrics found for this survey"
            }
        
        # Process each metric
        metric_results = []
        for metric_id in metric_ids:
            # Get the corresponding question-level aggregates
            # In a real implementation, we would have a mapping from metrics to questions
            # Here we're assuming metrics and questions have a 1:1 relationship for simplicity
            
            # Get question aggregate for this metric
            question_filter = {
                "must": [
                    {"key": "survey_id", "match": {"value": survey_id}},
                    {"key": "question_id", "match": {"value": metric_id}},
                    {"key": "embedding_type", "match": {"value": "question_aggregate"}}
                ]
            }
            
            question_aggs = await qdrant_service.scroll(
                collection_name=self.collections["question"],
                filter_condition=question_filter,
                limit=10  # There should be at most 1, but allow for some flexibility
            )
            
            if not question_aggs.points:
                # Try to get individual responses if we don't have question aggregates
                response_filter = {
                    "must": [
                        {"key": "survey_id", "match": {"value": survey_id}},
                        {"key": "question_id", "match": {"value": metric_id}}
                    ]
                }
                
                responses = await qdrant_service.scroll(
                    collection_name=self.collections["response"],
                    filter_condition=response_filter,
                    limit=1000
                )
                
                if not responses.points:
                    continue
                    
                # Calculate mean from individual responses
                vectors = [p.vector for p in responses.points if p.vector]
                if not vectors:
                    continue
                    
                vector = np.mean(vectors, axis=0).tolist()
                response_count = len(vectors)
            else:
                # Use the question aggregate
                vector = question_aggs.points[0].vector
                response_count = question_aggs.points[0].payload.get("response_count", 0)
            
            # Get metric metadata
            metric_meta = next((m.payload for m in metrics.points if m.payload.get("metric_id") == metric_id), {})
            
            # Create payload with metadata
            payload = {
                "survey_id": survey_id,
                "metric_id": metric_id,
                "embedding_type": "metric_aggregate",
                "response_count": response_count,
                "metric_name": metric_meta.get("metric_name", ""),
                "metric_type": metric_meta.get("metric_type", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # Create point for vector database
            point = PointStruct(
                id=f"agg_{survey_id}_{metric_id}",
                vector=vector,
                payload=payload
            )
            
            # Upsert to vector database
            result = await qdrant_service.upsert_vectors(
                collection_name=self.collections["metric"],
                points=[point]
            )
            
            metric_results.append({
                "metric_id": metric_id,
                "response_count": response_count,
                "status": "success"
            })
        
        return {
            "status": "success",
            "processed_metrics": len(metric_results),
            "metric_results": metric_results
        }
    
    async def _generate_demographic_level_embeddings(self, survey_id: int) -> Dict[str, Any]:
        """
        Generate aggregated embeddings for each demographic segment.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with results for demographic-level embeddings
        """
        # In a real implementation, we would get demographic info from the responses
        # For this example, we'll simulate a few demographic segments
        
        # First, ensure the demographic collection exists
        await self._ensure_collection_exists(self.collections["demographic"])
        
        # Get all responses for this survey
        filter_condition = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}}
            ]
        }
        
        responses = await qdrant_service.scroll(
            collection_name=self.collections["response"],
            filter_condition=filter_condition,
            limit=1000
        )
        
        if not responses.points:
            return {
                "status": "error",
                "error": "No responses found for this survey"
            }
        
        # For demonstration, create segments based on response timestamps
        # In a real implementation, we would use actual demographic info
        
        # Group responses by timestamp (month)
        time_segments = {}
        for point in responses.points:
            if not point.payload.get("response_date"):
                continue
                
            try:
                date = datetime.fromisoformat(point.payload["response_date"].replace("Z", "+00:00"))
                month = date.strftime("%Y-%m")
                
                if month not in time_segments:
                    time_segments[month] = []
                    
                time_segments[month].append(point)
            except (ValueError, TypeError):
                # Skip if we can't parse the date
                continue
        
        # Process each segment
        segment_results = []
        for segment, points in time_segments.items():
            if len(points) < 5:  # Skip if not enough data
                continue
                
            # Calculate mean embedding
            vectors = [p.vector for p in points if p.vector]
            if not vectors:
                continue
                
            mean_vector = np.mean(vectors, axis=0).tolist()
            
            # Create payload with metadata
            payload = {
                "survey_id": survey_id,
                "segment_id": f"time_{segment}",
                "segment_type": "time_period",
                "segment_value": segment,
                "embedding_type": "demographic_aggregate",
                "response_count": len(vectors),
                "timestamp": datetime.now().isoformat()
            }
            
            # Create point for vector database
            point = PointStruct(
                id=f"agg_{survey_id}_time_{segment}",
                vector=mean_vector,
                payload=payload
            )
            
            # Upsert to vector database
            result = await qdrant_service.upsert_vectors(
                collection_name=self.collections["demographic"],
                points=[point]
            )
            
            segment_results.append({
                "segment_id": f"time_{segment}",
                "segment_type": "time_period",
                "segment_value": segment,
                "response_count": len(vectors),
                "status": "success"
            })
        
        return {
            "status": "success",
            "processed_segments": len(segment_results),
            "segment_results": segment_results
        }
    
    async def _generate_survey_level_embedding(self, survey_id: int) -> Dict[str, Any]:
        """
        Generate a single embedding that represents the entire survey.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with results for survey-level embedding
        """
        # Ensure the survey collection exists
        await self._ensure_collection_exists(self.collections["survey"])
        
        # Get all question-level aggregates
        filter_condition = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}},
                {"key": "embedding_type", "match": {"value": "question_aggregate"}}
            ]
        }
        
        question_aggs = await qdrant_service.scroll(
            collection_name=self.collections["question"],
            filter_condition=filter_condition,
            limit=100
        )
        
        # If we don't have question aggregates, use response embeddings directly
        if not question_aggs.points:
            response_filter = {
                "must": [
                    {"key": "survey_id", "match": {"value": survey_id}}
                ]
            }
            
            responses = await qdrant_service.scroll(
                collection_name=self.collections["response"],
                filter_condition=response_filter,
                limit=1000
            )
            
            if not responses.points:
                return {
                    "status": "error",
                    "error": "No data found for this survey"
                }
                
            # Calculate mean embedding from responses
            vectors = [p.vector for p in responses.points if p.vector]
            if not vectors:
                return {
                    "status": "error",
                    "error": "No vectors found for this survey"
                }
                
            mean_vector = np.mean(vectors, axis=0).tolist()
            response_count = len(vectors)
        else:
            # Calculate mean embedding from question aggregates
            vectors = [p.vector for p in question_aggs.points if p.vector]
            if not vectors:
                return {
                    "status": "error",
                    "error": "No vectors found in question aggregates"
                }
                
            mean_vector = np.mean(vectors, axis=0).tolist()
            
            # Sum response counts from all questions
            response_count = sum(p.payload.get("response_count", 0) for p in question_aggs.points)
        
        # Create payload with metadata
        payload = {
            "survey_id": survey_id,
            "embedding_type": "survey_aggregate",
            "response_count": response_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create point for vector database
        point = PointStruct(
            id=f"agg_{survey_id}_survey",
            vector=mean_vector,
            payload=payload
        )
        
        # Upsert to vector database
        result = await qdrant_service.upsert_vectors(
            collection_name=self.collections["survey"],
            points=[point]
        )
        
        return {
            "status": "success",
            "survey_id": survey_id,
            "response_count": response_count
        }
    
    async def _ensure_collection_exists(self, collection_name: str) -> bool:
        """
        Ensure that a collection exists in the vector database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if the collection exists or was created, False otherwise
        """
        try:
            # Check if collection exists
            collections = await qdrant_service.client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            
            # Create if it doesn't exist
            if not exists:
                await qdrant_service.create_collection(
                    collection_name=collection_name,
                    vector_size=settings.EMBEDDING_DIMENSION
                )
                logger.info(f"Created collection {collection_name}")
                
            return True
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name} exists: {str(e)}")
            return False
    
    async def find_similar_segments(
        self, 
        survey_id: int, 
        segment_id: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Find demographic segments with similar response patterns.
        
        Args:
            survey_id: The ID of the survey
            segment_id: The ID of the segment to find similar segments for
            limit: Maximum number of similar segments to return
            
        Returns:
            Dictionary with similar segments and their similarity scores
        """
        # Get the segment embedding
        segment_filter = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}},
                {"key": "segment_id", "match": {"value": segment_id}}
            ]
        }
        
        segment = await qdrant_service.scroll(
            collection_name=self.collections["demographic"],
            filter_condition=segment_filter,
            limit=1
        )
        
        if not segment.points:
            return {
                "status": "error",
                "error": f"Segment {segment_id} not found for survey {survey_id}"
            }
        
        # Search for similar segments
        vector = segment.points[0].vector
        
        # Create a filter that excludes the original segment
        search_filter = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}},
                {"key": "embedding_type", "match": {"value": "demographic_aggregate"}}
            ],
            "must_not": [
                {"key": "segment_id", "match": {"value": segment_id}}
            ]
        }
        
        similar_segments = await qdrant_service.search(
            collection_name=self.collections["demographic"],
            query_vector=vector,
            filter_condition=search_filter,
            limit=limit
        )
        
        # Format the results
        results = []
        for result in similar_segments:
            results.append({
                "segment_id": result.payload.get("segment_id"),
                "segment_type": result.payload.get("segment_type"),
                "segment_value": result.payload.get("segment_value"),
                "response_count": result.payload.get("response_count"),
                "similarity_score": result.score
            })
        
        return {
            "status": "success",
            "original_segment": {
                "segment_id": segment_id,
                "segment_type": segment.points[0].payload.get("segment_type"),
                "segment_value": segment.points[0].payload.get("segment_value"),
                "response_count": segment.points[0].payload.get("response_count")
            },
            "similar_segments": results
        }
    
    async def compare_survey_embeddings(
        self, 
        survey_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Compare survey-level embeddings to identify similarities and differences.
        
        Args:
            survey_ids: List of survey IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        if len(survey_ids) < 2:
            return {
                "status": "error",
                "error": "Need at least two surveys to compare"
            }
        
        # Get survey embeddings
        surveys = []
        for survey_id in survey_ids:
            survey_filter = {
                "must": [
                    {"key": "survey_id", "match": {"value": survey_id}},
                    {"key": "embedding_type", "match": {"value": "survey_aggregate"}}
                ]
            }
            
            survey = await qdrant_service.scroll(
                collection_name=self.collections["survey"],
                filter_condition=survey_filter,
                limit=1
            )
            
            if not survey.points:
                # Generate survey embedding if it doesn't exist
                logger.info(f"Survey embedding for {survey_id} not found, generating it")
                result = await self._generate_survey_level_embedding(survey_id)
                
                if result["status"] != "success":
                    return {
                        "status": "error",
                        "error": f"Failed to generate survey embedding for {survey_id}: {result.get('error', 'Unknown error')}"
                    }
                
                # Retry fetching
                survey = await qdrant_service.scroll(
                    collection_name=self.collections["survey"],
                    filter_condition=survey_filter,
                    limit=1
                )
                
                if not survey.points:
                    return {
                        "status": "error",
                        "error": f"Survey embedding for {survey_id} not found after generation attempt"
                    }
            
            surveys.append({
                "survey_id": survey_id,
                "vector": survey.points[0].vector,
                "response_count": survey.points[0].payload.get("response_count", 0)
            })
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(surveys)):
            for j in range(i+1, len(surveys)):
                # Calculate cosine similarity
                vec1 = np.array(surveys[i]["vector"])
                vec2 = np.array(surveys[j]["vector"])
                
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                similarities.append({
                    "survey1_id": surveys[i]["survey_id"],
                    "survey2_id": surveys[j]["survey_id"],
                    "similarity_score": float(similarity),
                    "response_counts": [surveys[i]["response_count"], surveys[j]["response_count"]]
                })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "status": "success",
            "survey_count": len(surveys),
            "similarity_pairs": similarities
        }
    
    async def domain_adapt_embeddings(
        self,
        survey_id: int,
        domain_adaptation_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Simulate domain-specific adaptation of embeddings using enhanced prompts.
        In a full implementation, this would fine-tune the embedding model.
        
        Args:
            survey_id: The ID of the survey to adapt embeddings for
            domain_adaptation_prompts: Optional list of domain-specific prompts
            
        Returns:
            Dictionary with adaptation results
        """
        # In a production system, this would fine-tune or customize the embedding model
        # Here we'll simulate it by enhancing the context of embedding prompts
        
        logger.info(f"Simulating domain adaptation for survey {survey_id}")
        
        # Get some sample responses to understand the domain
        filter_condition = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}}
            ]
        }
        
        responses = await qdrant_service.scroll(
            collection_name=self.collections["response"],
            filter_condition=filter_condition,
            limit=50  # Sample 50 responses
        )
        
        if not responses.points:
            return {
                "status": "error",
                "error": "No responses found for domain adaptation"
            }
        
        # Extract text from responses
        sample_texts = [p.payload.get("answer", "") for p in responses.points if p.payload.get("answer")]
        
        # Get some question texts
        question_filter = {
            "must": [
                {"key": "survey_id", "match": {"value": survey_id}}
            ]
        }
        
        questions = await qdrant_service.scroll(
            collection_name=self.collections["question"],
            filter_condition=question_filter,
            limit=20
        )
        
        question_texts = [q.payload.get("question_text", "") for q in questions.points if q.payload.get("question_text")]
        
        # Combine with any provided domain prompts
        domain_context = []
        if domain_adaptation_prompts:
            domain_context.extend(domain_adaptation_prompts)
        
        domain_context.extend(question_texts[:5])  # Add some question texts
        domain_context.extend(sample_texts[:10])   # Add some response texts
        
        # Build simulated domain knowledge string
        domain_knowledge = "\n".join(domain_context)
        
        # In a real implementation, we would:
        # 1. Fine-tune the embedding model on this domain data
        # 2. Or use a domain-specific prompt template when generating embeddings
        
        # For simulation, store this as a cached prompt for future embedding requests
        domain_prompt_key = f"domain_prompt:{survey_id}"
        domain_prompt_data = {
            "survey_id": survey_id,
            "domain_knowledge": domain_knowledge,
            "timestamp": datetime.now().isoformat()
        }
        
        await metadata_store.store_analysis_result(domain_prompt_key, domain_prompt_data, self.cache_ttl)
        
        return {
            "status": "success",
            "survey_id": survey_id,
            "adapted_context_length": len(domain_knowledge),
            "context_samples": min(15, len(domain_context)),
            "simulation_note": "Domain adaptation simulated by enhancing context. In production, would fine-tune the embedding model."
        }


# Create a singleton instance
multi_level_embedding_service = MultiLevelEmbeddingService() 