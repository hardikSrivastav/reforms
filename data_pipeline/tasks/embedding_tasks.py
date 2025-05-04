"""
Embedding tasks for the data pipeline.
This module contains Celery tasks for generating and managing embeddings.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

from ..tasks.celery_app import app
from ..embeddings.semantic_search import semantic_search_service

logger = logging.getLogger(__name__)


@app.task(bind=True, name="generate_response_embeddings", max_retries=3)
def generate_response_embeddings(self, survey_id: int, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Celery task to generate embeddings for survey responses.
    
    Args:
        survey_id: The survey ID
        responses: List of survey responses
        
    Returns:
        Dictionary with embedding results
    """
    logger.info(f"Generating response embeddings for survey {survey_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            semantic_search_service.process_survey_responses(survey_id, responses)
        )
        logger.info(f"Completed response embedding generation for survey {survey_id}")
        return {
            "survey_id": survey_id,
            "status": "completed",
            "embeddings_count": len(result) if isinstance(result, list) else 0
        }
    except Exception as e:
        logger.error(f"Error generating response embeddings for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="generate_metric_embeddings", max_retries=3)
def generate_metric_embeddings(self, survey_id: int, metric_id: str, 
                              metric_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to generate embeddings for a specific metric.
    
    Args:
        survey_id: The survey ID
        metric_id: The metric ID
        metric_data: The metric definition data
        
    Returns:
        Dictionary with embedding results
    """
    logger.info(f"Generating metric embeddings for survey {survey_id}, metric {metric_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            semantic_search_service.process_metric(survey_id, metric_id, metric_data)
        )
        logger.info(f"Completed metric embedding generation for survey {survey_id}, metric {metric_id}")
        return {
            "survey_id": survey_id,
            "metric_id": metric_id,
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Error generating metric embeddings for survey {survey_id}, metric {metric_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="update_vector_index", max_retries=3)
def update_vector_index(self, survey_id: int, collection_name: str = None) -> Dict[str, Any]:
    """
    Celery task to update the vector index for a survey.
    
    Args:
        survey_id: The survey ID
        collection_name: Optional collection name to update
        
    Returns:
        Dictionary with update results
    """
    logger.info(f"Updating vector index for survey {survey_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            semantic_search_service.update_vector_index(survey_id, collection_name)
        )
        logger.info(f"Completed vector index update for survey {survey_id}")
        return {
            "survey_id": survey_id,
            "collection": collection_name or "all",
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Error updating vector index for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="process_survey_embeddings")
def process_survey_embeddings(self, survey_id: int, survey_data: Dict[str, Any]) -> str:
    """
    Celery task to process all embeddings for a survey.
    This task chains other embedding tasks together.
    
    Args:
        survey_id: The survey ID
        survey_data: Survey metadata including metrics
        
    Returns:
        Task ID for tracking
    """
    from celery import chain, group
    
    logger.info(f"Setting up embedding processing for survey {survey_id}")
    
    # Create metric embedding tasks
    metric_tasks = []
    for metric_id, metric_data in survey_data.get("metrics", {}).items():
        task = generate_metric_embeddings.s(survey_id, metric_id, metric_data)
        metric_tasks.append(task)
    
    # Create vector index update task
    update_task = update_vector_index.s(survey_id)
    
    # Chain the tasks together
    # First run metric embedding generation in parallel, then update the vector index
    embedding_chain = chain(
        group(metric_tasks),
        update_task
    )
    
    # Execute the chain
    result = embedding_chain.apply_async()
    
    logger.info(f"Survey embedding processing started for survey {survey_id}")
    return f"Embedding processing started with task ID: {result.id}" 