"""
Analysis tasks for the data pipeline.
This module contains Celery tasks for running analysis operations in the background.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from ..tasks.celery_app import app
from ..analysis.analysis_coordinator import AnalysisCoordinator

logger = logging.getLogger(__name__)
coordinator = AnalysisCoordinator()


@app.task(bind=True, name="run_base_analysis", max_retries=3, priority=9)
def run_base_analysis(self, survey_id: int, survey_data: Dict[str, Any], 
                      responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Celery task to run base analysis for a survey.
    
    Args:
        survey_id: The survey ID
        survey_data: Survey metadata and configuration
        responses: List of survey responses
        
    Returns:
        Dictionary with base analysis results
    """
    logger.info(f"Running base analysis task for survey {survey_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            coordinator._run_base_analysis(survey_id, survey_data, responses)
        )
        logger.info(f"Completed base analysis task for survey {survey_id}")
        return result
    except Exception as e:
        logger.error(f"Error in base analysis task for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="run_metric_analysis", max_retries=3, priority=5)
def run_metric_analysis(self, survey_id: int, metric_id: str, metric_data: Dict[str, Any],
                        responses: List[Dict[str, Any]], 
                        time_series_responses: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """
    Celery task to run metric analysis for a specific metric.
    
    Args:
        survey_id: The survey ID
        metric_id: The metric ID
        metric_data: The metric definition data
        responses: List of survey responses
        time_series_responses: Optional dictionary mapping time periods to response lists
        
    Returns:
        Dictionary with metric analysis results
    """
    logger.info(f"Running metric analysis task for survey {survey_id}, metric {metric_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            coordinator._run_metric_analysis(
                survey_id, metric_id, metric_data, responses, time_series_responses
            )
        )
        logger.info(f"Completed metric analysis task for survey {survey_id}, metric {metric_id}")
        return result
    except Exception as e:
        logger.error(f"Error in metric analysis task for survey {survey_id}, metric {metric_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="run_vector_enhanced_analysis", max_retries=3, priority=3)
def run_vector_enhanced_analysis(self, survey_id: int, metric_id: str,
                                 responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Celery task to run vector-enhanced analysis.
    
    Args:
        survey_id: Survey ID
        metric_id: Metric ID
        responses: Survey responses
        
    Returns:
        Dictionary with vector-enhanced analysis results
    """
    logger.info(f"Running vector-enhanced analysis task for survey {survey_id}, metric {metric_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            coordinator._run_vector_enhanced_analysis(survey_id, metric_id, responses)
        )
        logger.info(f"Completed vector-enhanced analysis task for survey {survey_id}, metric {metric_id}")
        return result
    except Exception as e:
        logger.error(f"Error in vector-enhanced analysis task for survey {survey_id}, metric {metric_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="run_cross_metric_analysis", max_retries=3, priority=2)
def run_cross_metric_analysis(self, survey_id: int, metrics_data: Dict[str, Any],
                             responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Celery task to run cross-metric analysis.
    
    Args:
        survey_id: The survey ID
        metrics_data: Dictionary of metric definitions
        responses: List of survey responses
        
    Returns:
        Dictionary with cross-metric analysis results
    """
    logger.info(f"Running cross-metric analysis task for survey {survey_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            coordinator._run_cross_metric_analysis(survey_id, metrics_data, responses)
        )
        logger.info(f"Completed cross-metric analysis task for survey {survey_id}")
        return result
    except Exception as e:
        logger.error(f"Error in cross-metric analysis task for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="generate_survey_summary", max_retries=3, priority=1)
def generate_survey_summary(self, survey_id: int, survey_data: Dict[str, Any], 
                           analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to generate a survey summary.
    
    Args:
        survey_id: The survey ID
        survey_data: Survey metadata and configuration
        analysis_results: Previous analysis results
        
    Returns:
        Dictionary with survey summary
    """
    logger.info(f"Running survey summary generation task for survey {survey_id}")
    try:
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            coordinator._generate_survey_summary(survey_id, survey_data, analysis_results)
        )
        logger.info(f"Completed survey summary generation task for survey {survey_id}")
        return result
    except Exception as e:
        logger.error(f"Error in survey summary generation task for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="run_analysis_pipeline")
def run_analysis_pipeline(self, survey_id: int, survey_data: Dict[str, Any], 
                         responses: List[Dict[str, Any]],
                         time_series_responses: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                         run_base_only: bool = False,
                         run_vector_analysis: bool = True) -> str:
    """
    Celery task to orchestrate the complete analysis pipeline.
    This task chains other tasks together.
    
    Args:
        survey_id: The survey ID
        survey_data: Survey metadata and configuration
        responses: List of survey responses
        time_series_responses: Optional dictionary mapping time periods to response lists
        run_base_only: Whether to run only base analysis
        run_vector_analysis: Whether to run vector-enhanced analysis
        
    Returns:
        Task ID for tracking
    """
    from celery import chain, group
    
    logger.info(f"Setting up analysis pipeline for survey {survey_id}")
    
    # Start with base analysis
    base_task = run_base_analysis.s(survey_id, survey_data, responses)
    
    if run_base_only:
        # Just run the base analysis
        result = base_task.apply_async()
        return f"Base analysis task started with ID: {result.id}"
    
    # Create metric analysis tasks
    metric_tasks = []
    for metric_id, metric_data in survey_data.get("metrics", {}).items():
        task = run_metric_analysis.s(
            survey_id, metric_id, metric_data, responses, time_series_responses
        )
        metric_tasks.append(task)
    
    # Create vector analysis tasks if needed
    vector_tasks = []
    if run_vector_analysis:
        for metric_id, metric_data in survey_data.get("metrics", {}).items():
            # Only run vector analysis for text and categorical metrics
            if metric_data.get("type") in ["text", "categorical", "single_choice", "multi_choice"]:
                task = run_vector_enhanced_analysis.s(
                    survey_id, metric_id, responses
                )
                vector_tasks.append(task)
    
    # Cross-metric analysis task
    cross_metric_task = run_cross_metric_analysis.s(
        survey_id, survey_data.get("metrics", {}), responses
    )
    
    # Survey summary task (placeholder - will be filled with real parameters later)
    summary_task = generate_survey_summary.s(survey_id, survey_data, {})
    
    # Chain the tasks together
    # First run base analysis, then a group of metric analyses in parallel
    # then cross-metric analysis, then summary generation
    analysis_chain = chain(
        base_task,
        group(metric_tasks),
        group(vector_tasks) if vector_tasks else None,
        cross_metric_task,
        summary_task
    )
    
    # Remove None tasks from the chain
    analysis_chain = [t for t in analysis_chain.tasks if t is not None]
    
    # Execute the chain
    result = chain(*analysis_chain).apply_async()
    
    logger.info(f"Analysis pipeline started for survey {survey_id}")
    return f"Analysis pipeline started with task ID: {result.id}" 