"""
Analysis tasks for the data pipeline.
This module contains Celery tasks for running analysis operations in the background.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json

from ..tasks.celery_app import app
from data_pipeline.services.metadata_store import metadata_store

logger = logging.getLogger(__name__)


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.run_comprehensive_analysis", max_retries=3, priority=9)
def run_comprehensive_analysis(self, survey_id: int, survey_data: str, responses: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Run a comprehensive analysis on survey data as a Celery task.
    
    Args:
        survey_id: The survey ID
        survey_data: The survey definition containing metrics (JSON string)
        responses: The survey responses (JSON string)
        force_refresh: Whether to skip the cache and force a fresh analysis
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"[CELERY] Starting comprehensive analysis for survey {survey_id}")
    logger.info(f"[CELERY] Task ID: {self.request.id}")
    logger.info(f"[CELERY] Running in worker: {self.request.hostname}")
    
    try:
        logger.info(f"[CELERY] Parsing JSON data for survey {survey_id}")
        # Parse JSON data
        survey_data_dict = json.loads(survey_data)
        responses_list = json.loads(responses)
        
        logger.info(f"[CELERY] Successfully parsed JSON data: {len(survey_data_dict.get('metrics', {}))} metrics, {len(responses_list)} responses")
        
        # Update analysis progress
        update_analysis_progress(survey_id, "running", "Analysis started")
        
        # Import AnalysisCoordinator here to avoid circular imports
        from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
        
        # Run the analysis using the coordinator
        logger.info(f"[CELERY] Running analysis_coordinator.analyze_survey for survey {survey_id}")
        result = asyncio.run(analysis_coordinator.analyze_survey(
            survey_id=survey_id,
            survey_data=survey_data_dict,
            responses=responses_list,
            force_refresh=force_refresh
        ))
        
        # Update progress with completed status
        update_analysis_progress(survey_id, "completed", "Analysis completed successfully")
        
        logger.info(f"[CELERY] Completed comprehensive analysis for survey {survey_id}")
        return {
            "status": "success",
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "metrics_analyzed": result.get("metrics_analyzed", 0)
        }
    
    except Exception as e:
        logger.error(f"[CELERY] Error in comprehensive analysis for survey {survey_id}: {str(e)}")
        import traceback
        logger.error(f"[CELERY] Traceback: {traceback.format_exc()}")
        
        # Update progress with error status
        update_analysis_progress(survey_id, "error", f"Analysis failed: {str(e)}")
        
        return {
            "status": "error",
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def update_analysis_progress(survey_id: int, status: str, message: str) -> None:
    """
    Update the analysis progress in the metadata store.
    
    Args:
        survey_id: The survey ID
        status: The status of the analysis (running, completed, error)
        message: A message describing the current state
    """
    progress = {
        "survey_id": survey_id,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    # Store progress in metadata store
    asyncio.run(metadata_store.store_analysis_result("analysis_progress", survey_id, progress))
    
    logger.info(f"[CELERY] Updated analysis progress for survey {survey_id}: {status}")


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.run_base_analysis", max_retries=3, priority=9)
def run_base_analysis(self, survey_id: int, survey_data: str, responses: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Run a basic analysis on survey data as a Celery task.
    
    Args:
        survey_id: The survey ID
        survey_data: The survey definition containing metrics (JSON string)
        responses: The survey responses (JSON string)
        force_refresh: Whether to skip the cache and force a fresh analysis
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"[CELERY] Starting base analysis for survey {survey_id}")
    
    try:
        # Parse JSON data
        survey_data_dict = json.loads(survey_data)
        responses_list = json.loads(responses)
        
        # Import analysis_coordinator here to avoid circular imports
        from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
        
        # Run the analysis using the coordinator
        result = asyncio.run(analysis_coordinator._run_base_analysis(
            survey_id=survey_id,
            survey_data=survey_data_dict,
            responses=responses_list,
            force_refresh=force_refresh
        ))
        
        logger.info(f"[CELERY] Completed base analysis for survey {survey_id}")
        return {
            "status": "success",
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "metrics_count": result.get("metrics_count", 0)
        }
    
    except Exception as e:
        logger.error(f"[CELERY] Error in base analysis for survey {survey_id}: {str(e)}")
        return {
            "status": "error",
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.run_metric_analysis", max_retries=3, priority=5)
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
        # Import analysis_coordinator here to avoid circular imports
        from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
        
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            analysis_coordinator._run_metric_analysis(
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


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.run_vector_enhanced_analysis", max_retries=3, priority=3)
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
        # Import analysis_coordinator here to avoid circular imports
        from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
        
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            analysis_coordinator._run_vector_enhanced_analysis(survey_id, metric_id, responses)
        )
        logger.info(f"Completed vector-enhanced analysis task for survey {survey_id}, metric {metric_id}")
        return result
    except Exception as e:
        logger.error(f"Error in vector-enhanced analysis task for survey {survey_id}, metric {metric_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.run_cross_metric_analysis", max_retries=3, priority=2)
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
        # Import analysis_coordinator here to avoid circular imports
        from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
        
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            analysis_coordinator._run_cross_metric_analysis(survey_id, metrics_data, responses)
        )
        logger.info(f"Completed cross-metric analysis task for survey {survey_id}")
        return result
    except Exception as e:
        logger.error(f"Error in cross-metric analysis task for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.generate_survey_summary", max_retries=3, priority=1)
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
        # Import analysis_coordinator here to avoid circular imports
        from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
        
        # Use asyncio to run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            analysis_coordinator._generate_survey_summary(survey_id, survey_data, analysis_results)
        )
        logger.info(f"Completed survey summary generation task for survey {survey_id}")
        return result
    except Exception as e:
        logger.error(f"Error in survey summary generation task for survey {survey_id}: {str(e)}")
        # Retry with exponential backoff
        retry_count = self.request.retries
        backoff = 2 ** retry_count
        self.retry(exc=e, countdown=backoff)


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.mark_analysis_completed")
def mark_analysis_completed(self, survey_id, task_ids=None):
    """
    Celery task to mark an analysis as completed.
    This task should be called after all analysis tasks have finished.
    
    Args:
        survey_id: The survey ID
        task_ids: Dictionary of task IDs that were part of this analysis
    """
    logger.info(f"Marking analysis as completed for survey_id={survey_id}")
    try:
        import asyncio
        
        # Create async function to gather all analysis results
        async def gather_and_update_completion():
            try:
                # Start with a base result structure
                complete_analysis = {
                    "survey_id": survey_id,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "task_ids": task_ids or {},
                    "completion_time": datetime.now().isoformat()
                }
                
                # Gather base analysis
                logger.info(f"Retrieving base analysis for survey_id={survey_id}")
                base_analysis = await metadata_store.get_analysis_result("base_analysis", survey_id)
                if base_analysis:
                    complete_analysis["base_analysis"] = base_analysis
                
                # Gather all metric analyses
                logger.info(f"Retrieving metric analyses for survey_id={survey_id}")
                metric_analyses = {}
                
                # First get base metrics to determine which metrics to look for
                if base_analysis and "metrics" in base_analysis:
                    for metric_id in base_analysis["metrics"].keys():
                        metric_analysis = await metadata_store.get_analysis_result("metric_analysis", survey_id, metric_id)
                        if metric_analysis:
                            metric_analyses[metric_id] = metric_analysis
                            logger.info(f"Retrieved analysis for metric {metric_id}")
                
                complete_analysis["metric_analyses"] = metric_analyses
                logger.info(f"Retrieved {len(metric_analyses)} metric analyses")
                
                # Gather cross-metric analysis
                logger.info(f"Retrieving cross-metric analysis for survey_id={survey_id}")
                cross_metric = await metadata_store.get_analysis_result("cross_metric_analysis", survey_id)
                if cross_metric:
                    complete_analysis["cross_metric_analysis"] = cross_metric
                
                # Gather survey summary if available
                logger.info(f"Retrieving survey summary for survey_id={survey_id}")
                summary = await metadata_store.get_analysis_result("survey_summary", survey_id)
                if summary:
                    complete_analysis["survey_summary"] = summary
                
                # Vector analyses if available
                logger.info(f"Retrieving vector analyses for survey_id={survey_id}")
                vector_analyses = {}
                if base_analysis and "metrics" in base_analysis:
                    for metric_id in base_analysis["metrics"].keys():
                        vector_analysis = await metadata_store.get_analysis_result(f"vector_analysis_{metric_id}", survey_id)
                        if vector_analysis:
                            vector_analyses[metric_id] = vector_analysis
                
                if vector_analyses:
                    complete_analysis["vector_analyses"] = vector_analyses
                    logger.info(f"Retrieved {len(vector_analyses)} vector analyses")
                
                # Store the complete analysis
                logger.info(f"Storing complete analysis result for survey_id={survey_id}")
                await metadata_store.store_analysis_result("analysis_progress", survey_id, complete_analysis)
                logger.info(f"Successfully stored complete analysis for survey_id={survey_id}")
                return True
            except Exception as e:
                logger.error(f"Error gathering analysis results: {str(e)}")
                # Store basic completion info as fallback
                basic_completion = {
                    "survey_id": survey_id,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "task_ids": task_ids or {},
                    "completion_time": datetime.now().isoformat(),
                    "error": f"Error gathering complete results: {str(e)}"
                }
                await metadata_store.store_analysis_result("analysis_progress", survey_id, basic_completion)
                logger.error(f"Stored fallback completion data due to error")
                raise
        
        # Execute the async function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(gather_and_update_completion())
        logger.info(f"Completion status update result: {result}")
        return {"status": "completed", "survey_id": survey_id}
    
    except Exception as e:
        logger.error(f"Error in mark_analysis_completed: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


@app.task(bind=True, name="data_pipeline.tasks.analysis_tasks.run_analysis_pipeline")
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
    from celery import chain, group, chord, signature
    import traceback
    
    logger.info(f"Setting up analysis pipeline for survey_id={survey_id}")
    
    # Start with base analysis
    base_task = signature("data_pipeline.tasks.analysis_tasks.run_base_analysis").s(survey_id, survey_data, responses)
    logger.info(f"Created base analysis task for survey_id={survey_id}")
    
    if run_base_only:
        # Just run the base analysis
        result = base_task.apply_async()
        logger.info(f"Running only base analysis with task ID: {result.id}")
        return f"Base analysis task started with ID: {result.id}"
    
    # Create metric analysis tasks
    metric_tasks = []
    for metric_id, metric_data in survey_data.get("metrics", {}).items():
        logger.info(f"Creating metric analysis task for survey {survey_id}, metric {metric_id}")
        task = signature("data_pipeline.tasks.analysis_tasks.run_metric_analysis").s(
            survey_id, metric_id, metric_data, responses, time_series_responses
        )
        metric_tasks.append(task)
    
    logger.info(f"Created {len(metric_tasks)} metric analysis tasks for survey {survey_id}")
    
    # Create vector analysis tasks if needed
    vector_tasks = []
    if run_vector_analysis:
        for metric_id, metric_data in survey_data.get("metrics", {}).items():
            # Only run vector analysis for text and categorical metrics
            if metric_data.get("type") in ["text", "categorical", "single_choice", "multi_choice"]:
                logger.info(f"Creating vector analysis task for survey {survey_id}, metric {metric_id}")
                task = signature("data_pipeline.tasks.analysis_tasks.run_vector_enhanced_analysis").s(
                    survey_id, metric_id, responses
                )
                vector_tasks.append(task)
    
    logger.info(f"Created {len(vector_tasks)} vector analysis tasks for survey {survey_id}")
    
    # Cross-metric analysis task
    cross_metric_task = signature("data_pipeline.tasks.analysis_tasks.run_cross_metric_analysis").s(
        survey_id, survey_data.get("metrics", {}), responses
    )
    logger.info(f"Created cross-metric analysis task for survey {survey_id}")
    
    # Survey summary task (placeholder - will be filled with real parameters later)
    summary_task = signature("data_pipeline.tasks.analysis_tasks.generate_survey_summary").s(survey_id, survey_data, {})
    logger.info(f"Created survey summary task for survey {survey_id}")

    # Completion marker task that will run at the end of the pipeline
    completion_task = signature("data_pipeline.tasks.analysis_tasks.mark_analysis_completed").s(survey_id)
    logger.info(f"Created completion marker task for survey {survey_id}")
    
    try:
        logger.info("Building task sequence")
        
        # First, let's mark the analysis as in-progress immediately
        try:
            from data_pipeline.services.metadata_store import metadata_store
            import asyncio
            
            async def mark_in_progress():
                in_progress_data = {
                    "survey_id": survey_id,
                    "status": "in_progress",
                    "timestamp": datetime.now().isoformat(),
                    "start_time": datetime.now().isoformat()
                }
                await metadata_store.store_analysis_result("analysis_progress", survey_id, in_progress_data)
                
            loop = asyncio.get_event_loop()
            loop.run_until_complete(mark_in_progress())
            logger.info(f"Marked analysis as in_progress for survey_id={survey_id}")
        except Exception as e:
            logger.error(f"Error marking analysis as in-progress: {str(e)}")
        
        # Execute base analysis task first
        base_result = base_task.apply_async()
        logger.info(f"Started base analysis task with ID: {base_result.id}")
        task_id = base_result.id  # Save for return value
        
        # Execute metric tasks (if any)
        if metric_tasks:
            if len(metric_tasks) > 1:
                # If multiple metrics, use a group
                logger.info(f"Executing group of {len(metric_tasks)} metric tasks")
                group(metric_tasks).apply_async()
                logger.info(f"Started metric tasks group")
            else:
                # If just one metric task, execute it directly
                logger.info("Executing single metric task")
                metric_tasks[0].apply_async()
                logger.info(f"Started metric task")
        
        # Execute vector tasks (if any)
        if vector_tasks:
            if len(vector_tasks) > 1:
                # If multiple vector tasks, use a group
                logger.info(f"Executing group of {len(vector_tasks)} vector tasks")
                group(vector_tasks).apply_async()
                logger.info(f"Started vector tasks group")
            else:
                # If just one vector task, execute it directly
                logger.info("Executing single vector task")
                vector_tasks[0].apply_async()
                logger.info(f"Started vector task")
        
        # Execute cross-metric analysis task
        cross_result = cross_metric_task.apply_async()
        logger.info(f"Started cross-metric task with ID: {cross_result.id}")
        
        # Execute summary task
        summary_result = summary_task.apply_async()
        logger.info(f"Started summary task with ID: {summary_result.id}")
        
        # Final task: mark analysis as completed
        # We need to run this as a separate task after a delay to ensure other tasks have had time to complete
        completion_kwargs = {
            "task_ids": {
                "base": base_result.id,
                "cross_metric": cross_result.id,
                "summary": summary_result.id
            },
            "countdown": 60  # Wait 60 seconds to allow other tasks to complete
        }
        completion_result = completion_task.apply_async(**completion_kwargs)
        logger.info(f"Scheduled completion marker task with ID: {completion_result.id} (will run after 60s delay)")
        
        logger.info(f"All analysis tasks started")
        
        # Also update the metadata store here as a fallback
        try:
            async def update_task_info():
                progress_data = {
                    "survey_id": survey_id,
                    "status": "in_progress",
                    "timestamp": datetime.now().isoformat(),
                    "task_ids": {
                        "base": base_result.id,
                        "cross_metric": cross_result.id, 
                        "summary": summary_result.id,
                        "completion": completion_result.id
                    },
                    "completion_scheduled": True
                }
                await metadata_store.store_analysis_result("analysis_progress", survey_id, progress_data)
                
            loop = asyncio.get_event_loop()
            loop.run_until_complete(update_task_info())
            logger.info(f"Updated task info in metadata store for survey_id={survey_id}")
        except Exception as e:
            logger.error(f"Error updating task info: {str(e)}")
        
        return f"Analysis pipeline started with base task ID: {task_id}"
        
    except Exception as e:
        logger.error(f"Error building analysis pipeline: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        raise 