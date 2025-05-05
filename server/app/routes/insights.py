from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Union, Coroutine
from datetime import datetime
import asyncio
from bson import ObjectId
import inspect
import json
import os
import logging
from collections import deque

from ..database import get_db, get_mongo_db
from ..models import SurveyIdMapping, Goal
from data_pipeline.analysis.analysis_coordinator import analysis_coordinator, convert_objectid_to_str
from data_pipeline.analysis.cross_metric_analysis import cross_metric_analysis_service
from data_pipeline.analysis.metric_analysis import metric_analysis_service
from data_pipeline.analysis.vector_trend_analysis import vector_trend_analysis_service
from data_pipeline.analysis.multimodal_ai_analysis import multimodal_ai_analysis_service
from data_pipeline.embeddings.multi_level_embedding_service import multi_level_embedding_service
from data_pipeline.embeddings.semantic_search import semantic_search_service
from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.utils.data_transformers import data_transformer

router = APIRouter(prefix="/api/insights", tags=["insights"])

async def resolve_coroutines(data: Any, path: str = "") -> Any:
    """
    Recursively resolve any coroutines in the data structure.
    
    Args:
        data: The data to check for coroutines
        path: The current path in the data structure (for logging)
        
    Returns:
        The data with all coroutines resolved
    """
    if inspect.iscoroutine(data) or hasattr(data, '__await__'):
        print(f"WARNING: Resolving coroutine at {path}")
        return await data
    
    if isinstance(data, dict):
        for key, value in list(data.items()):
            data[key] = await resolve_coroutines(value, f"{path}.{key}" if path else key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = await resolve_coroutines(item, f"{path}[{i}]")
    
    return data

@router.get("/survey/{survey_id}/base", response_model=Dict[str, Any])
async def get_base_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    force_refresh: bool = Query(False, description="Force refresh analysis"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get base insights for a survey (immediate, real-time analysis).
    This endpoint retrieves the fastest tier of analysis results.
    """
    try:
        # Validate survey exists
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="Survey not found")

        # Check if cached results exist
        if not force_refresh:
            cached_result = await metadata_store.get_analysis_result("base_analysis", survey_id)
            if cached_result:
                print(f"Got cached result for survey {survey_id}: {type(cached_result)}")
                # Resolve any coroutines in the cached result
                resolved_result = await resolve_coroutines(cached_result, "cached_result")
                if resolved_result:  # Only return if we have actual data
                    return {
                        "status": "success",
                        "message": "Retrieved cached base insights",
                        "data": resolved_result,
                        "source": "cache"
                    }
                print("Cached result resolved to null, forcing fresh analysis")
        else:
            print(f"Force refresh requested for survey {survey_id}")

        # Fetch survey data and responses
        mongo_id = mapping.mongo_id
        print(f"Fetching MongoDB data with ID: {mongo_id} (type: {type(mongo_id)})")
        survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
        if not survey_data:
            raise HTTPException(status_code=404, detail="Survey data not found")

        # Debug: print survey data structure keys
        print(f"Survey data keys: {survey_data.keys()}")
        
        # Check for metrics or questions in survey data
        metrics = []
        if "metrics" in survey_data and survey_data["metrics"]:
            metrics = survey_data["metrics"]
            print(f"Found {len(metrics)} metrics in survey data")
            print(f"First metric sample: {metrics[0] if metrics else 'None'}")
        elif "questions" in survey_data and survey_data["questions"]:
            metrics = survey_data["questions"]
            print(f"Found {len(metrics)} questions in survey data")
            print(f"First question sample: {metrics[0] if metrics else 'None'}")
        else:
            print("No metrics or questions found in survey data")
        
        # Fetch responses first to extract metric IDs
        cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
        responses = await cursor.to_list(length=1000)
        print(f"Found {len(responses)} responses for survey")
        
        # Extract all unique metric IDs from responses
        metric_ids = set()
        for response in responses:
            if "responses" in response:
                metric_ids.update(response["responses"].keys())
        
        print(f"Found {len(metric_ids)} unique metric IDs in responses: {list(metric_ids)[:5]}...")
        
        # Validate metrics format and extract metrics data
        metrics_data = {}
        
        # First check for metrics with IDs
        for m in metrics:
            # Check if metric has id
            if "id" in m:
                metrics_data[str(m["id"])] = m
            # Try alternative field names
            elif "_id" in m:
                m["id"] = m["_id"]  # Add id field for compatibility
                metrics_data[str(m["_id"])] = m
        
        # If no metrics have IDs, assign IDs based on response data and metric position
        if len(metrics_data) == 0 and metric_ids and len(metrics) > 0:
            print("No metrics with IDs found. Assigning IDs based on response data.")
            # Sort metric IDs to ensure consistent mapping
            sorted_ids = sorted(list(metric_ids))
            
            # Assign IDs to metrics in order
            for i, metric_id in enumerate(sorted_ids):
                if i < len(metrics):
                    metrics[i]["id"] = metric_id
                    metrics_data[metric_id] = metrics[i]
                    print(f"Assigned ID {metric_id} to metric {metrics[i].get('name', f'Metric {i}')}")
        
        print(f"Extracted {len(metrics_data)} metrics with valid IDs")
        if metrics_data:
            print(f"Sample metric data structure: {next(iter(metrics_data.values()))}")
        
        if responses:
            print(f"Sample response structure: {responses[0].keys()}")
            if 'responses' in responses[0]:
                print(f"Sample response data: {dict(list(responses[0]['responses'].items())[:2])}")
        
        # Skip analysis if no metrics or responses
        if len(metrics_data) == 0 or len(responses) == 0:
            print("WARNING: No valid metrics or responses found. Skipping analysis.")
            return {
                "status": "partial",
                "message": "Insufficient data for analysis",
                "data": {
                    "metrics_count": len(metrics_data),
                    "responses_count": len(responses),
                    "note": "Not enough data for analysis"
                },
                "source": "fallback"
            }
        
        # Prepare metrics format for analysis
        metrics_data_formatted = {"metrics": metrics_data}
        print(f"Calling _run_base_analysis with: survey_id={survey_id}, metrics_count={len(metrics_data)}, responses_count={len(responses)}")
        
        try:
            # Try a direct approach to bypass caching
            print(f"Generating fresh analysis with force_refresh={force_refresh}")
            
            # First generate a simple statistical summary as a fallback
            basic_insights = generate_basic_insights(metrics_data, responses)
            
            # Try to get insights from the analysis coordinator
            try:
                # Debug the metrics structure before calling analysis
                print(f"Metrics data structure: {metrics_data_formatted.keys()}")
                print(f"Sample metric in formatted data: {next(iter(metrics_data_formatted['metrics'].items())) if metrics_data_formatted['metrics'] else 'No metrics'}")
                
                insights_coroutine = analysis_coordinator._run_base_analysis(
                    survey_id, 
                    metrics_data_formatted, 
                    responses,
                    force_refresh=force_refresh
                )
                print(f"Analysis coroutine type: {type(insights_coroutine)}")
                
                # Resolve the coroutine
                insights = await insights_coroutine
                print(f"Resolved analysis result type: {type(insights)}")
                
                # Resolve any nested coroutines
                if insights:
                    resolved_insights = await resolve_coroutines(insights, "insights")
                    print(f"Final resolved insights type: {type(resolved_insights)}")
                    
                    # Check if metrics field is empty in the resolved insights
                    if resolved_insights and (not resolved_insights.get("metrics") or len(resolved_insights.get("metrics", {})) == 0):
                        print("WARNING: Metrics field is empty in resolved insights, using fallback")
                        # Merge base info with metrics from basic_insights
                        resolved_insights["metrics"] = basic_insights.get("metrics_summary", {})
                        
                    if resolved_insights:
                        print("Analysis successful, returning generated insights")
                        return {
                            "status": "success",
                            "message": "Generated base insights",
                            "data": resolved_insights,
                            "source": "generated"
                        }
                    else:
                        print("Resolved insights is null, using fallback")
                else:
                    print("Analysis returned null, using fallback")
            except Exception as analysis_error:
                print(f"Error in analysis: {type(analysis_error).__name__}: {str(analysis_error)}")
                import traceback
                traceback.print_exc()
            
            # If we get here, use the fallback analysis
            print(f"Using basic fallback insights")
            return {
                "status": "partial",
                "message": "Generated basic statistics (advanced analysis unavailable)",
                "data": basic_insights,
                "source": "fallback"
            }
        except Exception as analysis_error:
            print(f"Error in analysis: {type(analysis_error).__name__}: {str(analysis_error)}")
            import traceback
            traceback.print_exc()
            
            # Return a meaningful error to the client
            return {
                "status": "error",
                "message": f"Error analyzing survey data: {str(analysis_error)}",
                "data": {
                    "error_type": type(analysis_error).__name__,
                    "metrics_count": len(metrics_data),
                    "responses_count": len(responses)
                }
            }
    except Exception as e:
        print(f"Error in get_base_insights: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )

def generate_basic_insights(metrics_data, responses):
    """Generate basic insights from metrics and responses data."""
    basic_insights = {
        "response_count": len(responses),
        "metric_count": len(metrics_data),
        "timestamp": datetime.now().isoformat(),
        "metrics_summary": {},
        "note": "Basic statistics analysis"
    }
    
    # Calculate basic metrics for each question
    for metric_id, metric in metrics_data.items():
        metric_type = metric.get("type", "unknown")
        metric_name = metric.get("name", f"Metric {metric_id}")
        response_values = []
        
        # Extract values for this metric
        for response in responses:
            if "responses" in response and metric_id in response["responses"]:
                response_values.append(response["responses"][metric_id])
        
        # Sample the values to show in the response
        sample_values = response_values[:3] if response_values else []
        
        # Convert values to strings for the value_counts dictionary
        value_counts = {}
        for value in response_values:
            # Convert unhashable types (like lists) to their string representation
            if isinstance(value, (list, dict)):
                # Use JSON string representation for hashability
                key = json.dumps(value, sort_keys=True)
            else:
                key = str(value)
                
            if key in value_counts:
                value_counts[key] += 1
            else:
                value_counts[key] = 1
        
        # Basic summary statistics
        metric_summary = {
            "id": metric_id,
            "name": metric_name,
            "type": metric_type,
            "response_count": len(response_values),
            "sample_values": sample_values,
            "value_distribution": value_counts
        }
        
        # Add more statistics based on metric type
        if metric_type == "likert" or metric_type == "numeric":
            # Try to extract numeric values for numeric analysis
            numeric_values = []
            for value in response_values:
                try:
                    # For list values, skip them in numeric analysis
                    if isinstance(value, (list, dict)):
                        continue
                    
                    # Handle values like "4 - Agree" by extracting the number
                    if isinstance(value, str) and "-" in value:
                        numeric_part = value.split("-")[0].strip()
                        numeric_values.append(float(numeric_part))
                    else:
                        numeric_values.append(float(value))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    pass
            
            if numeric_values:
                metric_summary["numeric_analysis"] = {
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "values": numeric_values[:10]  # Include sample of values
                }
        
        basic_insights["metrics_summary"][metric_id] = metric_summary
    
    return basic_insights

@router.get("/survey/{survey_id}/full", response_model=Dict[str, Any])
async def get_full_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    force_refresh: bool = Query(False, description="Force refresh analysis"),
    check_celery: bool = Query(False, description="Check Celery task status without starting a new analysis"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get full insights for a survey (comprehensive analysis).
    This endpoint returns cached results or triggers a background analysis job.
    
    Parameters:
    - force_refresh: Whether to force a refresh of the analysis
    - check_celery: If true, only check if there's a Celery task running without starting a new one
    """
    # Set up logging
    logger = logging.getLogger("insights")
    logger.info(f"get_full_insights called for survey_id={survey_id}, force_refresh={force_refresh}, check_celery={check_celery}")
    
    try:
        # Validate survey exists
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        if not mapping:
            logger.warning(f"Survey ID {survey_id} not found in mapping table")
            raise HTTPException(status_code=404, detail="Survey not found")

        # Check if cached full analysis exists and we're not forcing refresh
        if not force_refresh:
            logger.info(f"Checking for cached analysis for survey_id={survey_id}")
            
            # First check for a stored comprehensive result
            cached_result = await metadata_store.get_analysis_result("comprehensive", survey_id)
            if cached_result:
                logger.info(f"Found completed cached analysis for survey_id={survey_id}")
                # Resolve any coroutines in the cached result
                resolved_result = await resolve_coroutines(cached_result, "cached_result")
                return {
                    "status": "success",
                    "message": "Retrieved cached full insights",
                    "data": resolved_result,
                    "source": "cache"
                }
            
            # Next check if Celery has stored results in the progress
            celery_progress = await metadata_store.get_analysis_result("analysis_progress", survey_id)
            if celery_progress and celery_progress.get("status") == "completed":
                # If Celery has stored the full analysis in the progress
                if "metric_analysis" in celery_progress or "cross_metric_analysis" in celery_progress:
                    logger.info(f"Found completed Celery analysis for survey_id={survey_id}")
                    resolved_result = await resolve_coroutines(celery_progress, "celery_progress")
                    return {
                        "status": "success",
                        "message": "Retrieved Celery analysis results",
                        "data": resolved_result,
                        "source": "celery"
                    }

        # Fetch survey data and responses
        mongo_id = mapping.mongo_id
        logger.info(f"Fetching survey data for survey_id={survey_id}, mongo_id={mongo_id}")
        survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
        if not survey_data:
            logger.warning(f"Survey data not found for mongo_id={mongo_id}")
            raise HTTPException(status_code=404, detail="Survey data not found")

        # Check for metrics or questions in survey data
        metrics = []
        if "metrics" in survey_data and survey_data["metrics"]:
            metrics = survey_data["metrics"]
            logger.info(f"Found {len(metrics)} metrics in survey data")
        elif "questions" in survey_data and survey_data["questions"]:
            metrics = survey_data["questions"]
            logger.info(f"Found {len(metrics)} questions in survey data")
        else:
            logger.warning("No metrics or questions found in survey data")
        
        # Validate metrics format and extract metrics data
        metrics_data = {}
        for m in metrics:
            # Check if metric has id
            if "id" in m:
                metrics_data[str(m["id"])] = m
            # Try alternative field names
            elif "_id" in m:
                m["id"] = m["_id"]  # Add id field for compatibility
                metrics_data[str(m["_id"])] = m
        
        logger.info(f"Extracted {len(metrics_data)} metrics with valid IDs")

        # Fetch responses
        logger.info(f"Fetching responses for survey_id={survey_id}, mongo_id={mongo_id}")
        cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
        responses = await cursor.to_list(length=1000)
        logger.info(f"Found {len(responses)} responses for survey_id={survey_id}")

        # Extract all unique metric IDs from responses if no metrics were found
        if len(metrics_data) == 0 and len(responses) > 0:
            logger.info("No metrics with valid IDs found. Extracting from responses.")
            metric_ids = set()
            for response in responses:
                if "responses" in response:
                    metric_ids.update(response["responses"].keys())
            
            logger.info(f"Found {len(metric_ids)} unique metric IDs in responses")
            
            # If no metrics have IDs, assign IDs based on response data and metric position
            if metric_ids and len(metrics) > 0:
                logger.info("Assigning IDs to metrics based on response data.")
                # Sort metric IDs to ensure consistent mapping
                sorted_ids = sorted(list(metric_ids))
                
                # Assign IDs to metrics in order
                for i, metric_id in enumerate(sorted_ids):
                    if i < len(metrics):
                        metrics[i]["id"] = metric_id
                        metrics_data[metric_id] = metrics[i]
                        logger.info(f"Assigned ID {metric_id} to metric {metrics[i].get('name', f'Metric {i}')}")
            
            logger.info(f"After ID assignment: {len(metrics_data)} metrics with valid IDs")
        
        # Validate there are some metrics and responses to analyze
        if len(metrics_data) == 0:
            logger.warning("No valid metrics found for analysis")
            return {
                "status": "partial",
                "message": "Insufficient data for analysis: no valid metrics found",
                "data": {
                    "metrics_count": 0,
                    "responses_count": len(responses),
                    "status": "error"
                },
                "source": "fallback"
            }

        # Convert MongoDB ObjectId fields to strings for serialization
        logger.info("Converting ObjectId fields to strings for serialization")
        serializable_metrics_data = convert_objectid_to_str(metrics_data)
        serializable_responses = convert_objectid_to_str(responses)

        # Check if analysis is in progress
        in_progress = False
        progress = None

        if not force_refresh:
            logger.info(f"Checking if analysis is already in progress for survey_id={survey_id}")
            progress = await metadata_store.get_analysis_result("analysis_progress", survey_id)
            if progress:
                # Resolve any coroutines in the progress
                progress = await resolve_coroutines(progress, "progress")
                
                # Check if a Celery task is in progress
                if progress.get("status") == "in_progress" and progress.get("task_ids"):
                    in_progress = True
                    logger.info(f"Analysis already in progress for survey_id={survey_id}")
                    
                    # If we're just checking status, return now
                    if check_celery:
                        return {
                            "status": "success",
                            "message": "Analysis in progress",
                            "data": progress,
                            "source": "in_progress"
                        }
                
                # Check if analysis was completed by Celery
                elif progress.get("status") == "completed" and not force_refresh:
                    logger.info(f"Found completed analysis from Celery for survey_id={survey_id}")
                    
                    # If comprehensive results are directly in the progress data
                    if "metric_analysis" in progress or "cross_metric_analysis" in progress:
                        return {
                            "status": "success", 
                            "message": "Retrieved completed analysis from Celery",
                            "data": progress,
                            "source": "celery_completed"
                        }
                    
                    # If there's a separate comprehensive result
                    comprehensive_result = await metadata_store.get_analysis_result("comprehensive", survey_id)
                    if comprehensive_result:
                        resolved_result = await resolve_coroutines(comprehensive_result, "comprehensive")
                        return {
                            "status": "success",
                            "message": "Retrieved comprehensive analysis results",
                            "data": resolved_result,
                            "source": "comprehensive"
                        }
                        
                    # If we're just checking status, return the progress
                    if check_celery:
                        return {
                            "status": "success",
                            "message": "Analysis completed, check /results endpoint for full data",
                            "data": progress,
                            "source": "celery_completed"
                        }

        if force_refresh or not in_progress:
            # Define a wrapper function for the background task with detailed logging
            async def run_analysis_with_logging():
                task_start_time = datetime.now()
                task_id = f"{survey_id}-{task_start_time.timestamp()}"
                logger.info(f"[TASK:{task_id}] Starting background analysis for survey_id={survey_id}")
                
                try:
                    # Log input sizes
                    logger.info(f"[TASK:{task_id}] Analysis input: metrics_count={len(serializable_metrics_data)}, responses_count={len(serializable_responses)}")
                    
                    # Log task start
                    logger.info(f"[TASK:{task_id}] Running analysis_coordinator.run_analysis_pipeline")
                    
                    # Execute the analysis
                    result = await analysis_coordinator.run_analysis_pipeline(
                        survey_id=survey_id,
                        survey_data={"metrics": serializable_metrics_data},
                        responses=serializable_responses,
                        use_celery=True,
                        force_refresh=force_refresh
                    )
                    
                    # Log completion and duration
                    task_end_time = datetime.now()
                    duration_seconds = (task_end_time - task_start_time).total_seconds()
                    
                    logger.info(f"[TASK:{task_id}] Analysis completed in {duration_seconds:.2f} seconds")
                    
                    # Log result details
                    if isinstance(result, dict):
                        logger.info(f"[TASK:{task_id}] Result keys: {list(result.keys())}")
                        logger.info(f"[TASK:{task_id}] Result status: {result.get('status', 'unknown')}")
                        
                        if 'task_id' in result:
                            logger.info(f"[TASK:{task_id}] Celery task ID: {result['task_id']}")
                    else:
                        logger.info(f"[TASK:{task_id}] Result type: {type(result)}")
                    
                    return result
                    
                except Exception as e:
                    # Log detailed error information
                    logger.error(f"[TASK:{task_id}] ERROR in background analysis: {type(e).__name__}: {str(e)}")
                    
                    import traceback
                    logger.error(f"[TASK:{task_id}] Traceback: {traceback.format_exc()}")
                    
                    # Log error context
                    logger.error(f"[TASK:{task_id}] Error context: survey_id={survey_id}, metrics_count={len(serializable_metrics_data)}, responses_count={len(serializable_responses)}")
                    
                    # Re-raise so the error is properly handled
                    raise
            
            # Add the wrapped task to background tasks
            logger.info(f"Adding background analysis task for survey_id={survey_id}")
            background_tasks.add_task(run_analysis_with_logging)
            
            try:
                # Return base insights while comprehensive analysis runs in background
                logger.info(f"Running base analysis for survey_id={survey_id}")
                base_insights = await analysis_coordinator._run_base_analysis(
                    survey_id, {"metrics": serializable_metrics_data}, serializable_responses
                )
                
                # Resolve any coroutines in the result
                logger.info("Resolving coroutines in base insights")
                resolved_insights = await resolve_coroutines(base_insights, "base_insights")
                
                # Check if metrics field is empty in the resolved insights
                if resolved_insights and (not resolved_insights.get("metrics") or len(resolved_insights.get("metrics", {})) == 0):
                    logger.info("Metrics field is empty in resolved insights, generating basic metrics")
                    # Generate basic insights with metrics data
                    basic_insights = generate_basic_insights(serializable_metrics_data, serializable_responses)
                    resolved_insights["metrics"] = basic_insights.get("metrics_summary", {})
                    logger.info(f"Added {len(resolved_insights['metrics'])} metrics from basic insights")
                
                logger.info(f"Returning base insights while comprehensive analysis runs in background")
                return {
                    "status": "success",
                    "message": "Analysis started, returning base insights while comprehensive analysis runs",
                    "data": {
                        "base_analysis": resolved_insights,
                        "status": "in_progress"
                    },
                    "source": "generated"
                }
            except Exception as analysis_error:
                logger.error(f"Error in base analysis: {type(analysis_error).__name__}: {str(analysis_error)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Return a meaningful error to the client
                return {
                    "status": "error",
                    "message": f"Error analyzing survey data: {str(analysis_error)}",
                    "data": {
                        "error_type": type(analysis_error).__name__,
                        "metrics_count": len(metrics_data),
                        "responses_count": len(responses),
                        "status": "error"
                    }
                }
        else:
            # Analysis already in progress, return current progress
            logger.info(f"Returning in-progress status for survey_id={survey_id}")
            return {
                "status": "success",
                "message": "Analysis in progress, returning current progress",
                "data": progress,
                "source": "in_progress"
            }
    except Exception as e:
        logger.error(f"Error in get_full_insights: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )

@router.get("/survey/{survey_id}/metric/{metric_id}", response_model=Dict[str, Any])
async def get_metric_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get detailed insights for a specific metric in a survey.
    """
    try:
        # Validate survey exists
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="Survey not found")

        # Check if cached results exist
        cached_result = await metadata_store.get_analysis_result("metric_analysis", survey_id, metric_id)
        if cached_result:
            # Resolve any coroutines in the cached result
            resolved_result = await resolve_coroutines(cached_result, "cached_metric_result")
            return {
                "status": "success",
                "message": "Retrieved cached metric insights",
                "data": resolved_result,
                "source": "cache"
            }

        # Fetch survey data and responses
        mongo_id = mapping.mongo_id
        survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
        if not survey_data:
            raise HTTPException(status_code=404, detail="Survey data not found")
        
        # Find the metric in either metrics or questions array
        metric_data = None
        
        # Check metrics array
        if "metrics" in survey_data:
            for m in survey_data["metrics"]:
                metric_id_value = m.get("id") or m.get("_id")
                if str(metric_id_value) == metric_id:
                    metric_data = m
                    # Ensure it has an id field
                    if "id" not in m and "_id" in m:
                        m["id"] = m["_id"]
                    break
        
        # If not found, check questions array
        if not metric_data and "questions" in survey_data:
            for m in survey_data["questions"]:
                metric_id_value = m.get("id") or m.get("_id")
                if str(metric_id_value) == metric_id:
                    metric_data = m
                    # Ensure it has an id field
                    if "id" not in m and "_id" in m:
                        m["id"] = m["_id"]
                    break
        
        if not metric_data:
            raise HTTPException(status_code=404, detail="Metric not found")

        # Fetch responses for this metric
        cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
        responses = await cursor.to_list(length=1000)
        
        # Extract responses for this metric
        metric_responses = []
        for response in responses:
            if "responses" in response and metric_id in response["responses"]:
                value = response["responses"][metric_id]
                if metric_data.get("type") == "numeric":
                    metric_responses.append({"value": value})
                elif metric_data.get("type") in ["categorical", "single_choice"]:
                    metric_responses.append({"category": value})
                elif metric_data.get("type") == "text":
                    metric_responses.append({"text": value})
                else:
                    metric_responses.append({"value": value})

        try:
            # Generate insights for this metric
            insights = await metric_analysis_service.analyze_metric(
                survey_id, metric_id, metric_data, metric_responses
            )
            
            # Resolve any coroutines in the result
            resolved_insights = await resolve_coroutines(insights, "metric_insights")

            return {
                "status": "success",
                "message": "Generated metric insights",
                "data": resolved_insights,
                "source": "generated"
            }
        except Exception as analysis_error:
            print(f"Error in metric analysis: {type(analysis_error).__name__}: {str(analysis_error)}")
            import traceback
            traceback.print_exc()
            
            # Return a meaningful error to the client
            return {
                "status": "error",
                "message": f"Error analyzing metric data: {str(analysis_error)}",
                "data": {
                    "error_type": type(analysis_error).__name__,
                    "metric_id": metric_id,
                    "responses_count": len(metric_responses)
                }
            }
    except Exception as e:
        print(f"Error in get_metric_insights: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )

@router.get("/survey/{survey_id}/cross-metrics", response_model=Dict[str, Any])
async def get_cross_metric_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get cross-metric correlation insights for a survey.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached results exist
    cached_result = await metadata_store.get_analysis_result("cross_metric_analysis", survey_id)
    if cached_result:
        return {
            "status": "success",
            "message": "Retrieved cached cross-metric insights",
            "data": cached_result,
            "source": "cache"
        }

    # Fetch survey data and responses
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")

    # Check for metrics or questions in survey data
    metrics = []
    if "metrics" in survey_data and survey_data["metrics"]:
        metrics = survey_data["metrics"]
    elif "questions" in survey_data and survey_data["questions"]:
        metrics = survey_data["questions"]
    
    # Validate metrics format and extract metrics data
    metrics_data = {}
    for m in metrics:
        # Check if metric has id
        if "id" in m:
            metrics_data[str(m["id"])] = m
        # Try alternative field names
        elif "_id" in m:
            m["id"] = m["_id"]  # Add id field for compatibility
            metrics_data[str(m["_id"])] = m

    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)

    # Generate cross-metric insights
    insights = await cross_metric_analysis_service.analyze_cross_metric_correlations(
        survey_id, metrics_data, responses
    )

    return {
        "status": "success",
        "message": "Generated cross-metric insights",
        "data": insights,
        "source": "generated"
    }

@router.get("/survey/{survey_id}/vector-analysis/{metric_id}", response_model=Dict[str, Any])
async def get_vector_analysis(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get vector-based trend analysis for a specific metric.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached results exist
    cache_key = f"vector_analysis_{metric_id}"
    cached_result = await metadata_store.get_analysis_result(cache_key, survey_id)
    if cached_result:
        return {
            "status": "success",
            "message": "Retrieved cached vector analysis",
            "data": cached_result,
            "source": "cache"
        }

    # Run vector trend analysis for clusters - make sure to await all async calls
    clusters = await vector_trend_analysis_service.detect_response_clusters(
        survey_id=survey_id,
        question_id=metric_id
    )
    
    # Run temporal trend analysis
    trends = await vector_trend_analysis_service.detect_temporal_trends(
        survey_id=survey_id,
        question_id=metric_id
    )
    
    # Run anomaly detection
    anomalies = await vector_trend_analysis_service.detect_anomalies(
        survey_id=survey_id,
        question_id=metric_id
    )
    
    # Combine results
    vector_analysis = {
        "survey_id": survey_id,
        "metric_id": metric_id,
        "timestamp": datetime.now().isoformat(),
        "response_clusters": clusters,
        "temporal_trends": trends,
        "anomaly_detection": anomalies
    }
    
    # Cache the results
    await metadata_store.store_analysis_result(cache_key, survey_id, vector_analysis)
    
    return {
        "status": "success",
        "message": "Generated vector-based analysis",
        "data": vector_analysis,
        "source": "generated"
    }

@router.get("/survey/{survey_id}/semantic-search", response_model=Dict[str, Any])
async def semantic_search(
    survey_id: int = Path(..., description="The ID of the survey"),
    query: str = Query(..., description="The search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    db: Session = Depends(get_db)
):
    """
    Perform semantic search across survey responses.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Perform semantic search
    search_results = await semantic_search_service.search_responses(
        survey_id=survey_id,
        query_text=query,
        limit=limit
    )
    
    return {
        "status": "success",
        "message": "Semantic search completed",
        "data": search_results
    }

@router.get("/survey/{survey_id}/embeddings/generate", response_model=Dict[str, Any])
async def generate_embeddings(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db)
):
    """
    Generate multi-level embeddings for a survey.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Generate embeddings - ensure we await the result
    result = await multi_level_embedding_service.generate_aggregate_embeddings(survey_id)
    
    return {
        "status": "success",
        "message": "Embeddings generation triggered",
        "data": result
    }

@router.get("/survey/{survey_id}/ai-insights/text/{metric_id}", response_model=Dict[str, Any])
async def get_text_sentiment_analysis(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get AI-powered sentiment analysis for text responses.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")
    
    # Fetch survey data
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")
    
    # Find the metric/question
    question_text = None
    
    # Check metrics array
    if "metrics" in survey_data:
        for m in survey_data["metrics"]:
            metric_id_value = m.get("id") or m.get("_id")
            if str(metric_id_value) == metric_id:
                question_text = m.get("name", "Unknown Question")
                break
    
    # If not found, check questions array
    if not question_text and "questions" in survey_data:
        for m in survey_data["questions"]:
            metric_id_value = m.get("id") or m.get("_id")
            if str(metric_id_value) == metric_id:
                question_text = m.get("name", "Unknown Question")
                break
    
    if not question_text:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)
    
    # Extract text responses for this metric
    text_responses = []
    for response in responses:
        if "responses" in response and metric_id in response["responses"]:
            text = response["responses"][metric_id]
            if isinstance(text, str):
                text_responses.append({"text": text})
    
    # Get sentiment analysis
    sentiment_analysis = await multimodal_ai_analysis_service.analyze_text_sentiment(
        survey_id=survey_id,
        question_id=metric_id,
        question_text=question_text,
        responses=text_responses
    )
    
    return {
        "status": "success",
        "message": "Generated text sentiment analysis",
        "data": sentiment_analysis
    }

@router.get("/survey/{survey_id}/ai-insights/free-text/{metric_id}", response_model=Dict[str, Any])
async def get_free_text_analysis(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    guidance: Optional[str] = Query(None, description="Optional guidance for the analysis"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get AI-powered analysis of free-text responses with optional guidance.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")
    
    # Fetch survey data
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")
    
    # Find the metric/question
    question_text = None
    
    # Check metrics array
    if "metrics" in survey_data:
        for m in survey_data["metrics"]:
            metric_id_value = m.get("id") or m.get("_id")
            if str(metric_id_value) == metric_id:
                question_text = m.get("name", "Unknown Question")
                break
    
    # If not found, check questions array
    if not question_text and "questions" in survey_data:
        for m in survey_data["questions"]:
            metric_id_value = m.get("id") or m.get("_id")
            if str(metric_id_value) == metric_id:
                question_text = m.get("name", "Unknown Question")
                break
    
    if not question_text:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)
    
    # Extract text responses for this metric
    text_responses = []
    for response in responses:
        if "responses" in response and metric_id in response["responses"]:
            text = response["responses"][metric_id]
            if isinstance(text, str):
                text_responses.append({"text": text})
    
    # Get free text analysis
    text_analysis = await multimodal_ai_analysis_service.analyze_free_responses(
        survey_id=survey_id,
        question_id=metric_id,
        question_text=question_text,
        responses=text_responses,
        guidance=guidance
    )
    
    return {
        "status": "success",
        "message": "Generated free-text analysis",
        "data": text_analysis
    }

@router.get("/logs/recent", response_model=Dict[str, Any])
async def get_recent_logs(
    lines: int = Query(100, description="Number of recent log lines to retrieve")
):
    """
    Get recent log lines from the insights pipeline.
    This is useful for debugging background tasks.
    """
    # Configure logging to a file if not already done
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "insights.log")
    
    # Create file handler if it doesn't exist
    logger = logging.getLogger("insights")
    has_file_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file):
            has_file_handler = True
            break
    
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Read recent logs
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                # Use deque to efficiently get the last N lines
                recent_logs = deque(f, lines)
                log_lines = list(recent_logs)
        else:
            log_lines = ["No log file found. Logs will appear after the first run."]
        
        return {
            "status": "success",
            "message": f"Retrieved {len(log_lines)} recent log lines",
            "data": {
                "log_file": log_file,
                "lines": log_lines
            }
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Error retrieving logs: {str(e)}",
            "data": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

@router.get("/survey/{survey_id}/status", response_model=Dict[str, Any])
async def get_analysis_status(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db)
):
    """
    Get the current status of any running analysis for a survey.
    
    This endpoint allows checking if an analysis is in progress, completed,
    or if an error occurred during analysis.
    """
    try:
        # Validate survey exists
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="Survey not found")
            
        # Check analysis progress
        progress = await metadata_store.get_analysis_result("analysis_progress", survey_id)
        
        if not progress:
            return {
                "status": "unknown",
                "message": "No analysis has been started for this survey"
            }
        
        # If analysis is completed, include a flag to indicate results are available
        if progress.get("status") == "completed":
            # Check if comprehensive results exist
            results_available = await metadata_store.get_analysis_result("comprehensive", survey_id) is not None
            progress["results_available"] = results_available
        
        return progress
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )

@router.get("/survey/{survey_id}/results", response_model=Dict[str, Any])
async def get_analysis_results(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db)
):
    """
    Get the completed analysis results for a survey.
    
    This endpoint retrieves the stored analysis results after background processing has completed.
    Use this endpoint after checking the analysis status is 'completed' through the status endpoint.
    """
    try:
        # Validate survey exists
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="Survey not found")
            
        # First check for comprehensive analysis results
        comprehensive_result = await metadata_store.get_analysis_result("comprehensive", survey_id)
        
        # If comprehensive results exist, return them
        if comprehensive_result:
            # Resolve any coroutines in the result
            resolved_result = await resolve_coroutines(comprehensive_result, "comprehensive_result")
            return {
                "status": "success",
                "message": "Retrieved comprehensive analysis results",
                "data": resolved_result,
                "source": "comprehensive"
            }
        
        # If no comprehensive results, check if there's data in analysis_progress
        progress = await metadata_store.get_analysis_result("analysis_progress", survey_id)
        
        if progress and progress.get("status") == "completed":
            # Extract any analysis data stored in the progress object
            analysis_data = {}
            
            # Check for various types of analysis data
            for key in ["base_analysis", "metric_analyses", "cross_metric_analysis", "survey_summary"]:
                if key in progress:
                    analysis_data[key] = progress[key]
            
            if analysis_data:
                # Resolve any coroutines in the result
                resolved_data = await resolve_coroutines(analysis_data, "progress_data")
                return {
                    "status": "success",
                    "message": "Retrieved analysis results from progress data",
                    "data": resolved_data,
                    "source": "progress"
                }
        
        # If no results found in either location
        return {
            "status": "not_found",
            "message": "No completed analysis results found for this survey",
            "data": {
                "survey_id": survey_id,
                "progress_status": progress.get("status") if progress else "unknown"
            }
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        ) 