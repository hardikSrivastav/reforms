"""
Analysis coordination service that orchestrates the various analysis modules.
This module manages the flow of data through different analysis services to generate comprehensive insights.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

from data_pipeline.analysis.metric_analysis import metric_analysis_service
from data_pipeline.analysis.cross_metric_analysis import cross_metric_analysis_service
from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.utils.data_transformers import data_transformer
from data_pipeline.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class AnalysisCoordinator:
    """Coordinates the analysis of survey data across different analysis services."""
    
    def __init__(self):
        """Initialize the analysis coordinator."""
        self.metric_service = metric_analysis_service
        self.cross_metric_service = cross_metric_analysis_service
        self.data_transformer = data_transformer
    
    async def analyze_survey(self, survey_id: int, survey_data: Dict[str, Any], responses: List[Dict[str, Any]], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze survey data using all available analysis services.
        
        Args:
            survey_id: The survey ID
            survey_data: The survey definition containing metrics
            responses: The survey responses
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Starting comprehensive analysis for survey {survey_id}")
        logger.info(f"Survey has {len(responses)} responses")
        
        # Log input data structure and sizes
        logger.info(f"Analysis input - survey_data keys: {list(survey_data.keys())}")
        metrics_count = len(survey_data.get("metrics", [])) if isinstance(survey_data.get("metrics"), list) else len(survey_data.get("metrics", {}))
        logger.info(f"Analysis input - metrics count: {metrics_count}")
        
        # Log sample of first response (without PII)
        if responses and len(responses) > 0:
            sample_response = responses[0].copy() if isinstance(responses[0], dict) else {"data": str(responses[0])}
            # Redact actual response values for privacy
            if "responses" in sample_response:
                sample_response["responses"] = {k: "..." for k in sample_response["responses"].keys()}
            logger.info(f"Analysis input - sample response structure: {sample_response}")
        
        # Check if we already have a recent analysis result
        if not force_refresh:
            cached_result = await metadata_store.get_analysis_result("comprehensive", survey_id)
            if cached_result:
                # Check if it's recent enough (within last day by default)
                if cached_result.get("timestamp"):
                    cache_time = datetime.fromisoformat(cached_result["timestamp"])
                    cache_age = (datetime.now() - cache_time).total_seconds()
                    if cache_age < settings.ANALYSIS_CACHE_TTL:
                        logger.info(f"Using cached comprehensive analysis for survey {survey_id}")
                        return cached_result
        else:
            logger.info(f"Force refresh requested, skipping cache for survey {survey_id}")
        
        # Transform data if needed
        transformed_survey = self.data_transformer.transform_survey_data(survey_data) if not survey_data.get("metrics") else survey_data
        transformed_responses = self.data_transformer.transform_responses(responses, transformed_survey) if "responses" not in responses[0] else responses
        
        # Extract metrics from survey data
        metrics = transformed_survey.get("metrics", {})
        if not metrics:
            return {"error": "No metrics found in survey data"}
        
        # Run metric analysis for each metric
        logger.info(f"Running individual metric analysis for {len(metrics)} metrics in survey {survey_id}")
        metric_results = await self._analyze_individual_metrics(survey_id, metrics, transformed_responses, force_refresh)
        logger.info(f"Individual metric analysis completed with {len(metric_results)} results")
        
        # Run cross-metric analysis
        logger.info(f"Running cross-metric analysis for survey {survey_id}")
        cross_results = await self._analyze_metric_relationships(survey_id, metrics, transformed_responses, force_refresh)
        logger.info(f"Cross-metric analysis completed")
        
        # Generate executive summary
        logger.info(f"Generating executive summary for survey {survey_id}")
        summary = await self._generate_executive_summary(metric_results, cross_results)
        logger.info(f"Executive summary generated with {len(summary.get('metric_insights', []))} metric insights and {len(summary.get('notable_correlations', []))} notable correlations")
        
        # Compile final results
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "response_count": len(responses),
            "metrics_analyzed": len(metrics),
            "metric_analysis": metric_results,
            "cross_metric_analysis": cross_results,
            "summary": summary
        }
        
        # Log result details
        logger.info(f"Comprehensive analysis completed for survey {survey_id} with {len(metric_results)} metrics analyzed")
        logger.info(f"Storing analysis result in metadata store for survey {survey_id}")
        
        # Cache the result
        await metadata_store.store_analysis_result("comprehensive", survey_id, result)
        
        logger.info(f"Completed comprehensive analysis for survey {survey_id}")
        return result
    
    async def _run_base_analysis(self, survey_id: int, survey_data: Dict[str, Any], responses: List[Dict[str, Any]], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run a faster, simplified analysis for immediate insights.
        
        Args:
            survey_id: The survey ID
            survey_data: The survey definition containing metrics 
            responses: The survey responses
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Dictionary with basic analysis results
        """
        logger.info(f"Running base analysis for survey {survey_id}")
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_result = await metadata_store.get_analysis_result("base_analysis", survey_id)
        if cached_result:
            logger.info(f"Using cached base analysis for survey {survey_id}")
            return cached_result
        else:
            logger.info(f"Force refresh requested, skipping cache for base analysis of survey {survey_id}")
        
        # Transform data if needed
        transformed_survey = self.data_transformer.transform_survey_data(survey_data) if not survey_data.get("metrics") else survey_data
        transformed_responses = self.data_transformer.transform_responses(responses, transformed_survey) if isinstance(responses, dict) or (responses and "responses" not in responses[0]) else responses
        
        # Extract metrics from survey data
        metrics_data = transformed_survey.get("metrics", {})
        if not metrics_data:
            return {"error": "No metrics found in survey data"}
        
        # Calculate basic statistics for each metric
        metrics_summary = {}
        
        for metric_id, metric in metrics_data.items():
            metric_type = metric.get("type", "unknown")
            metric_name = metric.get("name", f"Metric {metric_id}")
            
            # Extract values for this metric
            metric_values = []
            for response in transformed_responses:
                response_metrics = response.get("metrics", {})
                if metric_id in response_metrics:
                    metric_values.append(response_metrics[metric_id])
            
            # Skip if no values
            if not metric_values:
                continue
            
            # Perform basic analysis based on metric type
            if metric_type == "numeric":
                try:
                    numeric_values = [float(val) for val in metric_values if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().replace('.', '', 1).isdigit())]
                    if numeric_values:
                        metrics_summary[metric_id] = {
                            "type": "numeric",
                            "name": metric_name,
                            "count": len(numeric_values),
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "mean": sum(numeric_values) / len(numeric_values),
                            "values": numeric_values[:5]  # Include sample of values
                        }
                except Exception as e:
                    logger.warning(f"Error analyzing numeric metric {metric_id}: {str(e)}")
            
            elif metric_type == "categorical":
                category_counts = {}
                for val in metric_values:
                    val_str = str(val)
                    if val_str in category_counts:
                        category_counts[val_str] += 1
                    else:
                        category_counts[val_str] = 1
                
                metrics_summary[metric_id] = {
                    "type": "categorical",
                    "name": metric_name,
                    "count": len(metric_values),
                    "categories": len(category_counts),
                    "distribution": category_counts,
                    "values": list(metric_values[:5])  # Include sample of values
                }
            
            elif metric_type == "multi_choice":
                # For multi-choice, count occurrences of each option
                option_counts = {}
                
                for val in metric_values:
                    if isinstance(val, list):
                        for option in val:
                            option_str = str(option)
                            if option_str in option_counts:
                                option_counts[option_str] += 1
                            else:
                                option_counts[option_str] = 1
                    else:
                        # Handle case where multi-choice is stored as single value
                        val_str = str(val)
                        if val_str in option_counts:
                            option_counts[val_str] += 1
                        else:
                            option_counts[val_str] = 1
                
                metrics_summary[metric_id] = {
                    "type": "multi_choice",
                    "name": metric_name,
                    "count": len(metric_values),
                    "options": len(option_counts),
                    "distribution": option_counts,
                    "values": list(metric_values[:5]) if all(isinstance(v, (str, int, float)) for v in metric_values[:5]) 
                            else [str(v) for v in metric_values[:5]]
                }
            
            elif metric_type == "text":
                # For text, calculate basic text statistics
                text_lengths = [len(str(val)) for val in metric_values if val]
                word_counts = [len(str(val).split()) for val in metric_values if val]
                
                metrics_summary[metric_id] = {
                    "type": "text",
                    "name": metric_name,
                    "count": len(metric_values),
                    "avg_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                    "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
                    "samples": [str(val) for val in metric_values[:3] if val]
                }
            
            else:
                # Default case
                metrics_summary[metric_id] = {
                    "type": "unknown",
                    "name": metric_name,
                    "count": len(metric_values)
                }
        
        # Compile the result
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "response_count": len(transformed_responses),
            "metrics_count": len(metrics_data),
            "metrics": metrics_summary
        }
        
        # Cache the result
        await metadata_store.store_analysis_result("base_analysis", survey_id, result)
        
        logger.info(f"Completed base analysis for survey {survey_id}")
        return result
    
    async def _analyze_individual_metrics(
        self, 
        survey_id: int, 
        metrics: Union[List[Dict[str, Any]], Dict[str, Any]], 
        responses: List[Dict[str, Any]],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze each individual metric.
        
        Args:
            survey_id: The survey ID
            metrics: Dictionary of metric definitions or list of metric definitions
            responses: List of survey responses
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Dictionary mapping metric IDs to their analysis results
        """
        # Handle different formats of metrics data
        if isinstance(metrics, dict):
            logger.info(f"Analyzing {len(metrics)} individual metrics for survey {survey_id}")
            metrics_dict = metrics
        else:
            logger.info(f"Analyzing {len(metrics)} individual metrics (list format) for survey {survey_id}")
            # Convert list to dictionary if needed
            metrics_dict = {}
            for m in metrics:
                if isinstance(m, dict) and "id" in m:
                    metrics_dict[m["id"]] = m
                elif isinstance(m, str):
                    # If metrics is a list of strings (metric IDs), create dummy entries
                    metrics_dict[m] = {"id": m, "name": m}
        
        results = {}
        tasks = []
        
        for metric_id, metric in metrics_dict.items():
            metric_name = metric.get("name", metric_id)
            
            # Extract responses for this specific metric
            metric_responses = []
            for response_data in responses:
                # Get the responses object which contains question_id -> answer mapping
                response_answers = response_data.get("responses", {})
                
                if not response_answers:
                    continue
                    
                # Extract the response for this metric
                metric_value = response_answers.get(metric_name) or response_answers.get(metric_id)
                
                if metric_value is not None:
                    # Create a response object with the appropriate field based on metric type
                    metric_type = metric.get("type", "unknown")
                    
                    if metric_type == "numeric":
                        try:
                            metric_responses.append({"value": float(metric_value)})
                        except (ValueError, TypeError):
                            pass
                    elif metric_type == "categorical":
                        metric_responses.append({"category": str(metric_value)})
                    elif metric_type == "multi_choice":
                        if isinstance(metric_value, list):
                            metric_responses.append({"value": metric_value})
                        elif isinstance(metric_value, str) and metric_value.startswith('[') and metric_value.endswith(']'):
                            try:
                                metric_responses.append({"value": json.loads(metric_value)})
                            except:
                                metric_responses.append({"value": [metric_value]})
                    else:
                            metric_responses.append({"value": [metric_value]})
                elif metric_type == "text":
                    if isinstance(metric_value, str):
                        metric_responses.append({"text": metric_value})
            
            # Create a task for analyzing this metric
            task = asyncio.create_task(
                self.metric_service.analyze_metric(
                    survey_id=survey_id,
                    metric_id=metric_id,
                    metric_data=metric,
                    responses=metric_responses,
                    force_refresh=force_refresh
                )
            )
            tasks.append((metric_id, task))
        
        # Wait for all tasks to complete
        for metric_id, task in tasks:
            try:
                results[metric_id] = await task
            except Exception as e:
                logger.error(f"Error analyzing metric {metric_id}: {str(e)}")
                results[metric_id] = {"error": str(e)}
        
        logger.info(f"Completed individual metric analysis for survey {survey_id}")
        return results
    
    async def _analyze_metric_relationships(
        self, 
        survey_id: int, 
        metrics: Union[List[Dict[str, Any]], Dict[str, Any]], 
        responses: List[Dict[str, Any]],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze relationships between metrics.
        
        Args:
            survey_id: The survey ID
            metrics: Dictionary of metric definitions or list of metric definitions
            responses: List of survey responses
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Cross-metric analysis results
        """
        logger.info(f"Analyzing relationships between metrics for survey {survey_id}")
        
        # Convert metrics dict to list if needed for cross-metric service
        if isinstance(metrics, dict):
            metrics_list = list(metrics.values())
        else:
            metrics_list = metrics
        
        try:
            # Run cross-metric analysis
            cross_metric_results = await self.cross_metric_service.analyze_metric_correlations(
                survey_id=survey_id,
                metrics_data=metrics_list,
                responses=responses,
                force_refresh=force_refresh
            )
            
            return cross_metric_results
        except Exception as e:
            logger.error(f"Error in cross-metric analysis for survey {survey_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_executive_summary(
        self, 
        metric_results: Dict[str, Any], 
        cross_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an executive summary from all analysis results.
        
        Args:
            metric_results: Results from individual metric analysis
            cross_results: Results from cross-metric analysis
            
        Returns:
            Executive summary
        """
        # Extract key insights from metric analysis
        metric_insights = []
        for metric_id, result in metric_results.items():
            if "ai_insights" in result and "summary" in result["ai_insights"]:
                insight = {
                    "metric_id": metric_id,
                    "metric_name": result.get("metric_name", metric_id),
                    "insight": result["ai_insights"]["summary"]
                }
                metric_insights.append(insight)
        
        # Extract key insights from cross-metric analysis
        cross_metric_insights = []
        if "ai_insights" in cross_results and "summary" in cross_results["ai_insights"]:
            cross_metric_insights = [{"insight": cross_results["ai_insights"]["summary"]}]
        
        # Compile the executive summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "metric_insights": metric_insights[:5],  # Include top 5 metric insights
            "relationship_insights": cross_metric_insights,
            "total_metrics_analyzed": len(metric_results),
            "notable_correlations": []
        }
        
        # Add notable correlations if available
        if "numeric_correlations" in cross_results and "correlations" in cross_results["numeric_correlations"]:
            correlations = cross_results["numeric_correlations"]["correlations"]
            notable = [c for c in correlations if c.get("significant", False) and abs(c.get("correlation", 0)) > 0.5]
            
            for correlation in notable[:3]:  # Include top 3 notable correlations
                summary["notable_correlations"].append({
                    "metric1": correlation.get("metric1", ""),
                    "metric2": correlation.get("metric2", ""),
                    "correlation": correlation.get("correlation", 0),
                    "strength": correlation.get("strength", "")
                })
        
        return summary
    
    async def run_analysis_pipeline(
        self, 
        survey_id: int, 
        survey_data: Dict[str, Any],
        responses: List[Dict[str, Any]],
        use_celery: bool = False,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            survey_id: The survey ID
            survey_data: The survey definition data
            responses: List of survey responses
            use_celery: Whether to use Celery for async tasks
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Analysis results or Celery task info
        """
        logger.info(f"Running analysis pipeline for survey {survey_id} with use_celery={use_celery}, force_refresh={force_refresh}")
        
        if use_celery:
            # Import the Celery app and task directly
            from data_pipeline.tasks.celery_app import app
            
            # Convert data to JSON-serializable format
            serializable_survey_data = json.dumps(survey_data, default=lambda x: str(x) if hasattr(x, "__str__") else None)
            serializable_responses = json.dumps(responses, default=lambda x: str(x) if hasattr(x, "__str__") else None)
            
            # Submit task to Celery using the full task name
            logger.info(f"Submitting analysis task to Celery for survey {survey_id}")
            task = app.send_task(
                "data_pipeline.tasks.analysis_tasks.run_comprehensive_analysis",
                kwargs={
                    "survey_id": survey_id,
                    "survey_data": serializable_survey_data,
                    "responses": serializable_responses,
                    "force_refresh": force_refresh
                }
            )
            
            # Return task information
            return {
                "status": "submitted",
                "task_id": task.id,
                "survey_id": survey_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Analysis job submitted to Celery"
            }
        else:
            # Run analysis directly
            logger.info(f"Running analysis directly (not using Celery) for survey {survey_id}")
            return await self.analyze_survey(survey_id, survey_data, responses, force_refresh=force_refresh)


# Convert MongoDB ObjectId to strings for JSON serialization
def convert_objectid_to_str(data):
    """Convert ObjectId fields to strings in data structures recursively."""
    if isinstance(data, dict):
        return {k: convert_objectid_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif str(type(data)).endswith("ObjectId'>"):
        return str(data)
    else:
        return data


# Singleton instance
analysis_coordinator = AnalysisCoordinator() 