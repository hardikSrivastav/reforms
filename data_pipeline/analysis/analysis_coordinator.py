"""
Analysis coordinator for orchestrating the multi-tiered analysis process.
This module coordinates between different analysis components,
manages the progressive loading of results, and integrates statistical 
and AI-generated insights.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.analysis.trend_analysis import trend_analysis_service
from data_pipeline.analysis.correlation_analysis import correlation_analysis_service
from data_pipeline.services.ai_insights import ai_insights_service
from data_pipeline.analysis.vector_trend_analysis import vector_trend_analysis_service
from data_pipeline.embeddings.semantic_search import semantic_search_service

logger = logging.getLogger(__name__)


class AnalysisCoordinator:
    """Coordinates the multi-tiered analysis process."""
    
    def __init__(self):
        """Initialize the analysis coordinator."""
        logger.info("Initialized analysis coordinator")
    
    async def run_analysis_pipeline(
        self, 
        survey_id: int, 
        survey_data: Dict[str, Any], 
        responses: List[Dict[str, Any]],
        time_series_responses: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        run_base_only: bool = False,
        run_vector_analysis: bool = True,
        use_celery: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline for a survey.
        
        Args:
            survey_id: The survey ID
            survey_data: Survey metadata and configuration
            responses: List of survey responses
            time_series_responses: Optional dictionary mapping time periods to response lists
            run_base_only: Whether to run only base analysis
            run_vector_analysis: Whether to run vector-enhanced analysis
            use_celery: Whether to use Celery for distributed task processing
            
        Returns:
            Dictionary with all analysis results or task IDs if using Celery
        """
        logger.info(f"Running analysis pipeline for survey {survey_id}")
        
        # If using Celery, delegate to task queue
        if use_celery:
            from data_pipeline.tasks.analysis_tasks import run_analysis_pipeline as celery_run_pipeline
            
            # Submit task to Celery
            task_result = celery_run_pipeline.delay(
                survey_id, 
                survey_data, 
                responses, 
                time_series_responses, 
                run_base_only, 
                run_vector_analysis
            )
            
            # Return task information
            return {
                "survey_id": survey_id,
                "timestamp": datetime.now().isoformat(),
                "status": "queued",
                "task_id": task_result.id,
                "use_celery": True
            }
        
        # If not using Celery, run the original implementation
        # Initialize result structure
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress",
            "base_analysis": {},
            "metric_analysis": {},
            "vector_analysis": {},
            "cross_metric_analysis": None,
            "survey_summary": None
        }
        
        try:
            # First tier: Base analysis (always run)
            base_analysis = await self._run_base_analysis(survey_id, survey_data, responses)
            result["base_analysis"] = base_analysis
            
            # Save intermediate results
            self._save_progress(result)
            
            # Exit early if only base analysis is requested
            if run_base_only:
                result["status"] = "completed_base_only"
                self._save_progress(result)
                return result
            
            # Second tier: Per-metric analysis
            metric_analysis_tasks = []
            for metric_id, metric_data in survey_data.get("metrics", {}).items():
                task = self._run_metric_analysis(
                    survey_id, 
                    metric_id, 
                    metric_data, 
                    responses, 
                    time_series_responses
                )
                metric_analysis_tasks.append(task)
            
            # Run metric analysis tasks concurrently
            metric_results = await asyncio.gather(*metric_analysis_tasks, return_exceptions=True)
            
            # Process results, handling any exceptions
            for i, (metric_id, _) in enumerate(survey_data.get("metrics", {}).items()):
                if isinstance(metric_results[i], Exception):
                    logger.error(f"Error in metric analysis for {metric_id}: {str(metric_results[i])}")
                    result["metric_analysis"][metric_id] = {"error": str(metric_results[i])}
                else:
                    result["metric_analysis"][metric_id] = metric_results[i]
            
            # Save intermediate results
            self._save_progress(result)
            
            # Vector-enhanced analysis (optional)
            if run_vector_analysis:
                vector_analysis_tasks = []
                for metric_id, metric_data in survey_data.get("metrics", {}).items():
                    # Only run vector analysis for text and categorical metrics where it makes most sense
                    if metric_data.get("type") in ["text", "categorical", "single_choice", "multi_choice"]:
                        task = self._run_vector_enhanced_analysis(survey_id, metric_id, responses)
                        vector_analysis_tasks.append((metric_id, task))
                
                # Run vector analysis tasks concurrently
                for metric_id, task in vector_analysis_tasks:
                    try:
                        vector_result = await task
                        result["vector_analysis"][metric_id] = vector_result
                    except Exception as e:
                        logger.error(f"Error in vector analysis for {metric_id}: {str(e)}")
                        result["vector_analysis"][metric_id] = {"error": str(e)}
                
                # Save intermediate results
                self._save_progress(result)
            
            # Third tier: Cross-metric analysis
            cross_metric_result = await self._run_cross_metric_analysis(
                survey_id, 
                survey_data.get("metrics", {}), 
                responses
            )
            result["cross_metric_analysis"] = cross_metric_result
            
            # Save intermediate results
            self._save_progress(result)
            
            # Final tier: Generate survey summary
            summary_result = await self._generate_survey_summary(survey_id, survey_data, result)
            result["survey_summary"] = summary_result
            
            # Mark analysis as complete
            result["status"] = "completed"
            self._save_progress(result)
            
            logger.info(f"Completed analysis pipeline for survey {survey_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline for survey {survey_id}: {str(e)}")
            result["status"] = "error"
            result["error"] = str(e)
            self._save_progress(result)
            return result
    
    async def _run_base_analysis(
        self, 
        survey_id: int, 
        survey_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run base analysis for a survey.
        
        Args:
            survey_id: The survey ID
            survey_data: Survey metadata and configuration
            responses: List of survey responses
            
        Returns:
            Dictionary with base analysis results
        """
        logger.info(f"Running base analysis for survey {survey_id}")
        
        # Check cache first
        cached_result = metadata_store.get_analysis_result("base_analysis", survey_id)
        if cached_result:
            return cached_result
        
        # Initialize result
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "response_count": len(responses),
            "metrics": {}
        }
        
        # Basic analysis for each metric
        for metric_id, metric_data in survey_data.get("metrics", {}).items():
            metric_type = metric_data.get("type", "unknown")
            
            # Extract values for this metric
            values = []
            for response in responses:
                response_data = response.get("responses", {})
                value = response_data.get(metric_id)
                if value is not None:
                    values.append(value)
            
            # Calculate basic stats based on metric type
            if metric_type == "numeric":
                result["metrics"][metric_id] = self._calculate_numeric_stats(values)
            
            elif metric_type in ["categorical", "single_choice"]:
                result["metrics"][metric_id] = self._calculate_categorical_stats(values)
            
            elif metric_type == "multi_choice":
                result["metrics"][metric_id] = self._calculate_multi_choice_stats(values)
            
            else:
                # Default to treating as text
                result["metrics"][metric_id] = {
                    "count": len(values),
                    "response_rate": len(values) / len(responses) if len(responses) > 0 else 0
                }
        
        # Store in cache
        metadata_store.store_analysis_result("base_analysis", survey_id, result)
        
        logger.info(f"Completed base analysis for survey {survey_id}")
        return result
    
    def _calculate_numeric_stats(self, values: List[Any]) -> Dict[str, Any]:
        """
        Calculate basic statistics for numeric values.
        
        Args:
            values: List of values
            
        Returns:
            Dictionary with statistics
        """
        # Convert to numeric if possible
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                pass
        
        if not numeric_values:
            return {
                "count": 0,
                "valid_count": 0
            }
        
        # Calculate basic stats
        stats = {
            "count": len(values),
            "valid_count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": sum(numeric_values) / len(numeric_values),
            "median": sorted(numeric_values)[len(numeric_values) // 2]
        }
        
        return stats
    
    def _calculate_categorical_stats(self, values: List[Any]) -> Dict[str, Any]:
        """
        Calculate basic statistics for categorical values.
        
        Args:
            values: List of values
            
        Returns:
            Dictionary with statistics
        """
        if not values:
            return {
                "count": 0,
                "categories": {}
            }
        
        # Count categories
        category_counts = {}
        for v in values:
            if v is None:
                continue
                
            category = str(v)
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        # Calculate percentages
        total = len(values)
        category_percentages = {
            cat: (count / total) * 100
            for cat, count in category_counts.items()
        }
        
        # Find most common categories
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        most_common = sorted_categories[:3] if len(sorted_categories) > 3 else sorted_categories
        most_common = [{"category": cat, "count": count} for cat, count in most_common]
        
        return {
            "count": total,
            "unique_categories": len(category_counts),
            "categories": category_counts,
            "percentages": category_percentages,
            "most_common": most_common
        }
    
    def _calculate_multi_choice_stats(self, values: List[Any]) -> Dict[str, Any]:
        """
        Calculate basic statistics for multi-choice values.
        
        Args:
            values: List of values (each value should be a list)
            
        Returns:
            Dictionary with statistics
        """
        if not values:
            return {
                "count": 0,
                "options": {}
            }
        
        # Count options
        option_counts = {}
        valid_responses = 0
        
        for v in values:
            if not isinstance(v, list):
                continue
                
            valid_responses += 1
            for option in v:
                option_str = str(option)
                if option_str in option_counts:
                    option_counts[option_str] += 1
                else:
                    option_counts[option_str] = 1
        
        # Calculate percentages (of responses that chose each option)
        option_percentages = {
            opt: (count / valid_responses) * 100 if valid_responses > 0 else 0
            for opt, count in option_counts.items()
        }
        
        # Find most common options
        sorted_options = sorted(
            option_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        most_common = sorted_options[:3] if len(sorted_options) > 3 else sorted_options
        most_common = [{"option": opt, "count": count} for opt, count in most_common]
        
        # Calculate average selections per response
        total_selections = sum(option_counts.values())
        avg_selections = total_selections / valid_responses if valid_responses > 0 else 0
        
        return {
            "count": len(values),
            "valid_responses": valid_responses,
            "unique_options": len(option_counts),
            "options": option_counts,
            "percentages": option_percentages,
            "most_common": most_common,
            "average_selections": avg_selections
        }
    
    async def _run_metric_analysis(
        self, 
        survey_id: int, 
        metric_id: str, 
        metric_data: Dict[str, Any], 
        responses: List[Dict[str, Any]],
        time_series_responses: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Run detailed analysis for a specific metric.
        
        Args:
            survey_id: The survey ID
            metric_id: The metric ID
            metric_data: The metric definition data
            responses: List of survey responses
            time_series_responses: Optional dictionary mapping time periods to response lists
            
        Returns:
            Dictionary with metric analysis results
        """
        logger.info(f"Running metric analysis for survey {survey_id}, metric {metric_id}")
        
        # Check cache first
        cached_result = metadata_store.get_analysis_result("metric_analysis", survey_id, metric_id)
        if cached_result:
            return cached_result
        
        # Initialize result
        result = {
            "survey_id": survey_id,
            "metric_id": metric_id,
            "timestamp": datetime.now().isoformat(),
            "statistical_analysis": {},
            "trend_analysis": None,
            "ai_insights": None
        }
        
        # Extract values for this metric
        values = []
        for response in responses:
            response_data = response.get("responses", {})
            value = response_data.get(metric_id)
            if value is not None:
                values.append(value)
        
        # Response stats
        response_stats = {
            "total_responses": len(responses),
            "valid_responses": len(values),
            "response_rate": len(values) / len(responses) if len(responses) > 0 else 0
        }
        
        # Get metric type
        metric_type = metric_data.get("type", "unknown")
        
        # Perform appropriate statistical analysis based on metric type
        if metric_type == "numeric":
            stats = self._run_numeric_analysis(values)
        elif metric_type in ["categorical", "single_choice"]:
            stats = self._run_categorical_analysis(values)
        elif metric_type == "multi_choice":
            stats = self._run_multi_choice_analysis(values)
        else:
            stats = {"count": len(values)}
        
        result["statistical_analysis"] = stats
        
        # Run trend analysis if time series data is available
        if time_series_responses:
            trend_result = await trend_analysis_service.analyze_metric_trends(
                survey_id, 
                metric_id, 
                metric_data, 
                time_series_responses
            )
            result["trend_analysis"] = trend_result
        
        # Generate AI insights for the metric
        ai_result = await ai_insights_service.generate_metric_insights(
            survey_id,
            metric_id,
            metric_data,
            stats,
            response_stats
        )
        result["ai_insights"] = ai_result
        
        # Store in cache
        metadata_store.store_analysis_result("metric_analysis", survey_id, result, metric_id)
        
        logger.info(f"Completed metric analysis for survey {survey_id}, metric {metric_id}")
        return result
    
    def _run_numeric_analysis(self, values: List[Any]) -> Dict[str, Any]:
        """
        Run detailed analysis for numeric data.
        
        Args:
            values: List of values
            
        Returns:
            Dictionary with analysis results
        """
        import numpy as np
        from scipy import stats as scipystats
        
        # Convert to numeric
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                pass
        
        if not numeric_values:
            return {"count": 0, "valid_count": 0}
        
        # Basic stats
        np_array = np.array(numeric_values)
        percentiles = np.percentile(np_array, [25, 50, 75])
        
        analysis = {
            "count": len(values),
            "valid_count": len(numeric_values),
            "min": float(np.min(np_array)),
            "max": float(np.max(np_array)),
            "mean": float(np.mean(np_array)),
            "median": float(np.median(np_array)),
            "std_deviation": float(np.std(np_array)),
            "variance": float(np.var(np_array)),
            "quartiles": {
                "q1": float(percentiles[0]),
                "q2": float(percentiles[1]),
                "q3": float(percentiles[2])
            },
            "skewness": float(scipystats.skew(np_array)),
            "kurtosis": float(scipystats.kurtosis(np_array))
        }
        
        # Create histogram data
        hist, bin_edges = np.histogram(np_array, bins='auto')
        histogram_data = []
        for i in range(len(hist)):
            histogram_data.append({
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i+1]),
                "count": int(hist[i])
            })
        
        analysis["histogram"] = histogram_data
        
        return analysis
    
    def _run_categorical_analysis(self, values: List[Any]) -> Dict[str, Any]:
        """
        Run detailed analysis for categorical data.
        
        Args:
            values: List of values
            
        Returns:
            Dictionary with analysis results
        """
        if not values:
            return {"count": 0}
        
        # Count categories
        category_counts = {}
        for v in values:
            if v is None:
                continue
                
            category = str(v)
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        # Calculate percentages and sort
        total = len(values)
        categories_sorted = sorted(
            [
                {
                    "category": cat,
                    "count": count,
                    "percentage": (count / total) * 100
                }
                for cat, count in category_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True
        )
        
        # Calculate entropy (measure of diversity)
        import numpy as np
        probabilities = [count / total for count in category_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(len(category_counts)) if len(category_counts) > 0 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        analysis = {
            "count": total,
            "unique_categories": len(category_counts),
            "categories": categories_sorted,
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "dominance": 1.0 - float(normalized_entropy)  # How dominated by a few categories
        }
        
        return analysis
    
    def _run_multi_choice_analysis(self, values: List[Any]) -> Dict[str, Any]:
        """
        Run detailed analysis for multi-choice data.
        
        Args:
            values: List of values (each value should be a list)
            
        Returns:
            Dictionary with analysis results
        """
        if not values:
            return {"count": 0}
        
        # Count options and co-occurrences
        option_counts = {}
        valid_responses = 0
        co_occurrences = {}
        
        for v in values:
            if not isinstance(v, list) or not v:
                continue
                
            valid_responses += 1
            response_options = [str(opt) for opt in v]
            
            # Count individual options
            for option in response_options:
                if option in option_counts:
                    option_counts[option] += 1
                else:
                    option_counts[option] = 1
                    co_occurrences[option] = {}
            
            # Count co-occurrences
            for i, opt1 in enumerate(response_options):
                for opt2 in response_options[i+1:]:
                    if opt2 in co_occurrences[opt1]:
                        co_occurrences[opt1][opt2] += 1
                    else:
                        co_occurrences[opt1][opt2] = 1
                    
                    if opt1 in co_occurrences[opt2]:
                        co_occurrences[opt2][opt1] += 1
                    else:
                        co_occurrences[opt2][opt1] = 1
        
        # Calculate percentages and sort
        options_sorted = sorted(
            [
                {
                    "option": opt,
                    "count": count,
                    "percentage": (count / valid_responses) * 100 if valid_responses > 0 else 0
                }
                for opt, count in option_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True
        )
        
        # Format co-occurrences
        co_occurrence_list = []
        for opt1, occurrences in co_occurrences.items():
            for opt2, count in occurrences.items():
                if opt1 < opt2:  # Avoid duplicates
                    co_occurrence_list.append({
                        "option1": opt1,
                        "option2": opt2,
                        "count": count,
                        "percentage": (count / valid_responses) * 100 if valid_responses > 0 else 0
                    })
        
        # Sort by count
        co_occurrence_list.sort(key=lambda x: x["count"], reverse=True)
        
        # Calculate average selections per response
        total_selections = sum(option_counts.values())
        avg_selections = total_selections / valid_responses if valid_responses > 0 else 0
        
        analysis = {
            "count": len(values),
            "valid_responses": valid_responses,
            "unique_options": len(option_counts),
            "options": options_sorted,
            "co_occurrences": co_occurrence_list[:20],  # Limit to top 20
            "average_selections": float(avg_selections)
        }
        
        return analysis
    
    async def _run_cross_metric_analysis(
        self, 
        survey_id: int, 
        metrics_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run cross-metric correlation analysis.
        
        Args:
            survey_id: The survey ID
            metrics_data: Dictionary mapping metric IDs to metric definitions
            responses: List of survey responses
            
        Returns:
            Dictionary with cross-metric analysis results
        """
        logger.info(f"Running cross-metric analysis for survey {survey_id}")
        
        # Check cache first
        cached_result = metadata_store.get_analysis_result("cross_metric_analysis", survey_id)
        if cached_result:
            return cached_result
        
        # Run correlation analysis
        correlation_result = await correlation_analysis_service.analyze_cross_metric_correlations(
            survey_id, 
            metrics_data, 
            responses
        )
        
        # Generate AI insights
        ai_result = await ai_insights_service.generate_cross_metric_insights(
            survey_id,
            correlation_result,
            metrics_data
        )
        
        # Combine results
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "correlation_analysis": correlation_result,
            "ai_insights": ai_result
        }
        
        # Store in cache
        metadata_store.store_analysis_result("cross_metric_analysis", survey_id, result)
        
        logger.info(f"Completed cross-metric analysis for survey {survey_id}")
        return result
    
    async def _generate_survey_summary(
        self, 
        survey_id: int, 
        survey_data: Dict[str, Any], 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive survey summary.
        
        Args:
            survey_id: The survey ID
            survey_data: Survey metadata and configuration
            analysis_results: Results from all analysis steps
            
        Returns:
            Dictionary with survey summary
        """
        logger.info(f"Generating survey summary for survey {survey_id}")
        
        # Check cache first
        cached_result = metadata_store.get_analysis_result("survey_summary", survey_id)
        if cached_result:
            return cached_result
        
        # Generate summary using AI
        summary_result = await ai_insights_service.generate_survey_summary(
            survey_id,
            survey_data,
            analysis_results
        )
        
        # Store in cache
        metadata_store.store_analysis_result("survey_summary", survey_id, summary_result)
        
        logger.info(f"Completed survey summary for survey {survey_id}")
        return summary_result
    
    def _save_progress(self, result: Dict[str, Any]) -> None:
        """
        Save intermediate analysis progress.
        
        Args:
            result: Current analysis results
        """
        survey_id = result.get("survey_id")
        if survey_id:
            metadata_store.store_analysis_result("analysis_progress", survey_id, result)

    async def _run_vector_enhanced_analysis(
        self, 
        survey_id: int, 
        metric_id: str, 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run vector-enhanced analysis for a metric.
        
        Args:
            survey_id: Survey ID
            metric_id: Metric ID
            responses: Survey responses
            
        Returns:
            Dictionary with vector-enhanced analysis results
        """
        logger.info(f"Running vector-enhanced analysis for metric {metric_id} in survey {survey_id}")
        
        # Check cache first
        cache_key = f"vector_analysis_{metric_id}"
        cached_result = metadata_store.get_analysis_result(cache_key, survey_id)
        if cached_result:
            return cached_result
        
        try:
            # Initialize result
            result = {
                "survey_id": survey_id,
                "metric_id": metric_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Run cluster analysis
            clusters = await vector_trend_analysis_service.detect_response_clusters(
                survey_id=survey_id,
                question_id=metric_id
            )
            result["response_clusters"] = clusters
            
            # Run temporal trend analysis
            trends = await vector_trend_analysis_service.detect_temporal_trends(
                survey_id=survey_id,
                question_id=metric_id
            )
            result["temporal_trends"] = trends
            
            # Run anomaly detection
            anomalies = await vector_trend_analysis_service.detect_anomalies(
                survey_id=survey_id,
                question_id=metric_id
            )
            result["anomaly_detection"] = anomalies
            
            # Store in cache
            metadata_store.store_analysis_result(cache_key, survey_id, result)
            
            logger.info(f"Completed vector-enhanced analysis for metric {metric_id} in survey {survey_id}")
            return result
        except Exception as e:
            logger.error(f"Error in vector-enhanced analysis for metric {metric_id}: {str(e)}")
            return {
                "survey_id": survey_id,
                "metric_id": metric_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }

# Singleton instance
analysis_coordinator = AnalysisCoordinator() 