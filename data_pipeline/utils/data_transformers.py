"""
Utilities for transforming data between API formats and analysis formats.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataTransformer:
    """Transforms API data formats to analysis-compatible formats."""
    
    @staticmethod
    def transform_survey_data(survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform survey data from API format to analysis format.
        
        Args:
            survey_data: Survey data from API (survey form endpoint)
            
        Returns:
            Transformed survey data for analysis
        """
        # Extract relevant fields
        transformed_data = {
            "survey_id": survey_data.get("id"),
            "name": survey_data.get("title", "Untitled Survey"),
            "description": survey_data.get("description", ""),
            "metrics": {},
            "questions": {},
            "question_to_metric_map": {}
        }
        
        # Process metrics
        metrics = survey_data.get("metrics", [])
        for metric in metrics:
            metric_id = metric.get("name")  # Using name as ID
            if metric_id:
                transformed_data["metrics"][metric_id] = {
                    "id": metric_id,
                    "name": metric.get("name", "Unknown"),
                    "type": DataTransformer._normalize_metric_type(metric.get("type", "unknown")),
                    "description": metric.get("description", ""),
                    "weight": metric.get("weight", 1.0)
                }
        
        # Process questions and build mapping
        questions = survey_data.get("questions", [])
        for question in questions:
            question_id = question.get("id")
            metric_id = question.get("metric_id")
            
            if question_id and metric_id:
                # Store question details
                transformed_data["questions"][question_id] = {
                    "id": question_id,
                    "text": question.get("question", ""),
                    "type": question.get("type", "unknown"),
                    "options": question.get("options", []),
                    "metric_id": metric_id
                }
                
                # Create mapping from question to metric
                transformed_data["question_to_metric_map"][question_id] = metric_id
                
                # Add options to metric if applicable
                if metric_id in transformed_data["metrics"] and question.get("options"):
                    transformed_data["metrics"][metric_id]["options"] = question.get("options", [])
        
        return transformed_data
    
    @staticmethod
    def transform_responses(
        responses_data: Dict[str, Any],
        survey_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform response data from API format to analysis format.
        
        Args:
            responses_data: Response data from API (responses endpoint)
            survey_data: Transformed survey data for question-to-metric mapping
            
        Returns:
            Transformed response data for analysis
        """
        question_to_metric_map = survey_data.get("question_to_metric_map", {})
        metrics_info = survey_data.get("metrics", {})
        questions_info = survey_data.get("questions", {})
        
        # Initialize transformed responses
        transformed_data = {
            "survey_id": int(responses_data.get("data", {}).get("responses", [])[0].get("survey_id", 0)) 
                if responses_data.get("data", {}).get("responses") else 0,
            "total_responses": responses_data.get("data", {}).get("total", 0),
            "responses": []
        }
        
        # Process each response
        for response in responses_data.get("data", {}).get("responses", []):
            # Initialize a new response object in the expected format
            transformed_response = {
                "response_id": response.get("_id", ""),
                "timestamp": response.get("submitted_at", ""),
                "metrics": {},  # Will contain metric_id -> value mappings
                "metadata": {
                    "ip_address": response.get("ip_address"),
                    "user_agent": response.get("user_agent"),
                    "session_id": response.get("session_id")
                }
            }
            
            # Transform question responses to metric responses
            question_responses = response.get("responses", {})
            
            for question_id, value in question_responses.items():
                # Get corresponding metric ID
                metric_id = question_to_metric_map.get(question_id)
                
                if not metric_id:
                    logger.warning(f"No metric mapping found for question {question_id}")
                    continue
                
                # Get metric information
                metric_info = metrics_info.get(metric_id, {})
                metric_type = metric_info.get("type", "unknown")
                
                # Get question information
                question_info = questions_info.get(question_id, {})
                question_type = question_info.get("type", "unknown")
                
                # Transform value based on types
                transformed_value = DataTransformer._transform_response_value(
                    value, metric_type, question_type
                )
                
                # Add to metrics in transformed response
                transformed_response["metrics"][metric_id] = transformed_value
            
            transformed_data["responses"].append(transformed_response)
        
        return transformed_data
    
    @staticmethod
    def prepare_for_metric_analysis(
        survey_data: Dict[str, Any],
        responses_data: Dict[str, Any],
        metric_id: str
    ) -> Dict[str, Any]:
        """
        Prepare data for metric analysis service.
        
        Args:
            survey_data: Transformed survey data
            responses_data: Transformed response data
            metric_id: ID of the metric to analyze
            
        Returns:
            Data prepared for metric analysis
        """
        metrics = survey_data.get("metrics", {})
        metric_data = metrics.get(metric_id, {})
        
        # Extract responses for this specific metric
        metric_responses = []
        for response in responses_data.get("responses", []):
            if metric_id in response.get("metrics", {}):
                value = response["metrics"][metric_id]
                
                # Format response based on metric type
                if metric_data.get("type") == "numeric":
                    metric_responses.append({"value": value})
                elif metric_data.get("type") == "categorical":
                    metric_responses.append({"category": value})
                elif metric_data.get("type") == "text":
                    metric_responses.append({"text": value})
                else:
                    # Default format
                    metric_responses.append({"value": value})
        
        return {
            "survey_id": survey_data.get("survey_id"),
            "metric_id": metric_id,
            "metric_data": metric_data,
            "responses": metric_responses
        }
    
    @staticmethod
    def prepare_for_cross_metric_analysis(
        survey_data: Dict[str, Any],
        responses_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for cross-metric analysis service.
        
        Args:
            survey_data: Transformed survey data
            responses_data: Transformed response data
            
        Returns:
            Data prepared for cross-metric analysis
        """
        # The cross-metric analysis expects metrics_data as a dict and survey_responses as a list
        return {
            "survey_id": survey_data.get("survey_id"),
            "metrics_data": survey_data.get("metrics", {}),
            "survey_responses": responses_data.get("responses", [])
        }
    
    @staticmethod
    def prepare_for_ai_insights(
        survey_data: Dict[str, Any],
        responses_data: Dict[str, Any],
        metric_id: str,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for AI insights service.
        
        Args:
            survey_data: Transformed survey data
            responses_data: Transformed response data
            metric_id: ID of the metric to analyze
            analysis_results: Results from statistical analysis
            
        Returns:
            Data prepared for AI insights
        """
        metrics = survey_data.get("metrics", {})
        metric_data = metrics.get(metric_id, {})
        
        # Calculate response statistics
        total_responses = responses_data.get("total_responses", 0)
        valid_responses = sum(
            1 for r in responses_data.get("responses", [])
            if metric_id in r.get("metrics", {})
        )
        
        response_stats = {
            "total_responses": total_responses,
            "valid_responses": valid_responses,
            "response_rate": (valid_responses / total_responses) * 100 if total_responses > 0 else 0
        }
        
        return {
            "survey_id": survey_data.get("survey_id"),
            "metric_id": metric_id,
            "metric_data": metric_data,
            "analysis_results": analysis_results,
            "response_stats": response_stats
        }
    
    @staticmethod
    def _normalize_metric_type(metric_type: str) -> str:
        """
        Normalize metric type to standard types used in analysis.
        
        Args:
            metric_type: Original metric type from API
            
        Returns:
            Normalized metric type
        """
        metric_type = metric_type.lower()
        
        if metric_type in ["likert", "select", "radio", "boolean"]:
            return "categorical"
        elif metric_type in ["rating", "number"]:
            return "numeric"
        elif metric_type in ["text", "textarea"]:
            return "text"
        elif metric_type in ["multiple_choice", "checkbox"]:
            return "multi_choice"
        else:
            return "unknown"
    
    @staticmethod
    def _transform_response_value(value: Any, metric_type: str, question_type: str) -> Any:
        """
        Transform a response value based on metric and question types.
        
        Args:
            value: Original response value
            metric_type: Type of the metric
            question_type: Type of the question
            
        Returns:
            Transformed value
        """
        # Handle different value types
        if metric_type == "numeric":
            # Try to convert to numeric
            try:
                if isinstance(value, str) and value.strip():
                    # Remove non-numeric parts if it's a scale value like "5 - Very satisfied"
                    if "-" in value and value.split("-")[0].strip().isdigit():
                        return float(value.split("-")[0].strip())
                    return float(value)
                elif isinstance(value, (int, float)):
                    return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert value '{value}' to numeric")
                return None
                
        elif metric_type == "categorical":
            # Return as is for categorical
            return value
            
        elif metric_type == "multi_choice" and isinstance(value, list):
            # Return list of selected options
            return value
            
        elif metric_type == "text":
            # Ensure text is a string
            return str(value) if value else ""
            
        # Default: return value as is
        return value


# Create singleton instance
data_transformer = DataTransformer() 