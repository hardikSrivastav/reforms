"""
Base analysis service for generating real-time basic statistical metrics.
This module provides the first tier of analysis in the multi-tiered analysis engine.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from collections import Counter
import asyncio

from ..config import settings
from ..services.metadata_store import metadata_store

logger = logging.getLogger(__name__)

class BaseAnalysisService:
    """Service for generating basic real-time analysis of survey data."""
    
    def __init__(self):
        """Initialize the base analysis service."""
        logger.info("Initialized base analysis service")
    
    async def analyze_survey(
        self,
        survey_id: int,
        survey_data: Dict[str, Any],
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate base analysis for a survey.
        
        Args:
            survey_id: Survey ID
            survey_data: Survey data
            responses: List of survey responses
            
        Returns:
            Base analysis results
        """
        try:
            logger.info(f"Generating base analysis for survey {survey_id}")
            
            # Check cache first
            cached_result = metadata_store.get_analysis_result("base_analysis", survey_id)
            if cached_result:
                logger.info(f"Using cached base analysis for survey {survey_id}")
                return cached_result
            
            # Perform analysis
            result = {
                "survey_id": survey_id,
                "response_count": len(responses),
                "completion_stats": await self.calculate_completion_stats(responses),
                "question_stats": await self.calculate_question_stats(survey_data, responses),
                "time_series": await self.generate_time_series(responses)
            }
            
            # Store in cache
            metadata_store.store_analysis_result(
                "base_analysis",
                survey_id,
                result,
                ttl=settings.CACHE_TTL.get("base_analysis")
            )
            
            logger.info(f"Completed base analysis for survey {survey_id}")
            return result
        except Exception as e:
            logger.error(f"Error generating base analysis: {str(e)}")
            return {
                "survey_id": survey_id,
                "error": str(e),
                "response_count": len(responses) if responses else 0
            }
    
    async def calculate_completion_stats(
        self,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate completion statistics for survey responses.
        
        Args:
            responses: List of survey responses
            
        Returns:
            Completion statistics
        """
        if not responses:
            return {"completion_rate": 0, "partial_completions": 0, "full_completions": 0}
        
        total = len(responses)
        completed = 0
        partial = 0
        
        for response in responses:
            # Check completion status based on metadata or response data
            if response.get("completed", False):
                completed += 1
            elif response.get("responses") and len(response["responses"]) > 0:
                partial += 1
        
        return {
            "completion_rate": round(completed / total * 100, 2) if total > 0 else 0,
            "partial_completions": partial,
            "full_completions": completed,
            "abandoned": total - partial - completed
        }
    
    async def calculate_question_stats(
        self,
        survey_data: Dict[str, Any],
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate basic statistics for each question.
        
        Args:
            survey_data: Survey data
            responses: List of survey responses
            
        Returns:
            Question statistics
        """
        questions = survey_data.get("questions", [])
        question_stats = {}
        
        for question in questions:
            question_id = question.get("id")
            question_type = question.get("type")
            
            if not question_id:
                continue
            
            # Collect all responses for this question
            answers = []
            for response in responses:
                response_data = response.get("responses", {})
                if question_id in response_data:
                    answers.append(response_data[question_id])
            
            # Calculate statistics based on question type
            if question_type in ["text", "textarea"]:
                stats = self._text_question_stats(answers)
            elif question_type in ["radio", "select"]:
                stats = self._single_choice_stats(answers, question.get("options", []))
            elif question_type in ["checkbox", "multiselect"]:
                stats = self._multi_choice_stats(answers, question.get("options", []))
            elif question_type in ["rating", "scale"]:
                stats = self._numeric_question_stats(answers)
            else:
                stats = {"response_count": len(answers)}
            
            question_stats[question_id] = {
                "question_type": question_type,
                "question_text": question.get("question", ""),
                "response_rate": round(len(answers) / len(responses) * 100, 2) if responses else 0,
                **stats
            }
        
        return question_stats
    
    def _text_question_stats(self, answers: List[str]) -> Dict[str, Any]:
        """Calculate statistics for text questions."""
        # Filter out None values and convert non-strings to strings
        valid_answers = [str(a) for a in answers if a is not None]
        
        if not valid_answers:
            return {"response_count": 0, "avg_length": 0}
        
        # Calculate average length
        lengths = [len(a) for a in valid_answers]
        
        return {
            "response_count": len(valid_answers),
            "avg_length": round(np.mean(lengths), 2),
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0
        }
    
    def _single_choice_stats(
        self,
        answers: List[Any],
        options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics for single choice questions."""
        # Count frequencies
        counter = Counter(answers)
        total = len(answers)
        
        # Create frequency distribution
        distribution = {}
        for option in options:
            option_id = option.get("value")
            option_text = option.get("text", option_id)
            count = counter.get(option_id, 0)
            distribution[option_id] = {
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0,
                "text": option_text
            }
        
        # Add any answers not in the options
        for answer, count in counter.items():
            if answer not in distribution and answer is not None:
                distribution[answer] = {
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0,
                    "text": answer
                }
        
        return {
            "response_count": total,
            "distribution": distribution,
            "most_common": counter.most_common(1)[0][0] if counter else None
        }
    
    def _multi_choice_stats(
        self,
        answers: List[Any],
        options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics for multi-choice questions."""
        # Flatten the list of selected options
        all_selections = []
        for answer in answers:
            if isinstance(answer, list):
                all_selections.extend(answer)
            elif answer is not None:
                all_selections.append(answer)
        
        # Count frequencies
        counter = Counter(all_selections)
        total_responses = len(answers)
        total_selections = len(all_selections)
        
        # Create frequency distribution
        distribution = {}
        for option in options:
            option_id = option.get("value")
            option_text = option.get("text", option_id)
            count = counter.get(option_id, 0)
            distribution[option_id] = {
                "count": count,
                "percentage_of_responses": round(count / total_responses * 100, 2) if total_responses > 0 else 0,
                "percentage_of_selections": round(count / total_selections * 100, 2) if total_selections > 0 else 0,
                "text": option_text
            }
        
        # Add any selections not in the options
        for selection, count in counter.items():
            if selection not in distribution and selection is not None:
                distribution[selection] = {
                    "count": count,
                    "percentage_of_responses": round(count / total_responses * 100, 2) if total_responses > 0 else 0,
                    "percentage_of_selections": round(count / total_selections * 100, 2) if total_selections > 0 else 0,
                    "text": selection
                }
        
        return {
            "response_count": total_responses,
            "selection_count": total_selections,
            "avg_selections_per_response": round(total_selections / total_responses, 2) if total_responses > 0 else 0,
            "distribution": distribution,
            "most_common": counter.most_common(3) if counter else None
        }
    
    def _numeric_question_stats(self, answers: List[Any]) -> Dict[str, Any]:
        """Calculate statistics for numeric questions."""
        try:
            # Convert to numeric values and filter out non-numeric
            valid_answers = [float(a) for a in answers if a is not None and str(a).strip()]
            
            if not valid_answers:
                return {"response_count": 0}
            
            return {
                "response_count": len(valid_answers),
                "min": min(valid_answers),
                "max": max(valid_answers),
                "mean": round(np.mean(valid_answers), 2),
                "median": round(np.median(valid_answers), 2),
                "std_dev": round(np.std(valid_answers), 2) if len(valid_answers) > 1 else 0,
                "distribution": self._generate_histogram(valid_answers)
            }
        except Exception as e:
            logger.error(f"Error calculating numeric stats: {str(e)}")
            return {"response_count": len(answers), "error": str(e)}
    
    def _generate_histogram(self, values: List[float], bins: int = 5) -> Dict[str, Any]:
        """Generate a histogram for numeric values."""
        try:
            hist, bin_edges = np.histogram(values, bins=bins)
            result = {}
            
            for i in range(len(hist)):
                bin_label = f"{round(bin_edges[i], 2)}-{round(bin_edges[i+1], 2)}"
                result[bin_label] = {
                    "count": int(hist[i]),
                    "percentage": round(hist[i] / len(values) * 100, 2)
                }
            
            return result
        except Exception as e:
            logger.error(f"Error generating histogram: {str(e)}")
            return {}
    
    async def generate_time_series(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate time series data for responses.
        
        Args:
            responses: List of survey responses
            
        Returns:
            Time series data
        """
        try:
            if not responses:
                return {"daily": {}, "weekly": {}, "monthly": {}}
            
            # Extract timestamps and ensure they're in datetime format
            timestamps = []
            for response in responses:
                timestamp = response.get("submitted_at")
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            timestamps.append(pd.to_datetime(timestamp))
                        except:
                            continue
                    else:
                        timestamps.append(pd.to_datetime(timestamp))
            
            if not timestamps:
                return {"daily": {}, "weekly": {}, "monthly": {}}
            
            # Create a DataFrame with the timestamps
            df = pd.DataFrame({"timestamp": timestamps})
            
            # Generate time series at different granularities
            daily = df.groupby(df["timestamp"].dt.date).size().to_dict()
            weekly = df.groupby(df["timestamp"].dt.isocalendar().week).size().to_dict()
            monthly = df.groupby(df["timestamp"].dt.month).size().to_dict()
            
            # Convert to serializable format
            daily_result = {str(date): count for date, count in daily.items()}
            weekly_result = {str(week): count for week, count in weekly.items()}
            monthly_result = {str(month): count for month, count in monthly.items()}
            
            return {
                "daily": daily_result,
                "weekly": weekly_result,
                "monthly": monthly_result,
                "total_days": len(daily),
                "most_active_day": max(daily.items(), key=lambda x: x[1])[0].strftime("%Y-%m-%d") if daily else None,
                "least_active_day": min(daily.items(), key=lambda x: x[1])[0].strftime("%Y-%m-%d") if daily else None
            }
        except Exception as e:
            logger.error(f"Error generating time series: {str(e)}")
            return {"daily": {}, "weekly": {}, "monthly": {}, "error": str(e)}

# Create a singleton instance
base_analysis_service = BaseAnalysisService() 