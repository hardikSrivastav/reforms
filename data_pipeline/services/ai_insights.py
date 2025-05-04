"""
AI-powered insights service for survey data.
This service provides specialized AI-generated insights based on survey data.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from ..config import settings
from ..services.metadata_store import metadata_store

logger = logging.getLogger(__name__)

class AIInsightsService:
    """Service for generating AI-powered insights."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the AI insights service.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.completion_model = settings.COMPLETION_MODEL
        logger.info(f"Initialized AI insights service with model: {self.completion_model}")
    
    async def generate_metric_insights(
        self, 
        survey_id: int, 
        metric_id: str,
        metric_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        response_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI insights for a specific metric.
        
        Args:
            survey_id: The survey ID
            metric_id: The metric ID
            metric_data: The metric definition data
            analysis_results: Results from statistical analysis
            response_stats: Response statistics
            
        Returns:
            Dictionary with AI-generated insights
        """
        # Check cache first
        cached_result = metadata_store.get_analysis_result("ai_insights", survey_id, metric_id)
        if cached_result:
            return cached_result
        
        # Get metric type
        metric_type = metric_data.get("type", "unknown")
        
        # Select appropriate prompt template based on metric type
        prompt_template = self._get_prompt_template(metric_type)
        
        # Prepare context for the prompt
        context = self._prepare_metric_context(metric_data, analysis_results, response_stats)
        
        # Generate insights using AI
        insights = await self._generate_completion(prompt_template, context)
        
        # Parse and structure the insights
        structured_insights = self._parse_metric_insights(insights)
        
        # Create result
        result = {
            "survey_id": survey_id,
            "metric_id": metric_id,
            "metric_name": metric_data.get("name", "Unknown"),
            "metric_type": metric_type,
            "timestamp": datetime.now().isoformat(),
            "raw_insights": insights,
            "structured_insights": structured_insights
        }
        
        # Store in cache
        metadata_store.store_analysis_result("ai_insights", survey_id, result, metric_id)
        
        return result
    
    async def generate_cross_metric_insights(
        self, 
        survey_id: int,
        correlation_results: Dict[str, Any],
        metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI insights for cross-metric analysis.
        
        Args:
            survey_id: The survey ID
            correlation_results: Results from correlation analysis
            metrics_data: Dictionary mapping metric IDs to metric definitions
            
        Returns:
            Dictionary with AI-generated insights
        """
        # Check cache first
        cached_result = metadata_store.get_analysis_result("cross_metric_insights", survey_id)
        if cached_result:
            return cached_result
        
        # Prepare context for the prompt
        context = self._prepare_cross_metric_context(correlation_results, metrics_data)
        
        # Generate insights using AI
        insights = await self._generate_completion(self._get_cross_metric_prompt(), context)
        
        # Parse and structure the insights
        structured_insights = self._parse_cross_metric_insights(insights)
        
        # Create result
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "raw_insights": insights,
            "structured_insights": structured_insights
        }
        
        # Store in cache
        metadata_store.store_analysis_result("cross_metric_insights", survey_id, result)
        
        return result
    
    async def generate_survey_summary(
        self, 
        survey_id: int,
        survey_data: Dict[str, Any],
        all_analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of survey insights.
        
        Args:
            survey_id: The survey ID
            survey_data: Survey metadata
            all_analysis_results: Combined results from all analyses
            
        Returns:
            Dictionary with AI-generated summary
        """
        # Check cache first
        cached_result = metadata_store.get_analysis_result("survey_summary", survey_id)
        if cached_result:
            return cached_result
        
        # Prepare context for the prompt
        context = self._prepare_survey_summary_context(survey_data, all_analysis_results)
        
        # Generate insights using AI
        summary = await self._generate_completion(self._get_survey_summary_prompt(), context)
        
        # Parse and structure the summary
        structured_summary = self._parse_survey_summary(summary)
        
        # Create result
        result = {
            "survey_id": survey_id,
            "survey_name": survey_data.get("name", "Unknown Survey"),
            "timestamp": datetime.now().isoformat(),
            "raw_summary": summary,
            "structured_summary": structured_summary
        }
        
        # Store in cache
        metadata_store.store_analysis_result("survey_summary", survey_id, result)
        
        return result
    
    def _get_prompt_template(self, metric_type: str) -> str:
        """
        Get the appropriate prompt template based on metric type.
        
        Args:
            metric_type: Type of metric
            
        Returns:
            Prompt template as a string
        """
        if metric_type == "numeric":
            return """
            You are a data science expert analyzing numeric survey data for the metric "{metric_name}".
            
            Here is context about the metric:
            {metric_context}
            
            Here are the statistical analysis results:
            {analysis_results}
            
            Based on this information, provide insightful analysis in the following format:
            
            1. Key Findings: List the 2-3 most important insights about this metric
            2. Detailed Interpretation: Provide a deeper analysis of what the metric shows
            3. Actionable Recommendations: Suggest 2-3 concrete actions based on these findings
            
            Keep your response focused on the data provided. Explain any patterns or trends you see.
            Be specific, insightful, and actionable in your recommendations.
            """
        
        elif metric_type in ["categorical", "single_choice"]:
            return """
            You are a data science expert analyzing categorical survey data for the metric "{metric_name}".
            
            Here is context about the metric:
            {metric_context}
            
            Here are the statistical analysis results:
            {analysis_results}
            
            Based on this information, provide insightful analysis in the following format:
            
            1. Distribution Highlights: Identify the 2-3 most significant aspects of the category distribution
            2. Category Analysis: Analyze what the distribution reveals about respondent preferences/behavior
            3. Notable Segments: Identify any segments with distinct patterns
            4. Actionable Recommendations: Suggest 2-3 concrete actions based on these findings
            
            Focus on the most frequent and least frequent categories, and any surprising patterns.
            Be specific, insightful, and actionable in your recommendations.
            """
        
        elif metric_type == "multi_choice":
            return """
            You are a data science expert analyzing multi-choice survey data for the metric "{metric_name}".
            
            Here is context about the metric:
            {metric_context}
            
            Here are the statistical analysis results:
            {analysis_results}
            
            Based on this information, provide insightful analysis in the following format:
            
            1. Option Popularity: Identify the most and least selected options
            2. Option Combinations: Highlight any notable patterns in how options are selected together
            3. Segment Insights: Identify any segments with distinct selection patterns
            4. Actionable Recommendations: Suggest 2-3 concrete actions based on these findings
            
            Focus on meaningful patterns in option selection, not just frequencies.
            Be specific, insightful, and actionable in your recommendations.
            """
        
        else:
            return """
            You are a data science expert analyzing survey data for the metric "{metric_name}".
            
            Here is context about the metric:
            {metric_context}
            
            Here are the statistical analysis results:
            {analysis_results}
            
            Based on this information, provide insightful analysis in the following format:
            
            1. Key Findings: List the 2-3 most important insights about this metric
            2. Data Interpretation: Provide your analysis of what the data reveals
            3. Actionable Recommendations: Suggest 2-3 concrete actions based on these findings
            
            Keep your response focused on the data provided. Explain any patterns you see.
            Be specific, insightful, and actionable in your recommendations.
            """
    
    def _get_cross_metric_prompt(self) -> str:
        """
        Get the prompt template for cross-metric analysis.
        
        Returns:
            Prompt template as a string
        """
        return """
        You are a data science expert analyzing correlations between different survey metrics.
        
        Here is the context about the correlations found:
        {correlation_context}
        
        Based on this information, provide insightful analysis in the following format:
        
        1. Strongest Relationships: Identify the 2-3 most significant correlations and explain what they mean
        2. Potential Causality: For any relationships that might be causal, explain the potential mechanisms
        3. Unexpected Findings: Highlight any surprising or counter-intuitive relationships
        4. Strategic Implications: Explain how these relationships could inform strategy or decision-making
        5. Research Recommendations: Suggest specific follow-up research to better understand these relationships
        
        Focus on meaningful patterns that provide actionable insights, not just statistical significance.
        Be specific about the potential business or organizational implications of these relationships.
        """
    
    def _get_survey_summary_prompt(self) -> str:
        """
        Get the prompt template for comprehensive survey summary.
        
        Returns:
            Prompt template as a string
        """
        return """
        You are an expert data analyst creating an executive summary of a survey's results.
        
        Here is the context about the survey:
        {survey_context}
        
        Based on all the analysis results, provide a comprehensive summary in the following format:
        
        1. Executive Summary: 3-5 sentences that capture the most important overall findings
        
        2. Key Metrics Highlights: For each key metric, provide:
           - One sentence describing the main finding
           - Any significant trends or patterns
        
        3. Correlation Insights: Summarize the 2-3 most important relationships between metrics
        
        4. Strategic Recommendations: Provide 3-5 specific, actionable recommendations based on all findings
        
        5. Future Research: Suggest 2-3 areas where additional data collection or analysis would be valuable
        
        Your summary should be data-driven, concise, and focused on actionable insights that decision-makers can use.
        Avoid general statements that could apply to any survey - be specific to this data.
        """
    
    def _prepare_metric_context(
        self, 
        metric_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        response_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare context for metric insights prompt.
        
        Args:
            metric_data: The metric definition data
            analysis_results: Results from statistical analysis
            response_stats: Response statistics
            
        Returns:
            Dictionary with context data
        """
        metric_context = {
            "metric_name": metric_data.get("name", "Unknown"),
            "metric_description": metric_data.get("description", ""),
            "metric_type": metric_data.get("type", "unknown"),
            "options": metric_data.get("options", []),
            "total_responses": response_stats.get("total_responses", 0),
            "valid_responses": response_stats.get("valid_responses", 0),
            "response_rate": response_stats.get("response_rate", 0)
        }
        
        # Format the context as a string
        context_str = json.dumps(metric_context, indent=2)
        
        # Format the analysis results as a string
        analysis_str = json.dumps(analysis_results, indent=2)
        
        return {
            "metric_name": metric_data.get("name", "Unknown"),
            "metric_context": context_str,
            "analysis_results": analysis_str
        }
    
    def _prepare_cross_metric_context(
        self, 
        correlation_results: Dict[str, Any],
        metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare context for cross-metric insights prompt.
        
        Args:
            correlation_results: Results from correlation analysis
            metrics_data: Dictionary mapping metric IDs to metric definitions
            
        Returns:
            Dictionary with context data
        """
        # Extract significant correlations
        significant_correlations = correlation_results.get("significant_correlations", [])
        
        # Extract causal relationships
        causal_relationships = correlation_results.get("causal_relationships", [])
        
        # Create context dict with metric info
        metrics_info = {}
        for metric_id, metric_data in metrics_data.items():
            metrics_info[metric_id] = {
                "name": metric_data.get("name", "Unknown"),
                "description": metric_data.get("description", ""),
                "type": metric_data.get("type", "unknown")
            }
        
        # Combine into context
        context = {
            "metrics_info": metrics_info,
            "significant_correlations": significant_correlations,
            "causal_relationships": causal_relationships
        }
        
        # Format as string
        context_str = json.dumps(context, indent=2)
        
        return {
            "correlation_context": context_str
        }
    
    def _prepare_survey_summary_context(
        self, 
        survey_data: Dict[str, Any],
        all_analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare context for survey summary prompt.
        
        Args:
            survey_data: Survey metadata
            all_analysis_results: Combined results from all analyses
            
        Returns:
            Dictionary with context data
        """
        # Extract survey metadata
        survey_info = {
            "name": survey_data.get("name", "Unknown Survey"),
            "description": survey_data.get("description", ""),
            "total_respondents": survey_data.get("total_respondents", 0),
            "start_date": survey_data.get("start_date", ""),
            "end_date": survey_data.get("end_date", "")
        }
        
        # Extract metrics info
        metrics_info = {}
        for metric_id, metric_data in survey_data.get("metrics", {}).items():
            # Get metric insights if available
            metric_insights = all_analysis_results.get("metric_insights", {}).get(metric_id, {})
            structured_insights = metric_insights.get("structured_insights", {})
            
            metrics_info[metric_id] = {
                "name": metric_data.get("name", "Unknown"),
                "type": metric_data.get("type", "unknown"),
                "key_findings": structured_insights.get("key_findings", []),
                "recommendations": structured_insights.get("recommendations", [])
            }
        
        # Extract cross-metric insights
        cross_metric_insights = all_analysis_results.get("cross_metric_insights", {})
        significant_correlations = cross_metric_insights.get("structured_insights", {}).get("strongest_relationships", [])
        
        # Combine into context
        context = {
            "survey_info": survey_info,
            "metrics_info": metrics_info,
            "significant_correlations": significant_correlations
        }
        
        # Format as string
        context_str = json.dumps(context, indent=2)
        
        return {
            "survey_context": context_str
        }
    
    async def _generate_completion(
        self, 
        prompt_template: str, 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate completion using the OpenAI API.
        
        Args:
            prompt_template: Prompt template
            context: Context to fill in the template
            
        Returns:
            Generated completion text
        """
        try:
            # Format the prompt with context
            formatted_prompt = prompt_template.format(**context)
            
            # Call the OpenAI API
            response = await self.client.chat.completions.create(
                model=self.completion_model,
                messages=[
                    {"role": "system", "content": "You are a data science expert providing insights on survey data."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            # Extract the completion
            completion = response.choices[0].message.content
            return completion
        
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return "Error generating insights. Please try again later."
    
    def _parse_metric_insights(self, insights: str) -> Dict[str, Any]:
        """
        Parse AI-generated metric insights into structured format.
        
        Args:
            insights: Raw insights text
            
        Returns:
            Structured insights dictionary
        """
        structured_insights = {
            "key_findings": [],
            "interpretation": "",
            "recommendations": []
        }
        
        try:
            # Simple parsing logic - can be enhanced with regex or NLP techniques
            lines = insights.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections
                if "Key Findings:" in line or "Distribution Highlights:" in line or "Option Popularity:" in line:
                    current_section = "key_findings"
                    continue
                elif "Detailed Interpretation:" in line or "Category Analysis:" in line or "Option Combinations:" in line or "Data Interpretation:" in line:
                    current_section = "interpretation"
                    continue
                elif "Notable Segments:" in line or "Segment Insights:" in line:
                    current_section = "segments"
                    continue
                elif "Actionable Recommendations:" in line:
                    current_section = "recommendations"
                    continue
                
                # Add content to appropriate section
                if current_section == "key_findings":
                    # Extract bullet points
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        # Remove bullet or number
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_insights["key_findings"].append(clean_line)
                
                elif current_section == "interpretation":
                    if structured_insights["interpretation"]:
                        structured_insights["interpretation"] += " " + line
                    else:
                        structured_insights["interpretation"] = line
                
                elif current_section == "recommendations":
                    # Extract bullet points
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        # Remove bullet or number
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_insights["recommendations"].append(clean_line)
        
        except Exception as e:
            logger.error(f"Error parsing metric insights: {str(e)}")
        
        return structured_insights
    
    def _parse_cross_metric_insights(self, insights: str) -> Dict[str, Any]:
        """
        Parse AI-generated cross-metric insights into structured format.
        
        Args:
            insights: Raw insights text
            
        Returns:
            Structured insights dictionary
        """
        structured_insights = {
            "strongest_relationships": [],
            "potential_causality": [],
            "unexpected_findings": [],
            "strategic_implications": "",
            "research_recommendations": []
        }
        
        try:
            # Simple parsing logic
            lines = insights.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections
                if "Strongest Relationships:" in line:
                    current_section = "strongest_relationships"
                    continue
                elif "Potential Causality:" in line:
                    current_section = "potential_causality"
                    continue
                elif "Unexpected Findings:" in line:
                    current_section = "unexpected_findings"
                    continue
                elif "Strategic Implications:" in line:
                    current_section = "strategic_implications"
                    continue
                elif "Research Recommendations:" in line:
                    current_section = "research_recommendations"
                    continue
                
                # Add content to appropriate section
                if current_section == "strongest_relationships":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_insights["strongest_relationships"].append(clean_line)
                
                elif current_section == "potential_causality":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_insights["potential_causality"].append(clean_line)
                
                elif current_section == "unexpected_findings":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_insights["unexpected_findings"].append(clean_line)
                
                elif current_section == "strategic_implications":
                    if structured_insights["strategic_implications"]:
                        structured_insights["strategic_implications"] += " " + line
                    else:
                        structured_insights["strategic_implications"] = line
                
                elif current_section == "research_recommendations":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_insights["research_recommendations"].append(clean_line)
        
        except Exception as e:
            logger.error(f"Error parsing cross-metric insights: {str(e)}")
        
        return structured_insights
    
    def _parse_survey_summary(self, summary: str) -> Dict[str, Any]:
        """
        Parse AI-generated survey summary into structured format.
        
        Args:
            summary: Raw summary text
            
        Returns:
            Structured summary dictionary
        """
        structured_summary = {
            "executive_summary": "",
            "key_metrics": [],
            "correlation_insights": [],
            "recommendations": [],
            "future_research": []
        }
        
        try:
            # Simple parsing logic
            lines = summary.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections
                if "Executive Summary:" in line:
                    current_section = "executive_summary"
                    continue
                elif "Key Metrics Highlights:" in line:
                    current_section = "key_metrics"
                    continue
                elif "Correlation Insights:" in line:
                    current_section = "correlation_insights"
                    continue
                elif "Strategic Recommendations:" in line:
                    current_section = "recommendations"
                    continue
                elif "Future Research:" in line:
                    current_section = "future_research"
                    continue
                
                # Add content to appropriate section
                if current_section == "executive_summary":
                    if structured_summary["executive_summary"]:
                        structured_summary["executive_summary"] += " " + line
                    else:
                        structured_summary["executive_summary"] = line
                
                elif current_section == "key_metrics":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_summary["key_metrics"].append(clean_line)
                
                elif current_section == "correlation_insights":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_summary["correlation_insights"].append(clean_line)
                
                elif current_section == "recommendations":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_summary["recommendations"].append(clean_line)
                
                elif current_section == "future_research":
                    if line.startswith(('- ', '• ', '* ')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                        clean_line = line.lstrip('- •*0123456789. ')
                        structured_summary["future_research"].append(clean_line)
        
        except Exception as e:
            logger.error(f"Error parsing survey summary: {str(e)}")
        
        return structured_summary

# Singleton instance
ai_insights_service = AIInsightsService() 