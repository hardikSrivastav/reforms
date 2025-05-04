"""
Multi-modal AI analysis service.
Provides advanced AI-powered analysis for survey data using sophisticated prompt engineering.
"""

import logging
import json
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime
from enum import Enum
import os

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai.types.chat import ChatCompletionMessageParam

from ..config import settings
from ..services.metadata_store import metadata_store

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    TEXT_SENTIMENT = "text_sentiment"
    NUMERIC_TREND = "numeric_trend"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    SEGMENT_COMPARISON = "segment_comparison"
    FREE_RESPONSE = "free_response"
    IMAGE_ANALYSIS = "image_analysis"
    MULTI_MODAL = "multi_modal"


class AIModel(Enum):
    """AI models that can be used for analysis."""
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4_VISION = "gpt-4-vision-preview"


class MultimodalAIAnalysisService:
    """
    Advanced AI-powered analysis service with multi-modal capabilities and sophisticated prompt engineering.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the service.
        
        Args:
            api_key: OpenAI API key. If not provided, will use settings.OPENAI_API_KEY.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.cache_ttl = settings.ANALYSIS_CACHE_TTL
        logger.info(f"Initialized multi-modal AI analysis service")
        
        # Load prompt templates
        self.templates = {
            AnalysisType.TEXT_SENTIMENT: self._load_template("text_sentiment"),
            AnalysisType.NUMERIC_TREND: self._load_template("numeric_trend"),
            AnalysisType.CORRELATION: self._load_template("correlation"),
            AnalysisType.ANOMALY: self._load_template("anomaly"),
            AnalysisType.SEGMENT_COMPARISON: self._load_template("segment_comparison"),
            AnalysisType.FREE_RESPONSE: self._load_template("free_response"),
            AnalysisType.IMAGE_ANALYSIS: self._load_template("image_analysis"),
            AnalysisType.MULTI_MODAL: self._load_template("multi_modal"),
        }
        
    def _load_template(self, template_name: str) -> str:
        """
        Load a prompt template from the templates directory.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Template string
        """
        try:
            template_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "templates", 
                f"{template_name}_template.txt"
            )
            
            if os.path.exists(template_path):
                with open(template_path, "r") as f:
                    return f.read()
            else:
                # Return a default template if file doesn't exist
                logger.warning(f"Template {template_name} not found, using default")
                return self._get_default_template(template_name)
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {str(e)}")
            return self._get_default_template(template_name)
    
    def _get_default_template(self, analysis_type: str) -> str:
        """
        Get a default template for the given analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Default template string
        """
        # Simple default templates
        defaults = {
            "text_sentiment": """
                You are an expert data scientist analyzing survey response sentiment.
                
                Instructions:
                1. Analyze the sentiment of each response
                2. Categorize into positive, negative, or neutral
                3. Identify key themes and emotional drivers
                4. Provide a summary of overall sentiment
                5. Return a structured JSON with your findings
                
                Survey question: {{question}}
                
                Responses:
                {{responses}}
                
                Format your response as a JSON object with these fields:
                {
                    "sentiment_distribution": {"positive": float, "neutral": float, "negative": float},
                    "key_themes": [{"theme": string, "sentiment": string, "frequency": float}],
                    "emotional_drivers": [{"driver": string, "impact": string}],
                    "summary": string,
                    "recommendations": [string]
                }
            """,
            
            "numeric_trend": """
                You are an expert data scientist analyzing numeric survey data trends.
                
                Instructions:
                1. Analyze the time series data for patterns
                2. Identify significant trends and turning points
                3. Detect seasonality and cyclical patterns
                4. Forecast future values if possible
                5. Return a structured JSON with your findings
                
                Survey question: {{question}}
                
                Time series data:
                {{data}}
                
                Format your response as a JSON object with these fields:
                {
                    "trend_direction": string,
                    "trend_strength": float,
                    "significant_changes": [{"period": string, "change": float, "explanation": string}],
                    "seasonality": {"detected": boolean, "pattern": string},
                    "forecast": [{"period": string, "value": float, "confidence": float}],
                    "summary": string,
                    "recommendations": [string]
                }
            """,
            
            # Other default templates would go here...
            
            "multi_modal": """
                You are an expert data scientist analyzing multi-modal survey data.
                
                Instructions:
                1. Analyze both text responses and image data
                2. Identify patterns and insights across modalities
                3. Note any contradictions between text and visual data
                4. Provide an integrated analysis
                5. Return a structured JSON with your findings
                
                Survey question: {{question}}
                
                Text responses:
                {{text_data}}
                
                Image analysis:
                {{image_data}}
                
                Format your response as a JSON object with these fields:
                {
                    "cross_modal_themes": [{"theme": string, "text_support": float, "image_support": float}],
                    "contradictions": [{"topic": string, "text_signal": string, "image_signal": string}],
                    "integrated_insights": [string],
                    "summary": string,
                    "recommendations": [string]
                }
            """
        }
        
        return defaults.get(analysis_type, "Please provide detailed analysis of the data.")
    
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _generate_ai_response(
        self, 
        messages: List[ChatCompletionMessageParam],
        model: AIModel = AIModel.GPT4,
        temperature: float = 0.0,
        functions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a response from the AI using the OpenAI API.
        
        Args:
            messages: List of messages to send to the API
            model: Model to use
            temperature: Temperature for generation (0.0 = deterministic)
            functions: Optional function calling specifications
            
        Returns:
            AI response text
        """
        try:
            # Prepare the arguments for the API call
            kwargs = {
                "model": model.value,
                "messages": messages,
                "temperature": temperature,
            }
            
            # Add function calling if provided
            if functions:
                kwargs["functions"] = functions
                kwargs["function_call"] = "auto"
            
            # Make the API call
            response = await self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    async def analyze_text_sentiment(
        self,
        survey_id: int,
        question_id: str,
        question_text: str,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the sentiment of text responses.
        
        Args:
            survey_id: ID of the survey
            question_id: ID of the question
            question_text: Text of the question
            responses: List of text responses
            
        Returns:
            Sentiment analysis results
        """
        # Check cache first
        cache_key = f"sentiment:{survey_id}:{question_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached sentiment analysis for question {question_id}")
            return cached_result
        
        # Prepare the prompt
        template = self.templates[AnalysisType.TEXT_SENTIMENT]
        prompt = template.replace("{{question}}", question_text)
        
        # Format the responses for the prompt
        formatted_responses = "\n".join([
            f"- {r.get('text', '')}" for r in responses if r.get('text')
        ])
        prompt = prompt.replace("{{responses}}", formatted_responses)
        
        # Define the schema for structured output
        functions = [{
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of survey responses",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment_distribution": {
                        "type": "object",
                        "properties": {
                            "positive": {"type": "number"},
                            "neutral": {"type": "number"},
                            "negative": {"type": "number"}
                        }
                    },
                    "key_themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "sentiment": {"type": "string"},
                                "frequency": {"type": "number"}
                            }
                        }
                    },
                    "emotional_drivers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "driver": {"type": "string"},
                                "impact": {"type": "string"}
                            }
                        }
                    },
                    "summary": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }]
        
        # Create the messages
        messages = [
            {"role": "system", "content": "You are an expert data scientist specializing in sentiment analysis of survey responses."},
            {"role": "user", "content": prompt}
        ]
        
        # Get AI response with structured output
        ai_response = await self._generate_ai_response(messages, AIModel.GPT4, 0.0, functions)
        
        try:
            # Parse the JSON response
            if isinstance(ai_response, str):
                # Extract JSON if it's wrapped in markdown or text
                if "```json" in ai_response:
                    json_str = ai_response.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                elif "```" in ai_response:
                    json_str = ai_response.split("```")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                else:
                    result = json.loads(ai_response)
            else:
                # If it's already a dict/object
                result = ai_response
                
            # Add metadata
            result["survey_id"] = survey_id
            result["question_id"] = question_id
            result["analysis_type"] = AnalysisType.TEXT_SENTIMENT.value
            result["timestamp"] = datetime.now().isoformat()
            
            # Cache the result
            await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
            
            return result
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "error": f"Failed to parse AI response: {str(e)}",
                "survey_id": survey_id,
                "question_id": question_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_numeric_trends(
        self,
        survey_id: int,
        question_id: str,
        question_text: str,
        time_series_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze numeric trends in time series data.
        
        Args:
            survey_id: ID of the survey
            question_id: ID of the question
            question_text: Text of the question
            time_series_data: List of time series data points
            
        Returns:
            Trend analysis results
        """
        # Check cache first
        cache_key = f"numeric_trend:{survey_id}:{question_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached numeric trend analysis for question {question_id}")
            return cached_result
        
        # Prepare the prompt
        template = self.templates[AnalysisType.NUMERIC_TREND]
        prompt = template.replace("{{question}}", question_text)
        
        # Format the time series data for the prompt
        formatted_data = "\n".join([
            f"- Period: {d.get('period', '')}, Value: {d.get('value', '')}" 
            for d in time_series_data
        ])
        prompt = prompt.replace("{{data}}", formatted_data)
        
        # Define the schema for structured output
        functions = [{
            "name": "analyze_numeric_trend",
            "description": "Analyze numeric trends in time series data",
            "parameters": {
                "type": "object",
                "properties": {
                    "trend_direction": {"type": "string"},
                    "trend_strength": {"type": "number"},
                    "significant_changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "period": {"type": "string"},
                                "change": {"type": "number"},
                                "explanation": {"type": "string"}
                            }
                        }
                    },
                    "seasonality": {
                        "type": "object",
                        "properties": {
                            "detected": {"type": "boolean"},
                            "pattern": {"type": "string"}
                        }
                    },
                    "forecast": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "period": {"type": "string"},
                                "value": {"type": "number"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "summary": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }]
        
        # Create the messages
        messages = [
            {"role": "system", "content": "You are an expert data scientist specializing in time series analysis and forecasting."},
            {"role": "user", "content": prompt}
        ]
        
        # Get AI response with structured output
        ai_response = await self._generate_ai_response(messages, AIModel.GPT4, 0.0, functions)
        
        try:
            # Parse the JSON response
            if isinstance(ai_response, str):
                # Extract JSON if it's wrapped in markdown or text
                if "```json" in ai_response:
                    json_str = ai_response.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                elif "```" in ai_response:
                    json_str = ai_response.split("```")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                else:
                    result = json.loads(ai_response)
            else:
                # If it's already a dict/object
                result = ai_response
                
            # Add metadata
            result["survey_id"] = survey_id
            result["question_id"] = question_id
            result["analysis_type"] = AnalysisType.NUMERIC_TREND.value
            result["timestamp"] = datetime.now().isoformat()
            
            # Cache the result
            await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
            
            return result
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "error": f"Failed to parse AI response: {str(e)}",
                "survey_id": survey_id,
                "question_id": question_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_free_responses(
        self,
        survey_id: int,
        question_id: str,
        question_text: str,
        responses: List[Dict[str, Any]],
        guidance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze free-text responses using advanced prompt engineering techniques.
        
        Args:
            survey_id: ID of the survey
            question_id: ID of the question
            question_text: Text of the question
            responses: List of text responses
            guidance: Optional guidance for the analysis
            
        Returns:
            Free response analysis results
        """
        # Check cache first
        cache_key = f"free_response:{survey_id}:{question_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached free response analysis for question {question_id}")
            return cached_result
        
        # Prepare the prompt with chain-of-thought reasoning
        template = self.templates[AnalysisType.FREE_RESPONSE]
        prompt = template.replace("{{question}}", question_text)
        
        # Format the responses for the prompt
        formatted_responses = "\n".join([
            f"- {r.get('text', '')}" for r in responses if r.get('text')
        ])
        prompt = prompt.replace("{{responses}}", formatted_responses)
        
        # Add guidance if provided
        if guidance:
            prompt += f"\n\nSpecial guidance for this analysis: {guidance}"
        
        # Create system message that encourages chain-of-thought reasoning
        system_message = """
        You are an expert qualitative researcher specializing in analyzing survey responses. 
        
        Approach this analysis using the following steps:
        1. First, read all responses carefully to understand the overall themes
        2. Group responses into coherent categories and note their frequencies
        3. Identify key patterns, surprising insights, and outliers
        4. Consider how the responses relate to the question being asked
        5. Synthesize your findings into clear, actionable insights
        
        Explain your thought process for each step before providing your final analysis.
        """
        
        # Define the schema for structured output
        functions = [{
            "name": "analyze_free_responses",
            "description": "Analyze free-text survey responses",
            "parameters": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "frequency": {"type": "number"},
                                "sample_responses": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "insights": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "insight": {"type": "string"},
                                "supporting_evidence": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "outliers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "response": {"type": "string"},
                                "reason_for_flagging": {"type": "string"}
                            }
                        }
                    },
                    "summary": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "thought_process": {"type": "string"}
                }
            }
        }]
        
        # Create the messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Get AI response with structured output
        ai_response = await self._generate_ai_response(messages, AIModel.GPT4, 0.2, functions)
        
        try:
            # Parse the JSON response
            if isinstance(ai_response, str):
                # Extract JSON if it's wrapped in markdown or text
                if "```json" in ai_response:
                    json_str = ai_response.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                elif "```" in ai_response:
                    json_str = ai_response.split("```")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                else:
                    # Check if the entire response is JSON
                    try:
                        result = json.loads(ai_response)
                    except:
                        # Extract the thought process and convert to a structured format
                        thought_process = ai_response
                        result = {
                            "themes": [],
                            "insights": [],
                            "outliers": [],
                            "summary": "AI provided a narrative response instead of structured data",
                            "recommendations": [],
                            "thought_process": thought_process
                        }
            else:
                # If it's already a dict/object
                result = ai_response
                
            # Add metadata
            result["survey_id"] = survey_id
            result["question_id"] = question_id
            result["analysis_type"] = AnalysisType.FREE_RESPONSE.value
            result["timestamp"] = datetime.now().isoformat()
            
            # Cache the result
            await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
            
            return result
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "error": f"Failed to parse AI response: {str(e)}",
                "survey_id": survey_id,
                "question_id": question_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_image_responses(
        self,
        survey_id: int,
        question_id: str,
        question_text: str,
        image_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze image-based responses using GPT-4 Vision.
        
        Args:
            survey_id: ID of the survey
            question_id: ID of the question
            question_text: Text of the question
            image_responses: List of image responses with base64 encoded images
            
        Returns:
            Image analysis results
        """
        # Check cache first
        cache_key = f"image_analysis:{survey_id}:{question_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached image analysis for question {question_id}")
            return cached_result
        
        # For image analysis, we need to build a different type of message structure
        messages = [
            {
                "role": "system",
                "content": "You are an expert in analyzing visual survey responses. Extract insights from these images while considering the survey question context."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Survey question: {question_text}\n\nPlease analyze these images submitted in response to the question. Identify common themes, patterns, and insights."
                    }
                ]
            }
        ]
        
        # Add image content to the user message
        for idx, img_response in enumerate(image_responses[:10]):  # Limit to 10 images
            if "image_data" in img_response and img_response["image_data"]:
                # Add image to the last user message content
                messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_response['image_data']}",
                        "detail": "high"
                    }
                })
        
        # Get AI response
        ai_response = await self._generate_ai_response(
            messages, 
            AIModel.GPT4_VISION, 
            0.2
        )
        
        # Structure the response
        try:
            result = {
                "survey_id": survey_id,
                "question_id": question_id,
                "analysis_type": AnalysisType.IMAGE_ANALYSIS.value,
                "timestamp": datetime.now().isoformat(),
                "image_count": len(image_responses),
                "raw_analysis": ai_response,
                "structured_analysis": self._extract_structured_image_analysis(ai_response)
            }
            
            # Cache the result
            await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
            
            return result
        except Exception as e:
            logger.error(f"Error processing image analysis: {str(e)}")
            return {
                "error": f"Failed to process image analysis: {str(e)}",
                "survey_id": survey_id,
                "question_id": question_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_structured_image_analysis(self, raw_analysis: str) -> Dict[str, Any]:
        """
        Extract structured data from the raw image analysis text.
        
        Args:
            raw_analysis: Raw analysis text from GPT-4 Vision
            
        Returns:
            Structured analysis data
        """
        # Try to parse as JSON if it looks like JSON
        if raw_analysis.strip().startswith("{") and raw_analysis.strip().endswith("}"):
            try:
                return json.loads(raw_analysis)
            except:
                pass
                
        # Otherwise, extract key sections from the text
        structured = {
            "themes": [],
            "objects": [],
            "emotions": [],
            "summary": "",
            "recommendations": []
        }
        
        # Simple extraction based on section headers
        sections = {
            "themes": ["themes", "common themes", "patterns"],
            "objects": ["objects", "items", "elements"],
            "emotions": ["emotions", "sentiment", "mood"],
            "summary": ["summary", "conclusion", "overall"],
            "recommendations": ["recommendations", "suggestions", "next steps"]
        }
        
        lines = raw_analysis.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header
            lower_line = line.lower()
            for section, keywords in sections.items():
                if any(k in lower_line for k in keywords) and (":" in line or line.endswith(".")):
                    current_section = section
                    # Extract content after colon if present
                    if ":" in line:
                        content = line.split(":", 1)[1].strip()
                        if content and current_section != "summary":
                            structured[current_section].append(content)
                        elif content and current_section == "summary":
                            structured[current_section] = content
                    break
            
            # Add content to current section if we have one
            if current_section:
                # Skip lines that look like headers
                header_markers = ["#", "-", "*", "1.", "2."]
                is_header = any(line.lower().startswith(k) for k in header_markers)
                
                if not is_header:
                    if current_section == "summary":
                        structured[current_section] += " " + line
                    elif line.startswith("-") or line.startswith("•"):
                        structured[current_section].append(line[1:].strip())
                    
        return structured
    
    async def analyze_multi_modal(
        self,
        survey_id: int,
        question_id: str,
        question_text: str,
        text_responses: List[Dict[str, Any]],
        image_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform integrated analysis of both text and image responses.
        
        Args:
            survey_id: ID of the survey
            question_id: ID of the question
            question_text: Text of the question
            text_responses: List of text responses
            image_responses: List of image responses with base64 encoded images
            
        Returns:
            Multi-modal analysis results
        """
        # Check cache first
        cache_key = f"multi_modal:{survey_id}:{question_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached multi-modal analysis for question {question_id}")
            return cached_result
        
        # First, get separate analyses for text and images
        text_analysis_task = self.analyze_free_responses(
            survey_id, 
            question_id, 
            question_text, 
            text_responses
        )
        
        image_analysis_task = self.analyze_image_responses(
            survey_id,
            question_id,
            question_text,
            image_responses
        )
        
        # Run the analyses in parallel
        text_analysis, image_analysis = await asyncio.gather(
            text_analysis_task,
            image_analysis_task
        )
        
        # Now prepare a multi-modal integration analysis
        template = self.templates[AnalysisType.MULTI_MODAL]
        prompt = template.replace("{{question}}", question_text)
        
        # Prepare text data summary
        if "error" in text_analysis:
            text_data = "No text analysis available due to error."
        else:
            themes = "\n".join([f"- {t.get('theme', '')}" for t in text_analysis.get("themes", [])])
            insights = "\n".join([f"- {i.get('insight', '')}" for i in text_analysis.get("insights", [])])
            text_data = f"Text themes:\n{themes}\n\nText insights:\n{insights}\n\nText summary: {text_analysis.get('summary', '')}"
        
        prompt = prompt.replace("{{text_data}}", text_data)
        
        # Prepare image data summary
        if "error" in image_analysis:
            image_data = "No image analysis available due to error."
        else:
            structured = image_analysis.get("structured_analysis", {})
            image_themes = "\n".join([f"- {t}" for t in structured.get("themes", [])])
            image_objects = "\n".join([f"- {o}" for o in structured.get("objects", [])])
            image_data = f"Image themes:\n{image_themes}\n\nCommon objects:\n{image_objects}\n\nImage summary: {structured.get('summary', '')}"
        
        prompt = prompt.replace("{{image_data}}", image_data)
        
        # Define the schema for structured output
        functions = [{
            "name": "integrate_multi_modal_analysis",
            "description": "Integrate text and image analysis results",
            "parameters": {
                "type": "object",
                "properties": {
                    "cross_modal_themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "text_support": {"type": "number"},
                                "image_support": {"type": "number"}
                            }
                        }
                    },
                    "contradictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "text_signal": {"type": "string"},
                                "image_signal": {"type": "string"}
                            }
                        }
                    },
                    "integrated_insights": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "summary": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }]
        
        # Create the messages
        messages = [
            {"role": "system", "content": "You are an expert in multi-modal analysis of survey data, capable of integrating insights from both text and image responses."},
            {"role": "user", "content": prompt}
        ]
        
        # Get AI response with structured output
        ai_response = await self._generate_ai_response(messages, AIModel.GPT4, 0.0, functions)
        
        try:
            # Parse the JSON response
            if isinstance(ai_response, str):
                # Extract JSON if it's wrapped in markdown or text
                if "```json" in ai_response:
                    json_str = ai_response.split("```json")[1].split("```")[0].strip()
                    integrated_result = json.loads(json_str)
                elif "```" in ai_response:
                    json_str = ai_response.split("```")[1].split("```")[0].strip()
                    integrated_result = json.loads(json_str)
                else:
                    integrated_result = json.loads(ai_response)
            else:
                # If it's already a dict/object
                integrated_result = ai_response
                
            # Create the final result with all components
            result = {
                "survey_id": survey_id,
                "question_id": question_id,
                "analysis_type": AnalysisType.MULTI_MODAL.value,
                "timestamp": datetime.now().isoformat(),
                "text_analysis": text_analysis,
                "image_analysis": image_analysis,
                "integrated_analysis": integrated_result
            }
            
            # Cache the result
            await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
            
            return result
        except Exception as e:
            logger.error(f"Error parsing multi-modal AI response: {str(e)}")
            return {
                "error": f"Failed to parse multi-modal AI response: {str(e)}",
                "survey_id": survey_id,
                "question_id": question_id,
                "timestamp": datetime.now().isoformat(),
                "text_analysis": text_analysis,
                "image_analysis": image_analysis
            }


# Create a singleton instance
multimodal_ai_analysis_service = MultimodalAIAnalysisService() 