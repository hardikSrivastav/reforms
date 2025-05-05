"""
Metric-specific analysis service for detailed analysis of survey metrics.
This module provides statistical analysis, visualization, and AI-powered insights for metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from scipy import stats
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import json
from collections import Counter
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import logging

from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')


class MetricAnalysisService:
    """Service for analyzing specific metrics with statistical methods and AI insights."""

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the metric analysis service.
        
        Args:
            openai_api_key: OpenAI API key for AI-powered insights
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.model = settings.METRIC_ANALYSIS_MODEL

    async def analyze_metric(
        self, 
        survey_id: int, 
        metric_id: str, 
        metric_data: Dict[str, Any], 
        responses: List[Dict[str, Any]],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a specific metric in detail.
        
        Args:
            survey_id: The survey ID
            metric_id: The metric ID to analyze
            metric_data: The metric definition data
            responses: List of responses for this metric. Each response should have either
                       'value', 'category', or 'text' based on the metric type.
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Dictionary with detailed analysis results
        """
        logger.info(f"Analyzing metric {metric_id} for survey {survey_id}")
        logger.info(f"Metric type: {metric_data.get('type', 'unknown')}, Total responses: {len(responses)}")
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_result = await metadata_store.get_analysis_result("metric_analysis", survey_id, metric_id)
            if cached_result:
                logger.info(f"Using cached analysis for metric {metric_id}")
                return cached_result
        else:
            logger.info(f"Force refresh requested, skipping cache for metric {metric_id}")
        
        # Determine the type of metric and perform appropriate analysis
        metric_type = metric_data.get("type", "numeric")
        
        # Perform statistical analysis based on metric type
        if metric_type == "numeric":
            statistical_analysis = await self._analyze_numeric_metric(metric_data, responses)
            visualizations = await self._generate_numeric_visualizations(metric_data, responses)
        elif metric_type == "categorical":
            statistical_analysis = await self._analyze_categorical_metric(metric_data, responses)
            visualizations = await self._generate_categorical_visualizations(metric_data, responses)
        elif metric_type == "text":
            statistical_analysis = await self._analyze_text_metric(metric_data, responses)
            visualizations = await self._generate_text_visualizations(metric_data, responses)
        elif metric_type == "multi_choice":
            statistical_analysis = await self._analyze_multi_choice_metric(metric_data, responses)
            visualizations = await self._generate_categorical_visualizations(metric_data, responses)
        else:
            # Default to empty analysis for unsupported types
            statistical_analysis = {"type": metric_type, "error": "Unsupported metric type"}
            visualizations = {}
        
        # Generate AI insights
        ai_insights = await self._generate_ai_insights(metric_data, responses, statistical_analysis)
        
        # Compile the result
        result = {
            "survey_id": survey_id,
            "metric_id": metric_id,
            "metric_name": metric_data.get("name", "Unknown"),
            "metric_type": metric_type,
            "response_count": len(responses),
            "timestamp": datetime.now().isoformat(),
            "statistical_analysis": statistical_analysis,
            "visualizations": visualizations,
            "ai_insights": ai_insights
        }
        
        # Store in cache
        await metadata_store.store_analysis_result("metric_analysis", survey_id, result, metric_id)
        
        logger.info(f"Completed analysis for metric {metric_id}")
        return result

    async def _analyze_numeric_metric(
        self, metric_data: Dict[str, Any], responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on numeric metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with numeric values in the 'value' field
            
        Returns:
            Dictionary with statistical analysis
        """
        # Extract numeric values
        values = []
        for response in responses:
            value = response.get("value")
            if isinstance(value, (int, float)):
                values.append(value)
        
        if not values:
            return {"type": "numeric", "error": "No valid numeric values"}
        
        # Convert to numpy array for analysis
        values_array = np.array(values)
        
        # Calculate basic statistics
        stats_result = {
            "type": "numeric",
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std_dev": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75))
        }
        
        # Calculate confidence interval for the mean (95%)
        sem = stats.sem(values_array)
        ci_95 = stats.t.interval(0.95, len(values_array)-1, loc=np.mean(values_array), scale=sem)
        
        stats_result["confidence_interval"] = {
            "level": 95,
            "low": float(ci_95[0]),
            "high": float(ci_95[1])
        }
        
        # Test for normality (Shapiro-Wilk test)
        if len(values) >= 3:  # Shapiro-Wilk requires at least 3 values
            shapiro_test = stats.shapiro(values_array)
            stats_result["normality"] = {
                "test": "shapiro-wilk",
                "statistic": float(shapiro_test[0]),
                "p_value": float(shapiro_test[1]),
                "is_normal": shapiro_test[1] > 0.05
            }
        
        # Calculate distribution characteristics
        stats_result["skewness"] = float(stats.skew(values_array))
        stats_result["kurtosis"] = float(stats.kurtosis(values_array))
        
        # Outlier detection (values outside 1.5 * IQR)
        q1 = stats_result["q1"]
        q3 = stats_result["q3"]
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        stats_result["outliers"] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(values)) * 100 if values else 0,
            "values": outliers[:10]  # Limit to first 10 outliers
        }
        
        return stats_result

    async def _analyze_categorical_metric(
        self, metric_data: Dict[str, Any], responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on categorical metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with categorical values in the 'category' field
            
        Returns:
            Dictionary with statistical analysis
        """
        # Extract categorical values
        categories = []
        for response in responses:
            category = response.get("category")
            if isinstance(category, str):
                categories.append(category)
        
        if not categories:
            return {"type": "categorical", "error": "No valid categorical values"}
        
        # Count occurrences of each category
        category_counts = Counter(categories)
        
        # Calculate frequencies and percentages
        total_responses = len(categories)
        distribution = {}
        
        for category, count in category_counts.items():
            distribution[category] = {
                "count": count,
                "percentage": (count / total_responses) * 100
            }
        
        # Find most and least common categories
        most_common = category_counts.most_common(1)[0][0] if category_counts else None
        least_common = category_counts.most_common()[-1][0] if category_counts else None
        
        # Perform chi-square test for uniform distribution
        expected_freq = total_responses / len(category_counts)
        observed = list(category_counts.values())
        expected = [expected_freq] * len(category_counts)
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        stats_result = {
            "type": "categorical",
            "total_responses": total_responses,
            "unique_categories": len(category_counts),
            "distribution": distribution,
            "most_common": most_common,
            "least_common": least_common,
            "chi_squared_test": {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05
            }
        }
        
        # Calculate entropy (measure of distribution randomness)
        probabilities = [count / total_responses for count in category_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        max_entropy = np.log2(len(category_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        stats_result["entropy"] = {
            "value": float(entropy),
            "max_possible": float(max_entropy),
            "normalized": float(normalized_entropy)
        }
        
        return stats_result

    async def _analyze_multi_choice_metric(
        self, metric_data: Dict[str, Any], responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on multi-choice metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with multi-choice values in the 'value' field as lists
            
        Returns:
            Dictionary with statistical analysis
        """
        # Extract multi-choice values
        values = []
        for response in responses:
            value = response.get("value")
            if isinstance(value, list):
                values.append(value)
            # Handle string representation of list
            elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                try:
                    list_value = json.loads(value)
                    if isinstance(list_value, list):
                        values.append(list_value)
                except:
                    pass
        
        if not values:
            return {"type": "multi_choice", "error": "No valid multi-choice values"}
        
        # Count occurrences of each option
        option_counts = {}
        for value_list in values:
            for option in value_list:
                option_str = str(option)
                if option_str in option_counts:
                    option_counts[option_str] += 1
                else:
                    option_counts[option_str] = 1
        
        # Calculate frequencies and percentages
        total_responses = len(values)
        distribution = {}
        
        for option, count in option_counts.items():
            distribution[option] = {
                "count": count,
                "percentage": (count / total_responses) * 100
            }
        
        # Find most and least common options
        most_common = sorted(option_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        most_common = [{"option": option, "count": count} for option, count in most_common]
        
        # Calculate average selections per response
        avg_selections = sum(len(value_list) for value_list in values) / total_responses if total_responses > 0 else 0
        
        # Analyze co-occurrences between options
        co_occurrences = {}
        for value_list in values:
            if len(value_list) > 1:
                for i, option1 in enumerate(value_list):
                    for option2 in value_list[i+1:]:
                        pair = (str(option1), str(option2))
                        if pair in co_occurrences:
                            co_occurrences[pair] += 1
                        else:
                            co_occurrences[pair] = 1
        
        # Format co-occurrences for output
        co_occurrence_data = []
        for (option1, option2), count in sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:10]:
            co_occurrence_data.append({
                "option1": option1,
                "option2": option2,
                "count": count,
                "percentage": (count / total_responses) * 100
            })
        
        stats_result = {
            "type": "multi_choice",
            "total_responses": total_responses,
            "unique_options": len(option_counts),
            "distribution": distribution,
            "most_common": most_common,
            "average_selections": float(avg_selections),
            "co_occurrences": co_occurrence_data
        }
        
        return stats_result

    async def _analyze_text_metric(
        self, metric_data: Dict[str, Any], responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on text metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with text values in the 'text' field
            
        Returns:
            Dictionary with statistical analysis
        """
        # Extract text values
        texts = []
        for response in responses:
            text = response.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        
        if not texts:
            return {"type": "text", "error": "No valid text responses"}
        
        # Calculate basic text statistics
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        # Word statistics
        stats_result = {
            "type": "text",
            "response_count": len(texts),
            "word_stats": {
                "total_words": sum(word_counts),
                "avg_words": np.mean(word_counts),
                "min_words": min(word_counts),
                "max_words": max(word_counts),
                "std_dev_words": np.std(word_counts)
            },
            "char_stats": {
                "total_chars": sum(char_counts),
                "avg_chars": np.mean(char_counts),
                "min_chars": min(char_counts),
                "max_chars": max(char_counts),
                "std_dev_chars": np.std(char_counts)
            }
        }
        
        # Perform sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(text) for text in texts]
        
        # Average sentiment scores
        avg_sentiment = {
            "compound": np.mean([s["compound"] for s in sentiments]),
            "positive": np.mean([s["pos"] for s in sentiments]),
            "negative": np.mean([s["neg"] for s in sentiments]),
            "neutral": np.mean([s["neu"] for s in sentiments])
        }
        
        # Categorize responses by sentiment
        positive_count = sum(1 for s in sentiments if s["compound"] >= 0.05)
        negative_count = sum(1 for s in sentiments if s["compound"] <= -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        stats_result["sentiment"] = {
            **avg_sentiment,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_percentage": (positive_count / len(sentiments)) * 100,
            "negative_percentage": (negative_count / len(sentiments)) * 100,
            "neutral_percentage": (neutral_count / len(sentiments)) * 100
        }
        
        # Extract common words (exclude stopwords)
        try:
            all_words = []
            stop_words = set(stopwords.words('english'))
            
            for text in texts:
                words = word_tokenize(text.lower())
                # Filter out stopwords, punctuation, and short words
                filtered_words = [word for word in words 
                                  if word not in stop_words
                                  and word.isalnum()
                                  and len(word) > 2]
                all_words.extend(filtered_words)
            
            # Get most common words
            word_freq = Counter(all_words)
            common_words = word_freq.most_common(20)
            
            stats_result["common_words"] = [
                {"word": word, "count": count, "percentage": (count / len(all_words)) * 100}
                for word, count in common_words
            ]
        except Exception as e:
            # Handle case where NLTK resources are not available
            logger.warning(f"Error extracting common words: {str(e)}")
            stats_result["common_words"] = []
        
        return stats_result

    async def _generate_ai_insights(
        self, 
        metric_data: Dict[str, Any], 
        responses: List[Dict[str, Any]], 
        statistical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered insights for the metric.
        
        Args:
            metric_data: The metric definition
            responses: List of responses for this metric
            statistical_analysis: Statistical analysis results
            
        Returns:
            Dictionary with AI-generated insights
        """
        metric_name = metric_data.get("name", "Unknown Metric")
        metric_description = metric_data.get("description", "")
        metric_type = metric_data.get("type", "unknown")
        
        try:
            # Create a prompt based on metric type and statistical analysis
            prompt = f"""
            You are analyzing a survey metric: "{metric_name}" ({metric_description})
            Metric type: {metric_type}
            
            Statistical analysis:
            {json.dumps(statistical_analysis, indent=2)}
            
            Based on this data, please provide:
            1. A concise summary of key findings (2-3 sentences)
            2. The most notable patterns or trends
            3. Actionable recommendations based on the data
            """
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst examining survey metrics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            insight_text = response.choices[0].message.content
            
            # Parse insights into structured format
            structured_insights = self._parse_insights(insight_text)
            
            return {
                "summary": insight_text,
                "structured_insights": structured_insights,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            # Return error if AI insights generation fails
            logger.error(f"Error generating AI insights: {str(e)}")
            return {
                "error": str(e),
                "summary": f"Could not generate AI insights for {metric_name}",
                "generated_at": datetime.now().isoformat()
            }
    
    def _parse_insights(self, insight_text: str) -> Dict[str, Any]:
        """Parse AI-generated insights into a structured format."""
        result = {
            "key_findings": [],
            "patterns": [],
            "recommendations": []
        }
        
        current_section = None
        
        for line in insight_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if "key findings" in line.lower() or "summary" in line.lower():
                current_section = "key_findings"
                continue
            elif "patterns" in line.lower() or "trends" in line.lower() or "notable" in line.lower():
                current_section = "patterns"
                continue
            elif "recommendation" in line.lower() or "action" in line.lower():
                current_section = "recommendations"
                continue
                
            # Add content to the appropriate section
            if current_section:
                # Check if line starts with a bullet point or number
                if line.startswith(("- ", "• ", "* ", "1.", "2.", "3.")):
                    # Remove the bullet point or number
                    clean_line = re.sub(r"^[\-\•\*\d\.]+\s*", "", line)
                    result[current_section].append(clean_line)
                else:
                    # If the section is empty, add the line
                    # Otherwise append to the last item
                    if not result[current_section]:
                        result[current_section].append(line)
                    else:
                        result[current_section][-1] += " " + line
        
        return result

    async def _generate_numeric_visualizations(
        self, 
        metric_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate visualizations for numeric metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with numeric values
            
        Returns:
            Dictionary with visualization data
        """
        # Extract numeric values
        values = []
        for response in responses:
            value = response.get("value")
            if isinstance(value, (int, float)):
                values.append(value)
        
        if not values:
            return {"type": "numeric", "error": "No valid numeric values for visualization"}
        
        # Create histogram
        histogram = self._create_histogram(values)
        
        # Create boxplot
        boxplot = self._create_boxplot(values)
        
        # Calculate histogram bins for frontend rendering
        hist, edges = np.histogram(values, bins='auto')
        
        # Prepare visualization data
        visualization_data = {
            "type": "numeric",
            "histogram": histogram,
            "boxplot": boxplot,
            "chart_data": {
                "values": values,
                "histogram_bins": hist.tolist(),
                "histogram_edges": edges.tolist()
            }
        }
        
        return visualization_data

    async def _generate_categorical_visualizations(
        self, 
        metric_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate visualizations for categorical metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with categorical values
            
        Returns:
            Dictionary with visualization data
        """
        # Extract categorical values
        categories = []
        for response in responses:
            # Check both 'category' and 'value' fields to handle both categorical and multi-choice
            category = response.get("category", response.get("value"))
            
            # For multi-choice, each response might be a list
            if isinstance(category, list):
                categories.extend(category)
            elif isinstance(category, str):
                categories.append(category)
        
        if not categories:
            return {"type": "categorical", "error": "No valid categorical values for visualization"}
        
        # Count occurrences of each category
        counter = Counter(categories)
        
        # Create bar chart
        bar_chart = self._create_bar_chart(counter)
        
        # Create pie chart
        pie_chart = self._create_pie_chart(counter)
        
        # Prepare data for frontend rendering
        categories_list = list(counter.keys())
        counts_list = [counter[cat] for cat in categories_list]
        
        # Prepare visualization data
        visualization_data = {
            "type": "categorical",
            "bar_chart": bar_chart,
            "pie_chart": pie_chart,
            "chart_data": {
                "categories": categories_list,
                "counts": counts_list
            }
        }
        
        return visualization_data

    async def _generate_text_visualizations(
        self, 
        metric_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate visualizations for text metric data.
        
        Args:
            metric_data: The metric definition
            responses: List of responses with text values
            
        Returns:
            Dictionary with visualization data
        """
        # Extract text values
        texts = []
        for response in responses:
            text = response.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        
        if not texts:
            return {"type": "text", "error": "No valid text responses for visualization"}
        
        # Calculate word counts for histogram
        word_counts = [len(text.split()) for text in texts]
        
        # Create word count histogram
        word_count_histogram = self._create_histogram(word_counts)
        
        # Perform sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(text) for text in texts]
        
        # Extract compound scores for visualization
        compound_scores = [s["compound"] for s in sentiments]
        
        # Create sentiment distribution histogram
        sentiment_histogram = self._create_histogram(compound_scores)
        
        # Count sentiment categories
        sentiment_categories = {
            "Positive": sum(1 for s in compound_scores if s >= 0.05),
            "Neutral": sum(1 for s in compound_scores if -0.05 < s < 0.05),
            "Negative": sum(1 for s in compound_scores if s <= -0.05)
        }
        
        # Create sentiment pie chart
        sentiment_pie_chart = self._create_pie_chart(sentiment_categories)
        
        # Extract common words if possible
        word_cloud_data = {}
        try:
            all_words = []
            stop_words = set(stopwords.words('english'))
            
            for text in texts:
                words = word_tokenize(text.lower())
                # Filter out stopwords, punctuation, and short words
                filtered_words = [word for word in words 
                                  if word not in stop_words
                                  and word.isalnum()
                                  and len(word) > 2]
                all_words.extend(filtered_words)
            
            # Get word frequencies
            word_freq = Counter(all_words)
            
            # Convert to format suitable for word cloud
            word_cloud_data = {word: count for word, count in word_freq.most_common(50)}
        except Exception as e:
            # If NLTK resources are not available
            logger.warning(f"Error creating word cloud data: {str(e)}")
            word_cloud_data = {}
        
        # Prepare visualization data
        visualization_data = {
            "type": "text",
            "word_count_histogram": word_count_histogram,
            "sentiment_histogram": sentiment_histogram,
            "sentiment_pie_chart": sentiment_pie_chart,
            "chart_data": {
                "word_counts": word_counts,
                "sentiment_scores": compound_scores,
                "sentiment_categories": dict(sentiment_categories),
                "word_cloud_data": word_cloud_data
            }
        }
        
        return visualization_data

    def _create_histogram(self, values):
        """Create a histogram visualization and return as base64."""
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_boxplot(self, values):
        """Create a boxplot visualization and return as base64."""
        plt.figure(figsize=(8, 5))
        plt.boxplot(values, vert=False, widths=0.7)
        plt.xlabel('Value')
        plt.title('Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_bar_chart(self, counter):
        """Create a bar chart visualization and return as base64."""
        plt.figure(figsize=(10, 6))
        
        # Sort items by count for better visualization
        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*items) if items else ([], [])
        
        # Limit to top 15 categories if there are too many
        if len(labels) > 15:
            labels = labels[:15]
            values = values[:15]
            plt.title('Top 15 Categories')
        else:
            plt.title('Category Distribution')
        
        # Create bar chart
        plt.bar(labels, values, color='skyblue', edgecolor='black')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_pie_chart(self, counter):
        """Create a pie chart visualization and return as base64."""
        plt.figure(figsize=(8, 8))
        
        # Sort items by count
        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*items) if items else ([], [])
        
        # If there are too many categories, group smaller ones as "Other"
        if len(labels) > 8:
            top_labels = labels[:7]
            top_values = values[:7]
            other_value = sum(values[7:])
            
            if other_value > 0:  # Only add "Other" if it has a value
                labels = top_labels + ("Other",)
                values = top_values + (other_value,)
        
        # Create pie chart
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, shadow=False)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Category Distribution')
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str


# Singleton instance
metric_analysis_service = MetricAnalysisService() 