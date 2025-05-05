"""
Cross-metric analysis service for identifying correlations and patterns across metrics.
This module analyzes relationships between different metrics to identify patterns and insights.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import Counter

from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class CrossMetricAnalysisService:
    """Service for analyzing relationships between different metrics."""

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the cross-metric analysis service.
        
        Args:
            openai_api_key: OpenAI API key for AI-powered insights
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.model = settings.CROSS_METRIC_ANALYSIS_MODEL

    async def analyze_metric_correlations(
        self, 
        survey_id: int, 
        metrics_data: List[Dict[str, Any]], 
        responses: List[Dict[str, Any]],
        include_metrics: List[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics in the survey.
        
        Args:
            survey_id: The survey ID
            metrics_data: List of metric definitions
            responses: List of survey responses
            include_metrics: Optional list of metric IDs to include in analysis
            force_refresh: Whether to skip the cache and force a fresh analysis
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info(f"Analyzing cross-metric correlations for survey {survey_id}")
        logger.info(f"Total metrics: {len(metrics_data)}, Total responses: {len(responses)}")
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cache_key = f"correlations_{survey_id}"
            if include_metrics:
                # Sort to ensure consistent caching regardless of order
                cache_key += f"_{'_'.join(sorted(include_metrics))}"
            
            cached_result = await metadata_store.get_analysis_result("cross_metric", survey_id, cache_key)
        if cached_result:
            logger.info(f"Using cached correlation analysis for survey {survey_id}")
            return cached_result
        else:
            logger.info(f"Force refresh requested, skipping cache for cross-metric analysis of survey {survey_id}")
        
        # Filter metrics if include_metrics is specified
        if include_metrics:
            metrics_data = [m for m in metrics_data if m.get("name") in include_metrics]
            if not metrics_data:
                return {"error": "No valid metrics found with the specified names"}
        
        # Prepare data for correlation analysis
        df, metric_types = self._prepare_dataframe(metrics_data, responses)
        
        if df.empty or df.shape[1] < 2:
            return {
                "error": "Insufficient data for correlation analysis",
                "survey_id": survey_id,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate correlations between numeric metrics
        numeric_correlations = await self._analyze_numeric_correlations(df, metric_types)
        
        # Calculate associations between categorical metrics
        categorical_associations = await self._analyze_categorical_associations(df, metric_types)
        
        # Calculate numeric-categorical relationships
        mixed_relationships = await self._analyze_mixed_relationships(df, metric_types)
        
        # Generate visualizations
        visualizations = await self._generate_correlation_visualizations(df, metric_types, numeric_correlations)
        
        # Generate AI insights
        ai_insights = await self._generate_correlation_insights(
            numeric_correlations, 
            categorical_associations,
            mixed_relationships,
            metrics_data
        )
        
        # Prepare result
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "metrics_analyzed": [m.get("name") for m in metrics_data],
            "response_count": len(df),
            "numeric_correlations": numeric_correlations,
            "categorical_associations": categorical_associations,
            "mixed_relationships": mixed_relationships,
            "visualizations": visualizations,
            "ai_insights": ai_insights
        }
        
        # Store in cache
        await metadata_store.store_analysis_result("cross_metric", survey_id, result, cache_key)
        
        logger.info(f"Completed cross-metric correlation analysis for survey {survey_id}")
        return result

    def _prepare_dataframe(
        self, 
        metrics_data: List[Dict[str, Any]], 
        responses: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Prepare a pandas DataFrame for correlation analysis.
        
        Args:
            metrics_data: List of metric definitions
            responses: List of survey responses
            
        Returns:
            Tuple containing:
                - DataFrame with metrics as columns and responses as rows
                - Dictionary mapping metric IDs to their types
        """
        # Create a mapping of metric names to their types
        metric_types = {}
        metric_names = {}
        for metric in metrics_data:
            metric_name = metric.get("name")
            if metric_name:
                metric_types[metric_name] = metric.get("type", "unknown")
                metric_names[metric_name] = metric.get("label", metric_name)
        
        # Create a list to hold the data for each response
        data_rows = []
        
        # Process each response
        for response in responses:
            response_data = {}
            # Extract responses object which contains question_id -> answer mapping
            response_answers = response.get("responses", {})
            
            if not response_answers or not isinstance(response_answers, dict):
                continue
                
            # Process each metric
            for metric in metrics_data:
                metric_name = metric.get("name")
                metric_type = metric_types.get(metric_name, "unknown")
                
                # Get the response value for this metric
                response_value = response_answers.get(metric_name)
                
                if response_value is None:
                    continue
                
                # Process the value based on metric type
                if metric_type == "numeric":
                    try:
                        # Convert to float for numeric metrics
                        response_data[metric_name] = float(response_value)
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        pass
                
                elif metric_type == "categorical":
                    # Store categorical values as strings
                    if response_value:
                        response_data[metric_name] = str(response_value)
                
                elif metric_type == "multi_choice":
                    # For multi-choice, we encode as a string representation of the list
                    if isinstance(response_value, list):
                        response_data[metric_name] = json.dumps(response_value)
                    elif isinstance(response_value, str) and (response_value.startswith('[') and response_value.endswith(']')):
                        # It's already a JSON string representation
                        response_data[metric_name] = response_value
                    
                elif metric_type == "text":
                    # For text, we store as is
                    if response_value and isinstance(response_value, str):
                        response_data[metric_name] = response_value
            
            # Add this response data if it has at least one metric value
            if response_data:
                data_rows.append(response_data)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Log the shape of the resulting DataFrame
        logger.info(f"Created analysis dataframe with shape: {df.shape}")
        
        return df, metric_types

    async def _analyze_numeric_correlations(
        self, 
        df: pd.DataFrame, 
        metric_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Calculate correlations between numeric metrics.
        
        Args:
            df: DataFrame with metrics as columns
            metric_types: Dictionary mapping metric IDs to their types
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Filter numeric columns
        numeric_columns = [col for col in df.columns if metric_types.get(col) == "numeric"]
        
        if len(numeric_columns) < 2:
            return {"insufficient_data": True, "message": "Less than 2 numeric metrics available"}
        
        # Select only numeric columns
        numeric_df = df[numeric_columns].copy()
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr(method='pearson', min_periods=5)
        
        # Convert to dictionary format for JSON serialization
        correlation_data = []
        
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Only include each pair once
                    corr_value = correlation_matrix.loc[col1, col2]
                    
                    # Skip NaN values
                    if pd.isna(corr_value):
                        continue
                    
                    # Calculate p-value
                    if len(numeric_df[col1].dropna()) > 2 and len(numeric_df[col2].dropna()) > 2:
                        # Use pearson correlation test
                        correlation, p_value = stats.pearsonr(
                            numeric_df[col1].dropna(), 
                            numeric_df[col2].dropna()
                        )
                    else:
                        correlation, p_value = corr_value, np.nan
                    
                    # Add to results
                    correlation_data.append({
                        "metric1": col1,
                        "metric2": col2,
                        "correlation": float(corr_value),
                        "p_value": float(p_value) if not np.isnan(p_value) else None,
                        "significant": False if np.isnan(p_value) else p_value < 0.05,
                        "strength": self._interpret_correlation_strength(corr_value)
                    })
        
        # Sort by absolute correlation value
        correlation_data.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "correlations": correlation_data,
            "method": "pearson",
            "matrix": correlation_matrix.to_dict() if not correlation_matrix.empty else {}
        }

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret the strength of a correlation coefficient."""
        abs_corr = abs(correlation)
        
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.5:
            return "moderate"
        elif abs_corr < 0.7:
            return "strong"
        else:
            return "very strong"

    async def _analyze_categorical_associations(
        self, 
        df: pd.DataFrame, 
        metric_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Calculate associations between categorical metrics using Chi-square test.
        
        Args:
            df: DataFrame with metrics as columns
            metric_types: Dictionary mapping metric IDs to their types
            
        Returns:
            Dictionary with association analysis results
        """
        # Get categorical columns (including multi-choice treated as categorical)
        categorical_columns = [col for col in df.columns 
                               if metric_types.get(col) in ["categorical", "multi_choice"]]
        
        if len(categorical_columns) < 2:
            return {"insufficient_data": True, "message": "Less than 2 categorical metrics available"}
        
        association_data = []
        
        # Analyze each pair of categorical variables
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns):
                if i < j:  # Only include each pair once
                    # Create contingency table
                    contingency = pd.crosstab(df[col1].fillna('Missing'), 
                                           df[col2].fillna('Missing'))
                    
                    # Skip if not enough data
                    if contingency.size <= 1 or len(contingency) <= 1 or len(contingency.columns) <= 1:
                        continue
                    
                    # Calculate chi-square test
                    try:
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Calculate Cramer's V (measure of association)
                        n = contingency.sum().sum()
                        phi2 = chi2 / n
                        r, k = contingency.shape
                        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                        rcorr = r - ((r-1)**2)/(n-1)
                        kcorr = k - ((k-1)**2)/(n-1)
                        cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
                        
                        association_data.append({
                            "metric1": col1,
                            "metric2": col2,
                            "chi_square": float(chi2),
                            "p_value": float(p_value),
                            "degrees_of_freedom": int(dof),
                            "significant": p_value < 0.05,
                            "association": float(cramers_v),
                            "strength": self._interpret_association_strength(cramers_v)
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating chi-square for {col1} and {col2}: {str(e)}")
                        continue
        
        # Sort by association strength
        association_data.sort(key=lambda x: x["association"], reverse=True)
        
        return {
            "associations": association_data,
            "method": "chi_square_cramers_v"
        }

    def _interpret_association_strength(self, cramers_v: float) -> str:
        """Interpret the strength of Cramer's V association measure."""
        if cramers_v < 0.1:
            return "negligible"
        elif cramers_v < 0.2:
            return "weak"
        elif cramers_v < 0.3:
            return "moderate"
        elif cramers_v < 0.4:
            return "relatively strong"
        else:
            return "strong"

    async def _analyze_mixed_relationships(
        self, 
        df: pd.DataFrame, 
        metric_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze relationships between numeric and categorical metrics.
        
        Args:
            df: DataFrame with metrics as columns
            metric_types: Dictionary mapping metric IDs to their types
            
        Returns:
            Dictionary with relationship analysis results
        """
        # Get numeric and categorical columns
        numeric_columns = [col for col in df.columns if metric_types.get(col) == "numeric"]
        categorical_columns = [col for col in df.columns 
                              if metric_types.get(col) in ["categorical", "multi_choice"]]
        
        if not numeric_columns or not categorical_columns:
            return {"insufficient_data": True, "message": "Need both numeric and categorical metrics"}
        
        relationship_data = []
        
        # Analyze each numeric-categorical pair
        for num_col in numeric_columns:
            for cat_col in categorical_columns:
                # Group numeric values by category
                grouped_data = {}
                
                # For each category, collect the numeric values
                for category in df[cat_col].dropna().unique():
                    values = df[df[cat_col] == category][num_col].dropna()
                    if len(values) >= 3:  # Need at least 3 values for analysis
                        grouped_data[str(category)] = values.tolist()
                
                if len(grouped_data) >= 2:  # Need at least 2 categories
                    # Perform ANOVA test
                    try:
                        # Convert grouped data to format for ANOVA
                        anova_data = [values for values in grouped_data.values() if len(values) >= 3]
                        
                        if len(anova_data) >= 2:  # Need at least 2 groups for ANOVA
                            f_stat, p_value = stats.f_oneway(*anova_data)
                            
                            # Calculate effect size (eta-squared)
                            groups = [pd.Series(values) for values in anova_data]
                            grand_mean = np.mean([val for group in groups for val in group])
                            n_total = sum(len(group) for group in groups)
                            
                            # Between-group sum of squares
                            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
                            
                            # Total sum of squares
                            ss_total = sum((val - grand_mean)**2 for group in groups for val in group)
                            
                            # Calculate eta-squared
                            eta_squared = ss_between / ss_total if ss_total > 0 else 0
                            
                            relationship_data.append({
                                "numeric_metric": num_col,
                                "categorical_metric": cat_col,
                                "method": "ANOVA",
                                "f_statistic": float(f_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                                "effect_size": float(eta_squared),
                                "strength": self._interpret_effect_size(eta_squared),
                                "categories": len(grouped_data),
                                "category_stats": {
                                    cat: {
                                        "count": len(vals),
                                        "mean": float(np.mean(vals)),
                                        "std_dev": float(np.std(vals)),
                                        "min": float(np.min(vals)),
                                        "max": float(np.max(vals))
                                    }
                                    for cat, vals in grouped_data.items()
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Error in ANOVA for {num_col} by {cat_col}: {str(e)}")
                        continue
        
        # Sort by effect size
        relationship_data.sort(key=lambda x: x["effect_size"], reverse=True)
        
        return {
            "relationships": relationship_data,
            "method": "anova_eta_squared"
        }

    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size for ANOVA."""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"

    async def _generate_correlation_visualizations(
        self, 
        df: pd.DataFrame, 
        metric_types: Dict[str, str],
        numeric_correlations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate visualizations for correlation analysis.
        
        Args:
            df: DataFrame with metrics as columns
            metric_types: Dictionary mapping metric IDs to their types
            numeric_correlations: Results of numeric correlation analysis
            
        Returns:
            Dictionary with visualization data
        """
        visualization_data = {}
        
        # Generate correlation heatmap
        try:
            if "matrix" in numeric_correlations and numeric_correlations["matrix"]:
                # Convert dictionary back to DataFrame
                corr_matrix = pd.DataFrame.from_dict(numeric_correlations["matrix"])
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                
                # Add labels
                plt.colorbar(label='Correlation coefficient')
                plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
                plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
                
                # Add correlation values
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.index)):
                        text = plt.text(i, j, f'{corr_matrix.iloc[j, i]:.2f}',
                                       ha="center", va="center", color="black" if abs(corr_matrix.iloc[j, i]) < 0.7 else "white")
                
                plt.tight_layout()
                
                # Save to base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                visualization_data["correlation_heatmap"] = img_str
        except Exception as e:
            logger.warning(f"Error generating correlation heatmap: {str(e)}")
        
        # Generate top correlations bar chart
        try:
            correlations = numeric_correlations.get("correlations", [])
            if correlations:
                # Take top 10 correlations by absolute value
                top_correlations = sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)[:10]
                
                if top_correlations:
                    plt.figure(figsize=(10, 6))
                    
                    # Prepare data
                    labels = [f"{c['metric1']} vs {c['metric2']}" for c in top_correlations]
                    values = [c["correlation"] for c in top_correlations]
                    colors = ['#ff9999' if v < 0 else '#66b3ff' for v in values]
                    
                    # Create horizontal bar chart
                    plt.barh(range(len(labels)), values, color=colors)
                    plt.yticks(range(len(labels)), labels)
                    plt.xlabel('Correlation coefficient')
                    plt.title('Top 10 Correlations by Absolute Value')
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.grid(axis='x', alpha=0.3)
                    
                    # Add values to bars
                    for i, v in enumerate(values):
                        plt.text(v + (0.01 if v >= 0 else -0.01), 
                                i, 
                                f'{v:.2f}', 
                                va='center', 
                                ha='left' if v >= 0 else 'right')
        except:
            logger.warning("Error generating top correlations chart")
        
        plt.tight_layout()
        
                    # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
                    
        visualization_data["top_correlations_chart"] = img_str
        
        return visualization_data

    async def _generate_correlation_insights(
        self, 
        numeric_correlations: Dict[str, Any],
        categorical_associations: Dict[str, Any],
        mixed_relationships: Dict[str, Any],
        metrics_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered insights for correlation analysis.
        
        Args:
            numeric_correlations: Results of numeric correlation analysis
            categorical_associations: Results of categorical association analysis
            mixed_relationships: Results of mixed relationship analysis
            metrics_data: List of metric definitions
            
        Returns:
            Dictionary with AI-generated insights
        """
        # Create metric name mapping
        metric_names = {m.get("name"): m.get("label", m.get("name", "")) for m in metrics_data}
        
        # Get top correlations, associations, and relationships
        top_correlations = numeric_correlations.get("correlations", [])[:5]
        top_associations = categorical_associations.get("associations", [])[:5]
        top_relationships = mixed_relationships.get("relationships", [])[:5]
        
        if not top_correlations and not top_associations and not top_relationships:
            return {
                "error": "Insufficient data for generating insights",
                "summary": "Not enough significant relationships found between metrics.",
                "generated_at": datetime.now().isoformat()
            }
        
        try:
            # Create a prompt based on the analysis results
            prompt = f"""
            You are analyzing relationships between survey metrics.
            
            Top numeric correlations:
            {json.dumps([{
                "metric1": metric_names.get(c["metric1"], c["metric1"]),
                "metric2": metric_names.get(c["metric2"], c["metric2"]),
                "correlation": c["correlation"],
                "significant": c["significant"],
                "strength": c["strength"]
            } for c in top_correlations], indent=2)}
            
            Top categorical associations:
            {json.dumps([{
                "metric1": metric_names.get(a["metric1"], a["metric1"]),
                "metric2": metric_names.get(a["metric2"], a["metric2"]),
                "association": a["association"],
                "significant": a["significant"],
                "strength": a["strength"]
            } for a in top_associations], indent=2)}
            
            Top numeric-categorical relationships:
            {json.dumps([{
                "numeric_metric": metric_names.get(r["numeric_metric"], r["numeric_metric"]),
                "categorical_metric": metric_names.get(r["categorical_metric"], r["categorical_metric"]),
                "effect_size": r["effect_size"],
                "significant": r["significant"],
                "strength": r["strength"]
            } for r in top_relationships], indent=2)}
            
            Based on this data, please provide:
            1. A concise summary of the most important relationships between metrics (3-4 sentences)
            2. Key insights about how these metrics interact
            3. Actionable recommendations based on these relationships
            """
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst examining relationships between survey metrics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            insight_text = response.choices[0].message.content
            
            # Parse insights into structured format
            structured_insights = self._parse_correlation_insights(insight_text)
            
            return {
                "summary": insight_text,
                "structured_insights": structured_insights,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            # Return error if AI insights generation fails
            logger.error(f"Error generating correlation insights: {str(e)}")
            return {
                "error": str(e),
                "summary": "Could not generate insights for correlations analysis",
                "generated_at": datetime.now().isoformat()
            }
    
    def _parse_correlation_insights(self, insight_text: str) -> Dict[str, Any]:
        """Parse AI-generated correlation insights into a structured format."""
        result = {
            "summary": [],
            "key_insights": [],
            "recommendations": []
        }
        
        current_section = "summary"  # Default to summary for initial text
        
        for line in insight_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if any(keyword in line.lower() for keyword in ["key insight", "important relationship", "notable correlation"]):
                current_section = "key_insights"
                continue
            elif any(keyword in line.lower() for keyword in ["recommend", "action", "consider", "suggest"]):
                current_section = "recommendations"
                continue
            elif "summary" in line.lower():
                current_section = "summary"
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


# Singleton instance
cross_metric_analysis_service = CrossMetricAnalysisService() 