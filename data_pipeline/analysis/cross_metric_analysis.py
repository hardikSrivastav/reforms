"""
Cross-metric analysis service for analyzing relationships between different metrics in survey data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import json
from datetime import datetime

from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.analysis.vector_trend_analysis import vector_trend_analysis_service


class CrossMetricAnalysisService:
    """Service for analyzing relationships between different metrics in survey data."""

    def __init__(self):
        """Initialize the CrossMetricAnalysisService."""
        self.correlation_threshold = 0.3  # Threshold for considering correlations significant
        self.knowledge_graph = {}  # Store relationships between metrics

    async def analyze_cross_metric_correlations(
        self, survey_id: int, metrics_data: Dict[str, Any], survey_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics in a survey.
        
        Args:
            survey_id: The ID of the survey
            metrics_data: Dictionary of metric definitions
            survey_responses: List of survey responses with metric values
            
        Returns:
            Dictionary containing cross-metric correlation results
        """
        # Convert metrics_data dict to list for compatibility with analyze_cross_metrics
        metrics_list = []
        for metric_id, metric_info in metrics_data.items():
            metric_info["id"] = metric_id
            metrics_list.append(metric_info)
            
        return await self.analyze_cross_metrics(survey_id, metrics_list, survey_responses)

    async def analyze_cross_metrics(
        self, survey_id: int, metrics_data: List[Dict[str, Any]], survey_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze relationships between different metrics in a survey.
        
        Args:
            survey_id: The ID of the survey
            metrics_data: List of metric definitions
            survey_responses: List of survey responses with metric values
            
        Returns:
            Dictionary containing cross-metric analysis results
        """
        # Check if we have cached results
        cached_result = await metadata_store.get_analysis_result("cross_metric_analysis", survey_id)
        
        if cached_result:
            return cached_result
            
        # Calculate correlations between numeric metrics
        correlation_data = await self._calculate_correlations(metrics_data, survey_responses)
        
        # Analyze each pair of metrics
        pairwise_analyses = []
        
        # Get numeric metrics
        numeric_metrics = [m for m in metrics_data if m.get("type") == "numeric"]
        
        # Analyze all possible numeric metric pairs
        for i in range(len(numeric_metrics)):
            for j in range(i + 1, len(numeric_metrics)):
                metric1 = numeric_metrics[i]
                metric2 = numeric_metrics[j]
                
                pair_analysis = await self._analyze_metric_pair(metric1, metric2, survey_responses)
                pairwise_analyses.append(pair_analysis)
        
        # Analyze numeric-categorical pairs
        numeric_metrics = [m for m in metrics_data if m.get("type") == "numeric"]
        categorical_metrics = [m for m in metrics_data if m.get("type") == "categorical"]
        
        for numeric_metric in numeric_metrics:
            for categorical_metric in categorical_metrics:
                pair_analysis = await self._analyze_metric_pair(
                    numeric_metric, categorical_metric, survey_responses
                )
                pairwise_analyses.append(pair_analysis)
        
        # Generate key insights
        key_insights = await self._generate_key_insights(correlation_data, pairwise_analyses)
        
        # Generate visualizations
        visualizations = await self._generate_visualizations(correlation_data)
        
        # Generate hypotheses about relationships
        hypotheses = await self._generate_hypotheses(pairwise_analyses)
        
        # Create knowledge graph representation of relationships
        knowledge_graph = await self._build_knowledge_graph(pairwise_analyses, survey_id)
        
        # Integrate with vector-based analysis if available
        vector_insights = await self._integrate_vector_analysis(survey_id, metrics_data)
        
        # Prepare final result
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "correlation_matrix": correlation_data,
            "pairwise_analyses": pairwise_analyses,
            "key_insights": key_insights,
            "visualizations": visualizations,
            "hypotheses": hypotheses,
            "knowledge_graph": knowledge_graph,
            "vector_insights": vector_insights
        }
        
        # Store in cache
        await metadata_store.store_analysis_result("cross_metric_analysis", survey_id, result)
        
        return result

    async def _calculate_correlations(
        self, metrics_data: List[Dict[str, Any]], survey_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix between numeric metrics.
        
        Args:
            metrics_data: List of metric definitions
            survey_responses: List of survey responses with metric values
            
        Returns:
            Dictionary with correlation matrix and metric information
        """
        # Filter to numeric metrics only
        numeric_metrics = [m for m in metrics_data if m.get("type") == "numeric"]
        if not numeric_metrics:
            return {"metrics": [], "matrix": []}
            
        # Extract numeric metric values
        metric_values = {}
        for metric in numeric_metrics:
            metric_id = metric["id"]
            values = []
            
            for response in survey_responses:
                if "metrics" in response and metric_id in response["metrics"]:
                    value = response["metrics"][metric_id]
                    if isinstance(value, (int, float)):
                        values.append(value)
                    
            if values:
                metric_values[metric_id] = values
        
        # Ensure we have values for at least two metrics
        if len(metric_values) < 2:
            return {"metrics": numeric_metrics, "matrix": [[1.0] * len(numeric_metrics)] * len(numeric_metrics)}
        
        # Create a dataframe for correlation analysis
        df_data = {}
        metric_ids = []
        
        for metric in numeric_metrics:
            metric_id = metric["id"]
            if metric_id in metric_values and len(metric_values[metric_id]) > 0:
                metric_ids.append(metric_id)
                df_data[metric_id] = metric_values[metric_id]
        
        # Check if we have enough data
        if len(metric_ids) < 2:
            return {"metrics": numeric_metrics, "matrix": [[1.0] * len(numeric_metrics)] * len(numeric_metrics)}
        
        # Create a DataFrame with aligned values
        # For simplicity, we'll use the first min_length values for each metric
        min_length = min(len(values) for values in df_data.values())
        df_aligned = {metric_id: values[:min_length] for metric_id, values in df_data.items()}
        df = pd.DataFrame(df_aligned)
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method='pearson').values.tolist()
        
        # Create filtered list of metrics for which we have data
        metrics_with_data = [m for m in numeric_metrics if m["id"] in metric_ids]
        
        return {
            "metrics": metrics_with_data,
            "matrix": corr_matrix
        }

    async def _analyze_metric_pair(
        self, 
        metric1: Dict[str, Any], 
        metric2: Dict[str, Any], 
        survey_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the relationship between a pair of metrics.
        
        Args:
            metric1: First metric definition
            metric2: Second metric definition
            survey_responses: List of survey responses with metric values
            
        Returns:
            Dictionary with analysis of the metric pair relationship
        """
        metric1_id = metric1["id"]
        metric2_id = metric2["id"]
        metric1_type = metric1.get("type", "unknown")
        metric2_type = metric2.get("type", "unknown")
        
        # Extract paired values
        paired_data = []
        for response in survey_responses:
            if "metrics" not in response:
                continue
                
            metrics = response["metrics"]
            if metric1_id in metrics and metric2_id in metrics:
                value1 = metrics[metric1_id]
                value2 = metrics[metric2_id]
                
                if metric1_type == "numeric" and metric2_type == "numeric":
                    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                        paired_data.append((value1, value2))
                elif metric1_type == "numeric" and metric2_type == "categorical":
                    if isinstance(value1, (int, float)) and isinstance(value2, str):
                        paired_data.append((value1, value2))
                elif metric1_type == "categorical" and metric2_type == "numeric":
                    if isinstance(value1, str) and isinstance(value2, (int, float)):
                        paired_data.append((value1, value2))
                elif metric1_type == "categorical" and metric2_type == "categorical":
                    if isinstance(value1, str) and isinstance(value2, str):
                        paired_data.append((value1, value2))
        
        # If not enough paired data, return empty analysis
        if len(paired_data) < 5:
            return {
                "metric1_id": metric1_id,
                "metric2_id": metric2_id,
                "type": f"{metric1_type}-{metric2_type}",
                "insufficient_data": True,
                "sample_size": len(paired_data)
            }
        
        # Analyze based on metric types
        if metric1_type == "numeric" and metric2_type == "numeric":
            return await self._analyze_numeric_pair(metric1, metric2, paired_data)
        elif (metric1_type == "numeric" and metric2_type == "categorical") or \
             (metric1_type == "categorical" and metric2_type == "numeric"):
            # Make sure numeric is first, categorical is second
            if metric1_type == "categorical":
                metric1, metric2 = metric2, metric1
                metric1_id, metric2_id = metric2_id, metric1_id
                paired_data = [(b, a) for a, b in paired_data]
            return await self._analyze_numeric_categorical_pair(metric1, metric2, paired_data)
        else:  # Both categorical
            return await self._analyze_categorical_pair(metric1, metric2, paired_data)

    async def _analyze_numeric_pair(
        self, 
        metric1: Dict[str, Any], 
        metric2: Dict[str, Any], 
        paired_data: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze the relationship between two numeric metrics.
        
        Args:
            metric1: First metric definition
            metric2: Second metric definition
            paired_data: List of paired values (value1, value2)
            
        Returns:
            Dictionary with analysis of the numeric-numeric relationship
        """
        metric1_id = metric1["id"]
        metric2_id = metric2["id"]
        
        # Split paired data
        values1, values2 = zip(*paired_data)
        
        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(values1, values2)
        
        # Determine significance
        significance = "none"
        if p_value < 0.05:
            if abs(correlation) > 0.7:
                significance = "high"
            elif abs(correlation) > 0.4:
                significance = "medium"
            else:
                significance = "low"
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(values1, values2, alpha=0.7)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(values1, values2)
        x_range = np.linspace(min(values1), max(values1), 100)
        plt.plot(x_range, slope * x_range + intercept, 'r', alpha=0.7)
        
        plt.title(f'Relationship between {metric1["name"]} and {metric2["name"]}')
        plt.xlabel(metric1["name"])
        plt.ylabel(metric2["name"])
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64
        scatter_plot = self._figure_to_base64(plt)
        plt.close()
        
        # Prepare regression line data for frontend
        regression_line = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2
        }
        
        # Quadrant analysis (dividing by medians)
        median1 = np.median(values1)
        median2 = np.median(values2)
        
        quadrants = {
            "q1": 0,  # High metric1, High metric2
            "q2": 0,  # Low metric1, High metric2
            "q3": 0,  # Low metric1, Low metric2
            "q4": 0   # High metric1, Low metric2
        }
        
        for v1, v2 in paired_data:
            if v1 >= median1 and v2 >= median2:
                quadrants["q1"] += 1
            elif v1 < median1 and v2 >= median2:
                quadrants["q2"] += 1
            elif v1 < median1 and v2 < median2:
                quadrants["q3"] += 1
            else:  # v1 >= median1 and v2 < median2
                quadrants["q4"] += 1
        
        # Convert to percentages
        total = sum(quadrants.values())
        quadrant_analysis = {
            q: {"count": count, "percentage": (count / total) * 100}
            for q, count in quadrants.items()
        }
        
        return {
            "metric1_id": metric1_id,
            "metric2_id": metric2_id,
            "type": "numeric-numeric",
            "sample_size": len(paired_data),
            "correlation": correlation,
            "correlation_type": "pearson",
            "p_value": p_value,
            "significance": significance,
            "scatter_plot": scatter_plot,
            "regression_line": regression_line,
            "quadrant_analysis": quadrant_analysis
        }

    async def _analyze_numeric_categorical_pair(
        self, 
        numeric_metric: Dict[str, Any], 
        categorical_metric: Dict[str, Any], 
        paired_data: List[Tuple[float, str]]
    ) -> Dict[str, Any]:
        """
        Analyze the relationship between a numeric and a categorical metric.
        
        Args:
            numeric_metric: Numeric metric definition
            categorical_metric: Categorical metric definition
            paired_data: List of paired values (numeric_value, category)
            
        Returns:
            Dictionary with analysis of the numeric-categorical relationship
        """
        numeric_id = numeric_metric["id"]
        categorical_id = categorical_metric["id"]
        
        # Group data by category
        grouped_data = {}
        for numeric_value, category in paired_data:
            if category not in grouped_data:
                grouped_data[category] = []
            grouped_data[category].append(numeric_value)
        
        # Calculate summary statistics for each category
        category_stats = {}
        for category, values in grouped_data.items():
            category_stats[category] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
        
        # Perform ANOVA test
        categories = []
        category_values = []
        
        for category, values in grouped_data.items():
            categories.append(category)
            category_values.append(values)
        
        try:
            f_statistic, p_value = stats.f_oneway(*category_values)
            is_significant = p_value < 0.05
        except:
            # Handle case with insufficient data
            f_statistic = 0
            p_value = 1
            is_significant = False
        
        # Create box plots
        plt.figure(figsize=(10, 6))
        plt.boxplot(category_values, labels=categories)
        plt.title(f'{numeric_metric["name"]} by {categorical_metric["name"]}')
        plt.xlabel(categorical_metric["name"])
        plt.ylabel(numeric_metric["name"])
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64
        box_plots = self._figure_to_base64(plt)
        plt.close()
        
        # Create bar chart of means
        plt.figure(figsize=(10, 6))
        means = [stats["mean"] for stats in category_stats.values()]
        plt.bar(categories, means)
        plt.title(f'Mean {numeric_metric["name"]} by {categorical_metric["name"]}')
        plt.xlabel(categorical_metric["name"])
        plt.ylabel(f'Mean {numeric_metric["name"]}')
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64
        means_chart = self._figure_to_base64(plt)
        plt.close()
        
        return {
            "metric1_id": numeric_id,
            "metric2_id": categorical_id,
            "type": "numeric-categorical",
            "sample_size": len(paired_data),
            "category_stats": category_stats,
            "anova": {
                "f_statistic": f_statistic,
                "p_value": p_value,
                "is_significant": is_significant
            },
            "f_statistic": f_statistic,
            "p_value": p_value,
            "is_significant": is_significant,
            "box_plots": box_plots,
            "category_means": means_chart
        }

    async def _analyze_categorical_pair(
        self, 
        metric1: Dict[str, Any], 
        metric2: Dict[str, Any], 
        paired_data: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze the relationship between two categorical metrics.
        
        Args:
            metric1: First metric definition
            metric2: Second metric definition
            paired_data: List of paired values (category1, category2)
            
        Returns:
            Dictionary with analysis of the categorical-categorical relationship
        """
        metric1_id = metric1["id"]
        metric2_id = metric2["id"]
        
        # Create contingency table
        categories1 = list(set(val1 for val1, _ in paired_data))
        categories2 = list(set(val2 for _, val2 in paired_data))
        
        # Initialize contingency table with zeros
        contingency_table = {cat1: {cat2: 0 for cat2 in categories2} for cat1 in categories1}
        
        # Fill contingency table
        for cat1, cat2 in paired_data:
            contingency_table[cat1][cat2] += 1
        
        # Convert to 2D array for chi-square test
        observed = []
        for cat1 in categories1:
            observed.append([contingency_table[cat1][cat2] for cat2 in categories2])
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            cramer_v = np.sqrt(chi2 / (len(paired_data) * min(len(categories1)-1, len(categories2)-1)))
            is_significant = p_value < 0.05
        except:
            # Handle case with insufficient data
            chi2 = 0
            p_value = 1
            dof = 0
            cramer_v = 0
            is_significant = False
            expected = [[0] * len(categories2)] * len(categories1)
        
        # Create heatmap of contingency table
        plt.figure(figsize=(10, 8))
        plt.imshow(observed, cmap='YlOrRd')
        plt.colorbar(label='Count')
        plt.xticks(range(len(categories2)), categories2, rotation=45)
        plt.yticks(range(len(categories1)), categories1)
        plt.xlabel(metric2["name"])
        plt.ylabel(metric1["name"])
        plt.title(f'Contingency Table: {metric1["name"]} vs {metric2["name"]}')
        
        # Add text annotations
        for i in range(len(categories1)):
            for j in range(len(categories2)):
                text = plt.text(j, i, observed[i][j],
                               ha="center", va="center", color="black")
        
        plt.tight_layout()
        
        # Save plot to base64
        heatmap = self._figure_to_base64(plt)
        plt.close()
        
        return {
            "metric1_id": metric1_id,
            "metric2_id": metric2_id,
            "type": "categorical-categorical",
            "sample_size": len(paired_data),
            "contingency_table": contingency_table,
            "chi_squared_test": {
                "chi2": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "is_significant": is_significant
            },
            "association_strength": {
                "cramer_v": cramer_v,
                "interpretation": self._interpret_cramer_v(cramer_v)
            },
            "heatmap": heatmap
        }

    def _interpret_cramer_v(self, cramer_v: float) -> str:
        """Interpret Cramer's V value."""
        if cramer_v < 0.1:
            return "negligible"
        elif cramer_v < 0.2:
            return "weak"
        elif cramer_v < 0.4:
            return "moderate"
        elif cramer_v < 0.6:
            return "relatively strong"
        elif cramer_v < 0.8:
            return "strong"
        else:
            return "very strong"

    async def _generate_key_insights(
        self, 
        correlation_data: Dict[str, Any], 
        pairwise_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate key insights from correlation data and pairwise analyses.
        
        Args:
            correlation_data: Dictionary with correlation matrix and metric info
            pairwise_analyses: List of pairwise analysis results
            
        Returns:
            Dictionary with key insights
        """
        insights = {
            "strongest_correlations": [],
            "significant_category_relationships": [],
            "actionable_findings": []
        }
        
        # Find strongest correlations
        numeric_pairs = [p for p in pairwise_analyses if p.get("type") == "numeric-numeric"]
        numeric_pairs.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)
        
        for pair in numeric_pairs[:3]:  # Top 3 strongest correlations
            if "correlation" in pair and abs(pair["correlation"]) > self.correlation_threshold:
                insights["strongest_correlations"].append({
                    "metric1_id": pair["metric1_id"],
                    "metric2_id": pair["metric2_id"],
                    "correlation": pair["correlation"],
                    "p_value": pair.get("p_value", 1.0),
                    "significance": pair.get("significance", "none")
                })
        
        # Find significant category relationships
        for pair in pairwise_analyses:
            if pair.get("type") == "numeric-categorical" and pair.get("is_significant", False):
                insights["significant_category_relationships"].append({
                    "numeric_metric_id": pair["metric1_id"],
                    "categorical_metric_id": pair["metric2_id"],
                    "f_statistic": pair.get("f_statistic", 0),
                    "p_value": pair.get("p_value", 1.0)
                })
            elif pair.get("type") == "categorical-categorical":
                chi_squared = pair.get("chi_squared_test", {})
                if chi_squared.get("is_significant", False):
                    insights["significant_category_relationships"].append({
                        "metric1_id": pair["metric1_id"],
                        "metric2_id": pair["metric2_id"],
                        "chi2": chi_squared.get("chi2", 0),
                        "p_value": chi_squared.get("p_value", 1.0),
                        "cramer_v": pair.get("association_strength", {}).get("cramer_v", 0)
                    })
        
        # Generate actionable findings
        for pair in numeric_pairs:
            if "correlation" in pair and abs(pair["correlation"]) > 0.6 and pair.get("p_value", 1.0) < 0.05:
                direction = "positive" if pair["correlation"] > 0 else "negative"
                findings = {
                    "metric1_id": pair["metric1_id"],
                    "metric2_id": pair["metric2_id"],
                    "relationship": f"strong {direction} correlation",
                    "actionable": True,
                    "strength": abs(pair["correlation"])
                }
                insights["actionable_findings"].append(findings)
        
        return insights

    async def _generate_visualizations(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations for cross-metric analysis.
        
        Args:
            correlation_data: Dictionary with correlation matrix and metric info
            
        Returns:
            Dictionary with visualizations
        """
        visualizations = {}
        
        # Skip if no correlation data
        if not correlation_data or "matrix" not in correlation_data or not correlation_data["matrix"]:
            return {
                "heatmap": "",
                "correlation_data": {
                    "metric_names": [],
                    "matrix": []
                }
            }
        
        # Create correlation heatmap
        matrix = correlation_data["matrix"]
        metrics = correlation_data["metrics"]
        
        metric_names = [m.get("name", m.get("id", "Unknown")) for m in metrics]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha="right")
        plt.yticks(range(len(metric_names)), metric_names)
        plt.title('Correlation Matrix Between Metrics')
        
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                text = plt.text(j, i, f"{matrix[i][j]:.2f}",
                               ha="center", va="center", 
                               color="white" if abs(matrix[i][j]) > 0.5 else "black")
        
        plt.tight_layout()
        
        # Save plot to base64
        heatmap = self._figure_to_base64(plt)
        plt.close()
        
        # Prepare data for frontend visualization
        visualizations = {
            "heatmap": heatmap,
            "correlation_data": {
                "metric_names": metric_names,
                "matrix": matrix
            }
        }
        
        return visualizations

    def _figure_to_base64(self, plt) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

    async def _generate_hypotheses(self, pairwise_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses about relationships between metrics.
        
        Args:
            pairwise_analyses: List of pairwise analysis results
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        # Generate hypotheses for numeric-numeric relationships
        for pair in pairwise_analyses:
            if pair.get("type") == "numeric-numeric" and pair.get("significance") in ["medium", "high"]:
                correlation = pair.get("correlation", 0)
                
                if correlation > 0.7:
                    hypothesis = {
                        "metric1_id": pair["metric1_id"],
                        "metric2_id": pair["metric2_id"],
                        "confidence": "high",
                        "hypothesis": f"Improvements in {pair['metric1_id']} strongly predict improvements in {pair['metric2_id']}",
                        "evidence": f"Strong positive correlation (r={correlation:.2f}, p={pair.get('p_value', 1.0):.3f})"
                    }
                    hypotheses.append(hypothesis)
                elif correlation < -0.7:
                    hypothesis = {
                        "metric1_id": pair["metric1_id"],
                        "metric2_id": pair["metric2_id"],
                        "confidence": "high",
                        "hypothesis": f"Improvements in {pair['metric1_id']} strongly predict decreases in {pair['metric2_id']}",
                        "evidence": f"Strong negative correlation (r={correlation:.2f}, p={pair.get('p_value', 1.0):.3f})"
                    }
                    hypotheses.append(hypothesis)
                elif abs(correlation) > 0.4:
                    hypothesis = {
                        "metric1_id": pair["metric1_id"],
                        "metric2_id": pair["metric2_id"],
                        "confidence": "medium",
                        "hypothesis": f"Changes in {pair['metric1_id']} may be associated with {'increases' if correlation > 0 else 'decreases'} in {pair['metric2_id']}",
                        "evidence": f"Moderate {'positive' if correlation > 0 else 'negative'} correlation (r={correlation:.2f}, p={pair.get('p_value', 1.0):.3f})"
                    }
                    hypotheses.append(hypothesis)
        
        # Generate hypotheses for numeric-categorical relationships
        for pair in pairwise_analyses:
            if pair.get("type") == "numeric-categorical" and pair.get("is_significant", False):
                hypothesis = {
                    "metric1_id": pair["metric1_id"],
                    "metric2_id": pair["metric2_id"],
                    "confidence": "medium",
                    "hypothesis": f"Different categories of {pair['metric2_id']} show significantly different values of {pair['metric1_id']}",
                    "evidence": f"ANOVA test (F={pair.get('f_statistic', 0):.2f}, p={pair.get('p_value', 1.0):.3f})"
                }
                hypotheses.append(hypothesis)
                
                # Look for specific categories with notable differences
                if "category_stats" in pair:
                    categories = list(pair["category_stats"].keys())
                    if len(categories) >= 2:
                        # Find highest and lowest categories by mean value
                        highest_cat = max(categories, key=lambda c: pair["category_stats"][c]["mean"])
                        lowest_cat = min(categories, key=lambda c: pair["category_stats"][c]["mean"])
                        
                        if highest_cat != lowest_cat:
                            high_mean = pair["category_stats"][highest_cat]["mean"]
                            low_mean = pair["category_stats"][lowest_cat]["mean"]
                            
                            hypothesis = {
                                "metric1_id": pair["metric1_id"],
                                "metric2_id": pair["metric2_id"],
                                "confidence": "medium",
                                "hypothesis": f"Category '{highest_cat}' is associated with higher {pair['metric1_id']} compared to '{lowest_cat}'",
                                "evidence": f"Mean difference: {high_mean - low_mean:.2f} ({high_mean:.2f} vs {low_mean:.2f})"
                            }
                            hypotheses.append(hypothesis)
        
        return hypotheses

    async def _build_knowledge_graph(
        self, 
        pairwise_analyses: List[Dict[str, Any]], 
        survey_id: int
    ) -> Dict[str, Any]:
        """
        Build a knowledge graph of relationships between metrics.
        
        Args:
            pairwise_analyses: List of pairwise analysis results
            survey_id: Survey ID
            
        Returns:
            Knowledge graph representation
        """
        nodes = set()
        edges = []
        
        # Create nodes and edges from pairwise analyses
        for pair in pairwise_analyses:
            if pair.get("insufficient_data", False):
                continue
                
            metric1_id = pair["metric1_id"]
            metric2_id = pair["metric2_id"]
            
            # Add nodes
            nodes.add(metric1_id)
            nodes.add(metric2_id)
            
            # Add edge with relationship data
            edge = {
                "source": metric1_id,
                "target": metric2_id,
                "type": pair.get("type", "unknown"),
            }
            
            # Add relationship strength based on type
            if pair.get("type") == "numeric-numeric":
                edge["relationship"] = "correlation"
                edge["strength"] = abs(pair.get("correlation", 0))
                edge["direction"] = "positive" if pair.get("correlation", 0) > 0 else "negative"
                edge["significance"] = pair.get("significance", "none")
            
            elif pair.get("type") == "numeric-categorical":
                edge["relationship"] = "variation"
                edge["strength"] = min(1.0, pair.get("f_statistic", 0) / 10) if pair.get("f_statistic") else 0
                edge["significant"] = pair.get("is_significant", False)
            
            elif pair.get("type") == "categorical-categorical":
                edge["relationship"] = "association"
                edge["strength"] = pair.get("association_strength", {}).get("cramer_v", 0)
                edge["interpretation"] = pair.get("association_strength", {}).get("interpretation", "none")
            
            edges.append(edge)
        
        # Store in the instance knowledge graph
        self.knowledge_graph[survey_id] = {
            "nodes": list(nodes),
            "edges": edges
        }
        
        return {
            "nodes": list(nodes),
            "edges": edges
        }

    async def _integrate_vector_analysis(
        self, 
        survey_id: int, 
        metrics_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Integrate with vector-based analysis to enhance insights.
        
        Args:
            survey_id: Survey ID
            metrics_data: List of metric definitions
            
        Returns:
            Dictionary with vector-based insights
        """
        # Collect vector insights for text-based metrics
        text_metrics = [m for m in metrics_data if m.get("type") in ["text", "categorical"]]
        
        vector_insights = {
            "clusters": {},
            "temporal_trends": {},
            "cross_metric_patterns": []
        }
        
        # Skip if no text metrics
        if not text_metrics:
            return vector_insights
        
        # Get vector analysis for each text metric
        for metric in text_metrics:
            metric_id = metric["id"]
            
            try:
                # Get cluster analysis
                clusters = await vector_trend_analysis_service.detect_response_clusters(
                    survey_id=survey_id,
                    question_id=metric_id
                )
                
                if clusters and clusters.get("status") == "success":
                    vector_insights["clusters"][metric_id] = clusters
                
                # Get temporal trends
                trends = await vector_trend_analysis_service.detect_temporal_trends(
                    survey_id=survey_id,
                    question_id=metric_id
                )
                
                if trends and "drift_analysis" in trends:
                    vector_insights["temporal_trends"][metric_id] = trends
            
            except Exception as e:
                # Log error but continue with other metrics
                print(f"Error integrating vector analysis for metric {metric_id}: {str(e)}")
        
        # Identify cross-metric patterns (if we have multiple text metrics with clusters)
        if len(vector_insights["clusters"]) >= 2:
            # This is a simplified approach - a more advanced implementation would 
            # analyze semantic similarity between clusters across metrics
            cross_metric_patterns = []
            
            for metric1_id, cluster1 in vector_insights["clusters"].items():
                for metric2_id, cluster2 in vector_insights["clusters"].items():
                    if metric1_id != metric2_id:
                        pattern = {
                            "metric1_id": metric1_id,
                            "metric2_id": metric2_id,
                            "cluster_counts": [
                                len(cluster1.get("clusters", [])),
                                len(cluster2.get("clusters", []))
                            ],
                            "observation": f"Both {metric1_id} and {metric2_id} show distinct response clusters"
                        }
                        cross_metric_patterns.append(pattern)
            
            vector_insights["cross_metric_patterns"] = cross_metric_patterns
        
        return vector_insights


# Singleton instance
cross_metric_analysis_service = CrossMetricAnalysisService() 