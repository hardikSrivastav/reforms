"""
Correlation analysis service for analyzing relationships between metrics.
This module provides correlation detection, causality testing, and relationship visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
from datetime import datetime
import json
import networkx as nx
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.config import settings


class CorrelationAnalysisService:
    """Service for analyzing relationships between metrics."""

    def __init__(self, correlation_threshold: float = 0.3, causality_alpha: float = 0.05):
        """
        Initialize the correlation analysis service.
        
        Args:
            correlation_threshold: Threshold for determining significant correlations
            causality_alpha: Significance level for causality tests
        """
        self.correlation_threshold = correlation_threshold
        self.causality_alpha = causality_alpha
    
    async def analyze_cross_metric_correlations(
        self, 
        survey_id: int, 
        metrics_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics.
        
        Args:
            survey_id: The survey ID
            metrics_data: Dictionary mapping metric IDs to metric definitions
            responses: List of survey responses
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Check cache first
        cached_result = await metadata_store.get_analysis_result("cross_metric", survey_id)
        if cached_result:
            return cached_result
        
        # Extract metric values from responses
        metric_values = self._extract_metric_values(metrics_data, responses)
        
        # Calculate correlations
        correlation_matrix, p_values = self._calculate_correlations(metric_values)
        
        # Identify significant correlations
        significant_correlations = self._identify_significant_correlations(
            correlation_matrix, 
            p_values, 
            metrics_data
        )
        
        # Test for potential causality
        causal_relationships = await self._test_causality(metric_values, significant_correlations)
        
        # Create result structure
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "correlation_matrix": correlation_matrix.to_dict(),
            "significant_correlations": significant_correlations,
            "causal_relationships": causal_relationships,
            "visualizations": await self._generate_correlation_visualizations(
                correlation_matrix, 
                significant_correlations,
                metrics_data
            )
        }
        
        # Store in cache
        await metadata_store.store_analysis_result("cross_metric", survey_id, result)
        
        return result
    
    def _extract_metric_values(
        self, 
        metrics_data: Dict[str, Any], 
        responses: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Extract metric values from responses into a DataFrame.
        
        Args:
            metrics_data: Dictionary mapping metric IDs to metric definitions
            responses: List of survey responses
            
        Returns:
            DataFrame with metric values for each response
        """
        # Initialize data structure
        data = {metric_id: [] for metric_id in metrics_data.keys()}
        response_ids = []
        
        # Extract values from each response
        for response in responses:
            response_id = response.get("_id", "unknown")
            response_ids.append(response_id)
            
            # Get response values for each metric
            response_data = response.get("responses", {})
            
            for metric_id in metrics_data.keys():
                value = response_data.get(metric_id)
                
                # Handle different metric types
                metric_type = metrics_data[metric_id].get("type", "unknown")
                
                if metric_type == "numeric":
                    # Convert to float if possible
                    try:
                        if value is not None:
                            value = float(value)
                        else:
                            value = np.nan
                    except:
                        value = np.nan
                
                elif metric_type in ["categorical", "single_choice"]:
                    # Keep as is for categorical data
                    if value is None:
                        value = np.nan
                
                elif metric_type == "multi_choice":
                    # For multi-choice, use count of selected options
                    if isinstance(value, list):
                        value = len(value)
                    else:
                        value = np.nan
                
                else:
                    # For unknown types, try numeric conversion or use NaN
                    try:
                        if value is not None:
                            value = float(value)
                        else:
                            value = np.nan
                    except:
                        value = np.nan
                
                data[metric_id].append(value)
        
        # Create DataFrame
        df = pd.DataFrame(data, index=response_ids)
        return df
    
    def _calculate_correlations(
        self, 
        metric_values: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation coefficients and p-values between metrics.
        
        Args:
            metric_values: DataFrame with metric values
            
        Returns:
            Tuple of correlation matrix and p-value matrix
        """
        # Calculate Pearson correlation for numeric columns
        numeric_df = metric_values.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr(method='pearson')
        
        # Calculate p-values
        p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
        
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                if i == j:
                    p_values.loc[i, j] = 0.0
                else:
                    # Get valid pairs of values
                    valid_pairs = numeric_df[[i, j]].dropna()
                    
                    if len(valid_pairs) > 2:  # Need at least 3 pairs for meaningful correlation
                        _, p_value = stats.pearsonr(valid_pairs[i], valid_pairs[j])
                        p_values.loc[i, j] = p_value
                    else:
                        p_values.loc[i, j] = 1.0  # Not significant if too few data points
        
        return correlation_matrix, p_values
    
    def _identify_significant_correlations(
        self, 
        correlation_matrix: pd.DataFrame, 
        p_values: pd.DataFrame,
        metrics_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify statistically significant correlations.
        
        Args:
            correlation_matrix: Correlation matrix
            p_values: P-value matrix
            metrics_data: Dictionary mapping metric IDs to metric definitions
            
        Returns:
            List of significant correlation entries
        """
        significant_correlations = []
        
        # Check each pair of metrics
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                # Skip self-correlations and duplicate pairs
                if i >= j:
                    continue
                
                corr_value = correlation_matrix.loc[i, j]
                p_value = p_values.loc[i, j]
                
                # Check if correlation is significant
                if (abs(corr_value) >= self.correlation_threshold) and (p_value < 0.05):
                    # Get metric names
                    metric1_name = metrics_data.get(i, {}).get("name", i)
                    metric2_name = metrics_data.get(j, {}).get("name", j)
                    
                    # Determine correlation strength and direction
                    strength = "strong" if abs(corr_value) > 0.6 else "moderate"
                    direction = "positive" if corr_value > 0 else "negative"
                    
                    significant_correlations.append({
                        "metric1_id": i,
                        "metric2_id": j,
                        "metric1_name": metric1_name,
                        "metric2_name": metric2_name,
                        "correlation": float(corr_value),
                        "p_value": float(p_value),
                        "strength": strength,
                        "direction": direction,
                        "description": f"{strength} {direction} correlation between {metric1_name} and {metric2_name}"
                    })
        
        # Sort by absolute correlation value
        significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return significant_correlations
    
    async def _test_causality(
        self, 
        metric_values: pd.DataFrame,
        significant_correlations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Test for potential causal relationships using Granger causality.
        
        Args:
            metric_values: DataFrame with metric values
            significant_correlations: List of significant correlation entries
            
        Returns:
            List of potential causal relationships
        """
        causal_relationships = []
        
        # Only test on numeric data
        numeric_df = metric_values.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df) < 10:
            # Not enough data for meaningful causality testing
            return causal_relationships
        
        # Test each significant correlation
        for corr in significant_correlations:
            metric1_id = corr["metric1_id"]
            metric2_id = corr["metric2_id"]
            
            # Skip if either metric is not numeric
            if metric1_id not in numeric_df.columns or metric2_id not in numeric_df.columns:
                continue
            
            # Check for Granger causality in both directions
            # First direction: metric1 -> metric2
            try:
                data = numeric_df[[metric1_id, metric2_id]].dropna()
                
                if len(data) >= 10:  # Need sufficient data points
                    # Test with different lag values (1, 2, 3, 4)
                    results = grangercausalitytests(data, maxlag=4, verbose=False)
                    
                    # Check the minimum p-value across lags
                    min_p_value = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, 5))
                    
                    # If significant, add to causal relationships
                    if min_p_value < self.causality_alpha:
                        causal_relationships.append({
                            "cause_metric_id": metric1_id,
                            "effect_metric_id": metric2_id,
                            "cause_metric_name": corr["metric1_name"],
                            "effect_metric_name": corr["metric2_name"],
                            "p_value": float(min_p_value),
                            "confidence": 1.0 - float(min_p_value)
                        })
            except:
                # Skip if test fails
                pass
            
            # Second direction: metric2 -> metric1
            try:
                data = numeric_df[[metric2_id, metric1_id]].dropna()
                
                if len(data) >= 10:
                    results = grangercausalitytests(data, maxlag=4, verbose=False)
                    min_p_value = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, 5))
                    
                    if min_p_value < self.causality_alpha:
                        causal_relationships.append({
                            "cause_metric_id": metric2_id,
                            "effect_metric_id": metric1_id,
                            "cause_metric_name": corr["metric2_name"],
                            "effect_metric_name": corr["metric1_name"],
                            "p_value": float(min_p_value),
                            "confidence": 1.0 - float(min_p_value)
                        })
            except:
                pass
        
        return causal_relationships
    
    async def _generate_correlation_visualizations(
        self, 
        correlation_matrix: pd.DataFrame,
        significant_correlations: List[Dict[str, Any]],
        metrics_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate visualizations for correlation analysis.
        
        Args:
            correlation_matrix: Correlation matrix
            significant_correlations: List of significant correlation entries
            metrics_data: Dictionary mapping metric IDs to metric definitions
            
        Returns:
            Dictionary with visualization data
        """
        visualizations = {}
        
        try:
            # Create heatmap of correlation matrix
            if not correlation_matrix.empty:
                visualizations["correlation_heatmap"] = self._create_correlation_heatmap(
                    correlation_matrix
                )
            
            # Create network graph of significant correlations
            if significant_correlations:
                visualizations["correlation_network"] = self._create_correlation_network(
                    significant_correlations,
                    metrics_data
                )
        except Exception as e:
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> str:
        """
        Create a heatmap visualization of the correlation matrix.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Base64-encoded image string
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        cmap = plt.cm.RdBu_r  # Red-Blue colormap (reversed)
        plt.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(label='Pearson Correlation')
        
        # Add labels
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
        
        # Add correlation values
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                value = correlation_matrix.iloc[i, j]
                color = "white" if abs(value) > 0.5 else "black"
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)
        
        plt.title('Metric Correlation Heatmap')
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def _create_correlation_network(
        self, 
        significant_correlations: List[Dict[str, Any]],
        metrics_data: Dict[str, Any]
    ) -> str:
        """
        Create a network graph visualization of significant correlations.
        
        Args:
            significant_correlations: List of significant correlation entries
            metrics_data: Dictionary mapping metric IDs to metric definitions
            
        Returns:
            Base64-encoded image string
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes for each metric
        metric_ids = set()
        for corr in significant_correlations:
            metric_ids.add(corr["metric1_id"])
            metric_ids.add(corr["metric2_id"])
        
        for metric_id in metric_ids:
            metric_name = metrics_data.get(metric_id, {}).get("name", metric_id)
            G.add_node(metric_id, name=metric_name)
        
        # Add edges for correlations
        for corr in significant_correlations:
            G.add_edge(
                corr["metric1_id"],
                corr["metric2_id"],
                weight=abs(corr["correlation"]),
                correlation=corr["correlation"],
                color='green' if corr["correlation"] > 0 else 'red'
            )
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Calculate node sizes based on degree
        node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
        
        # Calculate positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
        
        # Draw edges with varying width based on correlation strength
        for u, v, data in G.edges(data=True):
            width = 2 * abs(data['correlation'])
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, edge_color=data['color'])
        
        # Draw labels
        labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        plt.title('Metric Correlation Network')
        plt.axis('off')
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

# Singleton instance
correlation_analysis_service = CorrelationAnalysisService() 