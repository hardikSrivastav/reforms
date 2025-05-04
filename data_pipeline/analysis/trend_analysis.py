"""
Trend analysis service for analyzing changes in metrics over time.
This module provides time series analysis, seasonality detection, and change point detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
from datetime import datetime, timedelta
import json
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import ruptures as rpt

from data_pipeline.services.metadata_store import metadata_store
from data_pipeline.config import settings


class TrendAnalysisService:
    """Service for analyzing changes in metrics over time."""

    def __init__(self, change_threshold: float = 0.1):
        """
        Initialize the trend analysis service.
        
        Args:
            change_threshold: Threshold for determining significant changes in metrics
        """
        self.change_threshold = change_threshold

    async def analyze_metric_trends(
        self, 
        survey_id: int, 
        metric_id: str, 
        metric_data: Dict[str, Any], 
        time_series_responses: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze how a metric changes over time.
        
        Args:
            survey_id: The survey ID
            metric_id: The metric ID to analyze
            metric_data: The metric definition data
            time_series_responses: Dictionary mapping time periods to response lists
            
        Returns:
            Dictionary with trend analysis results
        """
        # Check cache first
        cached_result = await metadata_store.get_analysis_result("trend_analysis", survey_id, metric_id)
        if cached_result:
            return cached_result
        
        # Determine the type of metric
        metric_type = metric_data.get("type", "numeric")
        
        # Sort time periods chronologically
        time_periods = sorted(time_series_responses.keys())
        
        # Prepare result structure
        result = {
            "survey_id": survey_id,
            "metric_id": metric_id,
            "metric_name": metric_data.get("name", "Unknown"),
            "metric_type": metric_type,
            "time_periods": time_periods,
            "timestamp": datetime.now().isoformat()
        }
        
        # Perform appropriate analysis based on metric type
        if metric_type == "numeric":
            # Extract time series data
            time_series_data = self._extract_numeric_time_series(time_series_responses)
            
            # Analyze the time series
            analysis_result = await self._analyze_numeric_time_series(time_series_data)
            result.update(analysis_result)
            
        elif metric_type in ["categorical", "single_choice", "multi_choice"]:
            # Extract categorical time series data
            time_series_data = self._extract_categorical_time_series(time_series_responses)
            
            # Analyze the categorical time series
            analysis_result = await self._analyze_categorical_time_series(time_series_data)
            result.update(analysis_result)
            
        else:
            result["error"] = f"Trend analysis not supported for metric type: {metric_type}"
        
        # Add visualizations
        result["visualizations"] = await self._generate_trend_visualizations(
            metric_type, 
            time_series_responses, 
            result
        )
        
        # Store in cache
        await metadata_store.store_analysis_result("trend_analysis", survey_id, metric_id, result)
        
        return result

    def _extract_numeric_time_series(
        self, 
        time_series_responses: Dict[str, List[Dict[str, Any]]]
    ) -> pd.Series:
        """
        Extract numeric time series data from responses.
        
        Args:
            time_series_responses: Dictionary mapping time periods to response lists
            
        Returns:
            Pandas Series with datetime index and numeric values
        """
        # Extract values for each time period
        data_points = {}
        
        for time_period, responses in time_series_responses.items():
            values = []
            for response in responses:
                value = response.get("value")
                if isinstance(value, (int, float)):
                    values.append(value)
            
            if values:
                # Convert time period string to datetime
                try:
                    # Assuming time_period is in ISO format
                    period_dt = pd.to_datetime(time_period)
                    # Use mean value for the period
                    data_points[period_dt] = np.mean(values)
                except:
                    # If conversion fails, use the string as is
                    data_points[time_period] = np.mean(values)
        
        # Create pandas Series with datetime index
        if all(isinstance(k, datetime) for k in data_points.keys()):
            # If all keys are datetime objects
            time_series = pd.Series(data_points)
            time_series.sort_index(inplace=True)
        else:
            # If keys are strings, convert to pandas series with string index
            time_series = pd.Series(data_points)
        
        return time_series

    def _extract_categorical_time_series(
        self, 
        time_series_responses: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, pd.Series]:
        """
        Extract categorical time series data from responses.
        
        Args:
            time_series_responses: Dictionary mapping time periods to response lists
            
        Returns:
            Dictionary mapping categories to time series
        """
        # First, identify all unique categories across all time periods
        all_categories = set()
        for responses in time_series_responses.values():
            for response in responses:
                category = response.get("category")
                if isinstance(category, str):
                    all_categories.add(category)
                categories = response.get("categories", [])
                if isinstance(categories, list):
                    all_categories.update(categories)
        
        # Initialize data structure for each category
        category_time_series = {category: {} for category in all_categories}
        
        # Extract counts for each category in each time period
        for time_period, responses in time_series_responses.items():
            # Count occurrences of each category
            period_counts = Counter()
            
            for response in responses:
                # Handle single category
                category = response.get("category")
                if isinstance(category, str):
                    period_counts[category] += 1
                
                # Handle multiple categories
                categories = response.get("categories", [])
                if isinstance(categories, list):
                    period_counts.update(categories)
            
            # Convert time period to datetime if possible
            try:
                period_dt = pd.to_datetime(time_period)
            except:
                period_dt = time_period
            
            # Add counts to each category's time series
            for category in all_categories:
                category_time_series[category][period_dt] = period_counts.get(category, 0)
        
        # Convert to pandas Series
        for category in all_categories:
            if all(isinstance(k, datetime) for k in category_time_series[category].keys()):
                series = pd.Series(category_time_series[category])
                series.sort_index(inplace=True)
                category_time_series[category] = series
            else:
                series = pd.Series(category_time_series[category])
                category_time_series[category] = series
        
        return category_time_series

    async def _analyze_numeric_time_series(
        self, 
        time_series: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze numeric time series data.
        
        Args:
            time_series: Time series data as pandas Series
            
        Returns:
            Dictionary with analysis results
        """
        if len(time_series) < 2:
            return {"error": "Not enough data points for trend analysis"}
        
        # Calculate basic trend statistics
        first_value = time_series.iloc[0]
        last_value = time_series.iloc[-1]
        min_value = time_series.min()
        max_value = time_series.max()
        overall_change = last_value - first_value
        percent_change = (overall_change / first_value) * 100 if first_value != 0 else float('inf')
        
        trend_data = {
            "trend_data": time_series.to_dict(),
            "stats": {
                "first_value": float(first_value),
                "last_value": float(last_value),
                "min_value": float(min_value),
                "max_value": float(max_value),
                "overall_change": float(overall_change),
                "overall_percent_change": float(percent_change)
            }
        }
        
        # Detect trend direction
        is_increasing = last_value > first_value
        is_significant_change = abs(percent_change) > (self.change_threshold * 100)
        
        trend_data["trend_direction"] = {
            "is_increasing": is_increasing,
            "is_significant_change": is_significant_change,
            "description": "increasing" if is_increasing else "decreasing"
        }
        
        # Check for sufficient data points for advanced analysis
        if len(time_series) >= 4:
            # Perform change point detection
            change_points = await self._detect_change_points_numeric(time_series)
            trend_data["change_points"] = change_points
            
            # Perform seasonal decomposition if enough data points
            if len(time_series) >= 6:
                try:
                    seasonal_results = await self._seasonal_decomposition(time_series)
                    trend_data["seasonal_decomposition"] = seasonal_results
                except Exception as e:
                    trend_data["seasonal_decomposition"] = {"error": str(e)}
        
        return trend_data

    async def _analyze_categorical_time_series(
        self, 
        category_time_series: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Analyze categorical time series data.
        
        Args:
            category_time_series: Dictionary mapping categories to time series
            
        Returns:
            Dictionary with analysis results
        """
        if not category_time_series:
            return {"error": "No categorical data for trend analysis"}
        
        # Initialize result structure
        result = {
            "categories": list(category_time_series.keys()),
            "category_trends": {},
            "distribution_shifts": []
        }
        
        # Analyze trend for each category
        for category, time_series in category_time_series.items():
            if len(time_series) < 2:
                continue
                
            # Calculate trend statistics
            first_value = time_series.iloc[0]
            last_value = time_series.iloc[-1]
            max_value = time_series.max()
            overall_change = last_value - first_value
            
            # Calculate percent change safely
            percent_change = 0
            if first_value > 0:
                percent_change = (overall_change / first_value) * 100
            
            # Store category trend data
            result["category_trends"][category] = {
                "trend_data": time_series.to_dict(),
                "stats": {
                    "first_value": int(first_value),
                    "last_value": int(last_value),
                    "max_value": int(max_value),
                    "overall_change": int(overall_change),
                    "overall_percent_change": float(percent_change)
                },
                "is_increasing": last_value > first_value,
                "is_significant_change": abs(percent_change) > (self.change_threshold * 100)
            }
        
        # Calculate distribution shifts between time periods
        time_periods = next(iter(category_time_series.values())).index.tolist()
        if len(time_periods) >= 2:
            for i in range(len(time_periods) - 1):
                period1 = time_periods[i]
                period2 = time_periods[i + 1]
                
                # Calculate distribution for each period
                dist1 = {cat: series[period1] for cat, series in category_time_series.items()}
                dist2 = {cat: series[period2] for cat, series in category_time_series.items()}
                
                # Calculate total responses in each period
                total1 = sum(dist1.values())
                total2 = sum(dist2.values())
                
                if total1 > 0 and total2 > 0:
                    # Convert to percentages
                    dist1_pct = {cat: (count / total1) * 100 for cat, count in dist1.items()}
                    dist2_pct = {cat: (count / total2) * 100 for cat, count in dist2.items()}
                    
                    # Calculate changes
                    changes = {}
                    for cat in category_time_series.keys():
                        pct1 = dist1_pct.get(cat, 0)
                        pct2 = dist2_pct.get(cat, 0)
                        changes[cat] = pct2 - pct1
                    
                    # Identify significant changes
                    significant_changes = {
                        cat: change for cat, change in changes.items() 
                        if abs(change) > self.change_threshold * 100
                    }
                    
                    if significant_changes:
                        shift = {
                            "from_period": str(period1),
                            "to_period": str(period2),
                            "significant_changes": significant_changes
                        }
                        result["distribution_shifts"].append(shift)
        
        return result

    async def _detect_change_points_numeric(
        self, 
        time_series: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect change points in numeric time series.
        
        Args:
            time_series: Time series data as pandas Series
            
        Returns:
            Dictionary with change point analysis
        """
        # Convert to numpy array for change point detection
        signal = time_series.values
        
        if len(signal) < 4:
            return {"error": "Not enough data points for change point detection"}
        
        try:
            # Use Pelt algorithm for change point detection
            algo = rpt.Pelt(model="rbf").fit(signal)
            # Get the change points with penalty parameter
            change_points = algo.predict(pen=2)
            
            # Convert change point indices to time period strings
            time_periods = time_series.index.tolist()
            change_point_periods = []
            
            for cp in change_points:
                if cp < len(time_periods):
                    change_point_periods.append(str(time_periods[cp]))
            
            # Calculate metrics for each change point
            change_point_metrics = []
            for i, cp in enumerate(change_points):
                if cp < len(signal):
                    # Determine the before and after segments
                    if i == 0:  # First change point
                        before_segment = signal[:cp]
                    else:
                        before_segment = signal[change_points[i-1]:cp]
                    
                    if cp == len(signal):  # Last point
                        after_segment = []
                    else:
                        after_segment = signal[cp:change_points[i+1] if i+1 < len(change_points) else None]
                    
                    # Calculate metrics if segments have data
                    if len(before_segment) > 0 and len(after_segment) > 0:
                        before_mean = float(np.mean(before_segment))
                        after_mean = float(np.mean(after_segment))
                        percent_change = ((after_mean - before_mean) / before_mean) * 100 if before_mean != 0 else float('inf')
                        
                        change_point_metrics.append({
                            "index": cp,
                            "time_period": str(time_periods[cp]) if cp < len(time_periods) else None,
                            "before_mean": before_mean,
                            "after_mean": after_mean,
                            "percent_change": percent_change,
                            "direction": "increase" if after_mean > before_mean else "decrease"
                        })
            
            return {
                "change_point_indices": change_points,
                "change_point_periods": change_point_periods,
                "change_point_metrics": change_point_metrics
            }
        except Exception as e:
            return {"error": str(e)}

    async def _seasonal_decomposition(
        self, 
        time_series: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform seasonal decomposition of time series.
        
        Args:
            time_series: Time series data as pandas Series
            
        Returns:
            Dictionary with seasonal decomposition results
        """
        if len(time_series) < 6:
            return {"error": "Not enough data points for seasonal decomposition"}
        
        try:
            # Ensure time series has a regular frequency
            if not time_series.index.is_monotonic_increasing:
                time_series = time_series.sort_index()
            
            # Try to infer frequency if not present
            if time_series.index.freq is None:
                # For basic weekly/monthly/quarterly analysis, try common frequencies
                for freq in ['D', 'W', 'M', 'Q']:
                    try:
                        # Resample to ensure regular frequency
                        resampled = time_series.resample(freq).mean().dropna()
                        if len(resampled) >= 6:  # Need enough data after resampling
                            time_series = resampled
                            break
                    except:
                        continue
            
            # Check if we have a valid frequency
            if len(time_series) < 6:
                return {"error": "Insufficient data after resampling"}
            
            # Use additive model for decomposition
            result = seasonal_decompose(time_series, model='additive')
            
            # Convert decomposition components to dictionaries
            trend = result.trend.dropna().to_dict()
            seasonal = result.seasonal.dropna().to_dict()
            residual = result.resid.dropna().to_dict()
            
            # Convert keys to strings for JSON serialization
            trend = {str(k): float(v) for k, v in trend.items()}
            seasonal = {str(k): float(v) for k, v in seasonal.items()}
            residual = {str(k): float(v) for k, v in residual.items()}
            
            # Calculate basic seasonality metrics
            seasonal_values = list(result.seasonal.dropna().values)
            seasonality_strength = float(np.std(seasonal_values))
            has_seasonality = seasonality_strength > (np.std(time_series.values) * 0.1)
            
            return {
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
                "seasonality_metrics": {
                    "seasonality_strength": seasonality_strength,
                    "has_seasonality": has_seasonality
                }
            }
        except Exception as e:
            return {"error": str(e)}

    async def _generate_trend_visualizations(
        self, 
        metric_type: str, 
        time_series_responses: Dict[str, List[Dict[str, Any]]],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate visualizations for trend analysis.
        
        Args:
            metric_type: Type of metric
            time_series_responses: Dictionary mapping time periods to response lists
            analysis_result: Analysis results from previous steps
            
        Returns:
            Dictionary with visualization data
        """
        visualizations = {}
        
        try:
            if metric_type == "numeric":
                # Create time series line chart
                if "trend_data" in analysis_result:
                    trend_data = analysis_result["trend_data"]
                    # Convert string keys to datetime for plotting
                    time_periods = []
                    values = []
                    for period, value in trend_data.items():
                        time_periods.append(period)
                        values.append(value)
                    
                    visualizations["trend_line_chart"] = self._create_time_series_chart(
                        time_periods, values, "Metric Value Over Time"
                    )
                
                # Create change point visualization if available
                if "change_points" in analysis_result and "change_point_indices" in analysis_result["change_points"]:
                    change_points = analysis_result["change_points"]["change_point_indices"]
                    if "trend_data" in analysis_result:
                        trend_data = analysis_result["trend_data"]
                        time_periods = []
                        values = []
                        for period, value in trend_data.items():
                            time_periods.append(period)
                            values.append(value)
                        
                        visualizations["change_point_chart"] = self._create_change_point_chart(
                            time_periods, values, change_points
                        )
                
                # Create seasonal decomposition visualization if available
                if "seasonal_decomposition" in analysis_result and "error" not in analysis_result["seasonal_decomposition"]:
                    seasonal_decomp = analysis_result["seasonal_decomposition"]
                    visualizations["seasonal_decomposition_chart"] = self._create_seasonal_decomposition_chart(
                        seasonal_decomp
                    )
            
            elif metric_type in ["categorical", "single_choice", "multi_choice"]:
                # Create stacked area chart for categorical data
                if "category_trends" in analysis_result:
                    category_trends = analysis_result["category_trends"]
                    visualizations["category_trend_chart"] = self._create_category_trend_chart(
                        category_trends
                    )
                    
                    # Create distribution shift visualization if there are shifts
                    if "distribution_shifts" in analysis_result and analysis_result["distribution_shifts"]:
                        visualizations["distribution_shift_chart"] = self._create_distribution_shift_chart(
                            analysis_result["distribution_shifts"][0]  # Use the first significant shift
                        )
        except Exception as e:
            visualizations["error"] = str(e)
        
        return visualizations

    def _create_time_series_chart(
        self, 
        time_periods: List[str], 
        values: List[float], 
        title: str
    ) -> str:
        """Create a time series chart and return as base64."""
        plt.figure(figsize=(10, 6))
        plt.plot(time_periods, values, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title(title)
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_change_point_chart(
        self, 
        time_periods: List[str], 
        values: List[float], 
        change_points: List[int]
    ) -> str:
        """Create a change point visualization chart and return as base64."""
        plt.figure(figsize=(12, 6))
        plt.plot(time_periods, values, marker='o', linestyle='-', linewidth=2)
        
        # Highlight change points
        for cp in change_points:
            if cp < len(time_periods) and cp != len(time_periods) - 1:  # Skip if it's the last point
                plt.axvline(x=time_periods[cp], color='red', linestyle='--', alpha=0.7)
                plt.text(time_periods[cp], max(values), 'Change Point', 
                         rotation=90, verticalalignment='bottom', color='red')
        
        plt.title('Metric Changes with Detected Change Points')
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_seasonal_decomposition_chart(
        self, 
        seasonal_decomp: Dict[str, Any]
    ) -> str:
        """Create a seasonal decomposition chart and return as base64."""
        # Extract components
        trend_data = seasonal_decomp.get("trend", {})
        seasonal_data = seasonal_decomp.get("seasonal", {})
        residual_data = seasonal_decomp.get("residual", {})
        
        # Convert to lists for plotting
        periods = list(trend_data.keys())
        trend_values = [trend_data[p] for p in periods]
        seasonal_values = [seasonal_data.get(p, 0) for p in periods]
        residual_values = [residual_data.get(p, 0) for p in periods]
        
        # Create plot with 4 subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original data
        original_values = [trend_values[i] + seasonal_values[i] + residual_values[i] 
                           for i in range(len(periods))]
        ax1.plot(periods, original_values)
        ax1.set_title('Original Time Series')
        ax1.grid(True, alpha=0.3)
        
        # Trend component
        ax2.plot(periods, trend_values, color='green')
        ax2.set_title('Trend Component')
        ax2.grid(True, alpha=0.3)
        
        # Seasonal component
        ax3.plot(periods, seasonal_values, color='red')
        ax3.set_title('Seasonal Component')
        ax3.grid(True, alpha=0.3)
        
        # Residual component
        ax4.plot(periods, residual_values, color='purple')
        ax4.set_title('Residual Component')
        ax4.grid(True, alpha=0.3)
        
        # Format
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_category_trend_chart(
        self, 
        category_trends: Dict[str, Any]
    ) -> str:
        """Create a stacked area chart for category trends and return as base64."""
        # Extract trend data for each category
        all_periods = set()
        for cat_data in category_trends.values():
            trend_data = cat_data.get("trend_data", {})
            all_periods.update(trend_data.keys())
        
        # Sort periods
        all_periods = sorted(all_periods)
        
        # Create dataframe for plotting
        data = {}
        for category, cat_data in category_trends.items():
            trend_data = cat_data.get("trend_data", {})
            # Fill in missing periods with zeros
            cat_values = [trend_data.get(period, 0) for period in all_periods]
            data[category] = cat_values
        
        df = pd.DataFrame(data, index=all_periods)
        
        # Plot stacked area chart
        plt.figure(figsize=(12, 8))
        df.plot.area(stacked=True, alpha=0.7, figsize=(12, 8))
        plt.title('Category Distribution Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_distribution_shift_chart(
        self, 
        shift_data: Dict[str, Any]
    ) -> str:
        """Create a visualization of distribution shift and return as base64."""
        from_period = shift_data.get("from_period", "")
        to_period = shift_data.get("to_period", "")
        changes = shift_data.get("significant_changes", {})
        
        # Sort categories by absolute change magnitude
        categories = sorted(changes.keys(), key=lambda x: abs(changes[x]), reverse=True)
        change_values = [changes[cat] for cat in categories]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, change_values, color=['green' if v > 0 else 'red' for v in change_values])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom' if height > 0 else 'top')
        
        plt.title(f'Distribution Shift from {from_period} to {to_period}')
        plt.xlabel('Category')
        plt.ylabel('Percentage Point Change')
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str


# Singleton instance
trend_analysis_service = TrendAnalysisService() 