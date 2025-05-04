"""
Predictive analysis service for forecasting trends and analyzing cohorts.
This module provides the fourth tier of analysis in the multi-tiered analysis engine.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
import asyncio
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

from ..config import settings
from ..services.metadata_store import metadata_store
from ..embeddings.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class PredictiveAnalysisService:
    """Service for generating predictive analyses for survey data."""
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the predictive analysis service.
        
        Args:
            openai_api_key: OpenAI API key for AI-powered analysis
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.model = settings.METRIC_ANALYSIS_MODEL  # Using the same model as metric analysis
        logger.info(f"Initialized predictive analysis service with model: {self.model}")
    
    async def generate_predictions(
        self,
        survey_id: int,
        metrics: List[Dict[str, Any]],
        responses: List[Dict[str, Any]],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate predictive analysis for a survey.
        
        Args:
            survey_id: Survey ID
            metrics: List of metric data
            responses: List of survey responses
            historical_data: Optional historical data for more accurate forecasting
            
        Returns:
            Predictive analysis results
        """
        try:
            logger.info(f"Generating predictive analysis for survey {survey_id}")
            
            # Check cache first
            cached_result = metadata_store.get_analysis_result("predictive_analysis", survey_id)
            if cached_result:
                logger.info(f"Using cached predictive analysis for survey {survey_id}")
                return cached_result
            
            # Generate time series forecasts
            time_series_forecasts = await self._forecast_time_series(survey_id, responses, historical_data)
            
            # Generate metric trend predictions
            metric_trends = await self._predict_metric_trends(survey_id, metrics, responses)
            
            # Perform cohort analysis
            cohort_analysis = await self._analyze_cohorts(survey_id, responses)
            
            # Generate AI-powered recommendations
            recommendations = await self._generate_recommendations(
                survey_id, 
                metrics, 
                time_series_forecasts, 
                metric_trends,
                cohort_analysis
            )
            
            # Compile the result
            result = {
                "survey_id": survey_id,
                "time_series_forecasts": time_series_forecasts,
                "metric_trends": metric_trends,
                "cohort_analysis": cohort_analysis,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
            # Store in cache
            metadata_store.store_analysis_result(
                "predictive_analysis",
                survey_id,
                result,
                ttl=settings.CACHE_TTL.get("cross_metric_analysis")  # Using same TTL as cross-metric for now
            )
            
            logger.info(f"Completed predictive analysis for survey {survey_id}")
            return result
        except Exception as e:
            logger.error(f"Error generating predictive analysis: {str(e)}")
            return {
                "survey_id": survey_id,
                "error": str(e)
            }
    
    async def _forecast_time_series(
        self,
        survey_id: int,
        responses: List[Dict[str, Any]],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Forecast future response rates based on historical patterns.
        
        Args:
            survey_id: Survey ID
            responses: List of survey responses
            historical_data: Optional historical data
            
        Returns:
            Time series forecasts
        """
        try:
            # Extract timestamps and create a time series
            timestamps = []
            for response in responses:
                ts = response.get("submitted_at")
                if ts:
                    if isinstance(ts, str):
                        try:
                            timestamps.append(pd.to_datetime(ts))
                        except:
                            continue
                    else:
                        timestamps.append(pd.to_datetime(ts))
            
            if not timestamps:
                return {"status": "insufficient_data"}
            
            # Create a DataFrame with timestamps
            df = pd.DataFrame({"timestamp": timestamps})
            df = df.sort_values("timestamp")
            
            # Group by day and count responses
            daily_counts = df.groupby(df["timestamp"].dt.date).size()
            
            # If we don't have enough data points, return a simple average projection
            if len(daily_counts) < 7:  # Need at least a week of data for meaningful forecast
                avg_daily = daily_counts.mean()
                forecast = {"mean": [float(avg_daily)] * 7}
                confidence_intervals = None
                method = "average"
            else:
                # Use ARIMA for forecasting if we have enough data
                daily_series = pd.Series(daily_counts.values, index=pd.DatetimeIndex(daily_counts.index))
                
                # Fit ARIMA model
                model = ARIMA(daily_series, order=(1, 1, 1))
                model_fit = model.fit()
                
                # Forecast next 7 days
                forecast = model_fit.forecast(steps=7)
                confidence_intervals = model_fit.get_forecast(steps=7).conf_int()
                
                method = "ARIMA(1,1,1)"
            
            # Create forecast result
            if method == "average":
                forecast_result = {
                    "method": method,
                    "forecast_days": 7,
                    "values": [float(forecast["mean"])] * 7,
                    "dates": [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
                }
            else:
                forecast_result = {
                    "method": method,
                    "forecast_days": 7,
                    "values": forecast.tolist(),
                    "dates": [(forecast.index[i]).strftime('%Y-%m-%d') for i in range(len(forecast))],
                    "confidence_intervals": {
                        "lower": confidence_intervals.iloc[:, 0].tolist(),
                        "upper": confidence_intervals.iloc[:, 1].tolist()
                    }
                }
            
            # Generate visualization
            forecast_chart = self._create_forecast_chart(daily_series, forecast, confidence_intervals)
            forecast_result["visualization"] = forecast_chart
            
            return forecast_result
        except Exception as e:
            logger.error(f"Error forecasting time series: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _create_forecast_chart(
        self,
        historical_data: pd.Series,
        forecast: Union[pd.Series, Dict[str, Any]],
        confidence_intervals: Optional[pd.DataFrame] = None
    ) -> str:
        """Create a forecast visualization chart."""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data.values, 'b-', label='Historical')
        
        # Plot forecast
        if isinstance(forecast, pd.Series):
            plt.plot(forecast.index, forecast.values, 'r--', label='Forecast')
            
            # Plot confidence intervals if available
            if confidence_intervals is not None:
                plt.fill_between(
                    forecast.index,
                    confidence_intervals.iloc[:, 0],
                    confidence_intervals.iloc[:, 1],
                    color='r', alpha=0.1,
                    label='95% Confidence Interval'
                )
        else:
            # Simple average forecast
            last_date = historical_data.index[-1]
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(7)]
            plt.plot(forecast_dates, [forecast["mean"]] * 7, 'r--', label='Forecast')
        
        plt.title('Response Rate Forecast')
        plt.xlabel('Date')
        plt.ylabel('Response Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    async def _predict_metric_trends(
        self,
        survey_id: int,
        metrics: List[Dict[str, Any]],
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict future trends for each metric.
        
        Args:
            survey_id: Survey ID
            metrics: List of metric data
            responses: List of survey responses
            
        Returns:
            Metric trend predictions
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, you would extract the actual metric values
            # from the responses based on your data model and timestamps
            
            # Extract responses with timestamps
            dated_responses = []
            for response in responses:
                ts = response.get("submitted_at")
                if ts:
                    try:
                        if isinstance(ts, str):
                            timestamp = pd.to_datetime(ts)
                        else:
                            timestamp = pd.to_datetime(ts)
                        
                        dated_responses.append({
                            "timestamp": timestamp,
                            "response": response
                        })
                    except Exception:
                        continue
            
            # Sort by timestamp
            dated_responses.sort(key=lambda x: x["timestamp"])
            
            # If we don't have enough data, return empty result
            if len(dated_responses) < 10:
                return {"status": "insufficient_data"}
            
            # Group responses by week
            weekly_data = {}
            for dated_response in dated_responses:
                week = dated_response["timestamp"].strftime("%Y-%U")
                if week not in weekly_data:
                    weekly_data[week] = []
                weekly_data[week].append(dated_response["response"])
            
            # Calculate metric values for each week
            metric_trends = {}
            for metric in metrics:
                metric_id = metric.get("id")
                if not metric_id:
                    continue
                
                # Initialize trend data for this metric
                metric_trends[metric_id] = {
                    "weeks": [],
                    "values": [],
                    "trend": None,
                    "forecast": []
                }
                
                # Extract weekly values for this metric
                for week, week_responses in weekly_data.items():
                    # This is a placeholder - you'd need to implement actual value extraction
                    # based on your data model
                    values = []
                    for response in week_responses:
                        if "metrics" in response and metric_id in response["metrics"]:
                            value = response["metrics"][metric_id]
                            if isinstance(value, (int, float)):
                                values.append(float(value))
                        elif "value" in response and isinstance(response["value"], (int, float)):
                            # Fallback - just use a generic value for testing
                            values.append(float(response["value"]))
                    
                    if values:
                        metric_trends[metric_id]["weeks"].append(week)
                        metric_trends[metric_id]["values"].append(np.mean(values))
                
                # Calculate trend if we have enough data points
                if len(metric_trends[metric_id]["values"]) >= 3:
                    # Simple linear regression for trend
                    X = np.arange(len(metric_trends[metric_id]["values"])).reshape(-1, 1)
                    y = np.array(metric_trends[metric_id]["values"])
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Determine trend direction
                    slope = model.coef_[0]
                    if abs(slope) < 0.01:
                        trend = "stable"
                    elif slope > 0:
                        trend = "increasing"
                    else:
                        trend = "decreasing"
                    
                    # Calculate trend strength (r-squared)
                    r_squared = model.score(X, y)
                    
                    # Make forecast for next 4 weeks
                    future_X = np.arange(len(X), len(X) + 4).reshape(-1, 1)
                    future_y = model.predict(future_X)
                    
                    # Store trend information
                    metric_trends[metric_id]["trend"] = {
                        "direction": trend,
                        "slope": float(slope),
                        "r_squared": float(r_squared)
                    }
                    
                    # Store forecast
                    metric_trends[metric_id]["forecast"] = future_y.tolist()
                    
                    # Generate next 4 week labels
                    last_week = pd.to_datetime(metric_trends[metric_id]["weeks"][-1] + "-0", format="%Y-%U-%w")
                    future_weeks = [(last_week + timedelta(weeks=i+1)).strftime("%Y-%U") for i in range(4)]
                    metric_trends[metric_id]["forecast_weeks"] = future_weeks
                    
                    # Generate visualization
                    trend_chart = self._create_trend_chart(
                        metric_trends[metric_id]["weeks"],
                        metric_trends[metric_id]["values"],
                        future_weeks,
                        future_y.tolist(),
                        metric.get("name", metric_id)
                    )
                    metric_trends[metric_id]["visualization"] = trend_chart
            
            return {"metric_trends": metric_trends}
        except Exception as e:
            logger.error(f"Error predicting metric trends: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _create_trend_chart(
        self,
        weeks: List[str],
        values: List[float],
        forecast_weeks: List[str],
        forecast_values: List[float],
        metric_name: str
    ) -> str:
        """Create a trend visualization chart."""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        x_values = range(len(weeks))
        plt.plot(x_values, values, 'b-', label='Historical')
        
        # Plot forecast
        forecast_x = range(len(weeks), len(weeks) + len(forecast_weeks))
        plt.plot(forecast_x, forecast_values, 'r--', label='Forecast')
        
        # Set x-axis labels
        all_weeks = weeks + forecast_weeks
        if len(all_weeks) > 10:
            # If too many weeks, show only some labels
            plt.xticks(
                range(0, len(all_weeks), len(all_weeks) // 10),
                [all_weeks[i] for i in range(0, len(all_weeks), len(all_weeks) // 10)],
                rotation=45
            )
        else:
            plt.xticks(range(len(all_weeks)), all_weeks, rotation=45)
        
        plt.title(f'Trend Analysis: {metric_name}')
        plt.xlabel('Week')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    async def _analyze_cohorts(
        self,
        survey_id: int,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze respondents by cohorts.
        
        Args:
            survey_id: Survey ID
            responses: List of survey responses
            
        Returns:
            Cohort analysis results
        """
        try:
            # Check if we have enough responses
            if len(responses) < 20:
                return {"status": "insufficient_data"}
            
            # Extract demographic information if available
            # This is a placeholder - you would extract actual demographics based on your data model
            demographics = {}
            for response in responses:
                # Try to extract demographic data
                demo_data = response.get("demographics", {})
                for key, value in demo_data.items():
                    if key not in demographics:
                        demographics[key] = {}
                    
                    str_value = str(value)
                    if str_value not in demographics[key]:
                        demographics[key][str_value] = 0
                    demographics[key][str_value] += 1
            
            # If no demographic data found, try to generate cohorts based on response patterns
            if not demographics:
                cohorts = await self._generate_response_cohorts(responses)
                return {
                    "cohort_type": "response_based",
                    "cohorts": cohorts
                }
            
            # Calculate most significant demographic segments
            segments = []
            for demo_key, values in demographics.items():
                # Calculate distribution
                total = sum(values.values())
                distribution = {}
                for value, count in values.items():
                    distribution[value] = {
                        "count": count,
                        "percentage": round(count / total * 100, 2)
                    }
                
                segments.append({
                    "demographic": demo_key,
                    "values": distribution,
                    "diversity_score": len(values) / total if total > 0 else 0
                })
            
            # Sort segments by diversity score (lower is more concentrated)
            segments.sort(key=lambda x: x["diversity_score"])
            
            # Generate visualizations for top demographics
            visualizations = {}
            for segment in segments[:3]:  # Show top 3 demographics
                viz = self._create_demographic_chart(segment["demographic"], segment["values"])
                visualizations[segment["demographic"]] = viz
            
            return {
                "cohort_type": "demographic",
                "segments": segments,
                "visualizations": visualizations
            }
        except Exception as e:
            logger.error(f"Error analyzing cohorts: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_response_cohorts(
        self,
        responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate cohorts based on response patterns."""
        try:
            # Extract features for clustering
            features = []
            for response in responses:
                # This is a placeholder - you'd extract relevant features for clustering
                # For example, average scores, response length, response completion, etc.
                response_data = response.get("responses", {})
                if not response_data:
                    continue
                
                # Simple feature extraction
                avg_value = 0
                count = 0
                for value in response_data.values():
                    if isinstance(value, (int, float)):
                        avg_value += value
                        count += 1
                
                if count > 0:
                    avg_value /= count
                
                # Calculate completion rate
                total_questions = 10  # Placeholder, should be determined from survey structure
                completion_rate = len(response_data) / total_questions if total_questions > 0 else 0
                
                # Store features
                features.append([avg_value, completion_rate])
            
            if not features:
                return []
            
            # Convert to numpy array
            X = np.array(features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters (using simple elbow method)
            max_clusters = min(5, len(X) // 10)  # Limit based on data size
            if max_clusters < 2:
                max_clusters = 2
            
            inertia = []
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            
            # Choose optimal k (for simplicity, just use k=3)
            k = 3
            if max_clusters < 3:
                k = max_clusters
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Analyze each cluster
            cluster_stats = {}
            for i in range(k):
                cluster_indices = np.where(clusters == i)[0]
                cluster_stats[i] = {
                    "size": len(cluster_indices),
                    "percentage": round(len(cluster_indices) / len(clusters) * 100, 2),
                    "avg_values": np.mean(X[cluster_indices], axis=0).tolist(),
                }
            
            # Generate cluster descriptions
            cluster_descriptions = []
            for i, stats in cluster_stats.items():
                avg_value, avg_completion = stats["avg_values"]
                
                # Generate a description based on the cluster characteristics
                if avg_completion < 0.3:
                    completion_desc = "low completion"
                elif avg_completion < 0.7:
                    completion_desc = "partial completion"
                else:
                    completion_desc = "high completion"
                
                if avg_value < 0.3:
                    value_desc = "low scores"
                elif avg_value < 0.7:
                    value_desc = "moderate scores"
                else:
                    value_desc = "high scores"
                
                description = f"Cohort {i+1}: {completion_desc}, {value_desc}"
                
                cluster_descriptions.append({
                    "cluster_id": i,
                    "description": description,
                    "size": stats["size"],
                    "percentage": stats["percentage"],
                    "characteristics": {
                        "avg_value": avg_value,
                        "completion_rate": avg_completion
                    }
                })
            
            # Generate visualization
            cohort_chart = self._create_cohort_chart(X, clusters, k)
            
            result = {
                "method": "k-means",
                "optimal_k": k,
                "cohorts": cluster_descriptions,
                "visualization": cohort_chart
            }
            
            return result
        except Exception as e:
            logger.error(f"Error generating response cohorts: {str(e)}")
            return {"error": str(e)}
    
    def _create_demographic_chart(
        self,
        demographic: str,
        values: Dict[str, Dict[str, Any]]
    ) -> str:
        """Create a demographic distribution chart."""
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        categories = list(values.keys())
        counts = [values[c]["count"] for c in categories]
        
        # Limit to top 10 categories if there are too many
        if len(categories) > 10:
            # Sort by count and take top 10
            sorted_indices = np.argsort(counts)[::-1]
            categories = [categories[i] for i in sorted_indices[:10]]
            counts = [counts[i] for i in sorted_indices[:10]]
        
        plt.bar(categories, counts, alpha=0.7, color='green')
        plt.title(f'Distribution by {demographic}')
        plt.xlabel(demographic)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def _create_cohort_chart(
        self,
        X: np.ndarray,
        clusters: np.ndarray,
        k: int
    ) -> str:
        """Create a cohort visualization chart."""
        plt.figure(figsize=(10, 8))
        
        # Plot each cluster with a different color
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i in range(k):
            cluster_points = X[clusters == i]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1], 
                c=colors[i % len(colors)],
                label=f'Cohort {i+1}',
                alpha=0.7
            )
        
        plt.title('Response Cohorts')
        plt.xlabel('Average Value')
        plt.ylabel('Completion Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    async def _generate_recommendations(
        self,
        survey_id: int,
        metrics: List[Dict[str, Any]],
        time_series_forecasts: Dict[str, Any],
        metric_trends: Dict[str, Any],
        cohort_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered recommendations based on predictive analysis.
        
        Args:
            survey_id: Survey ID
            metrics: List of metric data
            time_series_forecasts: Time series forecast results
            metric_trends: Metric trend prediction results
            cohort_analysis: Cohort analysis results
            
        Returns:
            AI-generated recommendations
        """
        try:
            # Create a map of metric IDs to names
            metric_names = {m["id"]: m["name"] for m in metrics if "id" in m and "name" in m}
            
            # Create prompt for the AI
            prompt = f"""
            Please analyze the predictive insights from this survey and provide recommendations:
            
            Time Series Forecast:
            {json.dumps(time_series_forecasts, indent=2)}
            
            Metric Trends:
            """
            
            # Add metric trends with proper names
            for metric_id, trend_data in metric_trends.get("metric_trends", {}).items():
                if "trend" in trend_data:
                    metric_name = metric_names.get(metric_id, metric_id)
                    prompt += f"\n{metric_name}: {trend_data['trend']['direction']} (slope: {trend_data['trend']['slope']})"
            
            # Add cohort information
            prompt += f"""
            
            Cohort Analysis:
            {json.dumps(cohort_analysis, indent=2)}
            
            Based on this predictive analysis, please provide the following:
            1. A summary of the key trends and patterns
            2. Expected developments in the next few weeks
            3. Actionable recommendations for survey administrators
            4. Suggested improvements or interventions based on the forecasts
            5. Specific actions to take for different cohorts of respondents
            """
            
            # Call the OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a survey data analyst providing predictive insights and recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            # Extract the response
            recommendation_text = response.choices[0].message.content
            
            # Structure the recommendations
            recommendations = {
                "text": recommendation_text,
                "generated_at": pd.Timestamp.now().isoformat()
            }
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}

# Create a singleton instance
predictive_analysis_service = PredictiveAnalysisService() 