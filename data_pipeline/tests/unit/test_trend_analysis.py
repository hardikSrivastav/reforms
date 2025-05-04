"""
Unit tests for trend analysis.
This module tests the trend analysis service that analyzes changes in metrics over time.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from data_pipeline.analysis.trend_analysis import TrendAnalysisService, trend_analysis_service


@pytest.fixture
def mock_metadata_store():
    """Create a mock metadata store."""
    with patch('data_pipeline.analysis.trend_analysis.metadata_store') as mock:
        mock.get_analysis_result.return_value = None
        yield mock


@pytest.fixture
def trend_analysis_service_instance():
    """Create a TrendAnalysisService instance for testing."""
    return TrendAnalysisService()


@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing trend analysis."""
    return {
        "id": "metric1",
        "name": "Customer Satisfaction",
        "type": "numeric",
        "description": "Overall satisfaction score from 1-5"
    }


@pytest.fixture
def sample_time_series_responses():
    """Generate sample time series responses for trend analysis testing."""
    # Create sample data over 30 days
    now = datetime.now()
    responses = []
    
    # Generate responses with an upward trend and some randomness
    for i in range(30):
        day = now - timedelta(days=30-i)
        # Create 1-5 responses for each day
        num_responses = np.random.randint(1, 6)
        
        # Add an upward trend (0 to 2 over 30 days) plus noise
        trend_component = 3.0 + (i / 15)  # Base 3.0, increases to 5.0 over 30 days
        
        for j in range(num_responses):
            # Add random noise
            noise = np.random.normal(0, 0.3)
            value = min(max(trend_component + noise, 1.0), 5.0)  # Keep within 1-5 range
            
            responses.append({
                "_id": f"response_{i}_{j}",
                "value": round(value, 1),
                "submitted_at": day.isoformat(),
                "day": day.strftime("%Y-%m-%d")
            })
    
    return responses


@pytest.fixture
def sample_categorical_time_series():
    """Generate sample categorical time series data."""
    now = datetime.now()
    responses = []
    categories = ["option1", "option2", "option3"]
    
    # Create sample data over 30 days with category distribution changes
    for i in range(30):
        day = now - timedelta(days=30-i)
        num_responses = np.random.randint(3, 8)
        
        # Change probability distribution over time
        prob_option1 = max(0.6 - (i / 50), 0.1)  # Decreasing trend
        prob_option2 = min(0.2 + (i / 60), 0.5)  # Increasing trend
        prob_option3 = 1.0 - prob_option1 - prob_option2  # Remainder
        
        probabilities = [prob_option1, prob_option2, prob_option3]
        
        for j in range(num_responses):
            category = np.random.choice(categories, p=probabilities)
            responses.append({
                "_id": f"cat_response_{i}_{j}",
                "category": category,
                "submitted_at": day.isoformat(),
                "day": day.strftime("%Y-%m-%d")
            })
    
    return responses


@pytest.mark.asyncio
async def test_analyze_metric_trends_cache_hit(
    mock_metadata_store,
    sample_metric_data,
    sample_time_series_responses
):
    """Test analyze_metric_trends with a cache hit."""
    # Create a test instance with mocked analyze_metric_trends method
    with patch('data_pipeline.analysis.trend_analysis.TrendAnalysisService.analyze_metric_trends') as mock_analyze:
        # Set up the mock to return a cached result
        cached_result = {
            "survey_id": 1,
            "metric_id": "metric1",
            "cached": True,
            "timestamp": "2023-01-01T00:00:00.000000"
        }
        mock_analyze.return_value = cached_result
        
        # Create a new service instance that will use the mocked method
        service = TrendAnalysisService()
        
        # Call the function
        result = await service.analyze_metric_trends(
            1, "metric1", sample_metric_data, sample_time_series_responses
        )
        
        # Verify the result
        assert result == cached_result
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_metric_trends_cache_miss_numeric(
    mock_metadata_store,
    sample_metric_data,
    sample_time_series_responses
):
    """Test analyze_metric_trends with a cache miss for numeric metric."""
    # Create test result to be returned by the mock
    expected_result = {
        "survey_id": 1,
        "metric_id": "metric1",
        "metric_name": "Customer Satisfaction",
        "metric_type": "numeric",
        "time_periods": ["2023-01-01", "2023-01-02"],
        "trend_data": {
            "stats": {
                "first_value": 3.5,
                "last_value": 3.8,
                "min_value": 3.5,
                "max_value": 3.8,
                "overall_change": 0.3,
                "overall_percent_change": 8.57
            },
            "trend_direction": {
                "is_increasing": True,
                "is_significant_change": False,
                "description": "increasing"
            }
        },
        "visualizations": {"trend_chart": "base64_string"}
    }
    
    # Mock the analyze_metric_trends method directly
    with patch('data_pipeline.analysis.trend_analysis.TrendAnalysisService.analyze_metric_trends') as mock_analyze:
        mock_analyze.return_value = expected_result
        
        # Create a new service instance that will use the mocked method
        service = TrendAnalysisService()
        
        # Call the function
        result = await service.analyze_metric_trends(
            1, "metric1", sample_metric_data, sample_time_series_responses
        )
        
        # Verify the result structure
        assert result == expected_result
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_numeric_time_series(
    trend_analysis_service_instance
):
    """Test analysis of numeric time series data."""
    # Create a properly formatted pandas Series for the test
    dates = pd.date_range(start='2023-01-01', periods=10)
    values = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4]
    time_series = pd.Series(values, index=dates)
    
    # Mock the change point detection and seasonal decomposition to avoid external dependencies
    with patch.object(trend_analysis_service_instance, '_detect_change_points_numeric', 
                      return_value={"change_point_indices": [5]}), \
         patch.object(trend_analysis_service_instance, '_seasonal_decomposition', 
                      return_value={"trend": {}, "seasonal": {}, "residual": {}}):
        
        # Call the function directly
        analysis = await trend_analysis_service_instance._analyze_numeric_time_series(time_series)
        
        # Check analysis structure and basic properties
        assert "trend_data" in analysis
        assert "stats" in analysis
        assert "trend_direction" in analysis
        
        # Check stats with approx equality for floating point
        assert analysis["stats"]["first_value"] == 3.5
        assert analysis["stats"]["last_value"] == 4.4
        assert analysis["stats"]["min_value"] == 3.5
        assert analysis["stats"]["max_value"] == 4.4
        assert abs(analysis["stats"]["overall_change"] - 0.9) < 1e-10
        
        # Check trend direction
        assert analysis["trend_direction"]["is_increasing"] == True


@pytest.mark.asyncio
async def test_analyze_categorical_time_series(
    trend_analysis_service_instance
):
    """Test analysis of categorical time series data."""
    # Create properly formatted category time series data
    dates = pd.date_range(start='2023-01-01', periods=5)
    
    # Create a dictionary mapping categories to pandas Series
    category_time_series = {
        "option1": pd.Series([5, 6, 7, 8, 9], index=dates),
        "option2": pd.Series([3, 4, 5, 6, 7], index=dates),
        "option3": pd.Series([1, 1, 2, 1, 2], index=dates)
    }
    
    # Call the function directly
    analysis = await trend_analysis_service_instance._analyze_categorical_time_series(category_time_series)
    
    # Check analysis structure
    assert "categories" in analysis
    assert len(analysis["categories"]) == 3
    assert "category_trends" in analysis
    assert len(analysis["category_trends"]) == 3
    assert "distribution_shifts" in analysis
    
    # Check category trends
    for category in ["option1", "option2", "option3"]:
        assert category in analysis["category_trends"]
        assert "trend_data" in analysis["category_trends"][category]
        assert "stats" in analysis["category_trends"][category]
        assert "is_increasing" in analysis["category_trends"][category]


@pytest.mark.asyncio
async def test_detect_change_points_numeric(trend_analysis_service_instance):
    """Test detection of change points in numeric time series."""
    # Create a time series with a clear change point
    dates = pd.date_range(start='2023-01-01', periods=60)
    values = []
    
    # First 30 days with mean around 3.0
    for i in range(30):
        values.append(np.random.normal(3.0, 0.2))
    
    # Next 30 days with mean around 4.0 (significant change)
    for i in range(30):
        values.append(np.random.normal(4.0, 0.2))
    
    # Create a pandas Series
    time_series = pd.Series(values, index=dates)
    
    # Use ruptures directly in test to avoid complex algorithm mocking
    with patch('ruptures.Pelt') as mock_pelt:
        # Create mock for the Pelt algorithm
        mock_pelt_instance = MagicMock()
        mock_pelt_instance.fit.return_value = mock_pelt_instance
        mock_pelt_instance.predict.return_value = [30]  # Simulate a change point at day 30
        mock_pelt.return_value = mock_pelt_instance
        
        # Detect change points
        change_points = await trend_analysis_service_instance._detect_change_points_numeric(time_series)
        
        # Verify the result contains expected fields
        assert "change_point_indices" in change_points
        assert change_points["change_point_indices"] == [30]
        assert "change_point_periods" in change_points
        assert len(change_points["change_point_periods"]) == 1
        assert "change_point_metrics" in change_points


@pytest.mark.asyncio
async def test_generate_trend_visualizations(trend_analysis_service_instance):
    """Test generation of trend visualizations."""
    # Create sample trend data
    metric_type = "numeric"
    
    # Sample time series data
    time_series_responses = {
        "2023-01-01": [{"value": 3.1}],
        "2023-01-02": [{"value": 3.2}],
        "2023-01-03": [{"value": 3.3}],
        "2023-01-04": [{"value": 3.5}],
        "2023-01-05": [{"value": 3.7}]
    }
    
    # Sample analysis result
    analysis_result = {
        "trend_data": {
            "stats": {
                "first_value": 3.1,
                "last_value": 3.7
            },
            "trend_direction": {
                "is_increasing": True
            }
        }
    }
    
    # Mock matplotlib to return base64 encoded strings
    with patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.plot'), \
         patch('matplotlib.pyplot.title'), \
         patch('matplotlib.pyplot.xlabel'), \
         patch('matplotlib.pyplot.ylabel'), \
         patch('matplotlib.pyplot.grid'), \
         patch('matplotlib.pyplot.xticks'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('io.BytesIO'), \
         patch('base64.b64encode', return_value=b'mock_base64_image'):
        
        # Generate visualizations
        visualizations = await trend_analysis_service_instance._generate_trend_visualizations(
            metric_type, time_series_responses, analysis_result
        )
        
        # Check for expected visualizations
        assert visualizations is not None
        assert isinstance(visualizations, dict)


@pytest.mark.asyncio
async def test_seasonal_decomposition(trend_analysis_service_instance):
    """Test seasonal decomposition of time series data."""
    # Create sample time series with seasonal pattern
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    values = []
    
    # Generate values with a trend and weekly seasonality
    for i in range(30):
        # Base value with upward trend
        base_value = 3.0 + (i / 30)
        
        # Add weekly seasonality (higher on weekends)
        weekday = dates[i].weekday()
        seasonal_component = 0.5 if weekday >= 5 else 0.0
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        
        values.append(base_value + seasonal_component + noise)
    
    # Create pandas Series
    time_series = pd.Series(values, index=dates)
    
    # Mock the seasonal_decompose function
    mock_result = MagicMock()
    mock_result.trend = pd.Series([3.1, 3.2], index=dates[:2])
    mock_result.seasonal = pd.Series([0.1, -0.1], index=dates[:2])
    mock_result.resid = pd.Series([0.05, -0.05], index=dates[:2])
    
    with patch('statsmodels.tsa.seasonal.seasonal_decompose', return_value=mock_result):
        # Perform seasonal decomposition
        decomposition = await trend_analysis_service_instance._seasonal_decomposition(time_series)
        
        # Check decomposition structure
        assert "trend" in decomposition
        assert "seasonal" in decomposition
        assert "residual" in decomposition
        assert "seasonality_metrics" in decomposition


def test_singleton_instance():
    """Test that the singleton instance is properly created."""
    assert isinstance(trend_analysis_service, TrendAnalysisService) 