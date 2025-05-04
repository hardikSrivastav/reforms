"""
Unit tests for base analysis visualization functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from data_pipeline.analysis.base_analysis import BaseAnalysisService

@pytest.fixture
def base_analysis_service_instance():
    """Create a BaseAnalysisService instance for testing."""
    return BaseAnalysisService()

@pytest.fixture
def numeric_data():
    """Sample numeric data for testing histograms."""
    return [2.5, 3.2, 3.7, 4.1, 2.9, 3.5, 3.8, 4.3, 2.7, 3.9]

@pytest.fixture
def time_series_data():
    """Sample time series data for testing."""
    now = datetime.now()
    dates = [(now - timedelta(days=i)).date() for i in range(7)]
    responses = []
    
    # Create responses for the past 7 days
    for i, date in enumerate(dates):
        # Add more responses for some days to create a pattern
        count = (i % 3) + 1
        for j in range(count):
            responses.append({
                "_id": f"r_{date}_{j}",
                "submitted_at": datetime.combine(date, datetime.min.time()).isoformat()
            })
    
    return responses

@pytest.mark.asyncio
async def test_generate_histogram(base_analysis_service_instance, numeric_data):
    """Test histogram generation function."""
    histogram = base_analysis_service_instance._generate_histogram(numeric_data)
    
    # Check if the histogram has the expected structure
    assert isinstance(histogram, dict)
    assert len(histogram) > 0
    
    # Check if all histogram bins have count and percentage
    for bin_label, bin_data in histogram.items():
        assert "count" in bin_data
        assert "percentage" in bin_data
        assert isinstance(bin_data["count"], int)
        assert isinstance(bin_data["percentage"], float)
        assert 0 <= bin_data["percentage"] <= 100  # Percentage should be between 0 and 100

@pytest.mark.asyncio
async def test_generate_time_series(base_analysis_service_instance, time_series_data):
    """Test time series generation function."""
    time_series = await base_analysis_service_instance.generate_time_series(time_series_data)
    
    # Check if the time series has the expected structure
    assert "daily" in time_series
    assert "weekly" in time_series
    assert "monthly" in time_series
    
    # Check daily data
    assert isinstance(time_series["daily"], dict)
    assert len(time_series["daily"]) > 0
    
    # Check if total days is calculated correctly
    assert time_series["total_days"] == len(time_series["daily"])
    
    # Check if most active day and least active day are included
    assert "most_active_day" in time_series
    assert "least_active_day" in time_series

@pytest.mark.asyncio
async def test_numeric_question_stats(base_analysis_service_instance):
    """Test numeric question statistics visualization data."""
    # Sample numeric responses
    answers = [2, 3, 4, 5, 3, 4, 2, 3, 5, 4]
    
    stats = base_analysis_service_instance._numeric_question_stats(answers)
    
    # Check basic statistics
    assert stats["response_count"] == len(answers)
    assert stats["min"] == min(answers)
    assert stats["max"] == max(answers)
    assert "mean" in stats
    assert "median" in stats
    assert "std_dev" in stats
    
    # Check if distribution data is included
    assert "distribution" in stats
    assert isinstance(stats["distribution"], dict)
    assert len(stats["distribution"]) > 0

@pytest.mark.asyncio
async def test_multi_choice_stats(base_analysis_service_instance):
    """Test multi-choice question statistics visualization data."""
    # Sample multi-choice answers and options
    answers = [
        ["option1", "option3"],
        ["option2"],
        ["option1", "option2"],
        ["option3"],
        ["option1", "option3"],
    ]
    
    options = [
        {"value": "option1", "text": "Option 1"},
        {"value": "option2", "text": "Option 2"},
        {"value": "option3", "text": "Option 3"},
    ]
    
    stats = base_analysis_service_instance._multi_choice_stats(answers, options)
    
    # Check basic statistics
    assert stats["response_count"] == len(answers)
    assert "selection_count" in stats
    assert "avg_selections_per_response" in stats
    
    # Check distribution data
    assert "distribution" in stats
    assert len(stats["distribution"]) == 3  # Should have 3 options
    
    # Check if all options are included
    for option in options:
        option_id = option["value"]
        assert option_id in stats["distribution"]
        assert "count" in stats["distribution"][option_id]
        assert "percentage_of_responses" in stats["distribution"][option_id]
        assert "percentage_of_selections" in stats["distribution"][option_id]
        assert "text" in stats["distribution"][option_id]

@pytest.mark.asyncio
async def test_single_choice_stats(base_analysis_service_instance):
    """Test single-choice question statistics visualization data."""
    # Sample single-choice answers and options
    answers = ["option1", "option2", "option1", "option3", "option1", "option2"]
    
    options = [
        {"value": "option1", "text": "Option 1"},
        {"value": "option2", "text": "Option 2"},
        {"value": "option3", "text": "Option 3"},
    ]
    
    stats = base_analysis_service_instance._single_choice_stats(answers, options)
    
    # Check basic statistics
    assert stats["response_count"] == len(answers)
    assert "distribution" in stats
    assert "most_common" in stats
    assert stats["most_common"] == "option1"  # option1 appears 3 times
    
    # Check distribution data
    assert len(stats["distribution"]) == 3  # Should have 3 options
    
    # Check if all options are included with correct data
    for option in options:
        option_id = option["value"]
        assert option_id in stats["distribution"]
        assert "count" in stats["distribution"][option_id]
        assert "percentage" in stats["distribution"][option_id]
        assert "text" in stats["distribution"][option_id]
        
    # Check specific values
    assert stats["distribution"]["option1"]["count"] == 3
    assert stats["distribution"]["option2"]["count"] == 2
    assert stats["distribution"]["option3"]["count"] == 1 