"""
Unit tests for the base analysis service.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from data_pipeline.analysis.base_analysis import BaseAnalysisService, base_analysis_service

@pytest.fixture
def mock_metadata_store():
    """Create a mock metadata store."""
    with patch('data_pipeline.analysis.base_analysis.metadata_store') as mock:
        yield mock

@pytest.fixture
def base_analysis_service_instance():
    """Create a BaseAnalysisService instance for testing."""
    return BaseAnalysisService()

@pytest.fixture
def sample_survey_data():
    """Sample survey data for testing."""
    return {
        "id": "test_survey",
        "title": "Test Survey",
        "questions": [
            {
                "id": "q1",
                "question": "How would you rate our service?",
                "type": "rating",
                "options": [
                    {"value": 1, "text": "Poor"},
                    {"value": 2, "text": "Fair"},
                    {"value": 3, "text": "Good"},
                    {"value": 4, "text": "Excellent"}
                ]
            },
            {
                "id": "q2",
                "question": "What did you like about our service?",
                "type": "text"
            },
            {
                "id": "q3",
                "question": "Which features did you use?",
                "type": "checkbox",
                "options": [
                    {"value": "feature1", "text": "Feature 1"},
                    {"value": "feature2", "text": "Feature 2"},
                    {"value": "feature3", "text": "Feature 3"}
                ]
            }
        ]
    }

@pytest.fixture
def sample_responses():
    """Sample survey responses for testing."""
    now = datetime.now()
    return [
        {
            "_id": "r1",
            "survey_mongo_id": "test_survey",
            "submitted_at": (now - timedelta(days=5)).isoformat(),
            "completed": True,
            "responses": {
                "q1": 4,
                "q2": "Great service! Very helpful.",
                "q3": ["feature1", "feature3"]
            }
        },
        {
            "_id": "r2",
            "survey_mongo_id": "test_survey",
            "submitted_at": (now - timedelta(days=3)).isoformat(),
            "completed": True,
            "responses": {
                "q1": 3,
                "q2": "Good overall experience.",
                "q3": ["feature2"]
            }
        },
        {
            "_id": "r3",
            "survey_mongo_id": "test_survey",
            "submitted_at": (now - timedelta(days=1)).isoformat(),
            "completed": False,
            "responses": {
                "q1": 2
            }
        }
    ]

@pytest.mark.asyncio
async def test_analyze_survey_cache_hit(
    base_analysis_service_instance, 
    mock_metadata_store, 
    sample_survey_data, 
    sample_responses
):
    """Test analyze_survey with a cache hit."""
    # Set up mock to return a cached result
    cached_result = {"survey_id": 1, "cached": True}
    mock_metadata_store.get_analysis_result.return_value = cached_result
    
    # Call the function
    result = await base_analysis_service_instance.analyze_survey(1, sample_survey_data, sample_responses)
    
    # Verify the result
    assert result == cached_result
    mock_metadata_store.get_analysis_result.assert_called_once_with("base_analysis", 1)
    # Store should not be called for a cache hit
    mock_metadata_store.store_analysis_result.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_survey_cache_miss(
    base_analysis_service_instance, 
    mock_metadata_store, 
    sample_survey_data, 
    sample_responses
):
    """Test analyze_survey with a cache miss."""
    # Set up mock to return no cached result
    mock_metadata_store.get_analysis_result.return_value = None
    
    # Call the function
    result = await base_analysis_service_instance.analyze_survey(1, sample_survey_data, sample_responses)
    
    # Verify the result
    assert result["survey_id"] == 1
    assert result["response_count"] == 3
    assert "completion_stats" in result
    assert "question_stats" in result
    assert "time_series" in result
    
    # Verify cache operations
    mock_metadata_store.get_analysis_result.assert_called_once_with("base_analysis", 1)
    mock_metadata_store.store_analysis_result.assert_called_once()

@pytest.mark.asyncio
async def test_calculate_completion_stats(
    base_analysis_service_instance,
    sample_responses
):
    """Test calculation of completion statistics."""
    stats = await base_analysis_service_instance.calculate_completion_stats(sample_responses)
    
    assert stats["full_completions"] == 2
    assert stats["partial_completions"] == 1
    assert stats["abandoned"] == 0
    # Use pytest.approx() for floating-point comparison to handle rounding differences
    assert stats["completion_rate"] == pytest.approx(2/3 * 100, abs=0.01)  # Approximately 66.67%

@pytest.mark.asyncio
async def test_calculate_question_stats(
    base_analysis_service_instance,
    sample_survey_data,
    sample_responses
):
    """Test calculation of question statistics."""
    stats = await base_analysis_service_instance.calculate_question_stats(sample_survey_data, sample_responses)
    
    # Check if all questions are included
    assert "q1" in stats
    assert "q2" in stats
    assert "q3" in stats
    
    # Check rating question stats
    assert stats["q1"]["question_type"] == "rating"
    assert stats["q1"]["response_rate"] == 100.0  # All 3 responses answered q1
    
    # Check text question stats
    assert stats["q2"]["question_type"] == "text"
    # Use pytest.approx() for floating-point comparison
    assert stats["q2"]["response_rate"] == pytest.approx(2/3 * 100, abs=0.01)  # Approximately 66.67%
    
    # Check checkbox question stats
    assert stats["q3"]["question_type"] == "checkbox"
    # Use pytest.approx() for floating-point comparison
    assert stats["q3"]["response_rate"] == pytest.approx(2/3 * 100, abs=0.01)  # Approximately 66.67%

@pytest.mark.asyncio
async def test_generate_time_series(
    base_analysis_service_instance,
    sample_responses
):
    """Test generation of time series data."""
    time_series = await base_analysis_service_instance.generate_time_series(sample_responses)
    
    assert "daily" in time_series
    assert "weekly" in time_series
    assert "monthly" in time_series
    assert len(time_series["daily"]) == 3  # 3 different days

def test_singleton_instance():
    """Test that the singleton instance is properly created."""
    assert isinstance(base_analysis_service, BaseAnalysisService) 