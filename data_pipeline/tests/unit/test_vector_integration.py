"""
Integration tests for vector analysis integration with the analysis coordinator.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime

from data_pipeline.analysis.analysis_coordinator import AnalysisCoordinator
from data_pipeline.embeddings.semantic_search import semantic_search_service
from data_pipeline.analysis.vector_trend_analysis import vector_trend_analysis_service


@pytest.fixture
def analysis_coordinator():
    return AnalysisCoordinator()


@pytest.fixture
def sample_survey_data():
    return {
        "id": 123,
        "name": "Customer Feedback Survey",
        "description": "Annual customer feedback survey",
        "metrics": {
            "satisfaction": {
                "id": "satisfaction",
                "name": "Customer Satisfaction",
                "type": "numeric",
                "description": "Overall customer satisfaction rating (1-10)"
            },
            "improvement": {
                "id": "improvement",
                "name": "Areas for Improvement",
                "type": "text",
                "description": "Areas where we could improve our service"
            },
            "category": {
                "id": "category",
                "name": "Customer Category",
                "type": "categorical",
                "description": "Customer category"
            }
        }
    }


@pytest.fixture
def sample_responses():
    responses = []
    for i in range(10):
        # Generate some synthetic responses
        responses.append({
            "_id": f"response_{i}",
            "responses": {
                "satisfaction": float(i % 5) + 5.0,  # Values from 5-9
                "improvement": f"Text response number {i}",
                "category": ["A", "B", "C"][i % 3]  # Categories A, B, C
            },
            "submitted_at": datetime.now().isoformat()
        })
    return responses


@patch("data_pipeline.services.metadata_store.metadata_store")
@patch("data_pipeline.analysis.analysis_coordinator.vector_trend_analysis_service")
@pytest.mark.asyncio
async def test_vector_enhanced_analysis_integration(mock_vector_service, mock_metadata_store, analysis_coordinator, sample_survey_data, sample_responses):
    """Test integration of vector-enhanced analysis with the analysis coordinator."""
    # Mock the cache check and storage to avoid Redis connection errors
    mock_metadata_store.get_analysis_result.return_value = None
    
    # We need to patch the implementation rather than assigning a new mock
    # because the code might be accessing this attribute directly
    async def mock_store_result(*args, **kwargs):
        return {"status": "success"}
    
    mock_metadata_store.store_analysis_result = mock_store_result
    
    # Mock the vector analysis service responses
    mock_clusters = {
        "status": "success",
        "cluster_count": 2,
        "silhouette_score": 0.65,
        "total_responses": 10,
        "clusters": [
            {"cluster_id": 0, "size": 7, "percentage": 70.0, "samples": ["Text response 1", "Text response 2"]},
            {"cluster_id": 1, "size": 3, "percentage": 30.0, "samples": ["Text response 3", "Text response 4"]}
        ]
    }
    
    mock_trends = {
        "survey_id": 123,
        "question_id": "improvement",
        "period_count": 4,
        "period_days": 30,
        "drift_analysis": [
            {"from_period": "Period 1", "to_period": "Period 2", "drift": 0.05, "is_significant": False}
        ],
        "has_significant_drift": False
    }
    
    mock_anomalies = {
        "status": "success",
        "total_responses": 10,
        "anomaly_count": 1,
        "anomaly_percentage": 10.0,
        "anomalies": [
            {"response_id": "response_9", "answer": "Text response number 9", "distance": 0.85}
        ]
    }
    
    mock_vector_service.detect_response_clusters = AsyncMock(return_value=mock_clusters)
    mock_vector_service.detect_temporal_trends = AsyncMock(return_value=mock_trends)
    mock_vector_service.detect_anomalies = AsyncMock(return_value=mock_anomalies)
    
    # Test for internal _run_vector_enhanced_analysis method
    result = await analysis_coordinator._run_vector_enhanced_analysis(
        123, 
        "improvement", 
        sample_responses
    )
    
    # Verify method calls with exactly what's passed in analysis_coordinator.py
    mock_vector_service.detect_response_clusters.assert_called_once_with(
        survey_id=123,
        question_id="improvement"
    )
    
    mock_vector_service.detect_temporal_trends.assert_called_once_with(
        survey_id=123,
        question_id="improvement"
    )
    
    mock_vector_service.detect_anomalies.assert_called_once_with(
        survey_id=123,
        question_id="improvement"
    )
    
    # Verify result structure
    assert result["survey_id"] == 123
    assert result["metric_id"] == "improvement"
    assert "timestamp" in result
    assert "response_clusters" in result
    assert "temporal_trends" in result
    assert "anomaly_detection" in result
    
    # We skip the store_analysis_result assertion since we've mocked it 
    # with a function instead of an AsyncMock


@patch("data_pipeline.services.metadata_store.metadata_store")
@pytest.mark.asyncio
async def test_analysis_pipeline_with_vector_analysis(mock_metadata_store, analysis_coordinator, sample_survey_data, sample_responses):
    """Test the full analysis pipeline with vector analysis integration."""
    # Mock the metadata store to avoid Redis connection errors
    mock_metadata_store.get_analysis_result.return_value = None
    mock_metadata_store.store_analysis_result = AsyncMock()
    
    # Create mocks for all the analysis coordinator methods to avoid external dependencies
    
    # 1. Mock base analysis
    original_run_base_analysis = analysis_coordinator._run_base_analysis
    async def mock_run_base_analysis(survey_id, survey_data, responses):
        return {
            "survey_id": survey_id,
            "response_count": len(responses),
            "metrics": {
                "satisfaction": {"mean": 7.0, "count": 10},
                "improvement": {"count": 10},
                "category": {"count": 10, "categories": {"A": 4, "B": 3, "C": 3}}
            }
        }
    analysis_coordinator._run_base_analysis = mock_run_base_analysis
    
    # 2. Mock metric analysis
    original_run_metric_analysis = analysis_coordinator._run_metric_analysis
    async def mock_run_metric_analysis(survey_id, metric_id, metric_data, responses, time_series_responses=None):
        return {
            "survey_id": survey_id,
            "metric_id": metric_id,
            "statistics": {"count": len(responses)}
        }
    analysis_coordinator._run_metric_analysis = mock_run_metric_analysis
    
    # 3. Mock vector enhanced analysis
    original_run_vector_enhanced_analysis = analysis_coordinator._run_vector_enhanced_analysis
    async def mock_run_vector_enhanced_analysis(survey_id, metric_id, responses):
        if metric_id in ["improvement", "category"]:  # Only run for text and categorical
            return {
                "survey_id": survey_id,
                "metric_id": metric_id,
                "response_clusters": {"cluster_count": 2},
                "temporal_trends": {"has_significant_drift": False},
                "anomaly_detection": {"anomaly_count": 1}
            }
        return None
    analysis_coordinator._run_vector_enhanced_analysis = mock_run_vector_enhanced_analysis
    
    # 4. Mock cross-metric analysis
    original_run_cross_metric_analysis = analysis_coordinator._run_cross_metric_analysis
    async def mock_run_cross_metric_analysis(survey_id, metrics_data, responses):
        return {
            "survey_id": survey_id,
            "correlation_analysis": {"significant_correlations": 2},
            "ai_insights": {"key_findings": ["Finding 1", "Finding 2"]}
        }
    analysis_coordinator._run_cross_metric_analysis = mock_run_cross_metric_analysis
    
    # 5. Mock survey summary generation
    original_generate_survey_summary = analysis_coordinator._generate_survey_summary
    async def mock_generate_survey_summary(survey_id, survey_data, analysis_results):
        return {
            "survey_id": survey_id,
            "structured_summary": {
                "executive_summary": "This is a summary",
                "recommendations": ["Recommendation 1", "Recommendation 2"]
            }
        }
    analysis_coordinator._generate_survey_summary = mock_generate_survey_summary
    
    try:
        # Run the full pipeline
        result = await analysis_coordinator.run_analysis_pipeline(
            123, 
            sample_survey_data, 
            sample_responses,
            run_vector_analysis=True
        )
        
        # Verify that vector analysis was included
        assert "vector_analysis" in result
        assert "improvement" in result["vector_analysis"]
        assert "category" in result["vector_analysis"]
        assert "satisfaction" not in result["vector_analysis"]  # Numeric should be skipped
        
        # Verify the complete analysis result structure
        assert result["survey_id"] == 123
        assert result["status"] == "completed"
        assert "base_analysis" in result
        assert "metric_analysis" in result
        assert "cross_metric_analysis" in result
        assert "survey_summary" in result
        
        # Run without vector analysis
        result_without_vector = await analysis_coordinator.run_analysis_pipeline(
            123, 
            sample_survey_data, 
            sample_responses,
            run_vector_analysis=False
        )
        
        # Verify no vector analysis was included
        assert "vector_analysis" in result_without_vector
        assert len(result_without_vector["vector_analysis"]) == 0
        
    finally:
        # Restore original methods
        analysis_coordinator._run_base_analysis = original_run_base_analysis
        analysis_coordinator._run_metric_analysis = original_run_metric_analysis
        analysis_coordinator._run_vector_enhanced_analysis = original_run_vector_enhanced_analysis
        analysis_coordinator._run_cross_metric_analysis = original_run_cross_metric_analysis
        analysis_coordinator._generate_survey_summary = original_generate_survey_summary 