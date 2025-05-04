"""
Unit tests for the vector trend analysis service.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta

from data_pipeline.analysis.vector_trend_analysis import VectorTrendAnalysisService, vector_trend_analysis_service


@pytest.fixture
def mock_qdrant_service():
    """Mock the Qdrant service."""
    with patch('data_pipeline.analysis.vector_trend_analysis.qdrant_service') as mock:
        # Setup client.scroll mock
        mock_point_1 = MagicMock()
        mock_point_1.vector = [0.1, 0.2, 0.3, 0.4]
        mock_point_1.payload = {
            "response_id": "resp1",
            "answer": "This is the first answer",
            "response_date": datetime.now().isoformat()
        }
        
        mock_point_2 = MagicMock()
        mock_point_2.vector = [0.15, 0.25, 0.35, 0.45]
        mock_point_2.payload = {
            "response_id": "resp2",
            "answer": "This is the second answer",
            "response_date": datetime.now().isoformat()
        }
        
        mock_point_3 = MagicMock()
        mock_point_3.vector = [0.9, 0.8, 0.7, 0.6]  # Very different vector - potential anomaly
        mock_point_3.payload = {
            "response_id": "resp3",
            "answer": "This is a very different answer",
            "response_date": datetime.now().isoformat()
        }
        
        # Create enough mock points for clustering
        mock_points = [mock_point_1, mock_point_2, mock_point_3]
        for i in range(7):  # Add 7 more points to reach 10 total
            mock_point = MagicMock()
            mock_point.vector = [0.1 + i*0.02, 0.2 + i*0.02, 0.3 + i*0.02, 0.4 + i*0.02]
            mock_point.payload = {
                "response_id": f"resp{i+4}",
                "answer": f"This is answer {i+4}",
                "response_date": datetime.now().isoformat()
            }
            mock_points.append(mock_point)
        
        mock_scroll_result = MagicMock()
        mock_scroll_result.points = mock_points
        
        mock.client = MagicMock()
        mock.client.scroll = AsyncMock(return_value=mock_scroll_result)
        
        yield mock


@pytest.fixture
def mock_sklearn_kmeans():
    """Mock sklearn KMeans."""
    with patch('data_pipeline.analysis.vector_trend_analysis.KMeans') as mock:
        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0])  # 7 in cluster 0, 3 in cluster 1
        mock.return_value = mock_kmeans
        yield mock


@pytest.fixture
def mock_silhouette_score():
    """Mock silhouette_score."""
    with patch('data_pipeline.analysis.vector_trend_analysis.silhouette_score') as mock:
        mock.return_value = 0.75  # Good silhouette score
        yield mock


@pytest.fixture
def vector_trend_analysis_service_instance():
    """Create a VectorTrendAnalysisService instance for testing."""
    return VectorTrendAnalysisService()


@pytest.mark.asyncio
async def test_detect_response_clusters(
    vector_trend_analysis_service_instance, 
    mock_qdrant_service, 
    mock_sklearn_kmeans,
    mock_silhouette_score
):
    """Test detecting clusters in responses."""
    # Call the method
    result = await vector_trend_analysis_service_instance.detect_response_clusters(
        survey_id=123,
        question_id="question1",
        max_clusters=5
    )
    
    # Verify scroll was called
    mock_qdrant_service.client.scroll.assert_called_once()
    
    # Verify kmeans was called
    mock_sklearn_kmeans.assert_called()
    
    # Verify result structure
    assert result["status"] == "success"
    assert result["cluster_count"] == 2  # From our mock
    assert result["silhouette_score"] == 0.75
    assert result["total_responses"] == 10
    assert len(result["clusters"]) == 2
    
    # Verify clusters
    cluster0 = next(c for c in result["clusters"] if c["cluster_id"] == 0)
    cluster1 = next(c for c in result["clusters"] if c["cluster_id"] == 1)
    
    assert cluster0["size"] == 7
    assert cluster1["size"] == 3
    assert cluster0["percentage"] == 70.0
    assert cluster1["percentage"] == 30.0


@pytest.mark.asyncio
async def test_detect_response_clusters_insufficient_data(
    vector_trend_analysis_service_instance, 
    mock_qdrant_service
):
    """Test detecting clusters with insufficient data."""
    # Modify mock to return fewer points
    mock_scroll_result = MagicMock()
    mock_scroll_result.points = [MagicMock() for _ in range(5)]  # Only 5 points, less than minimum 10
    mock_qdrant_service.client.scroll.return_value = mock_scroll_result
    
    # Call the method
    result = await vector_trend_analysis_service_instance.detect_response_clusters(
        survey_id=123,
        question_id="question1",
        max_clusters=5
    )
    
    # Verify result shows insufficient data
    assert result["status"] == "insufficient_data"
    assert len(result["clusters"]) == 0


@pytest.mark.asyncio
async def test_detect_temporal_trends(
    vector_trend_analysis_service_instance,
    mock_qdrant_service
):
    """Test detecting temporal trends in responses."""
    # Setup multiple period responses
    periods = [
        datetime.now() - timedelta(days=90),
        datetime.now() - timedelta(days=60),
        datetime.now() - timedelta(days=30),
        datetime.now()
    ]

    # Create different scroll results for different time periods
    mock_scroll_results = []

    # Period 1 - similar vectors
    period1_points = []
    for i in range(5):
        mock_point = MagicMock()
        mock_point.vector = [0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01]
        mock_point.payload = {
            "response_id": f"p1_resp{i}",
            "response_date": (periods[0] + timedelta(days=i)).isoformat()
        }
        period1_points.append(mock_point)

    # Period 2 - still similar vectors
    period2_points = []
    for i in range(5):
        mock_point = MagicMock()
        mock_point.vector = [0.15 + i*0.01, 0.25 + i*0.01, 0.35 + i*0.01, 0.45 + i*0.01]
        mock_point.payload = {
            "response_id": f"p2_resp{i}",
            "response_date": (periods[1] + timedelta(days=i)).isoformat()
        }
        period2_points.append(mock_point)

    # Period 3 - significant drift (much more different from period 2)
    period3_points = []
    for i in range(5):
        mock_point = MagicMock()
        # Create a more dramatically different vector to cause significant drift
        mock_point.vector = [-0.5 - i*0.01, -0.6 - i*0.01, -0.7 - i*0.01, -0.8 - i*0.01]
        mock_point.payload = {
            "response_id": f"p3_resp{i}",
            "response_date": (periods[2] + timedelta(days=i)).isoformat()
        }
        period3_points.append(mock_point)

    # Period 4 - similar to period 3
    period4_points = []
    for i in range(5):
        mock_point = MagicMock()
        mock_point.vector = [-0.52 - i*0.01, -0.62 - i*0.01, -0.72 - i*0.01, -0.82 - i*0.01]
        mock_point.payload = {
            "response_id": f"p4_resp{i}",
            "response_date": (periods[3] + timedelta(days=i)).isoformat()
        }
        period4_points.append(mock_point)

    # Setup the mock to avoid any scroll calls with actual Range objects
    # Just return the mock points for each call 
    mock_qdrant_service.client.scroll = AsyncMock()
    mock_qdrant_service.client.scroll.side_effect = [
        MagicMock(points=period1_points),
        MagicMock(points=period2_points),
        MagicMock(points=period3_points),
        MagicMock(points=period4_points)
    ]

    # Call the method
    result = await vector_trend_analysis_service_instance.detect_temporal_trends(
        survey_id=123,
        question_id="question1",
        time_periods=4,
        period_days=30
    )

    # Verify scroll was called 4 times (once per period)
    assert mock_qdrant_service.client.scroll.call_count == 4

    # Verify result structure
    assert "survey_id" in result
    assert "question_id" in result
    assert "period_count" in result
    assert "period_days" in result
    assert "drift_analysis" in result
    assert "has_significant_drift" in result
    
    # Verify drift analysis
    assert len(result["drift_analysis"]) == 3  # 3 transitions between 4 periods
    
    # Debug information
    for i, drift in enumerate(result["drift_analysis"]):
        print(f"Drift {i}: {drift.get('drift')} - Significant: {drift.get('is_significant')}")
    
    # Verify significant drift is detected between period 2 and 3
    assert any(drift.get("is_significant") for drift in result["drift_analysis"])


@pytest.mark.asyncio
async def test_detect_anomalies(
    vector_trend_analysis_service_instance,
    mock_qdrant_service
):
    """Test detecting anomalies in responses."""
    # Setup mock points with one clear anomaly
    normal_vectors = [[0.1, 0.2, 0.3, 0.4] for _ in range(9)]
    anomaly_vector = [0.9, 0.8, 0.7, 0.6]  # Very different from others

    mock_points = []
    for i, vector in enumerate(normal_vectors + [anomaly_vector]):
        mock_point = MagicMock()
        mock_point.vector = vector
        mock_point.payload = {
            "response_id": f"resp{i}",
            "answer": f"This is answer {i}" if i < 9 else "This is an anomalous answer",
            "response_date": datetime.now().isoformat()
        }
        mock_points.append(mock_point)

    mock_scroll_result = MagicMock()
    mock_scroll_result.points = mock_points
    mock_qdrant_service.client.scroll.return_value = mock_scroll_result

    # Call the method with a lower threshold to detect the anomaly
    result = await vector_trend_analysis_service_instance.detect_anomalies(
        survey_id=123,
        question_id="question1",
        threshold=0.3  # Set threshold low enough to detect our anomaly
    )

    # Verify scroll was called
    mock_qdrant_service.client.scroll.assert_called_once()

    # Verify result structure
    assert result["status"] == "success"
    assert result["total_responses"] == 10
    
    # Debug information
    print(f"Anomaly count: {result['anomaly_count']}")
    print(f"Anomalies: {result['anomalies']}")
    print(f"Threshold used: 0.3")  # Fixed reference to the threshold value
    
    assert result["anomaly_count"] == 1
    assert len(result["anomalies"]) == 1
    assert result["anomalies"][0]["response_id"] == "resp9"  # The anomaly is the last vector 


@pytest.mark.asyncio
async def test_detect_anomalies_insufficient_data(
    vector_trend_analysis_service_instance, 
    mock_qdrant_service
):
    """Test detecting anomalies with insufficient data."""
    # Modify mock to return fewer points
    mock_scroll_result = MagicMock()
    mock_scroll_result.points = [MagicMock() for _ in range(5)]  # Only 5 points, less than minimum 10
    mock_qdrant_service.client.scroll.return_value = mock_scroll_result
    
    # Call the method
    result = await vector_trend_analysis_service_instance.detect_anomalies(
        survey_id=123,
        question_id="question1",
        threshold=0.8
    )
    
    # Verify result shows insufficient data
    assert result["status"] == "insufficient_data"
    assert len(result["anomalies"]) == 0 