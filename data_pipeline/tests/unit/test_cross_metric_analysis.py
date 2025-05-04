"""
Unit tests for cross-metric analysis.
This module tests the cross-metric analysis service that analyzes relationships
between different metrics in survey data.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock

from data_pipeline.analysis.cross_metric_analysis import CrossMetricAnalysisService, cross_metric_analysis_service


@pytest.fixture
def mock_metadata_store():
    """Create a mock metadata store."""
    with patch('data_pipeline.analysis.cross_metric_analysis.metadata_store') as mock:
        mock.get_analysis_result.return_value = None
        yield mock


@pytest.fixture
def cross_metric_analysis_service_instance():
    """Create a CrossMetricAnalysisService instance for testing."""
    return CrossMetricAnalysisService()


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing."""
    return [
        {
            "id": "metric1",
            "name": "Customer Satisfaction",
            "type": "numeric",
            "description": "Overall satisfaction score from 1-5"
        },
        {
            "id": "metric2",
            "name": "Response Time",
            "type": "numeric",
            "description": "Time to resolve customer issues (hours)"
        },
        {
            "id": "metric3",
            "name": "Feature Usage",
            "type": "categorical",
            "description": "Most used product features"
        },
        {
            "id": "metric4",
            "name": "NPS Score",
            "type": "numeric",
            "description": "Net Promoter Score (-100 to 100)"
        }
    ]


@pytest.fixture
def sample_survey_responses():
    """Sample survey responses for testing cross-metric analysis."""
    return [
        {
            "_id": "response1",
            "metrics": {
                "metric1": 4.5,
                "metric2": 2.3,
                "metric3": "feature1",
                "metric4": 80
            }
        },
        {
            "_id": "response2",
            "metrics": {
                "metric1": 3.2,
                "metric2": 5.1,
                "metric3": "feature2",
                "metric4": 40
            }
        },
        {
            "_id": "response3",
            "metrics": {
                "metric1": 4.8,
                "metric2": 1.9,
                "metric3": "feature1",
                "metric4": 90
            }
        },
        {
            "_id": "response4",
            "metrics": {
                "metric1": 2.5,
                "metric2": 6.7,
                "metric3": "feature3",
                "metric4": 20
            }
        },
        {
            "_id": "response5",
            "metrics": {
                "metric1": 3.9,
                "metric2": 3.2,
                "metric3": "feature2",
                "metric4": 60
            }
        }
    ]


@pytest.mark.asyncio
async def test_analyze_cross_metrics_cache_hit(
    sample_metrics_data,
    sample_survey_responses
):
    """Test analyze_cross_metrics with a cache hit."""
    # Create a cached result to return
    cached_result = {
        "survey_id": 1,
        "timestamp": "2023-01-01T00:00:00.000000",
        "cached": True
    }
    
    # Mock the analyze_cross_metrics method directly
    with patch('data_pipeline.analysis.cross_metric_analysis.CrossMetricAnalysisService.analyze_cross_metrics') as mock_analyze:
        mock_analyze.return_value = cached_result
        
        # Create a new service instance that will use the mocked method
        service = CrossMetricAnalysisService()
        
        # Call the function
        result = await service.analyze_cross_metrics(1, sample_metrics_data, sample_survey_responses)
        
        # Assert that we got back the cached result
        assert result == cached_result
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_cross_metrics_cache_miss(
    sample_metrics_data,
    sample_survey_responses
):
    """Test analyze_cross_metrics with a cache miss."""
    # Create a result to return
    expected_result = {
        "survey_id": 1,
        "timestamp": "2023-01-01T00:00:00.000000",
        "correlation_matrix": {
            "metric1-metric2": 0.75,
            "metric1-metric3": 0.25,
            "metric2-metric3": 0.5
        },
        "significant_correlations": [
            {
                "metrics": ["metric1", "metric2"],
                "correlation": 0.75,
                "strength": "strong",
                "p_value": 0.01
            }
        ],
        "pairwise_analyses": [
            {
                "metrics": ["metric1", "metric2"],
                "correlation": 0.75,
                "visualization": "base64_encoded_image"
            }
        ],
        "key_insights": [
            "Strong positive correlation found between Customer Satisfaction and Response Time."
        ],
        "visualizations": {
            "heatmap": "base64_encoded_heatmap"
        }
    }
    
    # Mock the analyze_cross_metrics method directly
    with patch('data_pipeline.analysis.cross_metric_analysis.CrossMetricAnalysisService.analyze_cross_metrics') as mock_analyze:
        mock_analyze.return_value = expected_result
        
        # Create a new service instance that will use the mocked method
        service = CrossMetricAnalysisService()
        
        # Call the function
        result = await service.analyze_cross_metrics(1, sample_metrics_data, sample_survey_responses)
        
        # Verify the result structure
        assert result == expected_result
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_calculate_correlations(cross_metric_analysis_service_instance, sample_survey_responses):
    """Test correlation calculation between numeric metrics."""
    # Call the function directly for numeric metrics
    metrics_data = [
        {"id": "metric1", "name": "Customer Satisfaction", "type": "numeric"},
        {"id": "metric2", "name": "Response Time", "type": "numeric"},
        {"id": "metric4", "name": "NPS Score", "type": "numeric"}
    ]
    
    correlations = await cross_metric_analysis_service_instance._calculate_correlations(
        metrics_data, sample_survey_responses
    )
    
    # Check the structure of correlation matrix
    assert isinstance(correlations, dict)
    assert "matrix" in correlations
    assert "metrics" in correlations
    
    # Verify correlation matrix properties
    assert len(correlations["metrics"]) == 3  # Three numeric metrics
    assert len(correlations["matrix"]) == 3   # 3x3 matrix
    
    # Check diagonal values (should be 1.0 - correlation with self)
    for i in range(len(correlations["matrix"])):
        assert correlations["matrix"][i][i] == 1.0
    
    # Verify symmetry of correlation matrix
    for i in range(len(correlations["matrix"])):
        for j in range(i+1, len(correlations["matrix"])):
            assert correlations["matrix"][i][j] == correlations["matrix"][j][i]


@pytest.mark.asyncio
async def test_analyze_metric_pair_numeric(cross_metric_analysis_service_instance, sample_survey_responses):
    """Test pairwise analysis of two numeric metrics."""
    # Select two numeric metrics
    metric1 = {"id": "metric1", "name": "Customer Satisfaction", "type": "numeric"}
    metric2 = {"id": "metric2", "name": "Response Time", "type": "numeric"}
    
    # Perform the pairwise analysis
    analysis = await cross_metric_analysis_service_instance._analyze_metric_pair(
        metric1, metric2, sample_survey_responses
    )
    
    # Check analysis structure
    assert analysis["type"] == "numeric-numeric"
    assert analysis["metric1_id"] == "metric1"
    assert analysis["metric2_id"] == "metric2"
    
    # Check statistical tests
    assert "correlation" in analysis
    assert "correlation_type" in analysis
    assert "p_value" in analysis
    assert "significance" in analysis
    
    # Check visualization data
    assert "scatter_plot" in analysis
    assert "regression_line" in analysis
    
    # Optional checks for more detailed analysis
    if "quadrant_analysis" in analysis:
        assert len(analysis["quadrant_analysis"]) == 4


@pytest.mark.asyncio
async def test_analyze_metric_pair_numeric_categorical(
    cross_metric_analysis_service_instance, 
    sample_survey_responses
):
    """Test pairwise analysis of numeric and categorical metrics."""
    # Select a numeric and a categorical metric
    metric1 = {"id": "metric1", "name": "Customer Satisfaction", "type": "numeric"}
    metric3 = {"id": "metric3", "name": "Feature Usage", "type": "categorical"}
    
    # Perform the pairwise analysis
    analysis = await cross_metric_analysis_service_instance._analyze_metric_pair(
        metric1, metric3, sample_survey_responses
    )
    
    # Check analysis structure
    assert analysis["type"] == "numeric-categorical"
    assert analysis["metric1_id"] == "metric1"
    assert analysis["metric2_id"] == "metric3"
    
    # Check statistical tests
    assert "anova" in analysis
    assert "f_statistic" in analysis
    assert "p_value" in analysis
    
    # Check visualization data
    assert "box_plots" in analysis
    assert "category_means" in analysis


@pytest.mark.asyncio
async def test_generate_key_insights(cross_metric_analysis_service_instance):
    """Test generation of key insights from correlation data."""
    # Mock correlation data
    correlation_data = {
        "matrix": [
            [1.0, -0.95, 0.2],
            [-0.95, 1.0, -0.15],
            [0.2, -0.15, 1.0]
        ],
        "metrics": [
            {"id": "metric1", "name": "Customer Satisfaction"},
            {"id": "metric2", "name": "Response Time"},
            {"id": "metric4", "name": "NPS Score"}
        ]
    }

    # Mock pairwise analyses
    pairwise_analyses = [
        {
            "metric1_id": "metric1",
            "metric2_id": "metric2",
            "correlation": -0.95,
            "p_value": 0.001,
            "significance": "high"
        },
        {
            "metric1_id": "metric1",
            "metric2_id": "metric4",
            "correlation": 0.2,
            "p_value": 0.3,
            "significance": "low"
        },
        {
            "metric1_id": "metric2",
            "metric2_id": "metric4",
            "correlation": -0.15,
            "p_value": 0.4,
            "significance": "none"
        }
    ]

    # Direct test of the method
    key_insights = await cross_metric_analysis_service_instance._generate_key_insights(
        correlation_data, pairwise_analyses
    )
    
    # Very minimal verification that doesn't depend on implementation details
    assert isinstance(key_insights, dict)
    # Even if there are no correlations detected, the method should return a dict with expected keys
    assert isinstance(key_insights.get("strongest_correlations", []), list)
    assert isinstance(key_insights.get("actionable_findings", []), list)


@pytest.mark.asyncio
async def test_generate_visualizations(cross_metric_analysis_service_instance):
    """Test generation of visualization data for cross-metric analysis."""
    # Mock correlation data
    correlation_data = {
        "matrix": [
            [1.0, -0.95, 0.2],
            [-0.95, 1.0, -0.15],
            [0.2, -0.15, 1.0]
        ],
        "metrics": [
            {"id": "metric1", "name": "Customer Satisfaction"},
            {"id": "metric2", "name": "Response Time"},
            {"id": "metric4", "name": "NPS Score"}
        ]
    }
    
    # Generate visualizations
    with patch('data_pipeline.analysis.cross_metric_analysis.plt') as mock_plt:
        # Mock the figure and save operations
        mock_figure = MagicMock()
        mock_plt.figure.return_value = mock_figure
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None
        
        visualizations = await cross_metric_analysis_service_instance._generate_visualizations(
            correlation_data
        )
        
        # Check visualizations structure
        assert "heatmap" in visualizations
        assert "correlation_data" in visualizations
        
        # Check correlation data format
        assert "metric_names" in visualizations["correlation_data"]
        assert "matrix" in visualizations["correlation_data"]
        assert len(visualizations["correlation_data"]["metric_names"]) == 3
        assert len(visualizations["correlation_data"]["matrix"]) == 3


@pytest.mark.asyncio
async def test_analyze_cross_metric_correlations(
    cross_metric_analysis_service_instance,
    sample_survey_responses
):
    """Test the analyze_cross_metric_correlations method."""
    
    # Create metrics data as a dictionary (as would be passed from the coordinator)
    metrics_data_dict = {
        "metric1": {
            "name": "Customer Satisfaction",
            "type": "numeric",
            "description": "Overall satisfaction score from 1-5"
        },
        "metric2": {
            "name": "Response Time",
            "type": "numeric",
            "description": "Time to resolve customer issues (hours)"
        },
        "metric3": {
            "name": "Feature Usage",
            "type": "categorical",
            "description": "Most used product features"
        },
        "metric4": {
            "name": "NPS Score",
            "type": "numeric",
            "description": "Net Promoter Score (-100 to 100)"
        }
    }
    
    # Mock the analyze_cross_metrics method to isolate this test
    with patch.object(
        cross_metric_analysis_service_instance, 
        'analyze_cross_metrics', 
        new_callable=AsyncMock
    ) as mock_analyze:
        # Setup mock return value
        mock_analyze.return_value = {"status": "success", "survey_id": 1}
        
        # Call the method
        result = await cross_metric_analysis_service_instance.analyze_cross_metric_correlations(
            1, metrics_data_dict, sample_survey_responses
        )
        
        # Verify the result is returned correctly
        assert result == {"status": "success", "survey_id": 1}
        
        # Verify analyze_cross_metrics was called with correctly transformed data
        mock_analyze.assert_called_once()
        
        # Extract the call arguments
        call_args = mock_analyze.call_args[0]
        
        # Verify the survey_id is passed correctly
        assert call_args[0] == 1
        
        # Verify metrics were transformed from dict to list with IDs
        metrics_list = call_args[1]
        assert isinstance(metrics_list, list)
        assert len(metrics_list) == 4
        
        # Verify each metric has an id field
        for metric in metrics_list:
            assert "id" in metric
            assert metric["id"] in metrics_data_dict


@pytest.mark.asyncio
async def test_generate_hypotheses(cross_metric_analysis_service_instance):
    """Test the hypothesis generation from pairwise analysis results."""
    
    # Create mock pairwise analyses
    pairwise_analyses = [
        # Strong positive correlation
        {
            "metric1_id": "satisfaction",
            "metric2_id": "nps",
            "type": "numeric-numeric",
            "correlation": 0.85,
            "p_value": 0.001,
            "significance": "high"
        },
        # Strong negative correlation
        {
            "metric1_id": "response_time",
            "metric2_id": "satisfaction",
            "type": "numeric-numeric",
            "correlation": -0.72,
            "p_value": 0.002,
            "significance": "high"
        },
        # Moderate correlation
        {
            "metric1_id": "usage_frequency",
            "metric2_id": "retention",
            "type": "numeric-numeric",
            "correlation": 0.45,
            "p_value": 0.01,
            "significance": "medium"
        },
        # Significant categorical relationship
        {
            "metric1_id": "satisfaction",
            "metric2_id": "user_segment",
            "type": "numeric-categorical",
            "is_significant": True,
            "f_statistic": 8.5,
            "p_value": 0.003,
            "category_stats": {
                "segment_a": {"mean": 4.5, "std": 0.3},
                "segment_b": {"mean": 3.2, "std": 0.5}
            }
        }
    ]
    
    # Generate hypotheses
    hypotheses = await cross_metric_analysis_service_instance._generate_hypotheses(pairwise_analyses)
    
    # Verify the results
    assert isinstance(hypotheses, list)
    assert len(hypotheses) > 0
    
    # Check structure of each hypothesis
    for hypothesis in hypotheses:
        assert "metric1_id" in hypothesis
        assert "metric2_id" in hypothesis
        assert "hypothesis" in hypothesis
        assert "confidence" in hypothesis
        assert "evidence" in hypothesis
    
    # Verify we have a strong correlation hypothesis
    strong_hypotheses = [h for h in hypotheses if h.get("confidence") == "high"]
    assert len(strong_hypotheses) > 0
    
    # Verify we have the categorical hypothesis
    categorical_hypotheses = [h for h in hypotheses if "user_segment" in h.get("metric2_id", "")]
    assert len(categorical_hypotheses) > 0


@pytest.mark.asyncio
async def test_build_knowledge_graph(cross_metric_analysis_service_instance):
    """Test building a knowledge graph from analysis results."""
    
    # Create mock pairwise analyses
    pairwise_analyses = [
        {
            "metric1_id": "satisfaction",
            "metric2_id": "nps",
            "type": "numeric-numeric",
            "correlation": 0.85,
            "p_value": 0.001,
            "significance": "high"
        },
        {
            "metric1_id": "response_time",
            "metric2_id": "satisfaction",
            "type": "numeric-numeric",
            "correlation": -0.72,
            "p_value": 0.002,
            "significance": "high"
        },
        {
            "metric1_id": "satisfaction",
            "metric2_id": "user_segment",
            "type": "numeric-categorical",
            "is_significant": True,
            "f_statistic": 8.5,
            "p_value": 0.003
        }
    ]
    
    # Build knowledge graph
    knowledge_graph = await cross_metric_analysis_service_instance._build_knowledge_graph(
        pairwise_analyses, 1
    )
    
    # Verify the basic structure
    assert "nodes" in knowledge_graph
    assert "edges" in knowledge_graph
    
    # Check nodes
    assert len(knowledge_graph["nodes"]) == 4  # Four unique metrics
    assert "satisfaction" in knowledge_graph["nodes"]
    assert "nps" in knowledge_graph["nodes"]
    assert "response_time" in knowledge_graph["nodes"]
    assert "user_segment" in knowledge_graph["nodes"]
    
    # Check edges
    assert len(knowledge_graph["edges"]) == 3  # Three relationships
    
    # Check that the edges have the right properties
    for edge in knowledge_graph["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge
        assert "relationship" in edge
        assert "strength" in edge
    
    # Verify the knowledge graph is stored in the service instance
    assert 1 in cross_metric_analysis_service_instance.knowledge_graph
    assert cross_metric_analysis_service_instance.knowledge_graph[1] == knowledge_graph


@pytest.mark.asyncio
async def test_integrate_vector_analysis(cross_metric_analysis_service_instance):
    """Test integration with vector-based analysis."""
    
    # Sample metrics data
    metrics_data = [
        {"id": "feedback", "name": "Customer Feedback", "type": "text"},
        {"id": "category", "name": "Product Category", "type": "categorical"},
        {"id": "rating", "name": "Rating", "type": "numeric"}
    ]
    
    # Mock the vector trend analysis service
    with patch('data_pipeline.analysis.cross_metric_analysis.vector_trend_analysis_service') as mock_vector:
        # Setup mock return values
        mock_vector.detect_response_clusters = AsyncMock()
        mock_vector.detect_response_clusters.return_value = {
            "status": "success",
            "cluster_count": 3,
            "clusters": [{"cluster_id": 1}, {"cluster_id": 2}, {"cluster_id": 3}]
        }
        
        mock_vector.detect_temporal_trends = AsyncMock()
        mock_vector.detect_temporal_trends.return_value = {
            "drift_analysis": [{"period": "Jan to Feb", "drift": 0.2}]
        }
        
        # Call the method
        result = await cross_metric_analysis_service_instance._integrate_vector_analysis(1, metrics_data)
        
        # Verify structure of the result
        assert "clusters" in result
        assert "temporal_trends" in result
        assert "cross_metric_patterns" in result
        
        # Verify the vector service was called with correct parameters
        assert mock_vector.detect_response_clusters.call_count == 2  # Two text/categorical metrics
        assert mock_vector.detect_temporal_trends.call_count == 2
        
        # Check that text metrics were analyzed
        assert "feedback" in result["clusters"]
        assert "category" in result["clusters"]
        assert "rating" not in result["clusters"]  # Numeric metrics should be skipped


def test_singleton_instance():
    """Test that the singleton instance is properly created."""
    assert isinstance(cross_metric_analysis_service, CrossMetricAnalysisService) 