"""
Unit tests for visualization components.
This module tests the visualization generation functionality.
"""

import pytest
import base64
import io
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from data_pipeline.analysis.metric_analysis import MetricAnalysisService


@pytest.fixture
def sample_data():
    """Sample data for visualization testing."""
    return {
        "numeric_values": [1.5, 2.7, 3.2, 4.8, 2.9, 3.5, 4.2],
        "categorical_data": {
            "option1": 15,
            "option2": 25,
            "option3": 10,
            "option4": 30
        },
        "text_samples": [
            "This is a great product with excellent features.",
            "The service was good but could be improved.",
            "I really enjoyed using this platform.",
            "Needs some work but overall satisfied with results."
        ]
    }


def test_create_histogram():
    """Test histogram creation functionality."""
    # Initialize the service
    service = MetricAnalysisService()
    
    # Test data
    values = [1.5, 2.7, 3.2, 4.8, 2.9, 3.5, 4.2]
    
    # Create histogram
    histogram_base64 = service._create_histogram(values)
    
    # Check if result is a non-empty base64 string
    assert isinstance(histogram_base64, str)
    assert len(histogram_base64) > 0
    
    # Try to decode the base64 string to verify it's valid
    try:
        image_data = base64.b64decode(histogram_base64)
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 image: {str(e)}")


def test_create_boxplot():
    """Test box plot creation functionality."""
    # Initialize the service
    service = MetricAnalysisService()
    
    # Test data
    values = [1.5, 2.7, 3.2, 4.8, 2.9, 3.5, 4.2]
    
    # Create box plot
    boxplot_base64 = service._create_boxplot(values)
    
    # Check if result is a non-empty base64 string
    assert isinstance(boxplot_base64, str)
    assert len(boxplot_base64) > 0
    
    # Try to decode the base64 string to verify it's valid
    try:
        image_data = base64.b64decode(boxplot_base64)
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 image: {str(e)}")


def test_create_bar_chart():
    """Test bar chart creation functionality."""
    # Initialize the service
    service = MetricAnalysisService()
    
    # Test data
    counter = {
        "option1": 15,
        "option2": 25,
        "option3": 10,
        "option4": 30
    }
    
    # Create bar chart
    bar_chart_base64 = service._create_bar_chart(counter)
    
    # Check if result is a non-empty base64 string
    assert isinstance(bar_chart_base64, str)
    assert len(bar_chart_base64) > 0
    
    # Try to decode the base64 string to verify it's valid
    try:
        image_data = base64.b64decode(bar_chart_base64)
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 image: {str(e)}")


def test_create_pie_chart():
    """Test pie chart creation functionality."""
    # Initialize the service
    service = MetricAnalysisService()
    
    # Test data
    counter = {
        "option1": 15,
        "option2": 25,
        "option3": 10,
        "option4": 30
    }
    
    # Create pie chart
    pie_chart_base64 = service._create_pie_chart(counter)
    
    # Check if result is a non-empty base64 string
    assert isinstance(pie_chart_base64, str)
    assert len(pie_chart_base64) > 0
    
    # Try to decode the base64 string to verify it's valid
    try:
        image_data = base64.b64decode(pie_chart_base64)
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 image: {str(e)}")


@pytest.mark.asyncio
async def test_generate_numeric_visualizations():
    """Test visualization generation for numeric data."""
    # Initialize the service
    service = MetricAnalysisService()
    
    # Mock data
    metric_data = {"type": "numeric", "name": "Rating"}
    responses = [
        {"value": 3.5},
        {"value": 4.2},
        {"value": 2.8},
        {"value": 3.9},
        {"value": 4.7}
    ]
    
    # Generate visualizations
    result = await service._generate_numeric_visualizations(metric_data, responses)
    
    # Verify results
    assert result["type"] == "numeric"
    assert "histogram" in result
    assert "boxplot" in result
    assert "chart_data" in result
    assert "values" in result["chart_data"]
    assert "histogram_bins" in result["chart_data"]
    assert "histogram_edges" in result["chart_data"]


def test_matplotlib_save_to_base64():
    """Test the base functionality of saving matplotlib figures to base64."""
    # Create a simple figure
    plt.figure(figsize=(8, 5))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.title('Test Plot')
    
    # Save to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Verify result
    assert isinstance(img_str, str)
    assert len(img_str) > 0
    
    # Try to decode the base64 string to verify it's valid
    try:
        image_data = base64.b64decode(img_str)
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 image: {str(e)}") 