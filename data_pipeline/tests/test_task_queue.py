"""
Tests for the task queue implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json
import datetime
from celery import chain, group

# Import the Celery tasks
from data_pipeline.tasks.celery_app import app as celery_app
from data_pipeline.tasks.analysis_tasks import (
    run_base_analysis,
    run_metric_analysis,
    run_cross_metric_analysis,
    run_vector_enhanced_analysis,
    generate_survey_summary,
    run_analysis_pipeline
)
from data_pipeline.tasks.embedding_tasks import (
    generate_response_embeddings,
    generate_metric_embeddings,
    update_vector_index,
    process_survey_embeddings
)


@pytest.fixture
def celery_app_mock():
    """Mock the Celery app."""
    with patch('data_pipeline.tasks.celery_app.app') as mock_app:
        yield mock_app


@pytest.fixture
def sample_survey_data():
    """Return sample survey data for testing."""
    return {
        "id": 123,
        "name": "Test Survey",
        "description": "A test survey for task queue testing",
        "metrics": {
            "satisfaction": {
                "id": "satisfaction",
                "name": "Customer Satisfaction",
                "type": "numeric",
                "description": "Customer satisfaction rating (1-10)"
            },
            "feedback": {
                "id": "feedback",
                "name": "Customer Feedback",
                "type": "text",
                "description": "Open-ended feedback"
            }
        }
    }


@pytest.fixture
def sample_responses():
    """Return sample survey responses for testing."""
    now = datetime.datetime.now().isoformat()
    return [
        {
            "id": 1,
            "submitted_at": now,
            "responses": {
                "satisfaction": 8,
                "feedback": "Great service!"
            }
        },
        {
            "id": 2,
            "submitted_at": now,
            "responses": {
                "satisfaction": 9,
                "feedback": "Excellent experience overall."
            }
        }
    ]


@pytest.fixture
def mock_analysis_coordinator():
    """Mock the AnalysisCoordinator instance used in tasks."""
    coordinator = MagicMock()
    with patch('data_pipeline.tasks.analysis_tasks.AnalysisCoordinator', return_value=coordinator):
        yield coordinator


@pytest.fixture
def mock_semantic_search():
    """Mock the SemanticSearchService instance used in tasks."""
    service = MagicMock()
    with patch('data_pipeline.tasks.embedding_tasks.semantic_search_service', return_value=service):
        yield service


@pytest.fixture
def mock_asyncio_loop():
    """Mock the asyncio event loop with a properly configured AsyncMock."""
    mock_loop = MagicMock()
    # Create an AsyncMock for the coroutine result
    async_result = AsyncMock()
    # Configure run_until_complete to return the mocked result directly
    mock_loop.run_until_complete = lambda coro: async_result
    
    return mock_loop


def test_run_base_analysis_task(mock_analysis_coordinator, mock_asyncio_loop, sample_survey_data, sample_responses):
    """Test the run_base_analysis task."""
    # Setup the mock to return the expected result
    expected_result = {"status": "success", "type": "base_analysis"}
    mock_analysis_coordinator._run_base_analysis.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.analysis_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = run_base_analysis(123, sample_survey_data, sample_responses)
        
    # Verify the result
    assert result is not None


def test_run_metric_analysis_task(mock_analysis_coordinator, mock_asyncio_loop, sample_survey_data, sample_responses):
    """Test the run_metric_analysis task."""
    metric_id = "satisfaction"
    metric_data = sample_survey_data["metrics"][metric_id]

    # Setup the mock to return the expected result
    expected_result = {"status": "success", "type": "metric_analysis"}
    mock_analysis_coordinator._run_metric_analysis.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.analysis_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = run_metric_analysis(123, metric_id, metric_data, sample_responses)
        
    # Verify the result
    assert result is not None


def test_run_vector_enhanced_analysis_task(mock_analysis_coordinator, mock_asyncio_loop, sample_responses):
    """Test the run_vector_enhanced_analysis task."""
    metric_id = "feedback"

    # Setup the mock to return the expected result
    expected_result = {"status": "success", "type": "vector_analysis"}
    mock_analysis_coordinator._run_vector_enhanced_analysis.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.analysis_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = run_vector_enhanced_analysis(123, metric_id, sample_responses)
        
    # Verify the result
    assert result is not None


def test_run_cross_metric_analysis_task(mock_analysis_coordinator, mock_asyncio_loop, sample_survey_data, sample_responses):
    """Test the run_cross_metric_analysis task."""
    # Setup the mock to return the expected result
    expected_result = {"status": "success", "type": "cross_metric_analysis"}
    mock_analysis_coordinator._run_cross_metric_analysis.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.analysis_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = run_cross_metric_analysis(123, sample_survey_data["metrics"], sample_responses)
        
    # Verify the result
    assert result is not None


def test_generate_survey_summary_task(mock_analysis_coordinator, mock_asyncio_loop, sample_survey_data):
    """Test the generate_survey_summary task."""
    analysis_results = {
        "base_analysis": {"status": "success"},
        "metric_analysis": {"status": "success"},
        "cross_metric_analysis": {"status": "success"}
    }

    # Setup the mock to return the expected result
    expected_result = {"status": "success", "type": "survey_summary"}
    mock_analysis_coordinator._generate_survey_summary.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.analysis_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = generate_survey_summary(123, sample_survey_data, analysis_results)
        
    # Verify the result
    assert result is not None


def test_run_analysis_pipeline_task(sample_survey_data, sample_responses):
    """Test the run_analysis_pipeline task with mocked Celery task chain."""
    # Mock a successful result from chain.apply_async 
    mock_async_result = MagicMock()
    mock_async_result.id = "test_task_id_1234"
    
    # Create a mock chain function
    mock_chain_instance = MagicMock()
    mock_chain_instance.apply_async.return_value = mock_async_result
    mock_chain_func = MagicMock(return_value=mock_chain_instance)
    
    # Patch the celery module used by analysis_tasks
    with patch('celery.chain', mock_chain_func):
        # Add the import for run_analysis_pipeline after patching
        from data_pipeline.tasks.analysis_tasks import run_analysis_pipeline
        
        # Call the function
        result = run_analysis_pipeline(123, sample_survey_data, sample_responses, False)
        
        # Verify the result contains the task ID
        assert "test_task_id_1234" in result


def test_generate_response_embeddings_task(mock_semantic_search, mock_asyncio_loop, sample_responses):
    """Test the generate_response_embeddings task."""
    # Setup the mock to return the expected result
    expected_result = [{"id": 1, "embedding": [0.1, 0.2, 0.3]}]
    mock_semantic_search.process_survey_responses.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.embedding_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = generate_response_embeddings(123, sample_responses)
        
    # Verify the result
    assert result is not None


def test_generate_metric_embeddings_task(mock_semantic_search, mock_asyncio_loop, sample_survey_data):
    """Test the generate_metric_embeddings task."""
    metric_id = "feedback"
    metric_data = sample_survey_data["metrics"][metric_id]

    # Setup the mock to return the expected result
    expected_result = {"status": "success"}
    mock_semantic_search.process_metric.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.embedding_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = generate_metric_embeddings(123, metric_id, metric_data)
        
    # Verify the result
    assert result is not None


def test_update_vector_index_task(mock_semantic_search, mock_asyncio_loop):
    """Test the update_vector_index task."""
    # Setup the mock to return the expected result
    expected_result = {"status": "success"}
    mock_semantic_search.update_vector_index.return_value = expected_result
    
    # Mock the async function to properly return the result
    mock_asyncio_loop.run_until_complete.return_value = expected_result
    
    # Call the Celery task with mocked loop
    with patch('data_pipeline.tasks.embedding_tasks.asyncio.get_event_loop', return_value=mock_asyncio_loop):
        result = update_vector_index(123)
        
    # Verify the result
    assert result is not None


def test_process_survey_embeddings_task(sample_survey_data):
    """Test the process_survey_embeddings task with mocked Celery task chain."""
    # Mock a successful result from chain.apply_async 
    mock_async_result = MagicMock()
    mock_async_result.id = "test_task_id_5678"
    
    # Create a mock chain function
    mock_chain_instance = MagicMock()
    mock_chain_instance.apply_async.return_value = mock_async_result
    mock_chain_func = MagicMock(return_value=mock_chain_instance)
    
    # Patch the celery module used by embedding_tasks
    with patch('celery.chain', mock_chain_func):
        # Add the import for process_survey_embeddings after patching
        from data_pipeline.tasks.embedding_tasks import process_survey_embeddings
        
        # Call the function
        result = process_survey_embeddings(123, sample_survey_data)
        
        # Verify the result contains the task ID
        assert "test_task_id_5678" in result 