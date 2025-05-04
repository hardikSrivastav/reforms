"""
Unit tests for the semantic search service.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime

from data_pipeline.embeddings.semantic_search import SemanticSearchService, semantic_search_service


@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service."""
    with patch('data_pipeline.embeddings.semantic_search.embedding_service') as mock:
        mock.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        yield mock


@pytest.fixture
def mock_qdrant_service():
    """Mock the Qdrant service."""
    with patch('data_pipeline.embeddings.semantic_search.qdrant_service') as mock:
        # Setup create_survey_filter mock
        mock.create_survey_filter = AsyncMock(return_value={"must": [{"key": "survey_id", "match": {"value": 123}}]})
        
        # Setup search mock
        mock_result_1 = MagicMock()
        mock_result_1.payload = {
            "response_id": "resp1",
            "question_id": "question1",
            "answer": "This is answer 1",
            "response_date": datetime.now().isoformat()
        }
        mock_result_1.score = 0.95
        
        mock_result_2 = MagicMock()
        mock_result_2.payload = {
            "response_id": "resp2",
            "question_id": "question1",
            "answer": "This is answer 2",
            "response_date": datetime.now().isoformat()
        }
        mock_result_2.score = 0.85
        
        mock.search = AsyncMock(return_value=[mock_result_1, mock_result_2])
        
        # Setup client.retrieve mock
        mock_point = MagicMock()
        mock_vectors = {
            "resp1_question1": MagicMock(vector=[0.1, 0.2, 0.3, 0.4])
        }
        mock_point.vectors = mock_vectors
        mock.client = MagicMock()
        mock.client.retrieve = AsyncMock(return_value=mock_point)
        
        yield mock


@pytest.fixture
def semantic_search_service_instance(mock_embedding_service, mock_qdrant_service):
    """Create a SemanticSearchService instance for testing."""
    return SemanticSearchService()


@pytest.mark.asyncio
async def test_search_similar_responses(semantic_search_service_instance, mock_embedding_service, mock_qdrant_service):
    """Test searching for similar responses."""
    # Call the method
    results = await semantic_search_service_instance.search_similar_responses(
        query="test query",
        survey_id=123,
        limit=10
    )
    
    # Verify embedding was generated
    mock_embedding_service.get_embedding.assert_called_once_with("test query")
    
    # Verify filter was created
    mock_qdrant_service.create_survey_filter.assert_called_once_with(123)
    
    # Verify search was called
    mock_qdrant_service.search.assert_called_once()
    
    # Verify results
    assert len(results) == 2
    assert results[0]["response_id"] == "resp1"
    assert results[0]["question_id"] == "question1"
    assert results[0]["answer"] == "This is answer 1"
    assert results[0]["similarity_score"] == 0.95
    assert "response_date" in results[0]


@pytest.mark.asyncio
async def test_find_related_responses(semantic_search_service_instance, mock_embedding_service, mock_qdrant_service):
    """Test finding related responses."""
    # Call the method
    results = await semantic_search_service_instance.find_related_responses(
        response_id="resp1",
        question_id="question1",
        survey_id=123,
        limit=10
    )
    
    # Verify point was retrieved
    mock_qdrant_service.client.retrieve.assert_called_once_with(
        collection_name=mock_qdrant_service.search.call_args[1]["collection_name"],
        ids=["resp1_question1"]
    )
    
    # Verify filter was created
    mock_qdrant_service.create_survey_filter.assert_called_once_with(123)
    
    # Verify search was called
    mock_qdrant_service.search.assert_called_once()
    
    # Verify results
    assert len(results) == 1  # Only one result since we filter out the original
    assert results[0]["response_id"] == "resp2"
    assert results[0]["question_id"] == "question1"
    assert results[0]["answer"] == "This is answer 2"


@pytest.mark.asyncio
async def test_find_related_responses_missing_point(semantic_search_service_instance, mock_embedding_service, mock_qdrant_service):
    """Test finding related responses when the original point doesn't exist."""
    # Modify the mock to return None for retrieve
    mock_qdrant_service.client.retrieve = AsyncMock(return_value=None)
    
    # Call the method
    results = await semantic_search_service_instance.find_related_responses(
        response_id="resp99",  # Non-existent
        question_id="question99",
        survey_id=123,
        limit=10
    )
    
    # Verify empty results when point not found
    assert len(results) == 0
    
    # Verify search was not called
    mock_qdrant_service.search.assert_not_called()


@pytest.mark.asyncio
async def test_search_similar_questions(semantic_search_service_instance, mock_embedding_service, mock_qdrant_service):
    """Test searching for similar questions."""
    # Setup mock response for question search
    mock_result = MagicMock()
    mock_result.payload = {
        "question_id": "question1",
        "question_text": "What is your favorite color?",
        "question_type": "single_choice"
    }
    mock_result.score = 0.92
    mock_qdrant_service.search = AsyncMock(return_value=[mock_result])
    
    # Call the method
    results = await semantic_search_service_instance.search_similar_questions(
        query="favorite color",
        survey_id=123,
        limit=10
    )
    
    # Verify embedding was generated
    mock_embedding_service.get_embedding.assert_called_once_with("favorite color")
    
    # Verify filter was created
    mock_qdrant_service.create_survey_filter.assert_called_once_with(123)
    
    # Verify search was called
    mock_qdrant_service.search.assert_called_once()
    
    # Verify results
    assert len(results) == 1
    assert results[0]["question_id"] == "question1"
    assert results[0]["question_text"] == "What is your favorite color?"
    assert results[0]["question_type"] == "single_choice"
    assert results[0]["similarity_score"] == 0.92


@pytest.mark.asyncio
async def test_search_similar_metrics(semantic_search_service_instance, mock_embedding_service, mock_qdrant_service):
    """Test searching for similar metrics."""
    # Setup mock response for metric search
    mock_result = MagicMock()
    mock_result.payload = {
        "metric_id": "satisfaction",
        "metric_name": "Customer Satisfaction",
        "metric_description": "How satisfied are customers with our service",
        "metric_type": "numeric"
    }
    mock_result.score = 0.88
    mock_qdrant_service.search = AsyncMock(return_value=[mock_result])
    
    # Call the method
    results = await semantic_search_service_instance.search_similar_metrics(
        query="customer happiness",
        survey_id=123,
        limit=10
    )
    
    # Verify embedding was generated
    mock_embedding_service.get_embedding.assert_called_once_with("customer happiness")
    
    # Verify filter was created
    mock_qdrant_service.create_survey_filter.assert_called_once_with(123)
    
    # Verify search was called
    mock_qdrant_service.search.assert_called_once()
    
    # Verify results
    assert len(results) == 1
    assert results[0]["metric_id"] == "satisfaction"
    assert results[0]["metric_name"] == "Customer Satisfaction"
    assert results[0]["metric_description"] == "How satisfied are customers with our service"
    assert results[0]["metric_type"] == "numeric"
    assert results[0]["similarity_score"] == 0.88 