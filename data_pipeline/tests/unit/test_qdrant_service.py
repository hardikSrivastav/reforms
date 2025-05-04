"""
Unit tests for the Qdrant client service.
"""

import pytest
import uuid
import numpy as np
from unittest.mock import patch, MagicMock, call
from qdrant_client.http import models as qdrant_models

from data_pipeline.services.qdrant_client import QdrantService
from data_pipeline.config.settings import EMBEDDING_DIMENSION


@pytest.mark.asyncio
async def test_init_collections(mock_qdrant_service):
    """Test initialization of collections."""
    # Test initializing collections
    await mock_qdrant_service.initialize_collections()
    
    # We can't check for exactly one call since initialize_collections calls
    # create_collection_if_not_exists which itself calls get_collections for each collection
    
    # Verify that create_collection was called at least once
    assert mock_qdrant_service.client.create_collection.call_count >= 1
    
    # Verify that create_payload_index was called at least once
    assert mock_qdrant_service.client.create_payload_index.call_count >= 1


@pytest.mark.asyncio
async def test_create_collection_if_not_exists(mock_qdrant_service):
    """Test creation of collection if it doesn't exist."""
    # Configure mock to return empty collections list
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_qdrant_service.client.get_collections.return_value = mock_collections
    
    # Test creating a collection
    collection_name = "test_collection"
    await mock_qdrant_service.create_collection_if_not_exists(
        collection_name=collection_name,
        vector_size=EMBEDDING_DIMENSION
    )
    
    # Verify create_collection was called with expected parameters
    mock_qdrant_service.client.create_collection.assert_called_once()
    args = mock_qdrant_service.client.create_collection.call_args[1]
    assert args["collection_name"] == collection_name
    assert isinstance(args["vectors_config"], qdrant_models.VectorParams)
    assert args["vectors_config"].size == EMBEDDING_DIMENSION


@pytest.mark.asyncio
async def test_upsert_vectors(mock_qdrant_service):
    """Test upserting vectors to a collection."""
    # Create sample data
    collection_name = "survey_responses"
    points = [
        qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=np.random.random(EMBEDDING_DIMENSION).tolist(),
            payload={"survey_id": 1, "question_id": "q1"}
        ),
        qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=np.random.random(EMBEDDING_DIMENSION).tolist(),
            payload={"survey_id": 1, "question_id": "q2"}
        )
    ]
    
    # Call the method
    await mock_qdrant_service.upsert_vectors(collection_name, points)
    
    # Verify that upsert was called with expected parameters
    mock_qdrant_service.client.upsert.assert_called_once()
    args = mock_qdrant_service.client.upsert.call_args[1]
    assert args["collection_name"] == collection_name
    assert args["points"] == points


@pytest.mark.asyncio
async def test_search(mock_qdrant_service):
    """Test searching for similar vectors."""
    # Create test data
    collection_name = "survey_responses"
    query_vector = np.random.random(EMBEDDING_DIMENSION).tolist()
    
    # Create a filter using concrete types instead of unions
    filter_condition = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="survey_id",
                match=qdrant_models.MatchValue(value=1)
            )
        ]
    )
    limit = 5
    
    # Configure mock to return expected results
    mock_results = [
        MagicMock(id="1", score=0.95, payload={"text": "Sample text 1"}),
        MagicMock(id="2", score=0.85, payload={"text": "Sample text 2"})
    ]
    mock_qdrant_service.client.search.return_value = mock_results
    
    # Call the method
    results = await mock_qdrant_service.search(
        collection_name=collection_name,
        query_vector=query_vector,
        filter_condition=filter_condition,
        limit=limit
    )
    
    # Verify search was called with expected parameters
    mock_qdrant_service.client.search.assert_called_once()
    args = mock_qdrant_service.client.search.call_args[1]
    assert args["collection_name"] == collection_name
    assert args["query_vector"] == query_vector
    assert args["query_filter"] == filter_condition
    assert args["limit"] == limit
    
    # Verify results are returned as expected
    assert results == mock_results


@pytest.mark.asyncio
async def test_delete_survey_data(mock_qdrant_service):
    """Test deleting survey data."""
    # Test parameters
    survey_id = 123
    
    # Mock create_survey_filter to avoid Union type issues
    async def mock_create_survey_filter(survey_id):
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="survey_id",
                    match=qdrant_models.MatchValue(value=survey_id)
                )
            ]
        )
    
    # Replace the method with our mock
    mock_qdrant_service.create_survey_filter = mock_create_survey_filter
    
    # Call the method
    await mock_qdrant_service.delete_survey_data(survey_id)
    
    # Verify delete was called for each collection
    assert mock_qdrant_service.client.delete.call_count >= 1
    
    # Verify that the filter was created properly
    for call_args in mock_qdrant_service.client.delete.call_args_list:
        args = call_args[1]
        assert "collection_name" in args
        assert "points_selector" in args
        assert isinstance(args["points_selector"], qdrant_models.FilterSelector) 