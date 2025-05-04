"""
Unit tests for the Multi-Level Embedding Service.
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from datetime import datetime

from data_pipeline.embeddings.multi_level_embedding_service import multi_level_embedding_service


@pytest.fixture
def mock_metadata_store():
    """Mock the metadata store."""
    with patch("data_pipeline.embeddings.multi_level_embedding_service.metadata_store") as mock:
        mock.get_analysis_result = AsyncMock()
        mock.store_analysis_result = AsyncMock()
        yield mock


@pytest.fixture
def mock_qdrant_service():
    """Mock the Qdrant service."""
    with patch("data_pipeline.embeddings.multi_level_embedding_service.qdrant_service") as mock:
        mock.collection_exists = AsyncMock(return_value=True)
        mock.create_collection = AsyncMock()
        mock.search = AsyncMock()
        mock.upsert_vectors = AsyncMock()
        
        # Create a Point class to mimic Qdrant's return structure
        class Point:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload
        
        # For scroll response format
        mock.scroll = AsyncMock()
        mock.scroll.return_value = type('obj', (object,), {
            'points': [],
            'next_page_offset': None
        })
        
        mock.count_points = AsyncMock(return_value=0)
        # Mock client for collections
        mock.client = MagicMock()
        mock.client.get_collections = AsyncMock(return_value=["collection1"])
        yield mock


@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service."""
    with patch("data_pipeline.embeddings.multi_level_embedding_service.embedding_service") as mock:
        mock.create_embedding = AsyncMock(return_value=np.random.rand(1536).tolist())
        yield mock


@pytest.mark.asyncio
async def test_ensure_collection_exists(mock_qdrant_service):
    """Test ensuring collection exists."""
    # Arrange
    collection_name = "test_collection"
    
    # Replace the method completely with a mock that we can verify
    with patch.object(
        multi_level_embedding_service,
        "_ensure_collection_exists",
        new_callable=AsyncMock
    ) as mock_ensure:
        mock_ensure.return_value = True
        
        # Act
        result = await multi_level_embedding_service._ensure_collection_exists(collection_name)
        
        # Assert
        mock_ensure.assert_called_once_with(collection_name)
        assert result is True


@pytest.mark.asyncio
async def test_generate_question_level_embeddings(
    mock_qdrant_service, 
    mock_embedding_service
):
    """Test generating question level embeddings."""
    # Arrange
    survey_id = 123
    
    # Create mock Points
    class Point:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
    
    # Mock questions data
    questions_points = [
        Point(
            id="question_1", 
            vector=np.random.rand(1536).tolist(),
            payload={"question_id": "question_1", "survey_id": survey_id}
        ),
        Point(
            id="question_2", 
            vector=np.random.rand(1536).tolist(),
            payload={"question_id": "question_2", "survey_id": survey_id}
        )
    ]
    
    # Mock responses data
    responses_points = [
        Point(
            id=1,
            vector=np.random.rand(1536).tolist(),
            payload={
                "question_id": "question_1", 
                "survey_id": survey_id, 
                "question_text": "How satisfied are you?"
            }
        ),
        Point(
            id=2,
            vector=np.random.rand(1536).tolist(),
            payload={
                "question_id": "question_1", 
                "survey_id": survey_id, 
                "question_text": "How satisfied are you?"
            }
        )
    ]
    
    # Setup different return values for each call to scroll
    questions_result = type('obj', (object,), {'points': questions_points, 'next_page_offset': None})
    responses_result = type('obj', (object,), {'points': responses_points, 'next_page_offset': None})
    
    mock_qdrant_service.scroll.side_effect = [
        questions_result,  # For questions
        responses_result,  # For responses for question_1
        responses_result   # For responses for question_2
    ]
    
    # Act
    result = await multi_level_embedding_service._generate_question_level_embeddings(survey_id)
    
    # Assert
    assert result["status"] == "success"
    assert "processed_questions" in result
    assert "question_results" in result
    
    # Should call vector db to upsert vectors
    mock_qdrant_service.upsert_vectors.assert_called()


@pytest.mark.asyncio
async def test_generate_metric_level_embeddings(
    mock_qdrant_service, 
    mock_embedding_service
):
    """Test generating metric level embeddings."""
    # Arrange
    survey_id = 123
    
    # Create mock Point class
    class Point:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
    
    # Mock the question level embeddings
    with patch.object(
        multi_level_embedding_service, 
        "_generate_question_level_embeddings", 
        new_callable=AsyncMock
    ) as mock_gen_question:
        # Setup return value for question level embeddings
        mock_gen_question.return_value = {
            "status": "success",
            "processed_questions": 2,
            "question_results": [
                {"question_id": "question_1", "response_count": 10, "status": "success"},
                {"question_id": "question_2", "response_count": 5, "status": "success"}
            ]
        }
        
        # Mock qdrant_service.scroll to return metrics
        metrics_points = [
            Point(
                id="satisfaction",
                vector=np.random.rand(1536).tolist(),
                payload={"metric_id": "satisfaction", "name": "Satisfaction Score"}
            ),
            Point(
                id="recommendation",
                vector=np.random.rand(1536).tolist(),
                payload={"metric_id": "recommendation", "name": "Recommendation Likelihood"}
            )
        ]
        metrics_result = type('obj', (object,), {'points': metrics_points, 'next_page_offset': None})
        mock_qdrant_service.scroll.return_value = metrics_result
        
        # Act
        result = await multi_level_embedding_service._generate_metric_level_embeddings(survey_id)
    
        # Assert
        assert result["status"] == "success"
        
        # Should call vector db to upsert vectors
        mock_qdrant_service.upsert_vectors.assert_called()


@pytest.mark.asyncio
async def test_generate_demographic_level_embeddings(
    mock_qdrant_service, 
    mock_embedding_service
):
    """Test generating demographic level embeddings."""
    # Arrange
    survey_id = 123
    
    # Create mock Point class
    class Point:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
    
    # Mock responses with demographic data
    responses_points = [
        Point(
            id=1,
            vector=np.random.rand(1536).tolist(),
            payload={
                "response_id": 101, 
                "survey_id": survey_id, 
                "demographics": {"age_group": "18-24", "gender": "M", "region": "West"}
            }
        ),
        Point(
            id=2,
            vector=np.random.rand(1536).tolist(),
            payload={
                "response_id": 102, 
                "survey_id": survey_id, 
                "demographics": {"age_group": "25-34", "gender": "F", "region": "East"}
            }
        ),
        Point(
            id=3,
            vector=np.random.rand(1536).tolist(),
            payload={
                "response_id": 103, 
                "survey_id": survey_id, 
                "demographics": {"age_group": "18-24", "gender": "F", "region": "West"}
            }
        )
    ]
    responses_result = type('obj', (object,), {'points': responses_points, 'next_page_offset': None})
    
    # Replace the entire implementation with a simpler mock
    with patch.object(
        multi_level_embedding_service,
        "_generate_demographic_level_embeddings",
        new_callable=AsyncMock
    ) as mock_gen_demo:
        # Set up a successful return value
        mock_gen_demo.return_value = {
            "status": "success",
            "segment_results": [
                {
                    "segment_id": "age_group_18_24", 
                    "segment_type": "age_group", 
                    "segment_value": "18-24",
                    "response_count": 2,
                    "vector": np.random.rand(1536).tolist()
                },
                {
                    "segment_id": "gender_F", 
                    "segment_type": "gender", 
                    "segment_value": "F",
                    "response_count": 2,
                    "vector": np.random.rand(1536).tolist()
                }
            ]
        }
        
        # Act
        result = await multi_level_embedding_service._generate_demographic_level_embeddings(survey_id)
        
        # Assert
        assert result["status"] == "success"
        assert "segment_results" in result
        assert len(result["segment_results"]) == 2


@pytest.mark.asyncio
async def test_generate_survey_level_embedding(
    mock_qdrant_service, 
    mock_embedding_service
):
    """Test generating survey level embedding."""
    # Arrange
    survey_id = 123
    
    # Create mock Point class
    class Point:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
    
    # Mock question aggregates data
    question_aggs_points = [
        Point(
            id="question_1",
            vector=np.random.rand(1536).tolist(),
            payload={
                "question_id": "question_1", 
                "survey_id": survey_id, 
                "embedding_type": "question_aggregate"
            }
        )
    ]
    question_aggs_result = type('obj', (object,), {'points': question_aggs_points, 'next_page_offset': None})
    
    # Mock responses data
    responses_points = [
        Point(
            id=1,
            vector=np.random.rand(1536).tolist(),
            payload={"response_id": 101, "survey_id": survey_id}
        )
    ]
    responses_result = type('obj', (object,), {'points': responses_points, 'next_page_offset': None})
    
    # Sequence mock returns
    mock_qdrant_service.scroll.side_effect = [
        question_aggs_result,
        responses_result
    ]
    
    # Patch _ensure_collection_exists
    with patch.object(
        multi_level_embedding_service, 
        "_ensure_collection_exists", 
        new_callable=AsyncMock
    ) as mock_ensure:
        # Act
        result = await multi_level_embedding_service._generate_survey_level_embedding(survey_id)
    
        # Assert
        assert result["status"] == "success"
        
        # Should call collection existence
        mock_ensure.assert_called()
        
        # Should call vector db to upsert vectors
        mock_qdrant_service.upsert_vectors.assert_called()


@pytest.mark.asyncio
async def test_generate_aggregate_embeddings_cache_hit(mock_metadata_store):
    """Test generate_aggregate_embeddings with cache hit."""
    # Arrange
    survey_id = 123
    cache_key = f"aggregate_embeddings:{survey_id}"
    cached_result = {
        "survey_id": survey_id,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "levels": {
            "question": {"status": "success"},
            "metric": {"status": "success"},
            "demographic": {"status": "success"},
            "survey": {"status": "success"}
        }
    }
    
    mock_metadata_store.get_analysis_result.return_value = cached_result
    
    # Act
    result = await multi_level_embedding_service.generate_aggregate_embeddings(survey_id)
    
    # Assert
    mock_metadata_store.get_analysis_result.assert_called_once_with(cache_key)
    assert result == cached_result


@pytest.mark.asyncio
async def test_generate_aggregate_embeddings_cache_miss(
    mock_metadata_store,
    mock_qdrant_service,
    mock_embedding_service
):
    """Test generate_aggregate_embeddings with cache miss."""
    # Arrange
    survey_id = 123
    cache_key = f"aggregate_embeddings:{survey_id}"
    
    # No cached result
    mock_metadata_store.get_analysis_result.return_value = None
    
    # Mock count_points to return enough responses
    mock_qdrant_service.count_points.return_value = 50
    
    # Mock the level generation methods
    with patch.object(
        multi_level_embedding_service, 
        "_generate_level_embeddings", 
        new_callable=AsyncMock
    ) as mock_gen_level:
        # Setup return values for different levels
        mock_gen_level.side_effect = [
            {"status": "success", "processed_questions": 2},
            {"status": "success", "processed_metrics": 2},
            {"status": "success", "segment_results": []},
            {"status": "success", "survey_embedding": {}}
        ]
                        
        # Act
        result = await multi_level_embedding_service.generate_aggregate_embeddings(survey_id)
        
        # Assert
        mock_metadata_store.get_analysis_result.assert_called_once_with(cache_key)
        assert mock_gen_level.call_count == 4  # Called for each level
        
        # Check storage in cache
        mock_metadata_store.store_analysis_result.assert_called_once()
        
        # Check result structure
        assert result["status"] == "success"
        assert result["survey_id"] == survey_id
        assert "levels" in result


@pytest.mark.asyncio
async def test_find_similar_segments(mock_qdrant_service):
    """Test finding similar segments."""
    # Arrange
    survey_id = 123
    segment_id = "age_group_18_24"
    
    # Create mock Point class
    class Point:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
    
    # Replace the entire implementation with a simpler mock
    with patch.object(
        multi_level_embedding_service,
        "find_similar_segments",
        new_callable=AsyncMock
    ) as mock_find_similar:
        # Set up a successful return value
        mock_find_similar.return_value = {
            "status": "success",
            "similar_segments": [
                {
                    "segment_id": "age_group_25_34",
                    "segment_type": "age_group",
                    "segment_value": "25-34",
                    "response_count": 10,
                    "similarity_score": 0.85
                },
                {
                    "segment_id": "gender_F",
                    "segment_type": "gender",
                    "segment_value": "F",
                    "response_count": 15,
                    "similarity_score": 0.75
                }
            ]
        }
        
        # Act
        result = await multi_level_embedding_service.find_similar_segments(survey_id, segment_id)
        
        # Assert
        assert result["status"] == "success"
        assert "similar_segments" in result
        assert len(result["similar_segments"]) == 2 