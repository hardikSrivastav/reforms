"""
Test configuration and fixtures.
"""

import os
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any

import numpy as np
from qdrant_client.http.models import PointStruct

from data_pipeline.services.qdrant_client import QdrantService
from data_pipeline.embeddings.embedding_service import EmbeddingService


# Define MockResponse class at module level so it's accessible everywhere
class MockEmbeddingObject:
    def __init__(self, embedding):
        self.embedding = embedding

class MockResponse:
    def __init__(self, embeddings):
        self.data = [MockEmbeddingObject(emb) for emb in embeddings]


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = MagicMock()
    
    # Mock collections method
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections
    
    # Mock create_collection method
    mock_client.create_collection.return_value = None
    
    # Mock create_payload_index method
    mock_client.create_payload_index.return_value = None
    
    # Mock upsert method
    mock_client.upsert.return_value = None
    
    # Mock search method
    mock_client.search.return_value = []
    
    # Mock delete method
    mock_client.delete.return_value = None
    
    return mock_client


@pytest.fixture
def mock_qdrant_service(mock_qdrant_client):
    """Mock QdrantService with a mock client."""
    with patch('data_pipeline.services.qdrant_client.QdrantClient', return_value=mock_qdrant_client):
        service = QdrantService(url="http://mock-qdrant:6333")
        service.client = mock_qdrant_client
        yield service


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for embeddings."""
    # Create a fixed random vector of 1536 dimensions
    np.random.seed(42)
    vector = np.random.random(1536).tolist()
    
    return MockResponse([vector])


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    
    # Create mock embeddings object with create method
    mock_embeddings = MagicMock()
    mock_create = MagicMock()
    
    # Make create method return the mock response
    async def mock_create_async(**kwargs):
        if isinstance(kwargs.get('input'), list):
            # For batch embeddings, return a list of the same vector
            vectors = [mock_openai_response.data[0].embedding for _ in kwargs.get('input')]
            return MockResponse(vectors)
        return mock_openai_response
    
    mock_create.create = mock_create_async
    mock_client.embeddings = mock_create
    
    return mock_client


@pytest.fixture
def mock_embedding_service(mock_openai_client):
    """Mock EmbeddingService with a mock OpenAI client."""
    with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
        service = EmbeddingService(api_key="mock-api-key")
        service.client = mock_openai_client
        
        # Override the methods to use our mocks
        async def mock_get_embedding(text):
            response = await mock_openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
            return response.data[0].embedding
        
        async def mock_get_embeddings_batch(texts):
            response = await mock_openai_client.embeddings.create(input=texts, model="text-embedding-ada-002")
            return [item.embedding for item in response.data]
        
        # Replace the methods with our mocked versions
        service.get_embedding = mock_get_embedding
        service.get_embeddings_batch = mock_get_embeddings_batch
        
        yield service


@pytest.fixture
def sample_survey_data():
    """Sample survey data for testing."""
    return {
        "_id": "sample_survey_id",
        "title": "Sample Survey",
        "description": "A sample survey for testing",
        "questions": [
            {
                "id": "q1",
                "question": "How satisfied are you with our service?",
                "type": "radio",
                "options": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"]
            },
            {
                "id": "q2",
                "question": "What features do you like most?",
                "type": "checkbox",
                "options": ["UI", "Performance", "Reliability", "Support"]
            },
            {
                "id": "q3",
                "question": "Any additional feedback?",
                "type": "text"
            }
        ]
    }


@pytest.fixture
def sample_survey_responses():
    """Sample survey responses for testing."""
    return [
        {
            "_id": "response1",
            "survey_mongo_id": "sample_survey_id",
            "responses": {
                "q1": "Satisfied",
                "q2": ["UI", "Performance"],
                "q3": "Great product overall!"
            },
            "submitted_at": "2023-05-01T10:30:00Z"
        },
        {
            "_id": "response2",
            "survey_mongo_id": "sample_survey_id",
            "responses": {
                "q1": "Neutral",
                "q2": ["Support"],
                "q3": "Could use more features."
            },
            "submitted_at": "2023-05-02T14:15:00Z"
        }
    ]


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return [
        {
            "id": 1,
            "name": "Customer Satisfaction",
            "type": "likert",
            "description": "Overall satisfaction with our service"
        },
        {
            "id": 2,
            "name": "Feature Preference",
            "type": "multiple_choice",
            "description": "Which features users prefer"
        },
        {
            "id": 3,
            "name": "Feedback Quality",
            "type": "text",
            "description": "Quality of text feedback provided"
        }
    ]


@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client for testing."""
    mock_client = MagicMock()
    
    # Create mock db attribute
    mock_db = MagicMock()
    
    # Create mock collections
    mock_forms = MagicMock()
    mock_responses = MagicMock()
    
    # Mock find_one method for forms
    async def mock_find_one(*args, **kwargs):
        if kwargs.get("_id", None) == "sample_survey_id":
            return {
                "_id": "sample_survey_id",
                "title": "Sample Survey",
                "description": "A sample survey for testing",
                "questions": [
                    {
                        "id": "q1",
                        "question": "How satisfied are you with our service?",
                        "type": "radio"
                    }
                ]
            }
        return None
    
    mock_forms.find_one = mock_find_one
    
    # Mock find method for responses
    class MockCursor:
        def __init__(self, responses):
            self.responses = responses
            self.index = 0
        
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if self.index < len(self.responses):
                response = self.responses[self.index]
                self.index += 1
                return response
            raise StopAsyncIteration
    
    mock_responses.find = lambda *args, **kwargs: MockCursor([
        {
            "_id": "response1",
            "survey_mongo_id": "sample_survey_id",
            "responses": {"q1": "Satisfied"}
        }
    ])
    
    # Assign collections to db
    mock_db.forms = mock_forms
    mock_db.responses = mock_responses
    
    # Assign db to client
    mock_client.survey_db = mock_db
    
    return mock_client


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 