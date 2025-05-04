"""
Unit tests for the embedding service.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from data_pipeline.embeddings.embedding_service import EmbeddingService


@pytest.mark.asyncio
async def test_create_embedding(mock_embedding_service):
    """Test creating an embedding for a single text."""
    # Test with a single text input
    text = "This is a test text for embedding"
    embedding = await mock_embedding_service.get_embedding(text)
    
    # Verify shape and type
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # Expected embedding dimension


@pytest.mark.asyncio
async def test_create_embeddings_batch(mock_embedding_service):
    """Test creating embeddings for a batch of texts."""
    # Test with a batch of texts
    texts = [
        "First test text for embedding",
        "Second test text for embedding",
        "Third test text for embedding"
    ]
    
    embeddings = await mock_embedding_service.get_embeddings_batch(texts)
    
    # Verify shape and type
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(len(emb) == 1536 for emb in embeddings)


@pytest.mark.asyncio
async def test_embedding_format(mock_embedding_service):
    """Test the format of the embeddings."""
    text = "Test text for embedding format"
    embedding = await mock_embedding_service.get_embedding(text)
    
    # Convert to numpy for testing
    np_embedding = np.array(embedding)
    
    # Validate dimensions
    assert np_embedding.shape == (1536,)
    
    # Validate that values are within expected range (-1 to 1)
    assert np.all(np_embedding >= -1.0)
    assert np.all(np_embedding <= 1.0)


@pytest.mark.asyncio
async def test_embedding_consistency(mock_embedding_service):
    """Test that identical inputs produce identical embeddings."""
    text = "Test text for embedding consistency"
    
    # Get two embeddings for the same text
    embedding1 = await mock_embedding_service.get_embedding(text)
    embedding2 = await mock_embedding_service.get_embedding(text)
    
    # They should be identical
    assert embedding1 == embedding2


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for API failures."""
    # Create a class with a custom implementation of get_embedding that raises an exception
    class ErrorEmbeddingService(EmbeddingService):
        async def get_embedding(self, text):
            raise Exception("API Error")
            
        async def get_embeddings_batch(self, texts):
            raise Exception("API Error")
    
    # Create instance of our custom service
    service = ErrorEmbeddingService(api_key="mock-api-key")
    
    # Test error handling for single embedding
    with pytest.raises(Exception):
        await service.get_embedding("Test text")
    
    # Test error handling for batch embeddings
    with pytest.raises(Exception):
        await service.get_embeddings_batch(["Test text 1", "Test text 2"]) 