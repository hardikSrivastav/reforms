"""
Unit tests for the metadata store service.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import pickle
from datetime import datetime, timedelta

from data_pipeline.services.metadata_store import MetadataStore, metadata_store

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    with patch('data_pipeline.services.metadata_store.redis.from_url') as mock:
        mock_redis = MagicMock()
        mock.return_value = mock_redis
        yield mock_redis

@pytest.fixture
def metadata_store_instance(mock_redis):
    """Create a MetadataStore instance for testing."""
    return MetadataStore(redis_url="redis://localhost:6379/0")

def test_get_key(metadata_store_instance):
    """Test _get_key function generates the correct keys."""
    # Test with entity_id
    key = metadata_store_instance._get_key("base_analysis", 123, "metric456")
    assert key == "base_analysis:123:metric456"
    
    # Test without entity_id
    key = metadata_store_instance._get_key("cross_metric_analysis", 123)
    assert key == "cross_metric_analysis:123"

def test_store_analysis_result(metadata_store_instance, mock_redis):
    """Test storing analysis results."""
    # Test data
    key_type = "base_analysis"
    survey_id = 123
    entity_id = "metric456"
    result = {"key": "value", "nested": {"data": 42}}
    ttl = 3600
    
    # Store the result
    success = metadata_store_instance.store_analysis_result(
        key_type, survey_id, result, entity_id, ttl
    )
    
    # Verify
    assert success is True
    assert "timestamp" in result
    
    # Check that Redis set was called with the correct arguments
    mock_redis.set.assert_called_once()
    args, kwargs = mock_redis.set.call_args
    
    # First arg should be the key
    assert args[0] == "base_analysis:123:metric456"
    
    # Second arg should be the serialized result
    serialized_result = pickle.loads(args[1])
    assert "key" in serialized_result
    assert serialized_result["key"] == "value"
    assert "nested" in serialized_result
    assert serialized_result["nested"]["data"] == 42
    
    # TTL should be set
    assert kwargs["ex"] == ttl

def test_get_analysis_result(metadata_store_instance, mock_redis):
    """Test retrieving analysis results."""
    # Test data
    key_type = "base_analysis"
    survey_id = 123
    entity_id = "metric456"
    stored_result = {"key": "value", "timestamp": datetime.now().isoformat()}
    
    # Set up mock to return serialized data
    serialized = pickle.dumps(stored_result)
    mock_redis.get.return_value = serialized
    
    # Get the result
    result = metadata_store_instance.get_analysis_result(key_type, survey_id, entity_id)
    
    # Verify
    assert result is not None
    assert result["key"] == "value"
    assert "timestamp" in result
    
    # Check that Redis get was called with the correct key
    mock_redis.get.assert_called_once_with("base_analysis:123:metric456")

def test_get_analysis_result_not_found(metadata_store_instance, mock_redis):
    """Test retrieving non-existent analysis results."""
    # Set up mock to return None (key not found)
    mock_redis.get.return_value = None
    
    # Get the result
    result = metadata_store_instance.get_analysis_result("base_analysis", 123, "metric456")
    
    # Verify
    assert result is None
    mock_redis.get.assert_called_once()

def test_delete_analysis_result(metadata_store_instance, mock_redis):
    """Test deleting analysis results."""
    # Delete a result
    success = metadata_store_instance.delete_analysis_result("base_analysis", 123, "metric456")
    
    # Verify
    assert success is True
    mock_redis.delete.assert_called_once_with("base_analysis:123:metric456")

def test_delete_survey_data(metadata_store_instance, mock_redis):
    """Test deleting all data for a survey."""
    # Set up mock to return keys
    mock_redis.keys.return_value = ["base_analysis:123:m1", "metric_analysis:123:m2"]
    
    # Delete survey data
    success = metadata_store_instance.delete_survey_data(123)
    
    # Verify
    assert success is True
    mock_redis.keys.assert_called_once_with("*:123:*")
    mock_redis.delete.assert_called_once_with("base_analysis:123:m1", "metric_analysis:123:m2")

def test_is_cache_valid_fresh(metadata_store_instance, mock_redis):
    """Test cache validation with fresh data."""
    # Create a result with a recent timestamp
    now = datetime.now()
    stored_result = {
        "key": "value",
        "timestamp": now.isoformat()
    }
    
    # Set up mock to return serialized data
    serialized = pickle.dumps(stored_result)
    mock_redis.get.return_value = serialized
    
    # Check if cache is valid with default max_age
    is_valid = metadata_store_instance.is_cache_valid("base_analysis", 123, "metric456")
    
    # Should be valid since timestamp is recent
    assert is_valid is True

def test_is_cache_valid_stale(metadata_store_instance, mock_redis):
    """Test cache validation with stale data."""
    # Create a result with an old timestamp
    old_time = datetime.now() - timedelta(hours=2)
    stored_result = {
        "key": "value",
        "timestamp": old_time.isoformat()
    }
    
    # Set up mock to return serialized data
    serialized = pickle.dumps(stored_result)
    mock_redis.get.return_value = serialized
    
    # Check if cache is valid with a max_age of 1 hour
    is_valid = metadata_store_instance.is_cache_valid(
        "base_analysis", 123, "metric456", max_age=3600
    )
    
    # Should be invalid since timestamp is 2 hours old
    assert is_valid is False

def test_is_cache_valid_not_found(metadata_store_instance, mock_redis):
    """Test cache validation with non-existent data."""
    # Set up mock to return None (key not found)
    mock_redis.get.return_value = None
    
    # Check if cache is valid
    is_valid = metadata_store_instance.is_cache_valid("base_analysis", 123, "metric456")
    
    # Should be invalid since no data exists
    assert is_valid is False

def test_singleton_instance():
    """Test that the singleton instance is properly created."""
    assert isinstance(metadata_store, MetadataStore) 