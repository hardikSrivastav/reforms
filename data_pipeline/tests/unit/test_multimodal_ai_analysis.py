"""
Unit tests for the MultimodalAIAnalysisService.
"""

import pytest
import json
import os
import base64
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from data_pipeline.analysis.multimodal_ai_analysis import (
    MultimodalAIAnalysisService,
    AnalysisType,
    AIModel,
    multimodal_ai_analysis_service
)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch('data_pipeline.analysis.multimodal_ai_analysis.AsyncOpenAI') as mock_client:
        # Create a mock AsyncOpenAI instance
        mock_instance = AsyncMock()
        
        # Mock the chat completions API
        mock_chat = AsyncMock()
        mock_instance.chat.completions.create = mock_chat
        
        # Configure the mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # Default response is a JSON string
        mock_message.content = json.dumps({
            "key": "value",
            "analysis": "This is a mock analysis"
        })
        
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_chat.return_value = mock_response
        
        # Set the mocked instance to be returned
        mock_client.return_value = mock_instance
        
        yield mock_instance


@pytest.fixture
def mock_metadata_store():
    """Create a mock metadata store."""
    with patch('data_pipeline.analysis.multimodal_ai_analysis.metadata_store') as mock:
        # Default behavior: no cached results
        mock.get_analysis_result.return_value = None
        mock.store_analysis_result = AsyncMock()
        yield mock


@pytest.fixture
def multimodal_service(mock_openai_client):
    """Create a MultimodalAIAnalysisService instance for testing."""
    return MultimodalAIAnalysisService(api_key="test_api_key")


@pytest.fixture
def sample_text_responses():
    """Sample text responses for testing."""
    return [
        {"text": "I really enjoyed the product. It works very well!", "user_id": "user1"},
        {"text": "The service was good but could be improved.", "user_id": "user2"},
        {"text": "Not satisfied with my purchase. It broke after a week.", "user_id": "user3"},
        {"text": "Amazing experience overall, would recommend to others.", "user_id": "user4"},
        {"text": "It's okay, nothing special about it though.", "user_id": "user5"}
    ]


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing."""
    base_date = datetime.now() - timedelta(days=90)
    return [
        {"period": (base_date + timedelta(days=i*10)).strftime("%Y-%m-%d"), "value": 3.5 + (i * 0.2)}
        for i in range(10)
    ]


@pytest.fixture
def sample_image_responses():
    """Sample image responses for testing."""
    # Create a tiny 1x1 pixel JPEG as base64
    tiny_jpeg = "/9j/4AAQSkZJRgABAQEAYABgAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2ODApLCBxdWFsaXR5ID0gOTAK/9sAQwADAgIDAgIDAwMDBAMDBAUIBQUEBAUKBwcGCAwKDAwLCgsLDQ4SEA0OEQ4LCxAWEBETFBUVFQwPFxgWFBgSFBUU/9sAQwEDBAQFBAUJBQUJFA0LDRQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU/8AAEQgAAQABAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A+3KKKK/Nz9LP/9k="
    
    return [
        {"image_data": tiny_jpeg, "user_id": "user1"},
        {"image_data": tiny_jpeg, "user_id": "user2"},
        {"image_data": tiny_jpeg, "user_id": "user3"}
    ]


@pytest.mark.asyncio
async def test_template_loading(multimodal_service):
    """Test template loading functionality."""
    # Test with a mocked open to simulate template files
    mock_template_content = "Test template content with {{question}} placeholder"
    with patch("builtins.open", mock_open(read_data=mock_template_content)):
        with patch("os.path.exists", return_value=True):
            # Force reload of a template
            template = multimodal_service._load_template("text_sentiment")
    
    assert "{{question}}" in template
    assert template == mock_template_content
    
    # Test the default template fallback
    with patch("os.path.exists", return_value=False):
        template = multimodal_service._load_template("text_sentiment")
    
    assert isinstance(template, str)
    assert "{{question}}" in template
    assert "sentiment" in template.lower()


@pytest.mark.asyncio
async def test_ai_response_generation(multimodal_service, mock_openai_client):
    """Test AI response generation."""
    # Configure mock to return a specific response
    mock_message = MagicMock()
    mock_message.content = "Test AI response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test the response generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    response = await multimodal_service._generate_ai_response(messages, AIModel.GPT4)
    
    # Verify the response
    assert response == "Test AI response"
    
    # Verify the API was called with correct parameters
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == AIModel.GPT4.value
    assert call_args["messages"] == messages
    assert call_args["temperature"] == 0.0


@pytest.mark.asyncio
async def test_ai_response_with_functions(multimodal_service, mock_openai_client):
    """Test AI response generation with function calling."""
    # Configure mock to return a specific response
    mock_message = MagicMock()
    mock_message.content = json.dumps({"result": "function_result"})
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test the response generation with functions
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this data."}
    ]
    functions = [{
        "name": "test_function",
        "description": "A test function",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        }
    }]
    
    response = await multimodal_service._generate_ai_response(
        messages, AIModel.GPT4, 0.0, functions
    )
    
    # Verify the response
    assert "function_result" in response
    
    # Verify the API was called with correct parameters
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["functions"] == functions
    assert call_args["function_call"] == "auto"


@pytest.mark.asyncio
async def test_retry_on_error(multimodal_service, mock_openai_client):
    """Test retry functionality on API errors."""
    # Configure mock to raise an exception then succeed
    mock_openai_client.chat.completions.create.side_effect = [
        Exception("API error"),  # First call fails
        MagicMock(choices=[MagicMock(message=MagicMock(content="Success after retry"))])  # Second succeeds
    ]
    
    # Test the response generation with retry
    messages = [{"role": "user", "content": "Test retry"}]
    response = await multimodal_service._generate_ai_response(messages, AIModel.GPT4)
    
    # Verify the response after retry
    assert response == "Success after retry"
    
    # Verify API was called twice (fail + success)
    assert mock_openai_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_analyze_text_sentiment(
    multimodal_service, 
    sample_text_responses, 
    mock_openai_client,
    mock_metadata_store
):
    """Test text sentiment analysis."""
    # Configure mock to return specific sentiment analysis
    sentiment_analysis = {
        "sentiment_distribution": {"positive": 0.6, "neutral": 0.2, "negative": 0.2},
        "key_themes": [{"theme": "Product quality", "sentiment": "positive", "frequency": 0.4}],
        "summary": "Overall positive sentiment with some concerns about durability."
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(sentiment_analysis)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test the sentiment analysis
    result = await multimodal_service.analyze_text_sentiment(
        123, 
        "question_1", 
        "How satisfied are you with our product?",
        sample_text_responses
    )
    
    # Verify the result structure
    assert "sentiment_distribution" in result
    assert result["sentiment_distribution"]["positive"] == 0.6
    assert "key_themes" in result
    assert "summary" in result
    assert result["survey_id"] == 123
    assert result["question_id"] == "question_1"
    assert result["analysis_type"] == AnalysisType.TEXT_SENTIMENT.value
    
    # Verify the API was called with correct prompt
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    assert any("How satisfied are you with our product?" in msg.get("content", "") for msg in messages)
    
    # Verify result was cached
    mock_metadata_store.store_analysis_result.assert_called_once()
    call_args = mock_metadata_store.store_analysis_result.call_args[0]
    assert call_args[0] == f"sentiment:123:question_1"


@pytest.mark.asyncio
async def test_analyze_text_sentiment_cache_hit(
    multimodal_service, 
    sample_text_responses, 
    mock_openai_client,
    mock_metadata_store
):
    """Test text sentiment analysis with cache hit."""
    # Configure mock to return cached result
    cached_result = {
        "sentiment_distribution": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
        "key_themes": [{"theme": "Cached result", "sentiment": "positive", "frequency": 0.5}],
        "summary": "This is a cached result.",
        "survey_id": 123,
        "question_id": "question_1",
    }
    mock_metadata_store.get_analysis_result.return_value = cached_result
    
    # Test the sentiment analysis with cache hit
    result = await multimodal_service.analyze_text_sentiment(
        123, 
        "question_1", 
        "How satisfied are you with our product?",
        sample_text_responses
    )
    
    # Verify cached result was returned
    assert result == cached_result
    
    # Verify OpenAI API was not called
    mock_openai_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_numeric_trends(
    multimodal_service, 
    sample_time_series_data, 
    mock_openai_client,
    mock_metadata_store
):
    """Test numeric trend analysis."""
    # Configure mock to return specific trend analysis
    trend_analysis = {
        "trend_direction": "upward",
        "trend_strength": 0.8,
        "significant_changes": [
            {"period": "2023-03-01", "change": 0.5, "explanation": "Product launch"}
        ],
        "summary": "Strong upward trend over the past 90 days."
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(trend_analysis)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test the trend analysis
    result = await multimodal_service.analyze_numeric_trends(
        123, 
        "question_2", 
        "How would you rate our service over time?",
        sample_time_series_data
    )
    
    # Verify the result structure
    assert "trend_direction" in result
    assert result["trend_direction"] == "upward"
    assert "significant_changes" in result
    assert "summary" in result
    assert result["survey_id"] == 123
    assert result["question_id"] == "question_2"
    assert result["analysis_type"] == AnalysisType.NUMERIC_TREND.value
    
    # Verify the API was called with correct prompt
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    assert any("How would you rate our service over time?" in msg.get("content", "") for msg in messages)
    assert any("Period:" in msg.get("content", "") for msg in messages)
    
    # Verify result was cached
    mock_metadata_store.store_analysis_result.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_free_responses(
    multimodal_service, 
    sample_text_responses, 
    mock_openai_client,
    mock_metadata_store
):
    """Test free text response analysis with chain-of-thought reasoning."""
    # Configure mock to return detailed analysis with thought process
    free_response_analysis = {
        "thought_process": "I first noticed patterns around product quality...",
        "themes": [
            {"theme": "Product Quality", "frequency": 0.6, "sample_responses": ["Sample 1", "Sample 2"]}
        ],
        "insights": [
            {"insight": "Quality is the primary driver of satisfaction", "confidence": 0.8}
        ],
        "summary": "Users are primarily concerned with product quality."
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(free_response_analysis)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test the free response analysis
    result = await multimodal_service.analyze_free_responses(
        123, 
        "question_3", 
        "What do you think could be improved?",
        sample_text_responses,
        guidance="Focus on actionable improvements"
    )
    
    # Verify the result structure
    assert "thought_process" in result
    assert "themes" in result
    assert "insights" in result
    assert "summary" in result
    assert result["survey_id"] == 123
    assert result["question_id"] == "question_3"
    assert result["analysis_type"] == AnalysisType.FREE_RESPONSE.value
    
    # Verify the API was called with correct prompt
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    prompt = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
    
    assert "What do you think could be improved?" in prompt
    assert "Focus on actionable improvements" in prompt
    
    # Verify chain-of-thought was requested in system message
    system_msg = next((msg.get("content", "") for msg in messages if msg.get("role") == "system"), "")
    assert "step" in system_msg.lower()
    assert "process" in system_msg.lower()
    
    # Verify result was cached
    mock_metadata_store.store_analysis_result.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_image_responses(
    multimodal_service, 
    sample_image_responses, 
    mock_openai_client,
    mock_metadata_store
):
    """Test image response analysis."""
    # Configure mock to return image analysis
    image_analysis = "The images show product usage in various contexts. Common themes include outdoor usage and social settings."
    mock_message = MagicMock()
    mock_message.content = image_analysis
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test the image analysis
    result = await multimodal_service.analyze_image_responses(
        123, 
        "question_4", 
        "Please share a photo of how you use our product.",
        sample_image_responses
    )
    
    # Verify the result structure
    assert "raw_analysis" in result
    assert "structured_analysis" in result
    assert result["survey_id"] == 123
    assert result["question_id"] == "question_4"
    assert result["analysis_type"] == AnalysisType.IMAGE_ANALYSIS.value
    assert result["image_count"] == len(sample_image_responses)
    
    # Verify the API was called with correct message structure for vision
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == AIModel.GPT4_VISION.value
    
    messages = call_args["messages"]
    user_message = next((msg for msg in messages if msg.get("role") == "user"), None)
    assert user_message is not None
    assert isinstance(user_message["content"], list)
    
    # First item should be text, rest should be images
    assert user_message["content"][0]["type"] == "text"
    assert "Please share a photo" in user_message["content"][0]["text"]
    
    # Verify has image content
    image_content = [item for item in user_message["content"] if item.get("type") == "image_url"]
    assert len(image_content) == len(sample_image_responses)
    
    # Verify result was cached
    mock_metadata_store.store_analysis_result.assert_called_once()


@pytest.mark.asyncio
async def test_extract_structured_image_analysis(multimodal_service):
    """Test extracting structured data from raw image analysis text."""
    # Test with raw text containing sections
    raw_analysis = """
    # Analysis of Product Usage Images
    
    ## Common Themes
    The images show several common themes:
    - Outdoor usage (75% of images)
    - Social settings (50% of images)
    - Professional environments (25% of images)
    
    ## Objects Identified
    - Product clearly visible
    - Smartphones
    - Laptops
    
    ## Emotional Content
    The images generally convey positive emotions through smiling users.
    
    ## Summary
    Users primarily showcase the product in outdoor and social contexts, suggesting it's valued for its portability and social aspects.
    
    ## Recommendations
    - Highlight outdoor usage in marketing
    - Emphasize social aspects in advertising
    """
    
    result = multimodal_service._extract_structured_image_analysis(raw_analysis)
    
    # Verify the structured extraction
    assert isinstance(result, dict)
    assert "themes" in result
    assert "objects" in result
    assert "emotions" in result
    assert "summary" in result
    assert "recommendations" in result
    
    assert "outdoor usage" in " ".join(result["themes"]).lower()
    assert "product clearly visible" in " ".join(result["objects"]).lower()
    assert "outdoor and social contexts" in result["summary"].lower()
    
    # Test with JSON-like content
    json_analysis = '{"themes": ["Outdoor usage", "Social settings"], "summary": "Test summary"}'
    result = multimodal_service._extract_structured_image_analysis(json_analysis)
    
    assert isinstance(result, dict)
    assert "themes" in result
    assert "summary" in result


@pytest.mark.asyncio
async def test_analyze_multi_modal(
    multimodal_service,
    sample_text_responses,
    sample_image_responses,
    mock_openai_client,
    mock_metadata_store
):
    """Test multi-modal analysis combining text and image data."""
    # Mock the separate analysis methods
    text_analysis_result = {
        "themes": [{"theme": "Product quality", "frequency": 0.6}],
        "summary": "Users focused on quality."
    }
    
    image_analysis_result = {
        "structured_analysis": {
            "themes": ["Outdoor usage", "Social context"],
            "summary": "Images show product in use outdoors."
        }
    }
    
    # Mock the integration analysis
    integrated_analysis = {
        "cross_modal_themes": [
            {"theme": "Quality outdoor product", "text_support": 0.6, "image_support": 0.8}
        ],
        "contradictions": [],
        "integrated_insights": ["Product is primarily used outdoors"],
        "summary": "Overall positive view of product for outdoor use."
    }
    
    # Set up the mocks
    with patch.object(
        multimodal_service, 
        'analyze_free_responses', 
        AsyncMock(return_value=text_analysis_result)
    ) as mock_text_analysis:
        with patch.object(
            multimodal_service, 
            'analyze_image_responses', 
            AsyncMock(return_value=image_analysis_result)
        ) as mock_image_analysis:
            # Configure mock to return the integrated analysis
            mock_message = MagicMock()
            mock_message.content = json.dumps(integrated_analysis)
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            # Test the multi-modal analysis
            result = await multimodal_service.analyze_multi_modal(
                123, 
                "question_5", 
                "Please rate our product and share a photo of how you use it.",
                sample_text_responses,
                sample_image_responses
            )
            
            # Verify component analyses were called
            mock_text_analysis.assert_called_once()
            mock_image_analysis.assert_called_once()
            
            # Verify the result structure
            assert "text_analysis" in result
            assert "image_analysis" in result
            assert "integrated_analysis" in result
            assert result["survey_id"] == 123
            assert result["question_id"] == "question_5"
            assert result["analysis_type"] == AnalysisType.MULTI_MODAL.value
            
            # Verify the integrated analysis structure
            integrated = result["integrated_analysis"]
            assert "cross_modal_themes" in integrated
            assert "summary" in integrated
            
            # Verify result was cached
            mock_metadata_store.store_analysis_result.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling_in_analysis(
    multimodal_service,
    sample_text_responses,
    mock_openai_client,
    mock_metadata_store
):
    """Test error handling in analysis methods."""
    # Configure mock to return invalid JSON
    mock_message = MagicMock()
    mock_message.content = "This is not valid JSON"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Test error handling in sentiment analysis
    result = await multimodal_service.analyze_text_sentiment(
        123, 
        "question_1", 
        "How satisfied are you with our product?",
        sample_text_responses
    )
    
    # Verify we get an error result but no exception
    assert "error" in result
    assert "Failed to parse" in result["error"]
    assert result["survey_id"] == 123
    assert result["question_id"] == "question_1"
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_singleton_instance():
    """Test that singleton instance is properly configured."""
    assert multimodal_ai_analysis_service is not None
    assert isinstance(multimodal_ai_analysis_service, MultimodalAIAnalysisService) 