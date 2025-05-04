"""
Tests for the AI insights service.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime

from data_pipeline.services.ai_insights import AIInsightsService


@pytest.fixture
def ai_insights_service():
    return AIInsightsService(api_key="test_api_key")


@pytest.fixture
def sample_metric_data():
    return {
        "id": "satisfaction",
        "name": "Customer Satisfaction",
        "type": "numeric",
        "description": "Overall customer satisfaction rating (1-10)"
    }


@pytest.fixture
def sample_analysis_results():
    return {
        "count": 500,
        "valid_count": 485,
        "min": 2.0,
        "max": 10.0,
        "mean": 7.8,
        "median": 8.0,
        "std_deviation": 1.5,
        "variance": 2.25,
        "quartiles": {
            "q1": 7.0,
            "q2": 8.0,
            "q3": 9.0
        },
        "skewness": -0.75,
        "kurtosis": 0.25,
        "histogram": [
            {"bin_start": 2.0, "bin_end": 4.0, "count": 25},
            {"bin_start": 4.0, "bin_end": 6.0, "count": 70},
            {"bin_start": 6.0, "bin_end": 8.0, "count": 175},
            {"bin_start": 8.0, "bin_end": 10.0, "count": 215}
        ]
    }


@pytest.fixture
def sample_response_stats():
    return {
        "total_responses": 500,
        "valid_responses": 485,
        "response_rate": 0.97
    }


@pytest.fixture
def sample_correlation_results():
    return {
        "significant_correlations": [
            {
                "metric1_id": "satisfaction",
                "metric2_id": "loyalty",
                "metric1_name": "Customer Satisfaction",
                "metric2_name": "Customer Loyalty",
                "correlation": 0.85,
                "p_value": 0.0001,
                "strength": "strong",
                "direction": "positive",
                "description": "Strong positive correlation between Customer Satisfaction and Customer Loyalty"
            },
            {
                "metric1_id": "satisfaction",
                "metric2_id": "complaints",
                "metric1_name": "Customer Satisfaction",
                "metric2_name": "Number of Complaints",
                "correlation": -0.72,
                "p_value": 0.0005,
                "strength": "strong",
                "direction": "negative",
                "description": "Strong negative correlation between Customer Satisfaction and Number of Complaints"
            }
        ],
        "causal_relationships": [
            {
                "cause_metric_id": "satisfaction",
                "effect_metric_id": "loyalty",
                "cause_metric_name": "Customer Satisfaction",
                "effect_metric_name": "Customer Loyalty",
                "p_value": 0.001,
                "confidence": 0.999
            }
        ]
    }


@pytest.fixture
def sample_metrics_data():
    return {
        "satisfaction": {
            "id": "satisfaction",
            "name": "Customer Satisfaction",
            "type": "numeric",
            "description": "Overall customer satisfaction rating (1-10)"
        },
        "loyalty": {
            "id": "loyalty",
            "name": "Customer Loyalty",
            "type": "numeric",
            "description": "Customer loyalty score (1-10)"
        },
        "complaints": {
            "id": "complaints",
            "name": "Number of Complaints",
            "type": "numeric",
            "description": "Number of complaints submitted"
        }
    }


@pytest.fixture
def sample_survey_data():
    return {
        "id": 123,
        "name": "Customer Feedback Survey",
        "description": "Annual customer feedback survey",
        "total_respondents": 500,
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "metrics": {
            "satisfaction": {
                "id": "satisfaction",
                "name": "Customer Satisfaction",
                "type": "numeric",
                "description": "Overall customer satisfaction rating (1-10)"
            },
            "loyalty": {
                "id": "loyalty",
                "name": "Customer Loyalty",
                "type": "numeric",
                "description": "Customer loyalty score (1-10)"
            },
            "complaints": {
                "id": "complaints",
                "name": "Number of Complaints",
                "type": "numeric",
                "description": "Number of complaints submitted"
            }
        }
    }


@pytest.fixture
def sample_all_analysis_results():
    return {
        "metric_insights": {
            "satisfaction": {
                "structured_insights": {
                    "key_findings": [
                        "Overall satisfaction is high with an average of 7.8/10",
                        "15% of customers rate satisfaction below 6/10"
                    ],
                    "recommendations": [
                        "Focus improvement efforts on the 15% of dissatisfied customers"
                    ]
                }
            },
            "loyalty": {
                "structured_insights": {
                    "key_findings": [
                        "Customer loyalty has increased 5% year-over-year",
                        "Loyalty scores correlate strongly with satisfaction"
                    ],
                    "recommendations": [
                        "Create loyalty program for high-satisfaction customers"
                    ]
                }
            }
        },
        "cross_metric_insights": {
            "structured_insights": {
                "strongest_relationships": [
                    "Customer satisfaction strongly predicts customer loyalty (r=0.85)"
                ]
            }
        }
    }


@patch("data_pipeline.services.metadata_store.metadata_store")
@patch("data_pipeline.services.ai_insights.AsyncOpenAI")
@pytest.mark.asyncio
async def test_generate_metric_insights(mock_openai, mock_metadata_store, ai_insights_service, 
                                         sample_metric_data, sample_analysis_results, 
                                         sample_response_stats):
    """Test generating AI insights for a metric."""
    # Mock the cache check
    mock_metadata_store.get_analysis_result.return_value = None
    
    # Mock OpenAI response
    mock_openai_instance = AsyncMock()
    mock_chat_completions = AsyncMock()
    mock_openai.return_value = mock_openai_instance
    mock_openai_instance.chat.completions.create = mock_chat_completions
    
    # Set up mock response
    mock_message = MagicMock()
    mock_message.content = """
    1. Key Findings:
    - The average satisfaction rating is high at 7.8 out of 10.
    - Most customers (over 80%) rate their satisfaction as 6 or higher.
    
    2. Detailed Interpretation:
    The distribution is left-skewed, indicating generally positive sentiment. The highest concentration of responses is in the 8-10 range, which represents 44% of all responses. The negative skew value of -0.75 confirms this left skew.
    
    3. Actionable Recommendations:
    - Focus improvement efforts on understanding and addressing the concerns of the bottom 20% of respondents.
    - Investigate what drives scores in the 8-10 range to replicate these positive experiences.
    """
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_chat_completions.return_value = mock_response
    
    # Manually set the client to use our mock
    ai_insights_service.client = mock_openai_instance
    
    # Call the function
    result = await ai_insights_service.generate_metric_insights(
        123, "satisfaction", sample_metric_data, sample_analysis_results, sample_response_stats
    )
    
    # Check the function called the OpenAI API
    mock_chat_completions.assert_called_once()
    
    # Check the structure of the result
    assert "survey_id" in result
    assert "metric_id" in result
    assert "metric_name" in result
    assert "metric_type" in result
    assert "timestamp" in result
    assert "raw_insights" in result
    assert "structured_insights" in result
    
    # Skip checking cache was updated


@patch("data_pipeline.services.metadata_store.metadata_store")
@patch("data_pipeline.services.ai_insights.AsyncOpenAI")
@pytest.mark.asyncio
async def test_generate_cross_metric_insights(mock_openai, mock_metadata_store, ai_insights_service,
                                              sample_correlation_results, sample_metrics_data):
    """Test generating AI insights for cross-metric analysis."""
    # Mock the cache check
    mock_metadata_store.get_analysis_result.return_value = None
    
    # Mock OpenAI response
    mock_openai_instance = AsyncMock()
    mock_chat_completions = AsyncMock()
    mock_openai.return_value = mock_openai_instance
    mock_openai_instance.chat.completions.create = mock_chat_completions
    
    # Set up mock response
    mock_message = MagicMock()
    mock_message.content = """
    1. Strongest Relationships:
    - Customer Satisfaction and Customer Loyalty have a strong positive correlation (r=0.85), suggesting that satisfied customers are significantly more likely to be loyal.
    - Customer Satisfaction and Number of Complaints show a strong negative correlation (r=-0.72), indicating that higher satisfaction is associated with fewer complaints.
    
    2. Potential Causality:
    - The data suggests that Customer Satisfaction may directly influence Customer Loyalty. This makes intuitive business sense, as customers who are satisfied with their experience are more likely to return and demonstrate loyalty behaviors.
    
    3. Unexpected Findings:
    - None of the correlations are particularly surprising, as they align with common business understanding.
    
    4. Strategic Implications:
    Improving customer satisfaction should be a top priority as it appears to directly drive loyalty and reduce complaints. This suggests that investments in customer experience improvements could yield returns through increased loyalty and reduced service costs.
    
    5. Research Recommendations:
    - Conduct segmentation analysis to determine if these relationships vary across different customer segments.
    - Implement A/B testing of satisfaction improvement initiatives to confirm the causal relationship with loyalty.
    """
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_chat_completions.return_value = mock_response
    
    # Manually set the client to use our mock
    ai_insights_service.client = mock_openai_instance
    
    # Call the function
    result = await ai_insights_service.generate_cross_metric_insights(
        123, sample_correlation_results, sample_metrics_data
    )
    
    # Check the function called the OpenAI API
    mock_chat_completions.assert_called_once()
    
    # Check the structure of the result
    assert "survey_id" in result
    assert "timestamp" in result
    assert "raw_insights" in result
    assert "structured_insights" in result
    
    # Skip checking cache was updated


@patch("data_pipeline.services.metadata_store.metadata_store")
@patch("data_pipeline.services.ai_insights.AsyncOpenAI")
@pytest.mark.asyncio
async def test_generate_survey_summary(mock_openai, mock_metadata_store, ai_insights_service,
                                      sample_survey_data, sample_all_analysis_results):
    """Test generating comprehensive survey summary."""
    # Mock the cache check
    mock_metadata_store.get_analysis_result.return_value = None
    
    # Mock OpenAI response
    mock_openai_instance = AsyncMock()
    mock_chat_completions = AsyncMock()
    mock_openai.return_value = mock_openai_instance
    mock_openai_instance.chat.completions.create = mock_chat_completions
    
    # Set up mock response
    mock_message = MagicMock()
    mock_message.content = """
    1. Executive Summary:
    The Customer Feedback Survey reveals high overall satisfaction with an average rating of 7.8/10. Customer loyalty shows a strong positive correlation with satisfaction scores, indicating satisfied customers become loyal customers. While 85% of customers report positive experiences, there remains an opportunity to address the concerns of the 15% who are less satisfied.
    
    2. Key Metrics Highlights:
    - Customer Satisfaction: Average rating of 7.8/10 with 85% of customers rating 6 or above.
    - Customer Loyalty: Increased 5% year-over-year and strongly correlates with satisfaction scores.
    - Number of Complaints: Negatively correlates with satisfaction, as expected.
    
    3. Correlation Insights:
    - Customer satisfaction strongly predicts customer loyalty (r=0.85), suggesting improvements in satisfaction will likely increase loyalty.
    - Higher satisfaction is associated with fewer customer complaints (r=-0.72).
    
    4. Strategic Recommendations:
    - Implement a targeted intervention program for the 15% of customers reporting low satisfaction.
    - Create a tiered loyalty program that rewards highly satisfied customers.
    - Conduct follow-up research with dissatisfied customers to identify specific pain points.
    - Use high satisfaction scenarios as case studies for employee training.
    - Develop an early warning system to identify satisfaction drops before they impact loyalty.
    
    5. Future Research:
    - Conduct customer segmentation analysis to identify if satisfaction drivers differ across customer groups.
    - Implement longitudinal tracking to better understand the satisfaction-loyalty relationship over time.
    - Explore additional metrics like Net Promoter Score to complement current measures.
    """
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_chat_completions.return_value = mock_response
    
    # Manually set the client to use our mock
    ai_insights_service.client = mock_openai_instance
    
    # Call the function
    result = await ai_insights_service.generate_survey_summary(
        123, sample_survey_data, sample_all_analysis_results
    )
    
    # Check the function called the OpenAI API
    mock_chat_completions.assert_called_once()
    
    # Check the structure of the result
    assert "survey_id" in result
    assert "survey_name" in result
    assert "timestamp" in result
    assert "raw_summary" in result
    assert "structured_summary" in result
    
    # Skip checking cache was updated


def test_parse_metric_insights(ai_insights_service):
    """Test parsing raw AI output into structured format."""
    # Sample AI output
    raw_insights = """
    1. Key Findings:
    - Average satisfaction is 7.8/10, indicating generally positive sentiment.
    - 85% of customers rate their satisfaction as 6 or higher.
    - Only 15% of responses are below 6, suggesting isolated cases of dissatisfaction.
    
    2. Detailed Interpretation:
    The distribution is left-skewed, with most responses in the high range. This indicates that most customers are satisfied with the service. The negative skew value of -0.75 confirms this left skew. The concentration of responses in the 8-10 range (44%) suggests a strong positive sentiment overall.
    
    3. Actionable Recommendations:
    - Focus improvement efforts on understanding and addressing the concerns of the 15% of dissatisfied customers.
    - Investigate what drives scores in the 8-10 range to replicate these positive experiences.
    - Implement a follow-up program for customers scoring below 6 to identify specific pain points.
    """
    
    structured_insights = ai_insights_service._parse_metric_insights(raw_insights)
    
    # Check the structure
    assert "key_findings" in structured_insights
    assert "interpretation" in structured_insights
    assert "recommendations" in structured_insights
    
    # Check the key findings
    assert len(structured_insights["key_findings"]) == 3
    assert "Average satisfaction is 7.8/10" in structured_insights["key_findings"][0]
    
    # Check the interpretation
    assert "The distribution is left-skewed" in structured_insights["interpretation"]
    
    # Check the recommendations
    assert len(structured_insights["recommendations"]) == 3
    assert "Focus improvement efforts" in structured_insights["recommendations"][0]


def test_parse_cross_metric_insights(ai_insights_service):
    """Test parsing raw cross-metric AI output into structured format."""
    # Sample AI output
    raw_insights = """
    1. Strongest Relationships:
    - Customer Satisfaction and Customer Loyalty have a strong positive correlation (r=0.85).
    - Customer Satisfaction and Number of Complaints show a strong negative correlation (r=-0.72).
    
    2. Potential Causality:
    - Customer Satisfaction appears to directly influence Customer Loyalty.
    - Improvements in satisfaction likely lead to fewer complaints.
    
    3. Unexpected Findings:
    - No unexpected relationships were found in this analysis.
    
    4. Strategic Implications:
    Improving customer satisfaction should be a top priority as it appears to directly drive loyalty and reduce complaints. This suggests that investments in customer experience improvements could yield returns through increased loyalty and reduced service costs.
    
    5. Research Recommendations:
    - Conduct segmentation analysis to determine if these relationships vary across different customer segments.
    - Implement A/B testing of satisfaction improvement initiatives to confirm the causal relationship with loyalty.
    """
    
    structured_insights = ai_insights_service._parse_cross_metric_insights(raw_insights)
    
    # Check the structure
    assert "strongest_relationships" in structured_insights
    assert "potential_causality" in structured_insights
    assert "unexpected_findings" in structured_insights
    assert "strategic_implications" in structured_insights
    assert "research_recommendations" in structured_insights
    
    # Check the strongest relationships
    assert len(structured_insights["strongest_relationships"]) == 2
    assert "Customer Satisfaction and Customer Loyalty" in structured_insights["strongest_relationships"][0]
    
    # Check the potential causality
    assert len(structured_insights["potential_causality"]) == 2
    assert "Customer Satisfaction appears to directly influence" in structured_insights["potential_causality"][0]
    
    # Check the strategic implications
    assert "Improving customer satisfaction should be a top priority" in structured_insights["strategic_implications"]
    
    # Check the research recommendations
    assert len(structured_insights["research_recommendations"]) == 2
    assert "Conduct segmentation analysis" in structured_insights["research_recommendations"][0]


def test_parse_survey_summary(ai_insights_service):
    """Test parsing raw survey summary AI output into structured format."""
    # Sample AI output
    raw_summary = """
    1. Executive Summary:
    The Customer Feedback Survey reveals high overall satisfaction with an average rating of 7.8/10. Customer loyalty shows a strong positive correlation with satisfaction scores. While 85% of customers report positive experiences, there remains an opportunity to address the 15% who are less satisfied.
    
    2. Key Metrics Highlights:
    - Customer Satisfaction: Average rating of 7.8/10 with 85% of customers rating 6 or above.
    - Customer Loyalty: Increased 5% year-over-year and strongly correlates with satisfaction scores.
    - Number of Complaints: Negatively correlates with satisfaction, as expected.
    
    3. Correlation Insights:
    - Customer satisfaction strongly predicts customer loyalty (r=0.85).
    - Higher satisfaction is associated with fewer customer complaints (r=-0.72).
    
    4. Strategic Recommendations:
    - Implement a targeted intervention program for customers reporting low satisfaction.
    - Create a loyalty program that rewards highly satisfied customers.
    - Conduct follow-up research with dissatisfied customers to identify specific pain points.
    - Use high satisfaction scenarios as case studies for employee training.
    
    5. Future Research:
    - Conduct customer segmentation analysis to identify if satisfaction drivers differ across customer groups.
    - Implement longitudinal tracking to better understand the satisfaction-loyalty relationship over time.
    """
    
    structured_summary = ai_insights_service._parse_survey_summary(raw_summary)
    
    # Check the structure
    assert "executive_summary" in structured_summary
    assert "key_metrics" in structured_summary
    assert "correlation_insights" in structured_summary
    assert "recommendations" in structured_summary
    assert "future_research" in structured_summary
    
    # Check the executive summary
    assert "The Customer Feedback Survey reveals high overall satisfaction" in structured_summary["executive_summary"]
    
    # Check the key metrics
    assert len(structured_summary["key_metrics"]) == 3
    assert "Customer Satisfaction: Average rating of 7.8/10" in structured_summary["key_metrics"][0]
    
    # Check the correlation insights
    assert len(structured_summary["correlation_insights"]) == 2
    assert "Customer satisfaction strongly predicts" in structured_summary["correlation_insights"][0]
    
    # Check the recommendations
    assert len(structured_summary["recommendations"]) == 4
    assert "Implement a targeted intervention program" in structured_summary["recommendations"][0]
    
    # Check the future research
    assert len(structured_summary["future_research"]) == 2
    assert "Conduct customer segmentation analysis" in structured_summary["future_research"][0] 