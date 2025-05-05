"""
Utility modules for the data pipeline.
"""

# Import and expose the API fetcher and data transformer
from data_pipeline.utils.api_fetcher import APIFetcher
from data_pipeline.utils.data_transformers import data_transformer
from data_pipeline.config import settings

# Create singleton instance of API fetcher with settings
api_fetcher = APIFetcher(base_url=settings.API_BASE_URL, api_key=settings.API_KEY) 