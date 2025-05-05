"""
Utilities for fetching survey data from API endpoints.
"""

import asyncio
import aiohttp
import json
import os
import logging
from typing import Dict, Any, Optional, Tuple
import argparse

logger = logging.getLogger(__name__)

class APIFetcher:
    """Fetch survey data from API endpoints."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the API fetcher.
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def fetch_survey(self, survey_id: int) -> Dict[str, Any]:
        """
        Fetch survey data from the API.
        
        Args:
            survey_id: ID of the survey
            
        Returns:
            Survey data
        """
        url = f"{self.base_url}/api/survey/forms/{survey_id}"
        logger.info(f"Fetching survey data from {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    logger.error(f"Error fetching survey data: {response.status}")
                    logger.error(await response.text())
                    return {}
                
                return await response.json()
    
    async def fetch_responses(self, survey_id: int, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch survey responses from the API.
        
        Args:
            survey_id: ID of the survey
            skip: Number of responses to skip
            limit: Maximum number of responses to fetch
            
        Returns:
            Survey responses
        """
        url = f"{self.base_url}/api/survey/forms/{survey_id}/responses?skip={skip}&limit={limit}"
        logger.info(f"Fetching responses from {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    logger.error(f"Error fetching responses: {response.status}")
                    logger.error(await response.text())
                    return {}
                
                return await response.json()
    
    async def fetch_all_responses(self, survey_id: int, batch_size: int = 100) -> Dict[str, Any]:
        """
        Fetch all responses for a survey, handling pagination.
        
        Args:
            survey_id: ID of the survey
            batch_size: Number of responses to fetch per request
            
        Returns:
            All survey responses
        """
        # Fetch first batch to get total count
        first_batch = await self.fetch_responses(survey_id, 0, batch_size)
        
        if not first_batch or "data" not in first_batch:
            logger.error("Failed to fetch first batch of responses")
            return {}
        
        total_responses = first_batch.get("data", {}).get("total", 0)
        logger.info(f"Found {total_responses} total responses")
        
        if total_responses <= batch_size:
            # No need for additional fetches
            return first_batch
        
        # Initialize combined result with first batch
        all_responses = first_batch
        all_responses_list = first_batch.get("data", {}).get("responses", [])
        
        # Fetch remaining batches
        remaining_batches = (total_responses - 1) // batch_size
        for batch in range(1, remaining_batches + 1):
            skip = batch * batch_size
            logger.info(f"Fetching batch {batch+1}/{remaining_batches+1} (skip={skip}, limit={batch_size})")
            
            batch_data = await self.fetch_responses(survey_id, skip, batch_size)
            if batch_data and "data" in batch_data:
                batch_responses = batch_data.get("data", {}).get("responses", [])
                all_responses_list.extend(batch_responses)
        
        # Update combined result with all responses
        all_responses["data"]["responses"] = all_responses_list
        all_responses["data"]["total"] = len(all_responses_list)
        
        return all_responses
    
    async def submit_insights(self, survey_id: int, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit analysis insights back to the API.
        
        Args:
            survey_id: ID of the survey
            insights: Analysis insights to submit
            
        Returns:
            API response
        """
        url = f"{self.base_url}/api/insights/submit/{survey_id}"
        logger.info(f"Submitting insights to {url}")
        
        # Prepare headers for JSON content
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=headers, 
                json={"survey_id": survey_id, "insights": insights}
            ) as response:
                if response.status not in [200, 201]:
                    logger.error(f"Error submitting insights: {response.status}")
                    logger.error(await response.text())
                    return {"error": f"Failed to submit insights: {response.status}"}
                
                return await response.json()
    
    async def fetch_and_save(
        self, 
        survey_id: int,
        output_dir: str, 
        fetch_all: bool = True
    ) -> Tuple[str, str]:
        """
        Fetch survey data and responses and save to files.
        
        Args:
            survey_id: ID of the survey
            output_dir: Directory to save files to
            fetch_all: Whether to fetch all responses (True) or just the first batch (False)
            
        Returns:
            Tuple of (survey_file_path, responses_file_path)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch survey data
        survey_data = await self.fetch_survey(survey_id)
        
        if not survey_data:
            logger.error(f"Failed to fetch survey data for survey {survey_id}")
            return "", ""
        
        # Fetch responses
        if fetch_all:
            responses_data = await self.fetch_all_responses(survey_id)
        else:
            responses_data = await self.fetch_responses(survey_id)
        
        if not responses_data:
            logger.error(f"Failed to fetch responses for survey {survey_id}")
            return "", ""
        
        # Save to files
        survey_file = os.path.join(output_dir, f"survey_{survey_id}.json")
        responses_file = os.path.join(output_dir, f"responses_{survey_id}.json")
        
        with open(survey_file, 'w') as f:
            json.dump(survey_data, f, indent=2)
        
        with open(responses_file, 'w') as f:
            json.dump(responses_data, f, indent=2)
        
        logger.info(f"Saved survey data to {survey_file}")
        logger.info(f"Saved {len(responses_data.get('data', {}).get('responses', []))} responses to {responses_file}")
        
        return survey_file, responses_file

async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(description='Fetch survey data from API.')
    parser.add_argument('--survey_id', required=True, type=int, help='ID of the survey to fetch')
    parser.add_argument('--base_url', required=True, help='Base URL of the API')
    parser.add_argument('--api_key', help='API key for authentication')
    parser.add_argument('--output_dir', default='./data', help='Directory to save files to')
    parser.add_argument('--all', action='store_true', help='Fetch all responses (may be slow for large surveys)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize API fetcher
    fetcher = APIFetcher(args.base_url, args.api_key)
    
    # Fetch and save data
    survey_file, responses_file = await fetcher.fetch_and_save(
        args.survey_id,
        args.output_dir,
        args.all
    )
    
    if survey_file and responses_file:
        logger.info(f"Successfully fetched and saved survey {args.survey_id}")
        logger.info(f"Survey data: {survey_file}")
        logger.info(f"Responses data: {responses_file}")
    else:
        logger.error(f"Failed to fetch survey {args.survey_id}")
        return 1
    
    return 0

def main():
    """Main entry point."""
    exit_code = asyncio.run(main_async())
    exit(exit_code)

if __name__ == "__main__":
    main() 