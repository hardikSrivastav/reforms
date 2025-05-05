"""
Metadata storage service for caching analysis results.
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class MetadataStore:
    """Service for storing and retrieving analysis metadata and results."""
    
    def __init__(self, cache_dir: str = 'cache'):
        """
        Initialize the metadata store.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create subdirectories for different types of data
        self.analysis_dir = os.path.join(cache_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    async def store_analysis_result(
        self,
        analysis_type: str, 
        survey_id: int,
        result: Dict[str, Any],
        metric_id: Optional[str] = None
    ) -> bool:
        """
        Store analysis result in the cache.
        
        Args:
            analysis_type: Type of analysis (e.g., 'metric_analysis', 'cross_metric')
            survey_id: The survey ID
            result: The analysis result to store
            metric_id: Optional metric ID for metric-specific analysis
            
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Ensure timestamp is included
            if "timestamp" not in result:
                result["timestamp"] = datetime.now().isoformat()
            
            # Create the cache key
            cache_key = f"{analysis_type}_{survey_id}"
            if metric_id:
                cache_key += f"_{metric_id}"
            
            # Create the file path
            file_path = os.path.join(self.analysis_dir, f"{cache_key}.json")
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Stored analysis result for {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            return False
    
    async def get_analysis_result(
        self,
        analysis_type: str, 
        survey_id: int,
        metric_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get analysis result from the cache.
        
        Args:
            analysis_type: Type of analysis (e.g., 'metric_analysis', 'cross_metric')
            survey_id: The survey ID
            metric_id: Optional metric ID for metric-specific analysis
            
        Returns:
            The cached result or None if not found or expired
        """
        try:
            # Create the cache key
            cache_key = f"{analysis_type}_{survey_id}"
            if metric_id:
                cache_key += f"_{metric_id}"
            
            # Create the file path
            file_path = os.path.join(self.analysis_dir, f"{cache_key}.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                return None
            
            # Read from file
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Check if result is still valid (has timestamp)
            if "timestamp" not in result:
                return None
            
            logger.info(f"Retrieved analysis result for {cache_key}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving analysis result: {str(e)}")
            return None
    
    async def list_analysis_results(
        self,
        analysis_type: Optional[str] = None, 
        survey_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List available analysis results.
        
        Args:
            analysis_type: Optional type filter
            survey_id: Optional survey ID filter
            
        Returns:
            List of matching analysis result metadata
        """
        results = []
        
        try:
            for filename in os.listdir(self.analysis_dir):
                if not filename.endswith('.json'):
                    continue
                
                # Parse the cache key from filename
                cache_key = filename.replace('.json', '')
                key_parts = cache_key.split('_')
                
                if len(key_parts) < 2:
                    continue
                
                file_analysis_type = key_parts[0]
                file_survey_id = int(key_parts[1])
                
                # Apply filters
                if analysis_type and file_analysis_type != analysis_type:
                    continue
                
                if survey_id and file_survey_id != survey_id:
                    continue
                
                # Get basic metadata
                try:
                    with open(os.path.join(self.analysis_dir, filename), 'r') as f:
                        metadata = json.load(f)
                        
                    results.append({
                        "analysis_type": file_analysis_type,
                        "survey_id": file_survey_id,
                        "timestamp": metadata.get("timestamp", "unknown"),
                        "cache_key": cache_key
                    })
                except Exception as e:
                    logger.warning(f"Error reading metadata from {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing analysis results: {str(e)}")
        
        return results
    
    async def delete_analysis_result(
        self,
        analysis_type: str, 
        survey_id: int,
        metric_id: Optional[str] = None
    ) -> bool:
        """
        Delete an analysis result from the cache.
        
        Args:
            analysis_type: Type of analysis
            survey_id: The survey ID
            metric_id: Optional metric ID
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            # Create the cache key
            cache_key = f"{analysis_type}_{survey_id}"
            if metric_id:
                cache_key += f"_{metric_id}"
            
            # Create the file path
            file_path = os.path.join(self.analysis_dir, f"{cache_key}.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                return False
            
            # Delete the file
            os.remove(file_path)
            
            logger.info(f"Deleted analysis result for {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting analysis result: {str(e)}")
            return False


# Singleton instance
metadata_store = MetadataStore() 