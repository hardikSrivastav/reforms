"""
Main entry point for processing survey data.
This module provides functions to process survey data from MongoDB,
generate embeddings, and store them in the vector database.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from bson import ObjectId
import motor.motor_asyncio
import json

from .config import settings
from .services.qdrant_client import qdrant_service
from .services.metadata_store import metadata_store
from .embeddings.embedding_service import embedding_service
from .analysis.base_analysis import base_analysis_service
from .analysis.metric_analysis import metric_analysis_service
from .analysis.cross_metric_analysis import cross_metric_analysis_service
from .analysis.predictive_analysis import predictive_analysis_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def load_survey_data(mongo_client, survey_id: str) -> Dict[str, Any]:
    """
    Load survey data from MongoDB.
    
    Args:
        mongo_client: MongoDB client
        survey_id: Survey ID
        
    Returns:
        Survey data
    """
    try:
        db = mongo_client.survey_db
        
        # For testing purposes, allow both string IDs and ObjectIds
        if isinstance(survey_id, str):
            # Try to convert to ObjectId if it's a valid format
            try:
                object_id = ObjectId(survey_id)
                survey_data = await db.forms.find_one({"_id": object_id})
            except Exception:
                # If conversion fails, try direct string lookup for testing
                survey_data = await db.forms.find_one({"_id": survey_id})
        else:
            survey_data = await db.forms.find_one({"_id": survey_id})
            
        return survey_data
    except Exception as e:
        logger.error(f"Error loading survey data: {str(e)}")
        return None

async def load_survey_responses(mongo_client, survey_id: str) -> List[Dict[str, Any]]:
    """
    Load survey responses from MongoDB.
    
    Args:
        mongo_client: MongoDB client
        survey_id: Survey ID
        
    Returns:
        List of survey responses
    """
    try:
        db = mongo_client.survey_db
        cursor = db.responses.find({"survey_mongo_id": survey_id})
        responses = []
        async for response in cursor:
            # Convert ObjectId to string for JSON serialization
            response["_id"] = str(response["_id"])
            if "survey_mongo_id" in response:
                response["survey_mongo_id"] = str(response["survey_mongo_id"])
            responses.append(response)
        return responses
    except Exception as e:
        logger.error(f"Error loading survey responses: {str(e)}")
        return []

async def process_survey(survey_id: int, mongo_url: str = None):
    """
    Process a survey by generating embeddings for questions, responses, and metrics,
    and performing multi-tiered analysis.
    
    Args:
        survey_id: ID of the survey to process
        mongo_url: MongoDB connection URL
    """
    try:
        # Initialize MongoDB connection
        mongo_url = mongo_url or settings.MONGODB_URL
        client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        db = client.survey_db
        
        logger.info(f"Processing survey {survey_id}")
        
        # Get survey ID mapping
        from server.app.models import SurveyIdMapping
        from server.app.database import SessionLocal
        
        # Get MongoDB ID from SQL database
        db_session = SessionLocal()
        id_mapping = db_session.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        
        if not id_mapping:
            logger.error(f"Survey {survey_id} not found in ID mapping")
            db_session.close()
            return False
        
        mongo_id = id_mapping.mongo_id
        db_session.close()
        
        # Get survey data from MongoDB
        survey_data = await db.forms.find_one({"_id": ObjectId(mongo_id)})
        
        if not survey_data:
            logger.error(f"Survey not found in MongoDB with ID {mongo_id}")
            return False
        
        # Initialize Qdrant collections
        await qdrant_service.initialize_collections()
        
        # Process survey questions
        questions = survey_data.get("questions", [])
        await embedding_service.process_questions(survey_id, questions)
        
        # Get metrics from database
        from server.app.models import Metric
        
        db_session = SessionLocal()
        metrics = db_session.query(Metric).filter(Metric.goal_id == survey_id).all()
        metrics_data = [{"id": m.id, "name": m.name, "type": m.type, "description": m.description} for m in metrics]
        db_session.close()
        
        # Process metrics
        await embedding_service.process_metrics(survey_id, metrics_data)
        
        # Get responses
        cursor = db.responses.find({"survey_mongo_id": mongo_id})
        responses = []
        async for response in cursor:
            # Convert ObjectId to string for JSON serialization
            response["_id"] = str(response["_id"])
            if "survey_mongo_id" in response:
                response["survey_mongo_id"] = str(response["survey_mongo_id"])
            responses.append(response)
        
        # Process responses
        await embedding_service.process_survey_responses(survey_id, responses)
        
        # Perform base analysis (real-time)
        base_analysis = await base_analysis_service.analyze_survey(survey_id, survey_data, responses)
        logger.info(f"Completed base analysis for survey {survey_id}")
        
        # Perform metric-specific analysis (near real-time)
        for metric in metrics_data:
            metric_analysis = await metric_analysis_service.analyze_metric(
                survey_id, 
                metric["id"], 
                metric, 
                responses
            )
            logger.info(f"Completed metric analysis for survey {survey_id}, metric {metric['id']}")
        
        # Perform cross-metric analysis (background)
        asyncio.create_task(cross_metric_analysis_service.analyze_cross_metrics(
            survey_id,
            metrics_data,
            responses
        ))
        logger.info(f"Started cross-metric analysis for survey {survey_id}")
        
        # Perform predictive analysis (scheduled)
        asyncio.create_task(predictive_analysis_service.generate_predictions(
            survey_id,
            metrics_data,
            responses
        ))
        logger.info(f"Started predictive analysis for survey {survey_id}")
        
        logger.info(f"Successfully processed survey {survey_id}")
        return True
    except Exception as e:
        logger.error(f"Error processing survey {survey_id}: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m data_pipeline.process_survey <survey_id>")
        sys.exit(1)
    
    survey_id = int(sys.argv[1])
    asyncio.run(process_survey(survey_id)) 