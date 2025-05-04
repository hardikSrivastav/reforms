"""
Middleware for automatically processing new survey responses.
This triggers embedding generation and analysis when new responses are submitted.
"""

from fastapi import Request
from typing import Callable
import asyncio
import json
from starlette.types import ASGIApp, Receive, Scope, Send

from data_pipeline.embeddings.multi_level_embedding_service import multi_level_embedding_service
from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
from data_pipeline.services.metadata_store import metadata_store
from ..database import get_db, get_mongo_db
from ..models import SurveyIdMapping

class ResponseProcessorMiddleware:
    """Middleware that detects new response submissions and triggers analysis."""
    
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Check if this is a response submission endpoint
        if scope["path"] == "/api/survey/responses/submit" and scope["method"] == "POST":
            # We need to capture the request body without consuming it
            original_receive = receive
            
            async def modified_receive():
                message = await original_receive()
                if message["type"] == "http.request":
                    # Store the body for later processing
                    if "body" in message:
                        body = message.get("body", b"")
                        if not message.get("more_body", False):
                            # Complete body received
                            asyncio.create_task(self._process_response(body))
                return message
                
            # Process the request normally but with our modified receive
            await self.app(scope, modified_receive, send)
        else:
            # For all other endpoints, just pass through
            await self.app(scope, receive, send)
    
    async def _process_response(self, body_bytes):
        """Process a new survey response in the background."""
        try:
            # Parse the request body
            body = json.loads(body_bytes)
            survey_id = body.get("survey_id")
            
            if not survey_id:
                return
                
            print(f"Processing new response for survey {survey_id}")
            
            # Get database sessions
            db_gen = get_db()
            db = next(db_gen)
            
            mongo_db_gen = get_mongo_db()
            mongo_db = await anext(mongo_db_gen)
            
            try:
                # Start embedding generation
                print(f"Generating embeddings for survey {survey_id}")
                asyncio.create_task(multi_level_embedding_service.generate_aggregate_embeddings(survey_id))
                
                # Check if analysis is already running
                progress = await metadata_store.get_analysis_result("analysis_progress", survey_id)
                
                # Only start a new analysis if one isn't already running
                if not progress or progress.get("status") not in ["in_progress", "queued"]:
                    print(f"Starting analysis for survey {survey_id}")
                    
                    # Get survey data
                    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
                    if not mapping:
                        print(f"No mapping found for survey {survey_id}")
                        return
                        
                    mongo_id = mapping.mongo_id
                    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
                    
                    if not survey_data:
                        print(f"No survey data found for ID {mongo_id}")
                        return
                        
                    # Get all responses
                    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
                    responses = await cursor.to_list(length=1000)
                    
                    # Prepare data for analysis
                    metrics_data = {str(m["id"]): m for m in survey_data.get("metrics", [])}
                    
                    # Run analysis pipeline
                    asyncio.create_task(
                        analysis_coordinator.run_analysis_pipeline(
                            survey_id=survey_id,
                            survey_data={"metrics": metrics_data},
                            responses=responses,
                            use_celery=True
                        )
                    )
                    print(f"Analysis started for survey {survey_id}")
                else:
                    print(f"Analysis already in progress for survey {survey_id}")
                    
            finally:
                # Clean up database sessions
                db.close()
                
        except Exception as e:
            print(f"Error processing response: {str(e)}")

def add_response_processor(app):
    """Add the response processor middleware to the FastAPI app."""
    app.add_middleware(ResponseProcessorMiddleware) 