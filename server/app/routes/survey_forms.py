from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId

from ..database import get_db, get_mongo_db
from ..models import Goal, Metric, SurveyIdMapping, Form
from ..services.openai_service import openai_service

router = APIRouter(prefix="/api/survey", tags=["survey-forms"])

@router.get("/forms/{survey_id}", response_model=Dict[str, Any])
async def get_survey_form(survey_id: int, db: Session = Depends(get_db), mongo_db = Depends(get_mongo_db)):
    """
    Fetch survey form data using ID mapping between SQL and MongoDB
    """
    # Find the ID mapping to get MongoDB ID
    id_mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    
    if not id_mapping:
        raise HTTPException(status_code=404, detail="Survey not found in ID mapping")
    
    # Get MongoDB ID
    mongo_id = id_mapping.mongo_id
    
    # Fetch survey from MongoDB using the mapped ID
    survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
    
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey not found in MongoDB")
    
    # Convert ObjectId to string for JSON serialization
    survey_data["_id"] = str(survey_data["_id"])
    
    return {
        "status": "success",
        "message": "Survey retrieved successfully",
        "data": survey_data
    }

@router.put("/forms/update/{survey_id}", response_model=Dict[str, Any])
async def update_survey_form(
    survey_id: int, 
    data: Dict[str, Any] = Body(...), 
    db: Session = Depends(get_db), 
    mongo_db = Depends(get_mongo_db)
):
    """
    Update survey form data in MongoDB
    """
    survey_data = data.get("survey")
    
    if not survey_data:
        raise HTTPException(status_code=400, detail="No survey data provided")
    
    # Find the ID mapping to get MongoDB ID
    id_mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    
    if not id_mapping:
        raise HTTPException(status_code=404, detail="Survey not found in ID mapping")
    
    # Get MongoDB ID
    mongo_id = id_mapping.mongo_id
    
    # Prepare data for MongoDB
    mongo_data = {
        "title": survey_data.get("title"),
        "description": survey_data.get("description"),
        "metrics": survey_data.get("metrics", []),
        "questions": survey_data.get("questions", []),
        "updated_at": datetime.utcnow()
    }
    
    # Update MongoDB document
    result = await mongo_db.forms.update_one(
        {"_id": ObjectId(mongo_id)},
        {"$set": mongo_data}
    )
    
    if result.modified_count == 0:
        # Check if document exists
        document = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
        
        if not document:
            # If document doesn't exist, create it
            mongo_data["_id"] = ObjectId(mongo_id)
            mongo_data["created_at"] = datetime.utcnow()
            await mongo_db.forms.insert_one(mongo_data)
    
    return {
        "status": "success",
        "message": "Survey updated successfully"
    }

@router.post("/forms/generate-questions/{survey_id}", response_model=Dict[str, Any])
async def generate_questions(
    survey_id: int, 
    data: Dict[str, Any] = Body(...), 
    db: Session = Depends(get_db)
):
    """
    Generate questions for a survey using OpenAI
    """
    metrics = data.get("metrics", [])
    
    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics provided for question generation")
    
    # Find the ID mapping to ensure survey exists
    id_mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    
    if not id_mapping:
        raise HTTPException(status_code=404, detail="Survey not found in ID mapping")
    
    # Generate questions using OpenAI service
    questions = openai_service.generate_questions(metrics)
    
    # Add unique ID to each question
    for question in questions:
        question["id"] = str(ObjectId())
    
    return {
        "status": "success",
        "message": "Questions generated successfully",
        "questions": questions
    }

@router.post("/responses/submit", response_model=Dict[str, Any])
async def submit_response(
    request: Request,
    data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Submit a response to a survey form
    
    The response data will be stored in MongoDB, and the response count 
    will be incremented in the SQL database.
    """
    print("Received request to submit response")
    survey_id = data.get("survey_id")
    responses = data.get("responses")
    
    print(f"Processing submission for survey_id: {survey_id}")
    
    if not survey_id:
        print("Error: No survey ID provided")
        raise HTTPException(status_code=400, detail="Survey ID is required")
    
    if not responses or not isinstance(responses, dict):
        print(f"Error: Invalid response data for survey_id: {survey_id}")
        raise HTTPException(status_code=400, detail="Valid response data is required")
    
    try:
        print(f"Looking up ID mapping for survey_id: {survey_id}")
        # Find the ID mapping to get MongoDB ID
        id_mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        
        if not id_mapping:
            print(f"Survey not found for survey_id: {survey_id}")
            raise HTTPException(status_code=404, detail="Survey not found")
        
        # Get MongoDB ID
        mongo_id = id_mapping.mongo_id
        print(f"Found MongoDB ID: {mongo_id} for survey_id: {survey_id}")
        
        # Prepare response data for MongoDB
        response_data = {
            "survey_id": survey_id,
            "survey_mongo_id": mongo_id,
            "responses": responses,
            "submitted_at": datetime.utcnow(),
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "session_id": None  # Could be populated from auth or session middleware
        }
        print(f"Prepared response data for MongoDB insertion for survey_id: {survey_id}")
        
        # Insert the response into MongoDB
        print(f"Inserting response into MongoDB for survey_id: {survey_id}")
        result = await mongo_db.responses.insert_one(response_data)
        print(f"Response inserted with ID: {result.inserted_id}")
        
        # Update the response count in SQL
        print(f"Updating response count in SQL for survey_id: {survey_id}")
        form = db.query(Form).filter(Form.survey_id == survey_id).first()
        if form:
            form.responses_count += 1
            db.commit()
            print(f"Response count updated for survey_id: {survey_id}")
        else:
            print(f"Warning: Form not found in SQL database for survey_id: {survey_id}")
        
        print(f"Successfully processed response for survey_id: {survey_id}")
        return {
            "status": "success",
            "message": "Response submitted successfully",
            "response_id": str(result.inserted_id)
        }
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error submitting response for survey_id {survey_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting response: {str(e)}")

@router.get("/forms/{survey_id}/responses", response_model=Dict[str, Any])
async def get_survey_responses(
    survey_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Retrieve all responses for a specific survey form
    
    Returns a paginated list of responses for the survey.
    """
    print(f"Received request to get responses for survey_id: {survey_id}")
    print(f"Pagination parameters: skip={skip}, limit={limit}")
    
    # Find the ID mapping to get MongoDB ID
    print(f"Looking up ID mapping for survey_id: {survey_id}")
    id_mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    
    if not id_mapping:
        print(f"Survey not found for survey_id: {survey_id}")
        raise HTTPException(status_code=404, detail="Survey not found")
    
    # Get MongoDB ID
    mongo_id = id_mapping.mongo_id
    print(f"Found MongoDB ID: {mongo_id} for survey_id: {survey_id}")
    
    try:
        # Count total responses
        print(f"Counting total responses for survey_id: {survey_id}")
        total_responses = await mongo_db.responses.count_documents({"survey_mongo_id": mongo_id})
        print(f"Total responses found: {total_responses}")
        
        # Fetch responses with pagination
        print(f"Fetching responses with pagination for survey_id: {survey_id}")
        cursor = mongo_db.responses.find(
            {"survey_mongo_id": mongo_id}
        ).sort("submitted_at", -1).skip(skip).limit(limit)
        
        # Convert to list and process ObjectId
        print(f"Processing response data for survey_id: {survey_id}")
        responses = []
        async for response in cursor:
            # Convert ObjectId to string for JSON serialization
            response["_id"] = str(response["_id"])
            if "survey_mongo_id" in response:
                response["survey_mongo_id"] = str(response["survey_mongo_id"])
            responses.append(response)
        
        print(f"Successfully retrieved {len(responses)} responses for survey_id: {survey_id}")
        return {
            "status": "success",
            "message": "Responses retrieved successfully",
            "data": {
                "total": total_responses,
                "skip": skip,
                "limit": limit,
                "responses": responses
            }
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error retrieving responses for survey_id {survey_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving responses: {str(e)}")

@router.get("/responses/{response_id}", response_model=Dict[str, Any])
async def get_response(
    response_id: str,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Retrieve a single survey response by its ID
    """
    try:
        # Validate ObjectId format
        try:
            response_obj_id = ObjectId(response_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid response ID format")
        
        # Fetch the response from MongoDB
        response = await mongo_db.responses.find_one({"_id": response_obj_id})
        
        if not response:
            raise HTTPException(status_code=404, detail="Response not found")
        
        # Convert ObjectId to string for JSON serialization
        response["_id"] = str(response["_id"])
        if "survey_mongo_id" in response:
            response["survey_mongo_id"] = str(response["survey_mongo_id"])
        
        # Get survey form data for context
        survey_form = None
        if "survey_id" in response:
            id_mapping = db.query(SurveyIdMapping).filter(
                SurveyIdMapping.sql_id == response["survey_id"]
            ).first()
            
            if id_mapping:
                survey_data = await mongo_db.forms.find_one({"_id": ObjectId(id_mapping.mongo_id)})
                if survey_data:
                    survey_data["_id"] = str(survey_data["_id"])
                    survey_form = {
                        "title": survey_data.get("title", "Untitled Survey"),
                        "questions": survey_data.get("questions", [])
                    }
        
        return {
            "status": "success",
            "message": "Response retrieved successfully",
            "data": {
                "response": response,
                "survey_form": survey_form
            }
        }
    except Exception as e:
        if not isinstance(e, HTTPException):
            # Log the error for debugging
            print(f"Error retrieving response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving response: {str(e)}")
        raise e
