from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId

from ..database import get_db, get_mongo_db
from ..models import Goal, Metric, SurveyIdMapping
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
