from fastapi import APIRouter, Depends, HTTPException, Request, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
import logging
from ..services.openai_service import openai_service

from ..database import get_db, get_mongo_db
from ..models import Form, Goal, SurveyIdMapping, Metric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/survey/mongo", tags=["survey_forms"])

# MongoDB collection names
FORMS_COLLECTION = "forms"
FORMS_RESPONSES_COLLECTION = "form_responses" 

@router.get("/surveys/{survey_id}", response_model=Dict[str, Any])
async def get_survey(
    survey_id: str,
    request: Request,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """Get a specific survey form"""
    logger.info(f"Attempting to get survey with ID: {survey_id}")
    
    # Get auth context from wrapper
    auth_context = request.scope.get("auth_context", {})
    logger.info(f"Auth context: {auth_context}")
    
    # First try to find by numeric ID
    try:
        numeric_id = int(survey_id)
        logger.info(f"Looking up mapping for numeric ID: {numeric_id}")
        # Look up MongoDB ID from mapping
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == numeric_id).first()
        if mapping:
            logger.info(f"Found mapping: SQL ID {mapping.sql_id} -> MongoDB ID {mapping.mongo_id}")
            form = await mongo_db[FORMS_COLLECTION].find_one({"_id": ObjectId(mapping.mongo_id)})
            if form:
                logger.info(f"Successfully found form using mapping, {form}")
        else:
            logger.info("No mapping found, trying direct lookup")
            # If no mapping exists, try direct lookup
            form = await mongo_db[FORMS_COLLECTION].find_one({"id": numeric_id})
            if form:
                logger.info(f"Found form using direct lookup with numeric ID, {form}")
    except ValueError:
        logger.info("ID is not numeric, trying other formats")
        form = None
    
    # If not found by numeric ID, try UUID
    if not form:
        logger.info(f"Trying UUID lookup for: {survey_id}")
        form = await mongo_db[FORMS_COLLECTION].find_one({"id": survey_id})
        if form:
            logger.info(f"Found form using UUID, {form}")
    
    # If not found by UUID, try ObjectId
    if not form:
        try:
            logger.info(f"Trying ObjectId lookup for: {survey_id}")
            object_id = ObjectId(survey_id)
            form = await mongo_db[FORMS_COLLECTION].find_one({"_id": object_id})
            if form:
                logger.info("Found form using ObjectId")
        except:
            logger.info("Invalid ObjectId format")
            pass
    
    # If still not found, try access_key
    if not form:
        logger.info(f"Trying access_key lookup for: {survey_id}")
        form = await mongo_db[FORMS_COLLECTION].find_one({"access_key": survey_id})
        if form:
            logger.info("Found form using access_key")
    
    if not form:
        logger.error(f"Survey not found for ID: {survey_id}")
        raise HTTPException(
            status_code=404, 
            detail="Survey not found. Please check the survey ID and try again."
        )
    
    # Check access permissions
    has_access = False
    
    # Case 1: Authenticated user owns the form
    if auth_context.get("authenticated") and form.get("owner_id") == str(auth_context["user"]["_id"]):
        logger.info("Access granted: Authenticated user owns the form")
        has_access = True
    
    # Case 2: Non-authenticated user with matching session_id
    elif not auth_context.get("authenticated") and form.get("session_id") == auth_context.get("session_id"):
        logger.info("Access granted: Matching session_id")
        has_access = True
    
    # Case 3: Public access enabled for this form
    elif form.get("is_public", False):
        logger.info("Access granted: Form is public")
        has_access = True
    
    if not has_access:
        logger.warning(f"Access denied for survey {survey_id}")
        raise HTTPException(status_code=403, detail="You don't have access to this survey")
    
    # Convert ObjectId to string for JSON serialization
    if "_id" in form:
        form["_id"] = str(form["_id"])
    
    logger.info(f"Successfully retrieved survey: {form.get('title', 'Untitled')}")
    return {
        "status": "success",
        "message": "Survey retrieved successfully",
        "data": form
    }

@router.post("/forms", response_model=Dict[str, Any])
async def save_form(
    request: Request,
    form_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """Save a survey form"""
    try:
        # Get auth context from wrapper
        auth_context = request.scope.get("auth_context", {})
        
        # Create the form data
        form_data["created_at"] = datetime.utcnow()
        form_data["updated_at"] = datetime.utcnow()
        form_data["responses_count"] = 0
        
        # Add ownership data based on auth_context
        if auth_context.get("authenticated"):
            form_data["owner_id"] = str(auth_context["user"]["_id"])
        else:
            form_data["session_id"] = auth_context.get("session_id")
        
        # If we have a surveyId, fetch the metrics from the SQL database
        if "surveyId" in form_data:
            try:
                # Get the goal and its metrics from SQL
                goal = db.query(Goal).filter(Goal.id == form_data["surveyId"]).first()
                if goal:
                    metrics = db.query(Metric).filter(Metric.goal_id == goal.id).all()
                    # Convert metrics to dictionary format
                    form_data["metrics"] = [
                        {
                            "id": metric.id,
                            "name": metric.name,
                            "type": metric.type,
                            "description": metric.description,
                            "weight": metric.weight,
                            "options": metric.options
                        }
                        for metric in metrics
                    ]
            except Exception as e:
                logger.error(f"Error fetching metrics: {str(e)}")
                # Continue without metrics if there's an error
        
        # Insert into MongoDB
        result = await mongo_db[FORMS_COLLECTION].insert_one(form_data)
        mongo_id = str(result.inserted_id)
        
        # Create or update ID mapping if we have a SQL ID
        if "surveyId" in form_data:
            # Check if mapping already exists
            existing_mapping = db.query(SurveyIdMapping).filter(
                SurveyIdMapping.sql_id == form_data["surveyId"]
            ).first()
            
            if existing_mapping:
                # Update existing mapping
                existing_mapping.mongo_id = mongo_id
                existing_mapping.updated_at = datetime.utcnow()
            else:
                # Create new mapping
                mapping = SurveyIdMapping(
                    sql_id=form_data["surveyId"],
                    mongo_id=mongo_id
                )
                db.add(mapping)
            
            db.commit()
        
        return {
            "status": "success",
            "message": "Form saved successfully",
            "id": mongo_id
        }
        
    except Exception as e:
        logger.error(f"Error saving form: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving form: {str(e)}")

@router.get("/forms/{form_id}", response_model=Dict[str, Any])
async def get_form(
    form_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get a specific survey form"""
    # Get auth context from wrapper
    auth_context = request.scope.get("auth_context", {})
    
    # Get the form
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    
    # Check access permissions
    has_access = False
    
    # Case 1: Authenticated user owns the form
    if auth_context.get("authenticated") and form.owner_id == str(auth_context["user"]["_id"]):
        has_access = True
    
    # Case 2: Non-authenticated user with matching session_id
    elif not auth_context.get("authenticated") and form.session_id == auth_context.get("session_id"):
        has_access = True
    
    # Case 3: Public access enabled for this form
    elif form.is_public:
        has_access = True
    
    if not has_access:
        raise HTTPException(status_code=403, detail="You don't have access to this form")
    
    return {
        "status": "success",
        "message": "Form retrieved successfully",
        "data": form
    }

@router.post("/forms/{form_id}/submit", response_model=Dict[str, Any])
async def submit_form_response(
    form_id: int,
    request: Request,
    response_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Submit a response to a form"""
    # Get auth context from wrapper
    auth_context = request.scope.get("auth_context", {})
    
    # Check if form exists
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    
    # Create response document
    form_response = {
        "form_id": form_id,
        "responses": response_data.get("responses", {}),
        "submitted_at": datetime.utcnow(),
        "ip_address": request.client.host if request.client else None
    }
    
    # Add respondent info based on auth_context
    if auth_context.get("authenticated"):
        form_response["respondent_id"] = str(auth_context["user"]["_id"])
    else:
        form_response["session_id"] = auth_context.get("session_id")
    
    # Update response count in form
    form.responses_count += 1
    db.commit()
    
    return {
        "status": "success",
        "message": "Response submitted successfully"
    }

@router.get("/forms/{form_id}/responses", response_model=Dict[str, Any])
async def get_form_responses(
    form_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get all responses for a form"""
    # Get current user from wrapper
    current_user = request.scope.get("current_user")
    
    # Check if form exists and user has access
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    
    # Check if user owns the form
    if form.owner_id != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="You don't have access to this form's responses")
    
    # TODO: Implement response storage and retrieval
    # For now, return empty list
    return {
        "status": "success",
        "message": "Responses retrieved successfully",
        "data": []
    }

@router.get("/test", response_model=Dict[str, Any])
async def test_mongo_connection(
    mongo_db = Depends(get_mongo_db)
):
    """Test route to check MongoDB connection and data"""
    try:
        # Check MongoDB connection
        await mongo_db.command('ping')
        
        # Get all collections
        collections = await mongo_db.list_collection_names()
        
        # Initialize results dictionary
        results = {
            "connection": "ok",
            "collections": collections,
            "data": {}
        }
        
        # Get data from each collection
        for collection in collections:
            documents = await mongo_db[collection].find({}).to_list(length=1000)
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                    
            results["data"][collection] = documents
        
        return {
            "status": "success",
            "message": "MongoDB connection successful",
            "data": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"MongoDB connection failed: {str(e)}",
            "data": None
        }

@router.post("/forms/{form_id}/generate-questions", response_model=Dict[str, Any])
async def generate_survey_questions(
    form_id: str,
    request: Request,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """Generate survey questions using AI based on metrics"""
    try:
        # First try to find the form by numeric ID
        try:
            numeric_id = int(form_id)
            # Look up MongoDB ID from mapping
            mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == numeric_id).first()
            if mapping:
                form = await mongo_db[FORMS_COLLECTION].find_one({"_id": ObjectId(mapping.mongo_id)})
            else:
                # If no mapping exists, try direct lookup
                form = await mongo_db[FORMS_COLLECTION].find_one({"id": numeric_id})
        except ValueError:
            # If not a numeric ID, try as ObjectId
            try:
                form = await mongo_db[FORMS_COLLECTION].find_one({"_id": ObjectId(form_id)})
            except:
                # If not a valid ObjectId, try as a regular ID
                form = await mongo_db[FORMS_COLLECTION].find_one({"id": form_id})
        
        if not form:
            raise HTTPException(status_code=404, detail="Form not found")
        
        # Get metrics from the form
        metrics = form.get("metrics", [])
        if not metrics:
            raise HTTPException(status_code=400, detail="No metrics found in form")
        
        # Generate questions using OpenAI
        questions = []
        for metric in metrics:
            # Create a prompt for the metric
            prompt = f"""
            Based on the following metric, generate a survey question that effectively measures it:
            
            Metric Name: {metric['name']}
            Metric Type: {metric['type']}
            Description: {metric.get('description', 'No description provided')}
            
            Generate a clear, concise question that:
            1. Directly relates to the metric
            2. Is easy to understand
            3. Will yield meaningful responses
            4. Is appropriate for the metric type
            
            Return the question in JSON format with the following fields:
            - question: The actual question text
            - type: The question type (text, number, select, radio, etc.)
            - options: Array of options if type is select/radio
            - required: Boolean indicating if the question is required
            """
            
            # Call OpenAI service
            response = openai_service.generate_question(prompt)
            
            # Add the generated question
            questions.append({
                "id": str(ObjectId()),
                "metric_id": metric.get("id"),
                "question": response["question"],
                "type": response["type"],
                "options": response.get("options", []),
                "required": response.get("required", True),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
        
        # Update the form with generated questions
        update_query = {"_id": ObjectId(form["_id"])} if "_id" in form else {"id": form["id"]}
        await mongo_db[FORMS_COLLECTION].update_one(
            update_query,
            {
                "$set": {
                    "questions": questions,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return {
            "status": "success",
            "message": "Questions generated successfully",
            "data": {
                "questions": questions
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")