from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId

from ..database import get_db, get_mongo_db
from ..models import Goal, Metric, SurveyIdMapping, Form
from ..schemas import GoalCreate, Goal as GoalSchema, GoalResponse, MetricsGenerateRequest, MetricsGenerateResponse, MetricCreate
from ..services.openai_service import openai_service

router = APIRouter(prefix="/api/survey", tags=["survey"])

@router.post("/goals", response_model=GoalResponse)
async def create_goal(goal: GoalCreate, db: Session = Depends(get_db)):
    """Create a new survey goal"""
    # Create the goal
    db_goal = Goal(description=goal.description)
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)
    
    # Generate metrics using OpenAI
    metrics = openai_service.generate_metrics(goal.description)
    
    # Create metrics in database
    for metric_data in metrics:
        db_metric = Metric(
            name=metric_data["name"],
            type=metric_data["type"],
            description=metric_data.get("description"),
            goal_id=db_goal.id
        )
        db.add(db_metric)
    
    db.commit()
    db.refresh(db_goal)
    
    return {
        "status": "success",
        "message": "Goal created successfully",
        "data": db_goal
    }

@router.get("/goals/{goal_id}", response_model=GoalResponse)
async def get_goal(goal_id: int, db: Session = Depends(get_db)):
    """Get a specific goal with its metrics"""
    db_goal = db.query(Goal).filter(Goal.id == goal_id).first()
    if not db_goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    return {
        "status": "success",
        "message": "Goal retrieved successfully",
        "data": db_goal
    }

@router.get("/goals", response_model=List[GoalSchema])
async def get_goals(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all goals"""
    db_goals = db.query(Goal).offset(skip).limit(limit).all()
    return db_goals

@router.post("/metrics/generate", response_model=MetricsGenerateResponse)
async def generate_metrics(request: MetricsGenerateRequest):
    """Generate metrics for a goal using AI without saving to database"""
    metrics = openai_service.generate_metrics(request.goal_description)
    
    return {
        "status": "success",
        "message": "Metrics generated successfully",
        "data": metrics
    }

@router.put("/goals/{goal_id}/metrics", response_model=Dict[str, Any])
async def update_goal_metrics(
    goal_id: int,
    metrics_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """Update metrics for a specific goal"""
    # Check if goal exists
    goal = db.query(Goal).filter(Goal.id == goal_id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    try:
        # Delete existing metrics for this goal in SQL
        db.query(Metric).filter(Metric.goal_id == goal_id).delete()
        
        # Add new metrics to SQL
        for metric_data in metrics_data.get("metrics", []):
            metric = Metric(
                name=metric_data["name"],
                type=metric_data["type"],
                description=metric_data.get("description"),
                goal_id=goal_id
            )
            db.add(metric)
        
        # Update goal's updated_at timestamp
        goal.updated_at = datetime.utcnow()
        
        # Save changes to SQL
        db.commit()
        
        # Check if MongoDB mapping exists
        mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == goal_id).first()
        
        if not mapping:
            # Create a new MongoDB document
            mongo_doc = {
                "id": goal_id,
                "title": goal.description,
                "description": goal.description,
                "metrics": metrics_data.get("metrics", []),
                "is_public": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Insert into MongoDB
            result = await mongo_db.forms.insert_one(mongo_doc)
            mongo_id = str(result.inserted_id)
            
            # Create mapping
            mapping = SurveyIdMapping(
                sql_id=goal_id,
                mongo_id=mongo_id
            )
            db.add(mapping)

            new_form = Form(
                title=goal.description,
                description=goal.description,
                survey_id=goal_id,
                is_public=True,
                responses_count=0
            )
            db.add(new_form)
            
            db.commit()
        else:
            # Update existing MongoDB document
            mongo_update = {
                "$set": {
                    "metrics": metrics_data.get("metrics", []),
                    "updated_at": datetime.utcnow()
                }
            }
            
            result = await mongo_db.forms.update_one(
                {"_id": ObjectId(mapping.mongo_id)},
                mongo_update
            )
            
            if result.modified_count == 0:
                raise HTTPException(status_code=500, detail="Failed to update MongoDB document")
        
        return {
            "status": "success",
            "message": "Metrics updated successfully",
            "data": {
                "goal_id": goal_id,
                "metrics_count": len(metrics_data.get("metrics", [])),
                "mongo_id": mapping.mongo_id if mapping else None
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating metrics: {str(e)}") 