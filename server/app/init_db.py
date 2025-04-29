"""
Database initialization script to set up tables and add test data
Run this script once to create the database and seed it with test data
"""

import os
import sys
from sqlalchemy.orm import Session
from datetime import datetime
from bson import ObjectId

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, SessionLocal, connect_to_mongo_sync
from app.models import Base, Goal, Metric, Form

def init_db():
    """Initialize database tables and add test data"""
    # Create SQL tables
    Base.metadata.create_all(bind=engine)
    
    # Connect to MongoDB
    mongo_db = connect_to_mongo_sync()
    
    # Create a new session
    db = SessionLocal()
    
    try:
        # Check if we already have data
        existing_goals = db.query(Goal).count()
        if existing_goals > 0:
            print("Database already contains goals, skipping initialization")
            return
            
        # Add test goals and metrics
        test_goals = [
            {
                "description": "Measure customer satisfaction with our breakfast menu",
                "metrics": [
                    {"name": "Food Quality", "type": "likert", "description": "The taste, freshness, and appearance of the food"},
                    {"name": "Portion Size", "type": "likert", "description": "The amount of food served"},
                    {"name": "Value for Money", "type": "rating", "description": "Whether customers feel the price is fair for what they received"},
                    {"name": "Service Speed", "type": "likert", "description": "How quickly the food was served"},
                    {"name": "Menu Variety", "type": "likert", "description": "Range of options available on the breakfast menu"}
                ]
            },
            {
                "description": "Understand employee engagement in the marketing department",
                "metrics": [
                    {"name": "Job Satisfaction", "type": "likert", "description": "Overall satisfaction with current role"},
                    {"name": "Work-Life Balance", "type": "likert", "description": "Ability to balance work and personal life"},
                    {"name": "Career Growth", "type": "likert", "description": "Opportunities for advancement and skill development"},
                    {"name": "Team Collaboration", "type": "likert", "description": "Effectiveness of teamwork within the department"},
                    {"name": "Recognition", "type": "likert", "description": "Feeling appreciated for contributions"}
                ]
            }
        ]
        
        # Insert test data into SQL
        for goal_data in test_goals:
            goal = Goal(description=goal_data["description"])
            db.add(goal)
            db.flush()  # Flush to get the goal ID
            
            for metric_data in goal_data["metrics"]:
                metric = Metric(
                    name=metric_data["name"],
                    type=metric_data["type"],
                    description=metric_data["description"],
                    goal_id=goal.id
                )
                db.add(metric)
            
            # Create a test form for each goal
            form = Form(
                title=f"Survey: {goal.description[:30]}...",
                description=goal.description,
                survey_id=goal.id,
                is_public=True,
                responses_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(form)
        
        db.commit()
        
        # Create test survey in MongoDB
        test_survey = {
            "id": 4,
            "title": "Test Survey",
            "description": "This is a test survey",
            "metrics": [
                {
                    "id": 1,
                    "name": "Test Metric",
                    "type": "likert",
                    "description": "Test metric description"
                }
            ],
            "is_public": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Delete any existing survey with ID 4
        mongo_db.surveys.delete_one({"id": 4})
        
        # Insert the new survey
        result = mongo_db.surveys.insert_one(test_survey)
        print(f"Created test survey with ID: {result.inserted_id}")
        
        # Verify the survey was created
        survey = mongo_db.surveys.find_one({"id": 4})
        if survey:
            print("Successfully created test survey in MongoDB")
        else:
            print("Failed to create test survey in MongoDB")
        
        print("Database initialized with test data")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Done.") 