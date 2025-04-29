from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .routes import survey_router, survey_forms_router
from .models import Base
from .database import engine

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="rForms API",
    description="Goal-Oriented Adaptive Questionnaire API",
    version="0.1.0",
)

# Configure CORS
origins = ["http://localhost:3000"]  # Frontend URL
if os.getenv("ENVIRONMENT") == "production":
    # Add production origins
    origins.append("https://your-production-domain.com")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(survey_router)
app.include_router(survey_forms_router)

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "ok",
        "version": app.version,
        "service": "rForms API"
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint that returns a simple message."""
    return {
        "message": "Hello from rForms API!",
        "status": "success",
        "data": {
            "metrics": [
                {"id": 1, "name": "Food Quality", "type": "likert"},
                {"id": 2, "name": "Service Speed", "type": "likert"},
                {"id": 3, "name": "Staff Friendliness", "type": "likert"},
                {"id": 4, "name": "Value for Money", "type": "likert"},
                {"id": 5, "name": "Ambience", "type": "likert"}
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 