from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Determine environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Configure database URL based on environment
if ENVIRONMENT == "production":
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
else:
    # Use a writable location for SQLite in development
    DB_DIR = os.getenv("DB_DIR", "/data")
    os.makedirs(DB_DIR, exist_ok=True)
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_DIR}/dev.db"

# Create engine and session
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://mongodb:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
mongo_db = client.survey_db

# Synchronous MongoDB connection for initialization
def connect_to_mongo_sync():
    """Create a synchronous MongoDB connection for initialization"""
    sync_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    return sync_client.survey_db

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get MongoDB connection
async def get_mongo_db():
    try:
        yield mongo_db
    finally:
        pass  # Connection is managed by the client 