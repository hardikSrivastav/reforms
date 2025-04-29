from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Goal(Base):
    __tablename__ = "goals"
    
    id = Column(Integer, primary_key=True, index=True)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    unique_id = Column(String, default=lambda: str(uuid.uuid4()), unique=True)
    
    # Relationship to metrics
    metrics = relationship("Metric", back_populates="goal", cascade="all, delete-orphan")

    # Relationship to forms
    forms = relationship("Form", back_populates="survey", cascade="all, delete-orphan")

    # Relationship to survey id mapping
    id_mapping = relationship("SurveyIdMapping", back_populates="goal", uselist=False, cascade="all, delete-orphan")

class Metric(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, default="likert")  # likert, text, multiple_choice, etc.
    description = Column(Text, nullable=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    
    # Relationship to goal
    goal = relationship("Goal", back_populates="metrics") 

class Form(Base):
    __tablename__ = "forms"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    survey_id = Column(Integer, ForeignKey("goals.id"))
    is_public = Column(Boolean, default=False)
    responses_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    
    # Relationship to survey
    survey = relationship("Goal", back_populates="forms") 

class SurveyIdMapping(Base):
    __tablename__ = "survey_id_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    sql_id = Column(Integer, ForeignKey("goals.id"), unique=True)
    mongo_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to goal
    goal = relationship("Goal", back_populates="id_mapping") 