from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Goal schemas
class GoalBase(BaseModel):
    description: str

class GoalCreate(GoalBase):
    pass

class MetricBase(BaseModel):
    name: str
    type: str = "likert"
    description: Optional[str] = None

class MetricCreate(MetricBase):
    pass

class Metric(MetricBase):
    id: int
    goal_id: int
    
    class Config:
        from_attributes = True

class Goal(GoalBase):
    id: int
    created_at: datetime
    updated_at: datetime
    unique_id: str
    metrics: List[Metric] = []
    
    class Config:
        from_attributes = True

# Response schemas
class GoalResponse(BaseModel):
    status: str
    message: str
    data: Goal

class MetricsGenerateRequest(BaseModel):
    goal_description: str

class MetricsGenerateResponse(BaseModel):
    status: str
    message: str
    data: List[MetricBase]

class FormBase(BaseModel):
    title: str
    description: Optional[str] = None
    survey_id: int
    is_public: bool = False

class Form(FormBase):
    id: int
    responses_count: int
    created_at: datetime
    updated_at: datetime
    owner_id: Optional[str] = None
    session_id: Optional[str] = None
    
    class Config:
        from_attributes = True

class FormResponse(BaseModel):
    form_id: int
    responses: Dict[str, Any]
    submitted_at: datetime
    ip_address: Optional[str] = None
    respondent_id: Optional[str] = None
    session_id: Optional[str] = None 