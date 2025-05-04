from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from ..database import get_db, get_mongo_db
from ..models import SurveyIdMapping, Goal
from data_pipeline.analysis.analysis_coordinator import analysis_coordinator
from data_pipeline.analysis.cross_metric_analysis import cross_metric_analysis_service
from data_pipeline.analysis.metric_analysis import metric_analysis_service
from data_pipeline.analysis.vector_trend_analysis import vector_trend_analysis_service
from data_pipeline.analysis.multimodal_ai_analysis import multimodal_ai_analysis_service
from data_pipeline.embeddings.multi_level_embedding_service import multi_level_embedding_service
from data_pipeline.embeddings.semantic_search import semantic_search_service
from data_pipeline.services.metadata_store import metadata_store

router = APIRouter(prefix="/api/insights", tags=["insights"])

@router.get("/survey/{survey_id}/base", response_model=Dict[str, Any])
async def get_base_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get base insights for a survey (immediate, real-time analysis).
    This endpoint retrieves the fastest tier of analysis results.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached results exist
    cached_result = await metadata_store.get_analysis_result("base_analysis", survey_id)
    if cached_result:
        return {
            "status": "success",
            "message": "Retrieved cached base insights",
            "data": cached_result,
            "source": "cache"
        }

    # Fetch survey data and responses
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")

    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)

    # Generate base insights (real-time)
    metrics_data = {str(m["id"]): m for m in survey_data.get("metrics", [])}
    insights = await analysis_coordinator._run_base_analysis(survey_id, {"metrics": metrics_data}, responses)

    return {
        "status": "success",
        "message": "Generated base insights",
        "data": insights,
        "source": "generated"
    }

@router.get("/survey/{survey_id}/full", response_model=Dict[str, Any])
async def get_full_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    force_refresh: bool = Query(False, description="Force refresh analysis"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get full insights for a survey (comprehensive analysis).
    This endpoint returns cached results or triggers a background analysis job.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached full analysis exists and we're not forcing refresh
    if not force_refresh:
        cached_result = await metadata_store.get_analysis_result("analysis_progress", survey_id)
        if cached_result and cached_result.get("status") == "completed":
            return {
                "status": "success",
                "message": "Retrieved cached full insights",
                "data": cached_result,
                "source": "cache"
            }

    # Fetch survey data and responses
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")

    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)

    # Check if analysis is in progress
    in_progress = False
    if not force_refresh:
        progress = await metadata_store.get_analysis_result("analysis_progress", survey_id)
        if progress and progress.get("status") == "in_progress":
            in_progress = True

    if force_refresh or not in_progress:
        # Convert metrics to the format expected by analysis_coordinator
        metrics_data = {str(m["id"]): m for m in survey_data.get("metrics", [])}
        
        # Run analysis in background
        background_tasks.add_task(
            analysis_coordinator.run_analysis_pipeline,
            survey_id=survey_id,
            survey_data={"metrics": metrics_data},
            responses=responses,
            use_celery=True
        )
        
        # Return base insights while comprehensive analysis runs in background
        base_insights = await analysis_coordinator._run_base_analysis(
            survey_id, {"metrics": metrics_data}, responses
        )
        
        return {
            "status": "success",
            "message": "Analysis started, returning base insights while comprehensive analysis runs",
            "data": {
                "base_analysis": base_insights,
                "status": "in_progress"
            },
            "source": "generated"
        }
    else:
        # Analysis already in progress, return current progress
        return {
            "status": "success",
            "message": "Analysis in progress, returning current progress",
            "data": progress,
            "source": "in_progress"
        }

@router.get("/survey/{survey_id}/metric/{metric_id}", response_model=Dict[str, Any])
async def get_metric_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get detailed insights for a specific metric in a survey.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached results exist
    cached_result = await metadata_store.get_analysis_result("metric_analysis", survey_id, metric_id)
    if cached_result:
        return {
            "status": "success",
            "message": "Retrieved cached metric insights",
            "data": cached_result,
            "source": "cache"
        }

    # Fetch survey data and responses
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")
    
    # Find the metric
    metric_data = None
    for m in survey_data.get("metrics", []):
        if str(m.get("id")) == metric_id:
            metric_data = m
            break
    
    if not metric_data:
        raise HTTPException(status_code=404, detail="Metric not found")

    # Fetch responses for this metric
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)
    
    # Extract responses for this metric
    metric_responses = []
    for response in responses:
        if "responses" in response and metric_id in response["responses"]:
            value = response["responses"][metric_id]
            if metric_data.get("type") == "numeric":
                metric_responses.append({"value": value})
            elif metric_data.get("type") in ["categorical", "single_choice"]:
                metric_responses.append({"category": value})
            elif metric_data.get("type") == "text":
                metric_responses.append({"text": value})
            else:
                metric_responses.append({"value": value})

    # Generate insights for this metric
    insights = await metric_analysis_service.analyze_metric(
        survey_id, metric_id, metric_data, metric_responses
    )

    return {
        "status": "success",
        "message": "Generated metric insights",
        "data": insights,
        "source": "generated"
    }

@router.get("/survey/{survey_id}/cross-metrics", response_model=Dict[str, Any])
async def get_cross_metric_insights(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get cross-metric correlation insights for a survey.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached results exist
    cached_result = await metadata_store.get_analysis_result("cross_metric_analysis", survey_id)
    if cached_result:
        return {
            "status": "success",
            "message": "Retrieved cached cross-metric insights",
            "data": cached_result,
            "source": "cache"
        }

    # Fetch survey data and responses
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")

    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)

    # Generate cross-metric insights
    metrics_data = {str(m["id"]): m for m in survey_data.get("metrics", [])}
    insights = await cross_metric_analysis_service.analyze_cross_metric_correlations(
        survey_id, metrics_data, responses
    )

    return {
        "status": "success",
        "message": "Generated cross-metric insights",
        "data": insights,
        "source": "generated"
    }

@router.get("/survey/{survey_id}/vector-analysis/{metric_id}", response_model=Dict[str, Any])
async def get_vector_analysis(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get vector-based trend analysis for a specific metric.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Check if cached results exist
    cache_key = f"vector_analysis_{metric_id}"
    cached_result = await metadata_store.get_analysis_result(cache_key, survey_id)
    if cached_result:
        return {
            "status": "success",
            "message": "Retrieved cached vector analysis",
            "data": cached_result,
            "source": "cache"
        }

    # Run vector trend analysis for clusters
    clusters = await vector_trend_analysis_service.detect_response_clusters(
        survey_id=survey_id,
        question_id=metric_id
    )
    
    # Run temporal trend analysis
    trends = await vector_trend_analysis_service.detect_temporal_trends(
        survey_id=survey_id,
        question_id=metric_id
    )
    
    # Run anomaly detection
    anomalies = await vector_trend_analysis_service.detect_anomalies(
        survey_id=survey_id,
        question_id=metric_id
    )
    
    # Combine results
    vector_analysis = {
        "survey_id": survey_id,
        "metric_id": metric_id,
        "timestamp": datetime.now().isoformat(),
        "response_clusters": clusters,
        "temporal_trends": trends,
        "anomaly_detection": anomalies
    }
    
    # Cache the results
    await metadata_store.store_analysis_result(cache_key, survey_id, vector_analysis)
    
    return {
        "status": "success",
        "message": "Generated vector-based analysis",
        "data": vector_analysis,
        "source": "generated"
    }

@router.get("/survey/{survey_id}/semantic-search", response_model=Dict[str, Any])
async def semantic_search(
    survey_id: int = Path(..., description="The ID of the survey"),
    query: str = Query(..., description="The search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    db: Session = Depends(get_db)
):
    """
    Perform semantic search across survey responses.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Perform semantic search
    search_results = await semantic_search_service.search_responses(
        survey_id=survey_id,
        query_text=query,
        limit=limit
    )
    
    return {
        "status": "success",
        "message": "Semantic search completed",
        "data": search_results
    }

@router.get("/survey/{survey_id}/embeddings/generate", response_model=Dict[str, Any])
async def generate_embeddings(
    survey_id: int = Path(..., description="The ID of the survey"),
    db: Session = Depends(get_db)
):
    """
    Generate multi-level embeddings for a survey.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Generate embeddings
    result = await multi_level_embedding_service.generate_aggregate_embeddings(survey_id)
    
    return {
        "status": "success",
        "message": "Embeddings generation triggered",
        "data": result
    }

@router.get("/survey/{survey_id}/ai-insights/text/{metric_id}", response_model=Dict[str, Any])
async def get_text_sentiment_analysis(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get AI-powered sentiment analysis for text responses.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")
    
    # Fetch survey data
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")
    
    # Find the metric/question
    question_text = None
    for m in survey_data.get("metrics", []):
        if str(m.get("id")) == metric_id:
            question_text = m.get("name", "Unknown Question")
            break
    
    if not question_text:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)
    
    # Extract text responses for this metric
    text_responses = []
    for response in responses:
        if "responses" in response and metric_id in response["responses"]:
            text = response["responses"][metric_id]
            if isinstance(text, str):
                text_responses.append({"text": text})
    
    # Get sentiment analysis
    sentiment_analysis = await multimodal_ai_analysis_service.analyze_text_sentiment(
        survey_id=survey_id,
        question_id=metric_id,
        question_text=question_text,
        responses=text_responses
    )
    
    return {
        "status": "success",
        "message": "Generated text sentiment analysis",
        "data": sentiment_analysis
    }

@router.get("/survey/{survey_id}/ai-insights/free-text/{metric_id}", response_model=Dict[str, Any])
async def get_free_text_analysis(
    survey_id: int = Path(..., description="The ID of the survey"),
    metric_id: str = Path(..., description="The ID of the metric"),
    guidance: Optional[str] = Query(None, description="Optional guidance for the analysis"),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Get AI-powered analysis of free-text responses with optional guidance.
    """
    # Validate survey exists
    mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Survey not found")
    
    # Fetch survey data
    mongo_id = mapping.mongo_id
    survey_data = await mongo_db.forms.find_one({"_id": mongo_id})
    if not survey_data:
        raise HTTPException(status_code=404, detail="Survey data not found")
    
    # Find the metric/question
    question_text = None
    for m in survey_data.get("metrics", []):
        if str(m.get("id")) == metric_id:
            question_text = m.get("name", "Unknown Question")
            break
    
    if not question_text:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    # Fetch responses
    cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
    responses = await cursor.to_list(length=1000)
    
    # Extract text responses for this metric
    text_responses = []
    for response in responses:
        if "responses" in response and metric_id in response["responses"]:
            text = response["responses"][metric_id]
            if isinstance(text, str):
                text_responses.append({"text": text})
    
    # Get free text analysis
    text_analysis = await multimodal_ai_analysis_service.analyze_free_responses(
        survey_id=survey_id,
        question_id=metric_id,
        question_text=question_text,
        responses=text_responses,
        guidance=guidance
    )
    
    return {
        "status": "success",
        "message": "Generated free-text analysis",
        "data": text_analysis
    } 