from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
import random  # For generating sample data
import json
import hashlib

from ..database import get_db, get_mongo_db
from ..models import Goal, Metric, SurveyIdMapping, Form, SurveyInsightsCache
from ..services.openai_service import openai_service

router = APIRouter(prefix="/api/survey/insights", tags=["survey-insights"])

@router.get("/{survey_id}", response_model=Dict[str, Any])
async def get_survey_insights(
    survey_id: int,
    force_regenerate: bool = False,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo_db)
):
    """
    Generate AI-powered insights from survey responses
    
    This endpoint analyzes survey responses and returns recommended visualizations
    and insights based on the survey metrics and questions.
    
    Parameters:
    - survey_id: The ID of the survey to analyze
    - force_regenerate: If true, bypasses cache and forces regeneration of insights
    """
    try:
        # Find the ID mapping to get MongoDB ID
        id_mapping = db.query(SurveyIdMapping).filter(SurveyIdMapping.sql_id == survey_id).first()
        
        if not id_mapping:
            raise HTTPException(status_code=404, detail="Survey not found in ID mapping")
        
        # Get MongoDB ID
        mongo_id = id_mapping.mongo_id
        
        # Fetch survey from MongoDB
        survey_data = await mongo_db.forms.find_one({"_id": ObjectId(mongo_id)})
        
        if not survey_data:
            raise HTTPException(status_code=404, detail="Survey not found in MongoDB")
        
        # Fetch responses
        cursor = mongo_db.responses.find({"survey_mongo_id": mongo_id})
        
        # Convert to list and process ObjectId
        responses = []
        async for response in cursor:
            # Convert ObjectId to string for JSON serialization
            response["_id"] = str(response["_id"])
            if "survey_mongo_id" in response:
                response["survey_mongo_id"] = str(response["survey_mongo_id"])
            responses.append(response)
        
        # Calculate a hash of the response data to check if insights need to be regenerated
        # Only consider the actual response values, not metadata like IDs
        responses_data = []
        for response in responses:
            if "responses" in response:
                responses_data.append(response["responses"])
        
        # Create a hash of response data to detect changes
        response_hash = hashlib.md5(json.dumps(responses_data, sort_keys=True).encode()).hexdigest()
        response_count = len(responses)
        
        # Get the cache entry regardless of force_regenerate setting
        cache_entry = db.query(SurveyInsightsCache).filter(
            SurveyInsightsCache.survey_id == survey_id
        ).first()
        
        # Check if we have a valid cache, but only if force_regenerate is False
        if not force_regenerate:
            # If we have a cache entry with matching hash and response count, use it
            if cache_entry and cache_entry.response_hash == response_hash and cache_entry.response_count == response_count:
                print(f"Using cached insights for survey {survey_id}")
                return {
                    "status": "success",
                    "message": "Survey insights retrieved from cache",
                    "data": json.loads(cache_entry.insights_data),
                    "cached": True
                }
        else:
            print(f"Force regenerating insights for survey {survey_id}")
        
        # Get metrics from database
        metrics = db.query(Metric).filter(Metric.goal_id == survey_id).all()
        metrics_data = [{"id": m.id, "name": m.name, "type": m.type, "description": m.description} for m in metrics]
        
        # Create basic visualization suggestions based on question types
        visualizations = []
        response_data_by_question = {}
        
        # Process and organize response data by question for AI analysis
        for question in survey_data.get("questions", []):
            question_id = question.get("id")
            question_type = question.get("type")
            question_text = question.get("question", "Untitled Question")
            
            if not question_id or not question_type:
                continue
            
            # Collect actual response data for this question
            question_responses = []
            for response in responses:
                value = response.get("responses", {}).get(question_id)
                if value is not None:
                    question_responses.append(value)
            
            # Store data for AI analysis
            response_data_by_question[question_id] = {
                "question": question_text,
                "type": question_type,
                "responses": question_responses
            }
            
            # Create a visualization based on question type
            sample_data = generate_sample_data(question_type, responses, question_id)
            
            # Find the most relevant metric for this question using better matching
            matching_metric_id = None
            highest_match_score = 0
            
            for metric in metrics:
                match_score = 0
                
                # Check if metric name appears in the question text (case-insensitive)
                if metric.name.lower() in question_text.lower():
                    match_score += 3
                
                # Check for keyword matches between metric description and question text
                if metric.description and any(keyword in question_text.lower() 
                                           for keyword in metric.description.lower().split()):
                    match_score += 1
                
                # Check for metric type alignment with question type
                if metric.type == "likert" and question_type in ["radio", "scale"]:
                    match_score += 2
                elif metric.type == "multiple_choice" and question_type in ["select", "checkbox"]:
                    match_score += 2
                elif metric.type == "rating" and question_type in ["number", "rating"]:
                    match_score += 2
                elif metric.type == "text" and question_type in ["text", "textarea"]:
                    match_score += 2
                
                # Update the matching metric if this one has a higher score
                if match_score > highest_match_score:
                    highest_match_score = match_score
                    matching_metric_id = metric.id
            
            # Set default visualization type based on question type
            if question_type in ["radio", "select"]:
                viz_type = "bar"  # Bar charts are usually better for categorical single-select
                description = f"Distribution of responses for {question_type} question"
            elif question_type == "checkbox":
                viz_type = "pie"  # Pie charts work well for showing distribution of multi-select
                description = f"Distribution of responses for {question_type} question"
            elif question_type in ["number", "rating", "scale"]:
                # For numeric data, determine best chart type based on data
                if len(responses) > 5:  # If we have enough data points
                    viz_type = "line"   # Line chart for trending
                    description = f"Trend analysis for {question_type} responses"
                else:
                    viz_type = "bar"    # Bar chart for smaller datasets
                    description = f"Response distribution for {question_type} question"
            else:  # text/textarea
                viz_type = "text_analysis"
                description = "Word frequency and sentiment analysis"
            
            # Create the visualization object
            visualization = {
                "type": viz_type,
                "title": question_text,
                "description": description,
                "question_id": question_id,
                "metric_id": matching_metric_id,
                "data_processing": "categorical" if question_type in ["radio", "select", "checkbox"] else
                                  "numeric" if question_type in ["number", "rating", "scale"] else "text",
                "sample_data": sample_data
            }
            
            # Add to visualizations list
            visualizations.append(visualization)
        
        # Generate AI insights using OpenAI if we have responses
        ai_insights = []
        ai_recommendations = []
        
        if responses:
            # Use the OpenAI service to analyze responses
            try:
                # Prepare survey data for analysis
                analysis_survey_data = {
                    "title": survey_data.get("title", "Untitled Survey"),
                    "description": survey_data.get("description", ""),
                    "questions": survey_data.get("questions", [])
                }
                
                # Get AI analysis of the survey responses
                analysis_result = openai_service.analyze_survey_responses(analysis_survey_data, responses)
                
                # Extract insights and recommendations
                ai_insights = analysis_result.get("insights", [])
                ai_recommendations = analysis_result.get("recommendations", [])
                
                # Use AI-suggested visualizations if available
                ai_visualizations = analysis_result.get("visualizations", [])
                if ai_visualizations:
                    # Create a mapping of question_id to data format to check compatibility
                    data_format_map = {}
                    for viz in visualizations:
                        data_format_map[viz["question_id"]] = viz["data_processing"]
                    
                    # Update existing visualizations with AI suggestions
                    for ai_viz in ai_visualizations:
                        question_id = ai_viz.get("question_id")
                        chart_type = ai_viz.get("chart_type")
                        rationale = ai_viz.get("rationale", "")
                        
                        if not question_id or not chart_type:
                            continue
                        
                        # Find matching visualization to update
                        for viz in visualizations:
                            if viz["question_id"] == question_id:
                                # Ensure chart type is compatible with data format
                                data_format = viz["data_processing"]
                                
                                # Check compatibility and apply the suggested chart type if appropriate
                                if data_format == "categorical" and chart_type in ["bar", "pie"]:
                                    viz["type"] = chart_type
                                elif data_format == "numeric" and chart_type in ["line", "bar"]:
                                    viz["type"] = chart_type
                                
                                # Update description with AI rationale if provided
                                if rationale:
                                    viz["description"] = rationale
                                
                                break
            except Exception as e:
                print(f"Error using OpenAI service for analysis: {str(e)}")
                # Fall back to simple insight generation if OpenAI analysis fails
                ai_insights = generate_ai_insights({"response_data": response_data_by_question, "response_count": len(responses)})
        else:
            # Default insights if no responses
            ai_insights = [
                "No responses have been collected yet.",
                "Share your survey link to start gathering responses.",
                "Once responses are collected, AI-powered insights will be generated."
            ]
        
        # Convert ObjectId to string for JSON serialization
        survey_data["_id"] = str(survey_data["_id"])
        
        # Calculate some basic stats
        response_count = len(responses)
        completion_rates = {}
        
        for question in survey_data.get("questions", []):
            question_id = question.get("id")
            if question_id:
                answered = sum(1 for r in responses if r.get("responses", {}).get(question_id) is not None)
                completion_rates[question_id] = (answered / response_count) * 100 if response_count > 0 else 0
        
        # Create the insights data
        insights_data = {
            "survey": {
                "id": survey_id,
                "title": survey_data.get("title", "Untitled Survey"),
                "description": survey_data.get("description", ""),
                "response_count": response_count,
            },
            "metrics": metrics_data,
            "visualizations": visualizations,
            "stats": {
                "completion_rates": completion_rates,
                "average_completion_time": None,  # Would be calculated in a full implementation
                "response_trend": None,  # Would show responses over time in a full implementation
            },
            "ai_insights": ai_insights,
            "ai_recommendations": ai_recommendations
        }
        
        # Store the insights in cache
        if cache_entry:
            # Update existing cache entry
            cache_entry.response_hash = response_hash
            cache_entry.response_count = response_count
            cache_entry.insights_data = json.dumps(insights_data)
            cache_entry.updated_at = datetime.utcnow()
        else:
            # Create new cache entry
            new_cache = SurveyInsightsCache(
                survey_id=survey_id,
                response_hash=response_hash,
                response_count=response_count,
                insights_data=json.dumps(insights_data)
            )
            db.add(new_cache)
        
        # Commit changes to database
        db.commit()
        
        return {
            "status": "success",
            "message": "Survey insights generated successfully",
            "data": insights_data,
            "cached": False
        }
    except Exception as e:
        if not isinstance(e, HTTPException):
            print(f"Error generating insights: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")
        raise e

def generate_ai_insights(analysis_data: Dict[str, Any]) -> List[str]:
    """
    Generate AI-powered insights from survey response data using OpenAI
    
    Args:
        analysis_data: Dictionary containing survey data, questions, responses, and metrics
    
    Returns:
        List of insight statements about the survey results
    """
    # Make sure we have response data to analyze
    if not analysis_data.get("response_data") or analysis_data.get("response_count", 0) == 0:
        return []
    
    # Create a prompt for OpenAI that will analyze the survey data
    prompt = f"""
    You are an expert data analyst specializing in survey analysis. Analyze the following survey data and generate valuable insights.
    
    Survey Title: {analysis_data.get("survey", {}).get("title", "Untitled Survey")}
    Survey Description: {analysis_data.get("survey", {}).get("description", "")}
    Total Responses: {analysis_data.get("response_count", 0)}
    
    SURVEY DATA:
    {json.dumps(analysis_data.get("response_data", {}), indent=2)}
    
    Based on this data, generate 3-5 insightful observations focusing on:
    1. Key patterns or trends in the responses
    2. Surprising or unexpected findings
    3. Strategic recommendations based on the data
    4. Specific actions that could improve metrics
    
    Format your response as a JSON array of strings, with each string being a complete insight statement.
    Keep each insight concise (20-30 words) and actionable.
    """
    
    try:
        # Call OpenAI to analyze the data
        response = openai_service.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst expert in survey analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Extract insights array
        insights = data.get("insights", [])
        if not insights and isinstance(data, list):
            insights = data
            
        # Ensure we have at least a few insights
        if len(insights) < 3:
            insights.extend([
                "Consider adding more quantitative questions to measure specific metrics.",
                "Analyze the correlation between different questions to identify hidden patterns.",
                "Increasing your response rate could provide more statistical confidence in these results."
            ][:3-len(insights)])
            
        return insights
        
    except Exception as e:
        print(f"Error generating AI insights: {str(e)}")
        # Return fallback insights if OpenAI call fails
        return [
            "The data suggests you should consider refining some questions for clarity.",
            "Adding follow-up questions may help uncover deeper insights.",
            "Text responses show common themes that could benefit from dedicated questions."
        ]

def generate_sample_data(question_type: str, responses: List[Dict], question_id: str) -> List[Dict]:
    """
    Generate sample data for visualization based on question type.
    In a real implementation, this would use actual response data.
    """
    if not responses:
        # If no responses, generate placeholder data
        if question_type in ["radio", "select"]:
            options = ["Option A", "Option B", "Option C", "Option D"]
            return [{"label": option, "value": random.randint(1, 10)} for option in options]
        elif question_type == "checkbox":
            options = ["Choice 1", "Choice 2", "Choice 3", "Choice 4"]
            return [{"label": option, "value": random.randint(1, 10)} for option in options]
        elif question_type in ["number", "rating", "scale"]:
            # Generate time series data for line chart
            return [
                {"date": "Jan 1", "value": random.randint(1, 5)},
                {"date": "Jan 2", "value": random.randint(1, 5)},
                {"date": "Jan 3", "value": random.randint(1, 5)},
                {"date": "Jan 4", "value": random.randint(1, 5)},
                {"date": "Jan 5", "value": random.randint(1, 5)}
            ]
        else:
            return ["Sample text response 1", "Sample text response 2", "Sample text response 3"]
    
    # Analyze actual responses
    try:
        if question_type in ["radio", "select"]:
            # Count occurrences of each option
            option_counts = {}
            for response in responses:
                value = response.get("responses", {}).get(question_id)
                if value:
                    option_counts[value] = option_counts.get(value, 0) + 1
            return [{"label": str(option), "value": count} for option, count in option_counts.items()]
        
        elif question_type == "checkbox":
            # For checkbox, count each option independently
            option_counts = {}
            for response in responses:
                values = response.get("responses", {}).get(question_id, [])
                if isinstance(values, list):
                    for value in values:
                        if value:
                            option_counts[value] = option_counts.get(value, 0) + 1
            return [{"label": str(option), "value": count} for option, count in option_counts.items()]
        
        elif question_type in ["number", "rating", "scale"]:
            # Create time series data
            result = []
            for response in responses:
                value = response.get("responses", {}).get(question_id)
                if value is not None:
                    try:
                        date = datetime.fromisoformat(response.get("submitted_at")).strftime("%b %d")
                    except:
                        date = "Unknown"
                    result.append({"date": date, "value": float(value)})
            return result[:20]  # Limit to 20 data points
        
        else:  # text responses
            text_responses = []
            for response in responses:
                value = response.get("responses", {}).get(question_id)
                if value:
                    text_responses.append(str(value))
            return text_responses[:10]  # Limit to 10 text responses
    
    except Exception as e:
        print(f"Error generating sample data: {str(e)}")
        # Fallback to random data
        return generate_sample_data(question_type, [], question_id) 