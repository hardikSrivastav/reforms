"""
Main module for processing survey data and generating analysis results.
"""

import asyncio
import argparse
import json
import logging
from typing import Dict, List, Any, Optional
import os
import sys

from data_pipeline.utils.data_transformers import data_transformer
from data_pipeline.analysis.metric_analysis import metric_analysis_service
from data_pipeline.analysis.cross_metric_analysis import cross_metric_analysis_service
from data_pipeline.services.ai_insights import ai_insights_service
from data_pipeline.services.metadata_store import metadata_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def process_survey(survey_data: Dict[str, Any], responses_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process survey data and responses to generate analysis results.
    
    Args:
        survey_data: Raw survey data from API
        responses_data: Raw response data from API
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting survey analysis process")
    
    # Transform survey data
    transformed_survey = data_transformer.transform_survey_data(survey_data["data"])
    survey_id = transformed_survey["survey_id"]
    
    logger.info(f"Processing survey ID: {survey_id}")
    logger.info(f"Found {len(transformed_survey['metrics'])} metrics and {len(transformed_survey['questions'])} questions")
    
    # Transform responses data
    transformed_responses = data_transformer.transform_responses(responses_data, transformed_survey)
    logger.info(f"Processing {transformed_responses['total_responses']} responses")
    
    # Initialize results
    results = {
        "survey_id": survey_id,
        "survey_info": {
            "name": transformed_survey["name"],
            "description": transformed_survey["description"],
            "metrics_count": len(transformed_survey["metrics"]),
            "total_responses": transformed_responses["total_responses"]
        },
        "metric_analysis": {},
        "cross_metric_analysis": None,
        "ai_insights": {}
    }
    
    # Process each metric individually
    metrics = transformed_survey["metrics"]
    logger.info(f"Analyzing {len(metrics)} individual metrics")
    
    for metric_id, metric_data in metrics.items():
        logger.info(f"Processing metric: {metric_id}")
        
        # Prepare data for metric analysis
        metric_analysis_data = data_transformer.prepare_for_metric_analysis(
            transformed_survey, transformed_responses, metric_id
        )
        
        # Run metric analysis
        metric_result = await metric_analysis_service.analyze_metric(
            survey_id=metric_analysis_data["survey_id"],
            metric_id=metric_analysis_data["metric_id"],
            metric_data=metric_analysis_data["metric_data"],
            responses=metric_analysis_data["responses"]
        )
        
        # Store metric analysis result
        results["metric_analysis"][metric_id] = metric_result
        
        # Generate AI insights for the metric
        if metric_result and "statistical_analysis" in metric_result:
            ai_insights_data = data_transformer.prepare_for_ai_insights(
                transformed_survey, 
                transformed_responses, 
                metric_id,
                metric_result["statistical_analysis"]
            )
            
            # Run AI insights generation
            ai_insight = await ai_insights_service.generate_metric_insights(
                survey_id=ai_insights_data["survey_id"],
                metric_id=ai_insights_data["metric_id"],
                metric_data=ai_insights_data["metric_data"],
                analysis_results=ai_insights_data["analysis_results"],
                response_stats=ai_insights_data["response_stats"]
            )
            
            # Store AI insights result
            results["ai_insights"][metric_id] = ai_insight
    
    # Run cross-metric analysis if there are at least two metrics
    if len(metrics) >= 2:
        logger.info("Running cross-metric analysis")
        cross_analysis_data = data_transformer.prepare_for_cross_metric_analysis(
            transformed_survey, transformed_responses
        )
        
        # Run cross-metric analysis
        cross_result = await cross_metric_analysis_service.analyze_cross_metric_correlations(
            survey_id=cross_analysis_data["survey_id"],
            metrics_data=cross_analysis_data["metrics_data"],
            survey_responses=cross_analysis_data["survey_responses"]
        )
        
        # Store cross-metric analysis result
        results["cross_metric_analysis"] = cross_result
        
        # Generate AI insights for cross-metric analysis
        if cross_result:
            # Run AI insights for cross-metric analysis
            cross_insights = await ai_insights_service.generate_cross_metric_insights(
                survey_id=survey_id,
                correlation_results=cross_result,
                metrics_data=metrics
            )
            
            # Store cross-metric insights
            results["ai_insights"]["cross_metric"] = cross_insights
    
    # Generate overall survey summary
    logger.info("Generating survey summary")
    survey_summary = await ai_insights_service.generate_survey_summary(
        survey_id=survey_id,
        survey_data={
            "name": transformed_survey["name"],
            "description": transformed_survey["description"],
            "metrics": metrics,
            "total_respondents": transformed_responses["total_responses"]
        },
        all_analysis_results=results
    )
    
    # Add to results
    results["survey_summary"] = survey_summary
    
    logger.info("Analysis complete")
    return results

async def process_from_files(survey_file: str, responses_file: str, output_file: Optional[str] = None):
    """
    Process survey data from files.
    
    Args:
        survey_file: Path to survey data JSON file
        responses_file: Path to responses data JSON file
        output_file: Optional path to write results to
    """
    logger.info(f"Loading survey data from {survey_file}")
    with open(survey_file, 'r') as f:
        survey_data = json.load(f)
    
    logger.info(f"Loading responses data from {responses_file}")
    with open(responses_file, 'r') as f:
        responses_data = json.load(f)
    
    # Process the data
    results = await process_survey(survey_data, responses_data)
    
    # Write results to file if output_file is specified
    if output_file:
        logger.info(f"Writing results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process survey data and generate analysis results.')
    parser.add_argument('--survey', required=True, help='Path to survey data JSON file')
    parser.add_argument('--responses', required=True, help='Path to responses data JSON file')
    parser.add_argument('--output', help='Path to write results to (default: analysis_results.json)')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    output_file = args.output or 'analysis_results.json'
    
    # Run the async process
    asyncio.run(process_from_files(args.survey, args.responses, output_file))
    logger.info(f"Results written to {output_file}")

if __name__ == "__main__":
    main() 