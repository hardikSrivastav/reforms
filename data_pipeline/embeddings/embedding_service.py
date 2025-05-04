"""
Embedding service for generating vector embeddings.
This service provides methods to generate embeddings for survey data
including questions, responses, and metrics.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np
import asyncio
from openai import AsyncOpenAI

from ..config import settings
from ..services.qdrant_client import qdrant_service, PointStruct

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = settings.EMBEDDING_MODEL
        logger.info(f"Initialized embedding service with model: {self.model}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zeros vector if embedding fails
            return [0.0] * settings.EMBEDDING_DIMENSION
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            # Process in batches to avoid rate limits
            batch_size = settings.DEFAULT_BATCH_SIZE
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Sleep to avoid rate limits if not the last batch
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            # Return zeros vectors if embedding fails
            return [[0.0] * settings.EMBEDDING_DIMENSION for _ in range(len(texts))]
    
    async def process_survey_responses(self, survey_id: int, responses: List[Dict[str, Any]]):
        """
        Process and store embeddings for survey responses.
        
        Args:
            survey_id: Survey ID
            responses: List of survey responses
        """
        try:
            logger.info(f"Processing {len(responses)} responses for survey {survey_id}")
            
            # Process each response
            for response in responses:
                await self.process_single_response(survey_id, response)
                
            logger.info(f"Completed processing responses for survey {survey_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing survey responses: {str(e)}")
            return False
    
    async def process_single_response(self, survey_id: int, response: Dict[str, Any]):
        """
        Process a single survey response.
        
        Args:
            survey_id: Survey ID
            response: Survey response data
        """
        try:
            response_id = response.get("_id")
            responses_data = response.get("responses", {})
            
            # Process each question-answer pair
            for question_id, answer in responses_data.items():
                # Skip if answer is None or empty
                if answer is None or (isinstance(answer, str) and not answer.strip()):
                    continue
                
                # Convert answer to string if it's not already
                if not isinstance(answer, str):
                    answer = json.dumps(answer)
                
                # Generate embedding for the answer
                embedding = await self.get_embedding(answer)
                
                # Create point for the vector database
                point = PointStruct(
                    id=f"{response_id}_{question_id}",
                    vector=embedding,
                    payload={
                        "survey_id": survey_id,
                        "response_id": response_id,
                        "question_id": question_id,
                        "answer": answer,
                        "response_date": response.get("submitted_at")
                    }
                )
                
                # Upsert the point to the vector database
                await qdrant_service.upsert_vectors(
                    collection_name=settings.SURVEY_RESPONSES_COLLECTION,
                    points=[point]
                )
                
            logger.info(f"Processed response {response_id} for survey {survey_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing single response: {str(e)}")
            return False
    
    async def process_questions(self, survey_id: int, questions: List[Dict[str, Any]]):
        """
        Process and store embeddings for survey questions.
        
        Args:
            survey_id: Survey ID
            questions: List of survey questions
        """
        try:
            logger.info(f"Processing {len(questions)} questions for survey {survey_id}")
            
            # Extract question texts
            question_texts = []
            question_ids = []
            
            for q in questions:
                question_id = q.get("id")
                question_text = q.get("question", "")
                
                if question_id and question_text:
                    question_texts.append(question_text)
                    question_ids.append(question_id)
            
            # Generate embeddings for all questions
            embeddings = await self.get_embeddings_batch(question_texts)
            
            # Create points for the vector database
            points = []
            for i, (question_id, embedding) in enumerate(zip(question_ids, embeddings)):
                point = PointStruct(
                    id=f"{survey_id}_{question_id}",
                    vector=embedding,
                    payload={
                        "survey_id": survey_id,
                        "question_id": question_id,
                        "question_text": question_texts[i],
                        "question_type": questions[i].get("type", "unknown")
                    }
                )
                points.append(point)
            
            # Upsert the points to the vector database
            await qdrant_service.upsert_vectors(
                collection_name=settings.QUESTION_EMBEDDINGS_COLLECTION,
                points=points
            )
                
            logger.info(f"Completed processing questions for survey {survey_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing survey questions: {str(e)}")
            return False
    
    async def process_metrics(self, survey_id: int, metrics: List[Dict[str, Any]]):
        """
        Process and store embeddings for survey metrics.
        
        Args:
            survey_id: Survey ID
            metrics: List of survey metrics
        """
        try:
            logger.info(f"Processing {len(metrics)} metrics for survey {survey_id}")
            
            # Extract metric texts
            metric_texts = []
            metric_ids = []
            
            for m in metrics:
                metric_id = m.get("id")
                # Combine name and description for better semantic understanding
                metric_text = f"{m.get('name', '')}. {m.get('description', '')}"
                
                if metric_id and metric_text:
                    metric_texts.append(metric_text)
                    metric_ids.append(metric_id)
            
            # Generate embeddings for all metrics
            embeddings = await self.get_embeddings_batch(metric_texts)
            
            # Create points for the vector database
            points = []
            for i, (metric_id, embedding) in enumerate(zip(metric_ids, embeddings)):
                point = PointStruct(
                    id=f"{survey_id}_{metric_id}",
                    vector=embedding,
                    payload={
                        "survey_id": survey_id,
                        "metric_id": metric_id,
                        "metric_name": metrics[i].get("name", ""),
                        "metric_description": metrics[i].get("description", ""),
                        "metric_type": metrics[i].get("type", "unknown")
                    }
                )
                points.append(point)
            
            # Upsert the points to the vector database
            await qdrant_service.upsert_vectors(
                collection_name=settings.METRIC_EMBEDDINGS_COLLECTION,
                points=points
            )
                
            logger.info(f"Completed processing metrics for survey {survey_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing survey metrics: {str(e)}")
            return False

# Create a singleton instance
embedding_service = EmbeddingService() 