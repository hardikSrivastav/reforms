"""
Vector-based trend analysis service.
Provides methods for detecting trends and patterns in vector embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from qdrant_client.http.models import Filter, FieldCondition, Match, MatchValue, Range
from ..services.qdrant_client import qdrant_service
from ..config import settings

logger = logging.getLogger(__name__)

class VectorTrendAnalysisService:
    """Service for analyzing trends using vector embeddings."""
    
    async def detect_response_clusters(self, survey_id: int, question_id: str, max_clusters: int = 5):
        """
        Detect clusters in responses to identify common themes.
        
        Args:
            survey_id: Survey ID
            question_id: Question ID
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Dictionary with cluster analysis results
        """
        # Fetch all responses for the question
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="survey_id",
                    match=MatchValue(
                        value=survey_id
                    )
                ),
                FieldCondition(
                    key="question_id",
                    match=MatchValue(
                        value=question_id
                    )
                )
            ]
        )
        
        scroll_result = await qdrant_service.client.scroll(
            collection_name=settings.SURVEY_RESPONSES_COLLECTION,
            scroll_filter=filter_condition,
            limit=1000  # Adjust based on expected volume
        )
        
        points = scroll_result.points
        if not points or len(points) < 10:  # Need enough points for meaningful clustering
            return {"status": "insufficient_data", "clusters": []}
            
        # Extract vectors and metadata
        vectors = []
        payloads = []
        
        for point in points:
            vectors.append(point.vector)
            payloads.append(point.payload)
            
        vectors_array = np.array(vectors)
        
        # Determine optimal number of clusters
        optimal_clusters = 2  # Default
        best_score = -1
        
        for n_clusters in range(2, min(max_clusters + 1, len(vectors) // 5 + 1)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(vectors_array)
                
                if len(np.unique(cluster_labels)) > 1:  # Ensure we have at least 2 clusters
                    score = silhouette_score(vectors_array, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        optimal_clusters = n_clusters
            except Exception as e:
                logger.warning(f"Error in clustering with {n_clusters} clusters: {str(e)}")
        
        # Perform final clustering with optimal number
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors_array)
        
        # Organize results by cluster
        clusters = [[] for _ in range(optimal_clusters)]
        
        for i, (label, payload) in enumerate(zip(cluster_labels, payloads)):
            clusters[label].append({
                "response_id": payload.get("response_id"),
                "answer": payload.get("answer"),
                "response_date": payload.get("response_date")
            })
        
        # Generate cluster summaries
        result = {
            "status": "success",
            "cluster_count": optimal_clusters,
            "silhouette_score": best_score,
            "total_responses": len(vectors),
            "clusters": []
        }
        
        for i, cluster in enumerate(clusters):
            result["clusters"].append({
                "cluster_id": i,
                "size": len(cluster),
                "percentage": round(len(cluster) / len(vectors) * 100, 1),
                "samples": cluster[:5]  # Include sample responses
            })
        
        return result
    
    async def detect_temporal_trends(self, survey_id: int, question_id: str, 
                                    time_periods: int = 4, period_days: int = 30):
        """
        Detect trends over time in responses.
        
        Args:
            survey_id: Survey ID
            question_id: Question ID
            time_periods: Number of time periods to analyze
            period_days: Days per period
            
        Returns:
            Dictionary with temporal trend analysis
        """
        # Calculate time periods
        end_date = datetime.now()
        period_delta = timedelta(days=period_days)
        
        periods = []
        for i in range(time_periods):
            period_end = end_date - (i * period_delta)
            period_start = period_end - period_delta
            periods.append((period_start, period_end))
        
        # Get responses for each period
        period_responses = []
        
        for start, end in periods:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="survey_id",
                        match=MatchValue(
                            value=survey_id
                        )
                    ),
                    FieldCondition(
                        key="question_id",
                        match=MatchValue(
                            value=question_id
                        )
                    ),
                    FieldCondition(
                        key="response_date",
                        range=Range(
                            gte=start.isoformat(),
                            lt=end.isoformat()
                        )
                    )
                ]
            )
            
            try:
                scroll_result = await qdrant_service.client.scroll(
                    collection_name=settings.SURVEY_RESPONSES_COLLECTION,
                    scroll_filter=filter_condition,
                    limit=1000
                )
                
                period_responses.append({
                    "period": f"{start.date()} to {end.date()}",
                    "responses": scroll_result.points
                })
            except Exception as e:
                logger.error(f"Error querying responses for period {start.date()} to {end.date()}: {str(e)}")
                # Add empty responses to maintain period order
                period_responses.append({
                    "period": f"{start.date()} to {end.date()}",
                    "responses": []
                })
        
        # Calculate centroid vectors for each period
        centroids = []
        
        for period in period_responses:
            if not period["responses"]:
                centroids.append(None)
                continue
                
            vectors = [p.vector for p in period["responses"]]
            centroid = np.mean(vectors, axis=0).tolist()
            centroids.append(centroid)
        
        # Calculate drift between consecutive periods
        drift_analysis = []
        
        for i in range(len(centroids) - 1):
            if centroids[i] is None or centroids[i+1] is None:
                drift_analysis.append({
                    "from_period": period_responses[i]["period"],
                    "to_period": period_responses[i+1]["period"],
                    "drift": None,
                    "response_count": [
                        len(period_responses[i]["responses"]),
                        len(period_responses[i+1]["responses"])
                    ]
                })
                continue
                
            # Calculate cosine similarity between consecutive centroids
            vec1 = np.array(centroids[i])
            vec2 = np.array(centroids[i+1])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            drift = 1 - similarity
            
            drift_analysis.append({
                "from_period": period_responses[i]["period"],
                "to_period": period_responses[i+1]["period"],
                "drift": drift,
                "is_significant": drift > 0.05,  # Lower threshold to detect significant drift
                "response_count": [
                    len(period_responses[i]["responses"]),
                    len(period_responses[i+1]["responses"])
                ]
            })
        
        return {
            "survey_id": survey_id,
            "question_id": question_id,
            "period_count": time_periods,
            "period_days": period_days,
            "drift_analysis": drift_analysis,
            "has_significant_drift": any(a.get("is_significant", False) for a in drift_analysis if a.get("drift") is not None)
        }
    
    async def detect_anomalies(self, survey_id: int, question_id: str, threshold: float = 0.2):
        """
        Detect anomalous responses based on vector distance.
        
        Args:
            survey_id: Survey ID
            question_id: Question ID
            threshold: Threshold for anomaly detection (higher = more strict)
            
        Returns:
            List of anomalous responses
        """
        # Fetch all responses for the question
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="survey_id",
                    match=MatchValue(
                        value=survey_id
                    )
                ),
                FieldCondition(
                    key="question_id",
                    match=MatchValue(
                        value=question_id
                    )
                )
            ]
        )
        
        scroll_result = await qdrant_service.client.scroll(
            collection_name=settings.SURVEY_RESPONSES_COLLECTION,
            scroll_filter=filter_condition,
            limit=1000
        )
        
        points = scroll_result.points
        if not points or len(points) < 10:
            return {"status": "insufficient_data", "anomalies": []}
            
        # Extract vectors and convert to numpy array for more efficient computation
        vectors = np.array([p.vector for p in points])
        
        # Calculate pairwise cosine similarities between all vectors
        # We use matrix operations instead of loops for better performance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
        cosine_sim_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        # Average similarity for each vector with all others
        avg_similarities = np.sum(cosine_sim_matrix, axis=1) / (len(vectors) - 1)  # Subtract 1 to exclude self-similarity
        
        # Anomaly score is 1 - avg_similarity
        anomaly_scores = 1 - avg_similarities
        
        # Debug print
        print(f"DEBUG - Anomaly scores: {anomaly_scores}")
        
        # Find anomalies (ensure the threshold is properly applied)
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score > threshold:
                point = points[i]
                anomalies.append({
                    "response_id": point.payload.get("response_id"),
                    "answer": point.payload.get("answer"),
                    "distance": float(score),  # Convert from numpy type to native Python float
                    "response_date": point.payload.get("response_date")
                })
                
        # If no anomalies found and we have enough data, consider the most anomalous point
        if not anomalies and len(points) >= 10:
            # Find the index of maximum anomaly score
            max_index = np.argmax(anomaly_scores)
            max_score = anomaly_scores[max_index]
            
            # Add it as an anomaly regardless of threshold
            point = points[max_index]
            anomalies.append({
                "response_id": point.payload.get("response_id"),
                "answer": point.payload.get("answer"),
                "distance": float(max_score),
                "response_date": point.payload.get("response_date"),
                "note": "Forced anomaly detection - most divergent point"
            })
        
        return {
            "status": "success",
            "total_responses": len(points),
            "anomaly_count": len(anomalies),
            "anomaly_percentage": round(len(anomalies) / len(points) * 100, 1),
            "anomalies": anomalies
        }

# Create a singleton instance
vector_trend_analysis_service = VectorTrendAnalysisService() 