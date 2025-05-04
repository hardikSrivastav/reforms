"""
Knowledge graph integration service.
Links vector-based insights with statistical analysis to create a comprehensive knowledge graph.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime
import networkx as nx

from ..config import settings
from ..services.metadata_store import metadata_store
from ..embeddings.multi_level_embedding_service import multi_level_embedding_service
from ..analysis.cross_metric_analysis import cross_metric_analysis_service
from ..analysis.vector_trend_analysis import vector_trend_analysis_service

logger = logging.getLogger(__name__)

class KnowledgeGraphIntegrationService:
    """
    Service for integrating vector-based insights with statistical analysis.
    Creates and maintains a comprehensive knowledge graph of relationships.
    """
    
    def __init__(self):
        """Initialize the knowledge graph integration service."""
        self.cache_ttl = settings.ANALYSIS_CACHE_TTL
        # Store knowledge graphs by survey_id
        self.knowledge_graphs = {}
        logger.info(f"Initialized knowledge graph integration service")
    
    async def build_integrated_knowledge_graph(self, survey_id: int) -> Dict[str, Any]:
        """
        Build an integrated knowledge graph that combines vector and statistical insights.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with the integrated knowledge graph
        """
        logger.info(f"Building integrated knowledge graph for survey {survey_id}")
        
        # Check cache first
        cache_key = f"knowledge_graph:{survey_id}"
        cached_result = await metadata_store.get_analysis_result(cache_key)
        if cached_result:
            logger.info(f"Using cached knowledge graph for survey {survey_id}")
            self.knowledge_graphs[survey_id] = cached_result.get("graph", {})
            return cached_result
        
        # We need both statistical and vector analyses to build a complete graph
        # Run both in parallel
        tasks = [
            # Get cross-metric correlations from statistical analysis
            self._get_cross_metric_correlations(survey_id),
            # Get vector-based insights
            self._get_vector_insights(survey_id),
            # Get aggregate embeddings
            multi_level_embedding_service.generate_aggregate_embeddings(survey_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, handling any exceptions
        statistical_insights = results[0] if not isinstance(results[0], Exception) else None
        vector_insights = results[1] if not isinstance(results[1], Exception) else None
        aggregate_embeddings = results[2] if not isinstance(results[2], Exception) else None
        
        # Check if we have enough data to build a graph
        if not statistical_insights and not vector_insights:
            error_msg = "Failed to retrieve both statistical and vector insights"
            logger.error(error_msg)
            return {
                "survey_id": survey_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": error_msg
            }
        
        # Initialize a NetworkX graph
        G = nx.DiGraph()
        
        # Add survey node as root
        G.add_node(f"survey:{survey_id}", 
                   type="survey", 
                   id=survey_id, 
                   node_type="root")
        
        # Add insights from statistical analysis
        if statistical_insights:
            self._add_statistical_insights_to_graph(G, survey_id, statistical_insights)
        
        # Add insights from vector analysis
        if vector_insights:
            self._add_vector_insights_to_graph(G, survey_id, vector_insights)
        
        # Add insights from hierarchical embeddings
        if aggregate_embeddings and aggregate_embeddings.get("status") == "success":
            self._add_hierarchical_insights_to_graph(G, survey_id, aggregate_embeddings)
        
        # Cross-connect nodes if possible
        self._add_cross_connections(G, survey_id)
        
        # Convert NetworkX graph to serializable format
        graph_data = self._nx_to_serializable(G)
        
        # Store the graph in memory
        self.knowledge_graphs[survey_id] = graph_data
        
        # Prepare result
        result = {
            "survey_id": survey_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "node_count": len(G.nodes),
            "edge_count": len(G.edges),
            "statistical_insights_available": statistical_insights is not None,
            "vector_insights_available": vector_insights is not None,
            "hierarchical_embeddings_available": aggregate_embeddings is not None and aggregate_embeddings.get("status") == "success",
            "graph": graph_data
        }
        
        # Cache the result
        await metadata_store.store_analysis_result(cache_key, result, self.cache_ttl)
        
        return result
    
    async def _get_cross_metric_correlations(self, survey_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cross-metric correlations from the statistical analysis service.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with cross-metric correlations or None if not available
        """
        try:
            # Check for cached cross-metric analysis
            cache_key = f"cross_metric_analysis:{survey_id}"
            cached_result = await metadata_store.get_analysis_result(cache_key)
            
            if cached_result:
                return cached_result
            
            # If not in cache, this would typically trigger a call to generate correlations
            # But for now, we'll just return None to avoid a potential infinite loop
            # In practice, you'd want to have the coordinator pre-generate cross-metric analysis
            logger.info(f"Cross-metric correlations not found in cache for survey {survey_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting cross-metric correlations: {str(e)}")
            return None
    
    async def _get_vector_insights(self, survey_id: int) -> Optional[Dict[str, Any]]:
        """
        Get vector-based insights from all questions in the survey.
        
        Args:
            survey_id: The ID of the survey
            
        Returns:
            Dictionary with vector insights or None if not available
        """
        try:
            # This simulates getting insights from vector analysis
            # In practice, this would come from pre-computed results or
            # would trigger generation of new results
            
            # For now, let's check if there are any cached cluster analyses
            vector_insights = {
                "clusters": {},
                "temporal_trends": {},
                "anomalies": {}
            }
            
            # Simple data structure to track if we got any insights
            got_insights = False
            
            # Try getting insights for metrics or questions
            # In a real implementation, you'd have a proper way to list all metrics/questions
            # Here we'll just try some common IDs for demonstration
            for item_id in [f"metric_{i}" for i in range(1, 10)]:
                # Try getting clusters
                cluster_key = f"clusters:{survey_id}:{item_id}"
                cluster_result = await metadata_store.get_analysis_result(cluster_key)
                if cluster_result:
                    vector_insights["clusters"][item_id] = cluster_result
                    got_insights = True
                
                # Try getting temporal trends
                trend_key = f"temporal_trends:{survey_id}:{item_id}"
                trend_result = await metadata_store.get_analysis_result(trend_key)
                if trend_result:
                    vector_insights["temporal_trends"][item_id] = trend_result
                    got_insights = True
                
                # Try getting anomalies
                anomaly_key = f"anomalies:{survey_id}:{item_id}"
                anomaly_result = await metadata_store.get_analysis_result(anomaly_key)
                if anomaly_result:
                    vector_insights["anomalies"][item_id] = anomaly_result
                    got_insights = True
            
            return vector_insights if got_insights else None
        except Exception as e:
            logger.error(f"Error getting vector insights: {str(e)}")
            return None
    
    def _add_statistical_insights_to_graph(
        self, 
        G: nx.DiGraph, 
        survey_id: int, 
        insights: Dict[str, Any]
    ) -> None:
        """
        Add statistical insights to the knowledge graph.
        
        Args:
            G: NetworkX graph
            survey_id: The ID of the survey
            insights: Dictionary with statistical insights
        """
        # Add metric nodes
        correlation_data = insights.get("correlation_matrix", {})
        metrics = correlation_data.get("metrics", [])
        
        for metric in metrics:
            metric_id = metric.get("id")
            if not metric_id:
                continue
                
            # Add metric node
            G.add_node(f"metric:{metric_id}", 
                       type="metric", 
                       id=metric_id, 
                       name=metric.get("name", metric_id),
                       node_type="metric")
            
            # Connect to survey
            G.add_edge(f"survey:{survey_id}", 
                       f"metric:{metric_id}", 
                       type="contains", 
                       weight=1.0)
        
        # Add correlations as edges
        pairwise_analyses = insights.get("pairwise_analyses", [])
        
        for pair in pairwise_analyses:
            if pair.get("type") == "numeric-numeric":
                metric1_id = pair.get("metric1_id")
                metric2_id = pair.get("metric2_id")
                correlation = pair.get("correlation", 0)
                p_value = pair.get("p_value", 1.0)
                significance = pair.get("significance", "none")
                
                if not (metric1_id and metric2_id):
                    continue
                
                # Add correlation edge if significant
                if significance != "none":
                    G.add_edge(f"metric:{metric1_id}", 
                               f"metric:{metric2_id}", 
                               type="correlates_with", 
                               correlation=correlation,
                               p_value=p_value,
                               significance=significance,
                               weight=abs(correlation))
            
            elif pair.get("type") == "numeric-categorical":
                metric1_id = pair.get("metric1_id")
                metric2_id = pair.get("metric2_id")
                is_significant = pair.get("is_significant", False)
                
                if not (metric1_id and metric2_id):
                    continue
                
                # Add relationship edge if significant
                if is_significant:
                    G.add_edge(f"metric:{metric2_id}", 
                               f"metric:{metric1_id}", 
                               type="influences", 
                               f_statistic=pair.get("f_statistic", 0),
                               p_value=pair.get("p_value", 1.0),
                               weight=0.7)  # Default weight
        
        # Add hypotheses as special nodes
        hypotheses = insights.get("hypotheses", [])
        
        for i, hypothesis in enumerate(hypotheses):
            hypothesis_id = f"hypothesis_{i}"
            metric1_id = hypothesis.get("metric1_id")
            metric2_id = hypothesis.get("metric2_id")
            
            if not (metric1_id and metric2_id):
                continue
                
            # Add hypothesis node
            G.add_node(f"hypothesis:{hypothesis_id}", 
                       type="hypothesis", 
                       hypothesis=hypothesis.get("hypothesis", ""),
                       confidence=hypothesis.get("confidence", "low"),
                       evidence=hypothesis.get("evidence", ""),
                       node_type="insight")
            
            # Connect hypothesis to metrics
            G.add_edge(f"hypothesis:{hypothesis_id}", 
                       f"metric:{metric1_id}", 
                       type="references", 
                       weight=0.8)
            G.add_edge(f"hypothesis:{hypothesis_id}", 
                       f"metric:{metric2_id}", 
                       type="references", 
                       weight=0.8)
    
    def _add_vector_insights_to_graph(
        self, 
        G: nx.DiGraph, 
        survey_id: int, 
        insights: Dict[str, Any]
    ) -> None:
        """
        Add vector-based insights to the knowledge graph.
        
        Args:
            G: NetworkX graph
            survey_id: The ID of the survey
            insights: Dictionary with vector insights
        """
        # Add cluster nodes
        for item_id, clusters in insights.get("clusters", {}).items():
            cluster_data = clusters.get("clusters", [])
            
            for i, cluster in enumerate(cluster_data):
                cluster_id = f"cluster_{item_id}_{i}"
                
                # Add cluster node
                G.add_node(f"cluster:{cluster_id}", 
                           type="cluster", 
                           size=cluster.get("size", 0),
                           percentage=cluster.get("percentage", 0),
                           samples=cluster.get("samples", []),
                           node_type="insight")
                
                # Connect to metric/question
                G.add_edge(f"cluster:{cluster_id}", 
                           f"metric:{item_id}", 
                           type="derived_from", 
                           weight=cluster.get("percentage", 0) / 100)
        
        # Add temporal trend nodes
        for item_id, trends in insights.get("temporal_trends", {}).items():
            drift_analyses = trends.get("drift_analysis", [])
            
            for i, drift in enumerate(drift_analyses):
                if drift.get("is_significant", False):
                    drift_id = f"drift_{item_id}_{i}"
                    
                    # Add drift node
                    G.add_node(f"drift:{drift_id}", 
                               type="temporal_drift", 
                               from_period=drift.get("from_period", ""),
                               to_period=drift.get("to_period", ""),
                               drift=drift.get("drift", 0),
                               node_type="insight")
                    
                    # Connect to metric/question
                    G.add_edge(f"drift:{drift_id}", 
                               f"metric:{item_id}", 
                               type="observed_in", 
                               weight=min(1.0, drift.get("drift", 0) * 10))
        
        # Add anomaly nodes
        for item_id, anomalies in insights.get("anomalies", {}).items():
            anomaly_data = anomalies.get("anomalies", [])
            
            for i, anomaly in enumerate(anomaly_data):
                anomaly_id = f"anomaly_{item_id}_{i}"
                
                # Add anomaly node
                G.add_node(f"anomaly:{anomaly_id}", 
                           type="anomaly", 
                           distance=anomaly.get("distance", 0),
                           response=anomaly.get("answer", ""),
                           node_type="insight")
                
                # Connect to metric/question
                G.add_edge(f"anomaly:{anomaly_id}", 
                           f"metric:{item_id}", 
                           type="detected_in", 
                           weight=min(1.0, anomaly.get("distance", 0) * 2))
    
    def _add_hierarchical_insights_to_graph(
        self, 
        G: nx.DiGraph, 
        survey_id: int, 
        embeddings: Dict[str, Any]
    ) -> None:
        """
        Add insights from hierarchical embeddings to the knowledge graph.
        
        Args:
            G: NetworkX graph
            survey_id: The ID of the survey
            embeddings: Dictionary with hierarchical embedding results
        """
        # Add segment nodes based on demographic embeddings
        demographic_results = embeddings.get("levels", {}).get("demographic", {})
        segment_results = demographic_results.get("segment_results", [])
        
        for segment in segment_results:
            segment_id = segment.get("segment_id")
            if not segment_id:
                continue
                
            # Add segment node
            G.add_node(f"segment:{segment_id}", 
                       type="demographic_segment", 
                       segment_type=segment.get("segment_type", ""),
                       segment_value=segment.get("segment_value", ""),
                       response_count=segment.get("response_count", 0),
                       node_type="segment")
            
            # Connect to survey
            G.add_edge(f"segment:{segment_id}", 
                       f"survey:{survey_id}", 
                       type="part_of", 
                       weight=segment.get("response_count", 0) / 100)
    
    def _add_cross_connections(self, G: nx.DiGraph, survey_id: int) -> None:
        """
        Add cross-connections between different types of nodes in the graph.
        
        Args:
            G: NetworkX graph
            survey_id: The ID of the survey
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated methods to determine connections
        
        # Identify insight nodes
        insight_nodes = [n for n, attrs in G.nodes(data=True) 
                         if attrs.get("node_type") == "insight"]
        
        # For each insight, try to connect to related segments if any
        for insight_node in insight_nodes:
            # Check what metric this insight is connected to
            metric_nodes = []
            for _, target, edge_attrs in G.out_edges(insight_node, data=True):
                if target.startswith("metric:") and edge_attrs.get("type") in ["derived_from", "observed_in", "detected_in", "references"]:
                    metric_nodes.append(target)
            
            # For each connected metric, see if there are segments that might relate
            segment_nodes = [n for n, attrs in G.nodes(data=True) 
                            if attrs.get("node_type") == "segment"]
            
            for metric_node in metric_nodes:
                for segment_node in segment_nodes:
                    # For simplicity, we'll create connections with a random "relevance" score
                    # In a real system, you would compute this based on actual data
                    import random
                    relevance = random.uniform(0.5, 0.9)
                    
                    # Only add connection if relevance is high enough
                    if relevance > 0.7:
                        G.add_edge(insight_node, 
                                  segment_node, 
                                  type="relevant_to", 
                                  relevance=relevance,
                                  weight=relevance)
    
    def _nx_to_serializable(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Convert a NetworkX graph to a serializable format.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary representation of the graph
        """
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            node_data = {
                "id": node_id,
                **attrs
            }
            nodes.append(node_data)
        
        edges = []
        for source, target, attrs in G.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                **attrs
            }
            edges.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    async def query_knowledge_graph(
        self, 
        survey_id: int, 
        query_type: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph to answer specific questions.
        
        Args:
            survey_id: The ID of the survey
            query_type: The type of query (e.g., 'path', 'neighbors', 'subgraph')
            params: Parameters for the query
            
        Returns:
            Query results
        """
        # Ensure we have the knowledge graph built
        if survey_id not in self.knowledge_graphs:
            # Try to get from cache
            cache_key = f"knowledge_graph:{survey_id}"
            cached_result = await metadata_store.get_analysis_result(cache_key)
            
            if cached_result:
                self.knowledge_graphs[survey_id] = cached_result.get("graph", {})
            else:
                # Build the graph if not in cache
                result = await self.build_integrated_knowledge_graph(survey_id)
                if result.get("status") != "success":
                    return {
                        "status": "error",
                        "error": "Failed to build knowledge graph",
                        "query_type": query_type
                    }
        
        # Get the graph data
        graph_data = self.knowledge_graphs.get(survey_id, {})
        
        # Rebuild the NetworkX graph from the serialized format
        G = nx.DiGraph()
        
        for node in graph_data.get("nodes", []):
            node_id = node.pop("id")
            G.add_node(node_id, **node)
        
        for edge in graph_data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            G.add_edge(source, target, **edge)
        
        # Process the query
        if query_type == "path":
            return self._query_path(G, params)
        elif query_type == "neighbors":
            return self._query_neighbors(G, params)
        elif query_type == "subgraph":
            return self._query_subgraph(G, params)
        else:
            return {
                "status": "error",
                "error": f"Unknown query type: {query_type}"
            }
    
    def _query_path(self, G: nx.DiGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find paths between nodes in the knowledge graph.
        
        Args:
            G: NetworkX graph
            params: Query parameters (source_node, target_node)
            
        Returns:
            Dictionary with paths between nodes
        """
        source_node = params.get("source_node")
        target_node = params.get("target_node")
        
        if not (source_node and target_node):
            return {
                "status": "error",
                "error": "Both source_node and target_node are required"
            }
        
        # Check if nodes exist
        if source_node not in G:
            return {
                "status": "error",
                "error": f"Source node not found: {source_node}"
            }
        
        if target_node not in G:
            return {
                "status": "error",
                "error": f"Target node not found: {target_node}"
            }
        
        # Try to find all simple paths
        try:
            all_paths = list(nx.all_simple_paths(G, source_node, target_node, cutoff=5))
            
            # If no direct paths, try paths in the other direction
            if not all_paths:
                all_paths = list(nx.all_simple_paths(G, target_node, source_node, cutoff=5))
                # Reverse the paths
                all_paths = [path[::-1] for path in all_paths]
            
            # Format the paths with edge information
            formatted_paths = []
            for path in all_paths:
                path_edges = []
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    edge_attrs = G.get_edge_data(source, target) or {}
                    path_edges.append({
                        "source": source,
                        "target": target,
                        **edge_attrs
                    })
                
                formatted_paths.append({
                    "path": path,
                    "edges": path_edges,
                    "length": len(path) - 1
                })
            
            # Sort paths by length
            formatted_paths.sort(key=lambda p: p["length"])
            
            return {
                "status": "success",
                "query_type": "path",
                "source_node": source_node,
                "target_node": target_node,
                "path_count": len(formatted_paths),
                "paths": formatted_paths
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error finding paths: {str(e)}"
            }
    
    def _query_neighbors(self, G: nx.DiGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find neighbors of a node in the knowledge graph.
        
        Args:
            G: NetworkX graph
            params: Query parameters (node_id, depth, edge_types)
            
        Returns:
            Dictionary with neighbor nodes
        """
        node_id = params.get("node_id")
        depth = params.get("depth", 1)
        edge_types = params.get("edge_types", [])
        
        if not node_id:
            return {
                "status": "error",
                "error": "node_id is required"
            }
        
        # Check if node exists
        if node_id not in G:
            return {
                "status": "error",
                "error": f"Node not found: {node_id}"
            }
        
        # Get neighbors up to specified depth
        neighbors = set()
        current_nodes = {node_id}
        
        for _ in range(depth):
            next_nodes = set()
            
            for current_node in current_nodes:
                # Get outgoing neighbors
                for _, neighbor, edge_attrs in G.out_edges(current_node, data=True):
                    edge_type = edge_attrs.get("type", "")
                    if not edge_types or edge_type in edge_types:
                        if neighbor != node_id and neighbor not in neighbors:
                            next_nodes.add(neighbor)
                
                # Get incoming neighbors
                for neighbor, _, edge_attrs in G.in_edges(current_node, data=True):
                    edge_type = edge_attrs.get("type", "")
                    if not edge_types or edge_type in edge_types:
                        if neighbor != node_id and neighbor not in neighbors:
                            next_nodes.add(neighbor)
            
            # Add new neighbors
            neighbors.update(next_nodes)
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        # Format the results
        neighbor_nodes = []
        for neighbor in neighbors:
            node_attrs = G.nodes[neighbor]
            neighbor_nodes.append({
                "id": neighbor,
                **node_attrs
            })
        
        # Get the connecting edges
        connecting_edges = []
        
        for source, target, attrs in G.edges(data=True):
            if (source == node_id and target in neighbors) or (target == node_id and source in neighbors):
                edge_type = attrs.get("type", "")
                if not edge_types or edge_type in edge_types:
                    connecting_edges.append({
                        "source": source,
                        "target": target,
                        **attrs
                    })
        
        return {
            "status": "success",
            "query_type": "neighbors",
            "node_id": node_id,
            "depth": depth,
            "neighbor_count": len(neighbor_nodes),
            "neighbors": neighbor_nodes,
            "edges": connecting_edges
        }
    
    def _query_subgraph(self, G: nx.DiGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a subgraph based on specified criteria.
        
        Args:
            G: NetworkX graph
            params: Query parameters (node_types, edge_types, min_weight)
            
        Returns:
            Dictionary with subgraph data
        """
        node_types = params.get("node_types", [])
        edge_types = params.get("edge_types", [])
        min_weight = params.get("min_weight", 0.0)
        
        # Filter nodes by type
        nodes_to_keep = []
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get("type", "")
            if not node_types or node_type in node_types:
                nodes_to_keep.append(node)
        
        # Create subgraph with filtered nodes
        subgraph = G.subgraph(nodes_to_keep).copy()
        
        # Filter edges by type and weight
        edges_to_remove = []
        for source, target, attrs in subgraph.edges(data=True):
            edge_type = attrs.get("type", "")
            edge_weight = attrs.get("weight", 0.0)
            
            if (edge_types and edge_type not in edge_types) or edge_weight < min_weight:
                edges_to_remove.append((source, target))
        
        # Remove filtered edges
        for source, target in edges_to_remove:
            subgraph.remove_edge(source, target)
        
        # Convert subgraph to serializable format
        subgraph_data = self._nx_to_serializable(subgraph)
        
        return {
            "status": "success",
            "query_type": "subgraph",
            "node_count": len(subgraph.nodes),
            "edge_count": len(subgraph.edges),
            "subgraph": subgraph_data
        }
    
    async def get_interactive_exploration(
        self, 
        survey_id: int,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Provide interactive exploration capabilities for hypothesis testing.
        
        Args:
            survey_id: The ID of the survey
            query: The exploration query (natural language)
            context: Optional exploration context
            
        Returns:
            Dictionary with exploration results
        """
        # Ensure we have the knowledge graph built
        if survey_id not in self.knowledge_graphs:
            await self.build_integrated_knowledge_graph(survey_id)
        
        # This would be implemented with more sophisticated NLP/reasoning
        # For now, we'll provide a simple implementation that returns some relevant data
        
        # Parse the query to determine what kind of exploration is needed
        query = query.lower()
        
        if "correlation" in query or "relationship" in query or "related" in query:
            # Get correlation data from cross-metric analysis
            cross_metric_data = await self._get_cross_metric_correlations(survey_id)
            
            # Extract relevant correlations
            if cross_metric_data:
                pairwise_analyses = cross_metric_data.get("pairwise_analyses", [])
                
                # Sort by correlation strength
                numeric_pairs = [p for p in pairwise_analyses if p.get("type") == "numeric-numeric"]
                numeric_pairs.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)
                
                return {
                    "status": "success",
                    "query_type": "relationship_exploration",
                    "query": query,
                    "result_type": "correlations",
                    "results": numeric_pairs[:5]  # Return top 5 correlations
                }
        
        elif "trend" in query or "time" in query or "temporal" in query:
            # Get vector-based trend analysis
            vector_insights = await self._get_vector_insights(survey_id)
            
            if vector_insights:
                temporal_trends = vector_insights.get("temporal_trends", {})
                
                # Flatten the trend data for easier consumption
                trend_results = []
                for item_id, trends in temporal_trends.items():
                    drift_analyses = trends.get("drift_analysis", [])
                    
                    for drift in drift_analyses:
                        if drift.get("is_significant", False):
                            trend_results.append({
                                "metric_id": item_id,
                                "from_period": drift.get("from_period", ""),
                                "to_period": drift.get("to_period", ""),
                                "drift": drift.get("drift", 0),
                                "is_significant": True
                            })
                
                return {
                    "status": "success",
                    "query_type": "trend_exploration",
                    "query": query,
                    "result_type": "temporal_trends",
                    "results": trend_results
                }
        
        elif "cluster" in query or "group" in query or "segment" in query:
            # Get vector-based cluster analysis
            vector_insights = await self._get_vector_insights(survey_id)
            
            if vector_insights:
                clusters = vector_insights.get("clusters", {})
                
                # Flatten the cluster data for easier consumption
                cluster_results = []
                for item_id, cluster_data in clusters.items():
                    cluster_list = cluster_data.get("clusters", [])
                    
                    for cluster in cluster_list:
                        cluster_results.append({
                            "metric_id": item_id,
                            "cluster_size": cluster.get("size", 0),
                            "percentage": cluster.get("percentage", 0),
                            "samples": cluster.get("samples", [])[:2]  # Limit to 2 samples
                        })
                
                return {
                    "status": "success",
                    "query_type": "cluster_exploration",
                    "query": query,
                    "result_type": "response_clusters",
                    "results": cluster_results
                }
        
        # Default response if no specific exploration matched
        return {
            "status": "success",
            "query_type": "general_exploration",
            "query": query,
            "result_type": "suggestion",
            "results": [
                {
                    "suggestion": "Try asking about correlations between metrics",
                    "example": "What metrics are strongly correlated?"
                },
                {
                    "suggestion": "Try asking about temporal trends",
                    "example": "Are there any significant trends over time?"
                },
                {
                    "suggestion": "Try asking about response clusters",
                    "example": "What clusters exist in the responses?"
                }
            ]
        }
    
    async def simulate_scenario(
        self, 
        survey_id: int,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate a scenario based on the knowledge graph.
        
        Args:
            survey_id: The ID of the survey
            scenario: Scenario configuration
            
        Returns:
            Dictionary with simulation results
        """
        # Ensure we have the knowledge graph built
        if survey_id not in self.knowledge_graphs:
            await self.build_integrated_knowledge_graph(survey_id)
        
        # In a real implementation, this would use the knowledge graph structure
        # to simulate effects of changes in certain metrics
        
        # For now, implement a simplified version that uses correlations
        
        # Get cross-metric correlations
        cross_metric_data = await self._get_cross_metric_correlations(survey_id)
        
        if not cross_metric_data:
            return {
                "status": "error",
                "error": "No correlation data available for simulation"
            }
        
        # Extract the variables to change from the scenario
        variables = scenario.get("variables", {})
        if not variables:
            return {
                "status": "error",
                "error": "No variables specified for the scenario"
            }
        
        # Get pairwise correlations
        pairwise_analyses = cross_metric_data.get("pairwise_analyses", [])
        numeric_pairs = [p for p in pairwise_analyses if p.get("type") == "numeric-numeric"]
        
        # Build a simple graph of correlations
        correlation_graph = {}
        
        for pair in numeric_pairs:
            metric1 = pair.get("metric1_id")
            metric2 = pair.get("metric2_id")
            correlation = pair.get("correlation", 0)
            
            if not (metric1 and metric2):
                continue
                
            # Only use significant correlations
            if pair.get("significance") == "none":
                continue
                
            # Add to graph
            if metric1 not in correlation_graph:
                correlation_graph[metric1] = {}
            
            if metric2 not in correlation_graph:
                correlation_graph[metric2] = {}
            
            correlation_graph[metric1][metric2] = correlation
            correlation_graph[metric2][metric1] = correlation
        
        # Simulate changes
        simulated_effects = {}
        
        for metric, change in variables.items():
            # Direct effect
            simulated_effects[metric] = change
            
            # Propagate effects through correlations
            if metric in correlation_graph:
                for related_metric, correlation in correlation_graph[metric].items():
                    # Skip if the related metric is also directly changed
                    if related_metric in variables:
                        continue
                        
                    # Calculate indirect effect
                    indirect_effect = change * correlation
                    
                    # Add to simulated effects (or update if already affected)
                    if related_metric in simulated_effects:
                        simulated_effects[related_metric] += indirect_effect
                    else:
                        simulated_effects[related_metric] = indirect_effect
        
        # Format the results
        effects = []
        for metric, effect in simulated_effects.items():
            effects.append({
                "metric_id": metric,
                "effect": effect,
                "is_direct": metric in variables
            })
        
        # Sort effects by magnitude
        effects.sort(key=lambda x: abs(x["effect"]), reverse=True)
        
        return {
            "status": "success",
            "scenario_name": scenario.get("name", "Unnamed Scenario"),
            "variables": variables,
            "effects": effects,
            "note": "This is a simplified simulation based on direct correlations. A full implementation would use more sophisticated causal modeling."
        }


# Create a singleton instance
knowledge_graph_integration = KnowledgeGraphIntegrationService() 