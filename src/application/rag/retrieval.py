"""
RAG Retrieval Module

Implements dense retrieval with metadata filtering, query expansion,
and result ranking for the Customer Issue Resolution Copilot.

This module is critical for hackathon scoring on retrieval quality.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from src.infrastructure.embeddings.embedding_service import IEmbeddingService
from src.infrastructure.vector_store.chromadb_adapter import (
    ChromaDBAdapter,
    SearchResult,
    DistanceMetric
)


logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Retrieval strategy types"""
    DENSE = "dense"  # Pure vector similarity
    HYBRID = "hybrid"  # Vector + metadata filtering
    METADATA_FIRST = "metadata_first"  # Filter then search
    MULTI_QUERY = "multi_query"  # Query expansion


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    min_score: float = 0.0
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    enable_query_expansion: bool = True
    max_expanded_queries: int = 3
    hybrid_alpha: float = 0.7  # Weight for vector vs metadata
    metadata_boost: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default metadata boost weights"""
        if not self.metadata_boost:
            self.metadata_boost = {
                "doc_type": 1.2,  # Boost exact doc type matches
                "domain": 1.1,    # Boost domain matches
                "source": 1.0,    # Neutral for source
            }


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata and scoring"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    doc_type: str
    domain: Optional[str] = None
    section: Optional[str] = None
    timestamp: Optional[datetime] = None
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "doc_type": self.doc_type,
            "domain": self.domain,
            "section": self.section,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "rank": self.rank,
        }


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality monitoring"""
    query: str
    num_results: int
    avg_score: float
    max_score: float
    min_score: float
    retrieval_time_ms: float
    strategy_used: RetrievalStrategy
    metadata_filters: Dict[str, Any]
    expanded_queries: List[str] = field(default_factory=list)


class DenseRetriever:
    """
    Dense retrieval implementation using embeddings and vector search.
    
    Supports:
    - Pure vector similarity search
    - Hybrid search with metadata filtering
    - Query expansion for better recall
    - Metadata boosting for precision
    - Result deduplication and ranking
    """
    
    def __init__(
        self,
        embedding_service: IEmbeddingService,
        vector_store: ChromaDBAdapter,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize dense retriever.
        
        Args:
            embedding_service: Service for generating query embeddings
            vector_store: Vector database for similarity search
            config: Retrieval configuration
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        
        logger.info(
            f"Initialized DenseRetriever with strategy={self.config.strategy}, "
            f"top_k={self.config.top_k}"
        )
    
    def retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query or issue description
            metadata_filters: Optional metadata filters (doc_type, domain, etc.)
            top_k: Number of results to return (overrides config)
            min_score: Minimum similarity score (overrides config)
        
        Returns:
            Tuple of (results, metrics)
        """
        start_time = datetime.now()
        
        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.min_score
        
        logger.info(f"Retrieving for query: '{query[:100]}...'")
        
        # Select retrieval strategy
        if self.config.strategy == RetrievalStrategy.DENSE:
            results = self._dense_retrieve(query, top_k, min_score)
            expanded_queries = []
        
        elif self.config.strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_retrieve(
                query, metadata_filters, top_k, min_score
            )
            expanded_queries = []
        
        elif self.config.strategy == RetrievalStrategy.METADATA_FIRST:
            results = self._metadata_first_retrieve(
                query, metadata_filters, top_k, min_score
            )
            expanded_queries = []
        
        elif self.config.strategy == RetrievalStrategy.MULTI_QUERY:
            results, expanded_queries = self._multi_query_retrieve(
                query, metadata_filters, top_k, min_score
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Deduplicate and rank
        results = self._deduplicate_results(results)
        results = self._rank_results(results, query, metadata_filters)
        
        # Calculate metrics
        end_time = datetime.now()
        retrieval_time_ms = (end_time - start_time).total_seconds() * 1000
        
        metrics = RetrievalMetrics(
            query=query,
            num_results=len(results),
            avg_score=sum(r.score for r in results) / len(results) if results else 0.0,
            max_score=max((r.score for r in results), default=0.0),
            min_score=min((r.score for r in results), default=0.0),
            retrieval_time_ms=retrieval_time_ms,
            strategy_used=self.config.strategy,
            metadata_filters=metadata_filters or {},
            expanded_queries=expanded_queries
        )
        
        logger.info(
            f"Retrieved {len(results)} results in {retrieval_time_ms:.2f}ms, "
            f"avg_score={metrics.avg_score:.3f}"
        )
        
        return results, metrics
    
    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float
    ) -> List[RetrievalResult]:
        """Pure vector similarity search"""
        # Generate query embedding
        embedding_result = self.embedding_service.embed_texts([query])
        query_embedding = embedding_result.embeddings[0]
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where=None,
            distance_metric=self.config.distance_metric
        )
        
        # Convert to RetrievalResult
        results = []
        for sr in search_results:
            if sr.score >= min_score:
                results.append(self._convert_search_result(sr))
        
        return results
    
    def _hybrid_retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]],
        top_k: int,
        min_score: float
    ) -> List[RetrievalResult]:
        """Hybrid search combining vector similarity and metadata filtering"""
        # Generate query embedding
        embedding_result = self.embedding_service.embed_texts([query])
        query_embedding = embedding_result.embeddings[0]
        
        # Perform hybrid search
        # ChromaDB requires None instead of {} for no filters
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            where=metadata_filters if metadata_filters else None,
            n_results=top_k,
            alpha=self.config.hybrid_alpha
        )
        
        # Convert and filter by score
        results = []
        for sr in search_results:
            if sr.score >= min_score:
                results.append(self._convert_search_result(sr))
        
        return results
    
    def _metadata_first_retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]],
        top_k: int,
        min_score: float
    ) -> List[RetrievalResult]:
        """Filter by metadata first, then search within filtered set"""
        if not metadata_filters:
            # Fall back to dense retrieval if no filters
            return self._dense_retrieve(query, top_k, min_score)
        
        # First, get candidates by metadata
        metadata_results = self.vector_store.search_by_metadata(
            where=metadata_filters,
            n_results=top_k * 3  # Get more candidates
        )
        
        if not metadata_results:
            logger.warning("No results found with metadata filters")
            return []
        
        # Then rank by vector similarity
        embedding_result = self.embedding_service.embed_texts([query])
        query_embedding = embedding_result.embeddings[0]
        
        # Re-search within filtered set
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where=metadata_filters
        )
        
        results = []
        for sr in search_results:
            if sr.score >= min_score:
                results.append(self._convert_search_result(sr))
        
        return results
    
    def _multi_query_retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]],
        top_k: int,
        min_score: float
    ) -> Tuple[List[RetrievalResult], List[str]]:
        """
        Query expansion for better recall.
        
        Generates multiple query variations and combines results.
        """
        # Generate expanded queries
        expanded_queries = self._expand_query(query)
        
        logger.info(f"Expanded query into {len(expanded_queries)} variations")
        
        # Retrieve for each query
        all_results = []
        for expanded_query in expanded_queries:
            embedding_result = self.embedding_service.embed_texts([expanded_query])
            query_embedding = embedding_result.embeddings[0]
            
            search_results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                where=metadata_filters or {},
                n_results=top_k,
                alpha=self.config.hybrid_alpha
            )
            
            for sr in search_results:
                if sr.score >= min_score:
                    all_results.append(self._convert_search_result(sr))
        
        return all_results, expanded_queries
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple variations.
        
        For POC, uses simple rule-based expansion.
        In production, could use LLM for query rewriting.
        """
        expanded = [query]  # Original query
        
        # Add question variations
        if not query.endswith("?"):
            expanded.append(f"{query}?")
        
        # Add context variations for common hotel issues
        if "refund" in query.lower():
            expanded.append(f"cancellation policy for {query}")
            expanded.append(f"how to process {query}")
        
        if "room" in query.lower():
            expanded.append(f"room management {query}")
            expanded.append(f"booking modification {query}")
        
        if "checkout" in query.lower():
            expanded.append(f"late checkout policy {query}")
        
        # Limit to max expanded queries
        return expanded[:self.config.max_expanded_queries]
    
    def _convert_search_result(self, sr: SearchResult) -> RetrievalResult:
        """Convert SearchResult to RetrievalResult"""
        # Debug: Log metadata to see what we're getting
        logger.debug(f"Converting SearchResult - metadata keys: {list(sr.metadata.keys())}")
        logger.debug(f"Source field value: {sr.metadata.get('source', 'MISSING')}")
        
        return RetrievalResult(
            chunk_id=sr.chunk_id,
            content=sr.content,
            score=sr.score,
            metadata=sr.metadata,
            source=sr.metadata.get("source", "Unknown Source"),  # Match TPAO's default
            doc_type=sr.metadata.get("doc_type", "document"),  # Match TPAO's default
            domain=sr.metadata.get("domain"),
            section=sr.metadata.get("section"),
            timestamp=datetime.fromisoformat(sr.metadata["timestamp"])
            if "timestamp" in sr.metadata else None
        )
    
    def _deduplicate_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Remove duplicate chunks based on content similarity.
        
        Keeps the highest scoring version of each unique chunk.
        """
        seen_ids = set()
        deduplicated = []
        
        # Sort by score descending
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        for result in sorted_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                deduplicated.append(result)
        
        return deduplicated
    
    def _rank_results(
        self,
        results: List[RetrievalResult],
        query: str,
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """
        Re-rank results with metadata boosting.
        
        Applies boost factors for exact metadata matches.
        """
        if not metadata_filters:
            # Just sort by score
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            for i, result in enumerate(sorted_results):
                result.rank = i + 1
            return sorted_results
        
        # Apply metadata boosting
        for result in results:
            boost = 1.0
            
            # Boost for doc_type match
            if "doc_type" in metadata_filters:
                if result.doc_type == metadata_filters["doc_type"]:
                    boost *= self.config.metadata_boost.get("doc_type", 1.0)
            
            # Boost for domain match
            if "domain" in metadata_filters:
                if result.domain == metadata_filters["domain"]:
                    boost *= self.config.metadata_boost.get("domain", 1.0)
            
            # Apply boost to score
            result.score *= boost
        
        # Sort by boosted score
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        
        return sorted_results
    
    def retrieve_by_domain(
        self,
        query: str,
        domain: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Convenience method to retrieve by domain.
        
        Args:
            query: User query
            domain: Domain to filter (billing, room_management, etc.)
            top_k: Number of results
        
        Returns:
            List of retrieval results
        """
        metadata_filters = {"domain": domain}
        results, _ = self.retrieve(
            query=query,
            metadata_filters=metadata_filters,
            top_k=top_k
        )
        return results
    
    def retrieve_by_doc_type(
        self,
        query: str,
        doc_type: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Convenience method to retrieve by document type.
        
        Args:
            query: User query
            doc_type: Document type (policy, booking, issue, resolution)
            top_k: Number of results
        
        Returns:
            List of retrieval results
        """
        metadata_filters = {"doc_type": doc_type}
        results, _ = self.retrieve(
            query=query,
            metadata_filters=metadata_filters,
            top_k=top_k
        )
        return results
    
    def retrieve_recent(
        self,
        query: str,
        days: int = 30,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve recent documents (useful for issues and resolutions).
        
        Args:
            query: User query
            days: Number of days to look back
            top_k: Number of results
        
        Returns:
            List of retrieval results
        """
        # Note: This requires timestamp metadata in the vector store
        # For POC, we'll retrieve all and filter in memory
        results, _ = self.retrieve(query=query, top_k=top_k or 50)
        
        # Filter by timestamp
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_results = [
            r for r in results
            if r.timestamp and r.timestamp.timestamp() >= cutoff
        ]
        
        return recent_results[:top_k or self.config.top_k]
