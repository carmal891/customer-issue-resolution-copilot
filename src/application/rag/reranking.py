"""
RAG Reranking Module

Implements reranking strategies to improve retrieval precision.
This is CRITICAL for hackathon scoring on retrieval quality.

Supports:
- Cross-encoder reranking (local models)
- Cohere Rerank API
- Score normalization
- Diversity-aware reranking
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

from src.application.rag.retrieval import RetrievalResult


logger = logging.getLogger(__name__)


class RerankStrategy(str, Enum):
    """Reranking strategy types"""
    CROSS_ENCODER = "cross_encoder"  # Local cross-encoder model
    COHERE = "cohere"  # Cohere Rerank API
    DIVERSITY = "diversity"  # Diversity-aware reranking
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class RerankConfig:
    """Configuration for reranking"""
    strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    min_score: float = 0.0
    normalize_scores: bool = True
    diversity_lambda: float = 0.5  # For MMR-style diversity
    cohere_api_key: Optional[str] = None
    cohere_model: str = "rerank-english-v2.0"


@dataclass
class RerankResult:
    """Reranked result with updated score"""
    retrieval_result: RetrievalResult
    rerank_score: float
    original_score: float
    rank: int
    relevance_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.retrieval_result.chunk_id,
            "content": self.retrieval_result.content,
            "rerank_score": self.rerank_score,
            "original_score": self.original_score,
            "rank": self.rank,
            "metadata": self.retrieval_result.metadata,
            "relevance_explanation": self.relevance_explanation,
        }


class IReranker(ABC):
    """Abstract interface for reranking implementations"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank retrieval results.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        pass


class CrossEncoderReranker(IReranker):
    """
    Cross-encoder reranking using sentence-transformers.

    Cross-encoders jointly encode query and document for better
    relevance scoring than bi-encoders used in retrieval.
    """

    def __init__(self, config: RerankConfig):
        """
        Initialize cross-encoder reranker.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self.model = None

        logger.info(
            f"Initialized CrossEncoderReranker with model={config.model_name}"
        )

    def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.config.model_name)
                logger.info(f"Loaded cross-encoder model: {self.config.model_name}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using cross-encoder.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results with cross-encoder scores
        """
        if not results:
            return []

        self._load_model()

        top_k = top_k or self.config.top_k

        logger.info(f"Reranking {len(results)} results with cross-encoder")

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Normalize scores if configured
        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)

        # Create rerank results
        rerank_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            if score >= self.config.min_score:
                rerank_results.append(
                    RerankResult(
                        retrieval_result=result,
                        rerank_score=float(score),
                        original_score=result.score,
                        rank=0  # Will be set after sorting
                    )
                )

        # Sort by rerank score
        rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Assign ranks and limit to top_k
        for i, result in enumerate(rerank_results[:top_k]):
            result.rank = i + 1

        logger.info(
            f"Reranked to {len(rerank_results[:top_k])} results, "
            f"top score={rerank_results[0].rerank_score:.3f}"
        )

        return rerank_results[:top_k]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        # Handle numpy arrays and lists properly
        if isinstance(scores, (list, tuple)):
            if len(scores) == 0:
                return []
        else:
            # numpy array
            if scores.size == 0:
                return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            return [1.0] * len(scores)

        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

        return normalized


class CohereReranker(IReranker):
    """
    Reranking using Cohere Rerank API.

    Cohere's rerank models are specifically trained for relevance scoring
    and often outperform open-source alternatives.
    """

    def __init__(self, config: RerankConfig):
        """
        Initialize Cohere reranker.

        Args:
            config: Reranking configuration with API key
        """
        self.config = config
        self.client = None

        if not config.cohere_api_key:
            logger.warning(
                "Cohere API key not provided. "
                "Set COHERE_API_KEY environment variable."
            )

        logger.info(f"Initialized CohereReranker with model={config.cohere_model}")

    def _get_client(self):
        """Lazy load Cohere client"""
        if self.client is None:
            try:
                import cohere
                self.client = cohere.Client(self.config.cohere_api_key)
                logger.info("Initialized Cohere client")
            except ImportError:
                logger.error(
                    "cohere not installed. Install with: pip install cohere"
                )
                raise

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using Cohere Rerank API.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results with Cohere scores
        """
        if not results:
            return []

        self._get_client()

        top_k = top_k or self.config.top_k

        logger.info(f"Reranking {len(results)} results with Cohere API")

        # Prepare documents
        documents = [result.content for result in results]

        # Call Cohere Rerank API
        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model=self.config.cohere_model
            )

            # Create rerank results
            rerank_results = []
            for i, rerank_item in enumerate(response.results):
                original_result = results[rerank_item.index]

                rerank_results.append(
                    RerankResult(
                        retrieval_result=original_result,
                        rerank_score=rerank_item.relevance_score,
                        original_score=original_result.score,
                        rank=i + 1,
                        relevance_explanation=None  # Cohere doesn't provide this
                    )
                )

            logger.info(
                f"Reranked to {len(rerank_results)} results, "
                f"top score={rerank_results[0].rerank_score:.3f}"
            )

            return rerank_results

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            # Fall back to original ranking
            return self._fallback_ranking(results, top_k)

    def _fallback_ranking(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        """Fallback to original ranking if API fails"""
        logger.warning("Using fallback ranking (original scores)")

        rerank_results = []
        for i, result in enumerate(results[:top_k]):
            rerank_results.append(
                RerankResult(
                    retrieval_result=result,
                    rerank_score=result.score,
                    original_score=result.score,
                    rank=i + 1
                )
            )

        return rerank_results


class DiversityReranker(IReranker):
    """
    Diversity-aware reranking using Maximal Marginal Relevance (MMR).

    Balances relevance with diversity to avoid redundant results.
    Useful when multiple chunks from the same document are retrieved.
    """

    def __init__(self, config: RerankConfig):
        """
        Initialize diversity reranker.

        Args:
            config: Reranking configuration
        """
        self.config = config
        logger.info(
            f"Initialized DiversityReranker with lambda={config.diversity_lambda}"
        )

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using MMR for diversity.

        MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Diverse reranked results
        """
        if not results:
            return []

        top_k = top_k or self.config.top_k
        lambda_param = self.config.diversity_lambda

        logger.info(
            f"Reranking {len(results)} results with MMR (lambda={lambda_param})"
        )

        # Start with empty selected set
        selected = []
        remaining = list(results)

        # Iteratively select diverse results
        while len(selected) < top_k and remaining:
            if not selected:
                # First result: highest relevance
                best_idx = 0
                best_result = remaining[0]
            else:
                # Subsequent results: balance relevance and diversity
                best_idx = -1
                best_score = float('-inf')

                for i, candidate in enumerate(remaining):
                    # Relevance score
                    relevance = candidate.score

                    # Max similarity to already selected
                    max_sim = max(
                        self._content_similarity(candidate, selected_result)
                        for selected_result in selected
                    )

                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                        best_result = candidate

            # Move best result to selected
            selected.append(remaining.pop(best_idx))

        # Create rerank results
        rerank_results = []
        for i, result in enumerate(selected):
            rerank_results.append(
                RerankResult(
                    retrieval_result=result,
                    rerank_score=result.score,  # Keep original score
                    original_score=result.score,
                    rank=i + 1
                )
            )

        logger.info(f"Selected {len(rerank_results)} diverse results")

        return rerank_results

    def _content_similarity(
        self,
        result1: RetrievalResult,
        result2: RetrievalResult
    ) -> float:
        """
        Calculate content similarity between two results.

        For POC, uses simple heuristics:
        - Same source: high similarity
        - Same section: medium similarity
        - Different: low similarity

        In production, could use embedding similarity.
        """
        # Same chunk
        if result1.chunk_id == result2.chunk_id:
            return 1.0

        # Same source document
        if result1.source == result2.source:
            # Same section within document
            if result1.section and result2.section:
                if result1.section == result2.section:
                    return 0.8
                else:
                    return 0.5
            return 0.6

        # Different sources
        return 0.2


class HybridReranker(IReranker):
    """
    Hybrid reranking combining multiple strategies.

    Uses cross-encoder for relevance and diversity for coverage.
    """

    def __init__(self, config: RerankConfig):
        """
        Initialize hybrid reranker.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self.cross_encoder = CrossEncoderReranker(config)
        self.diversity = DiversityReranker(config)

        logger.info("Initialized HybridReranker")

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using hybrid approach.

        1. First pass: cross-encoder for relevance
        2. Second pass: diversity selection from top results

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Hybrid reranked results
        """
        if not results:
            return []

        top_k = top_k or self.config.top_k

        logger.info(f"Hybrid reranking {len(results)} results")

        # First pass: cross-encoder (get more than top_k)
        ce_results = self.cross_encoder.rerank(
            query=query,
            results=results,
            top_k=min(top_k * 2, len(results))
        )

        # Convert back to RetrievalResult for diversity pass
        ce_retrieval_results = [r.retrieval_result for r in ce_results]

        # Second pass: diversity selection
        final_results = self.diversity.rerank(
            query=query,
            results=ce_retrieval_results,
            top_k=top_k
        )

        logger.info(f"Hybrid reranking complete: {len(final_results)} results")

        return final_results


class RerankingPipeline:
    """
    Main reranking pipeline with strategy selection.

    Factory for creating and using different reranking strategies.
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        """
        Initialize reranking pipeline.

        Args:
            config: Reranking configuration
        """
        self.config = config or RerankConfig()
        self.reranker = self._create_reranker()

        logger.info(f"Initialized RerankingPipeline with strategy={self.config.strategy}")

    def _create_reranker(self) -> IReranker:
        """Create reranker based on strategy"""
        if self.config.strategy == RerankStrategy.CROSS_ENCODER:
            return CrossEncoderReranker(self.config)

        elif self.config.strategy == RerankStrategy.COHERE:
            return CohereReranker(self.config)

        elif self.config.strategy == RerankStrategy.DIVERSITY:
            return DiversityReranker(self.config)

        elif self.config.strategy == RerankStrategy.HYBRID:
            return HybridReranker(self.config)

        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank retrieval results.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        return self.reranker.rerank(query, results, top_k)

    def rerank_with_metrics(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> Tuple[List[RerankResult], Dict[str, Any]]:
        """
        Rerank with metrics tracking.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Tuple of (reranked results, metrics)
        """
        from datetime import datetime

        start_time = datetime.now()

        reranked = self.rerank(query, results, top_k)

        end_time = datetime.now()
        rerank_time_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate metrics
        metrics = {
            "strategy": self.config.strategy.value,
            "num_input": len(results),
            "num_output": len(reranked),
            "rerank_time_ms": rerank_time_ms,
            "avg_rerank_score": sum(r.rerank_score for r in reranked) / len(reranked) if reranked else 0.0,
            "avg_original_score": sum(r.original_score for r in reranked) / len(reranked) if reranked else 0.0,
            "score_improvement": 0.0,
        }

        # Calculate score improvement (top result)
        if reranked and results:
            original_top_score = results[0].score
            reranked_top_score = reranked[0].rerank_score
            metrics["score_improvement"] = reranked_top_score - original_top_score

        return reranked, metrics
