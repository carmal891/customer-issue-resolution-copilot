"""
RAG Evaluation Metrics

Implements comprehensive metrics for evaluating RAG system quality:
- Retrieval metrics: Precision, Recall, MRR, NDCG
- RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Custom metrics: Coverage, Diversity, Latency

Critical for hackathon scoring on RAG quality.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math

from src.application.rag.retrieval import RetrievalResult
from src.application.rag.reranking import RerankResult


logger = logging.getLogger(__name__)


@dataclass
class GroundTruthItem:
    """Ground truth for evaluation"""
    query: str
    relevant_chunk_ids: List[str]  # IDs of relevant chunks
    expected_answer: Optional[str] = None  # For answer quality metrics
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics"""
    precision: float
    recall: float
    f1_score: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    map_score: float  # Mean Average Precision
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int


@dataclass
class RAGASMetrics:
    """RAGAS framework metrics"""
    faithfulness: float  # Answer grounded in context
    answer_relevancy: float  # Answer addresses query
    context_precision: float  # Retrieved chunks are relevant
    context_recall: float  # All relevant chunks retrieved


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    query: str
    retrieval_metrics: RetrievalMetrics
    ragas_metrics: Optional[RAGASMetrics]
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "retrieval_metrics": {
                "precision": self.retrieval_metrics.precision,
                "recall": self.retrieval_metrics.recall,
                "f1_score": self.retrieval_metrics.f1_score,
                "mrr": self.retrieval_metrics.mrr,
                "ndcg": self.retrieval_metrics.ndcg,
                "map": self.retrieval_metrics.map_score,
            },
            "ragas_metrics": {
                "faithfulness": self.ragas_metrics.faithfulness,
                "answer_relevancy": self.ragas_metrics.answer_relevancy,
                "context_precision": self.ragas_metrics.context_precision,
                "context_recall": self.ragas_metrics.context_recall,
            } if self.ragas_metrics else None,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    Metrics:
    - Precision@K: Fraction of retrieved docs that are relevant
    - Recall@K: Fraction of relevant docs that are retrieved
    - F1 Score: Harmonic mean of precision and recall
    - MRR: Mean Reciprocal Rank (position of first relevant doc)
    - NDCG: Normalized Discounted Cumulative Gain (ranking quality)
    - MAP: Mean Average Precision (precision at each relevant doc)
    """

    def __init__(self):
        """Initialize retrieval evaluator"""
        logger.info("Initialized RetrievalEvaluator")

    def evaluate(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: Optional[int] = None
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.

        Args:
            retrieved_ids: IDs of retrieved chunks (in rank order)
            relevant_ids: IDs of relevant chunks (ground truth)
            k: Evaluate at top-k (None = all)

        Returns:
            Retrieval metrics
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        # Convert to sets for intersection
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        # Calculate basic metrics
        num_retrieved = len(retrieved_ids)
        num_relevant = len(relevant_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_set)

        # Precision: fraction of retrieved that are relevant
        precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0

        # Recall: fraction of relevant that are retrieved
        recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0

        # F1 Score: harmonic mean
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # MRR: Mean Reciprocal Rank
        mrr = self._calculate_mrr(retrieved_ids, relevant_set)

        # NDCG: Normalized Discounted Cumulative Gain
        ndcg = self._calculate_ndcg(retrieved_ids, relevant_set)

        # MAP: Mean Average Precision
        map_score = self._calculate_map(retrieved_ids, relevant_set)

        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mrr=mrr,
            ndcg=ndcg,
            map_score=map_score,
            num_retrieved=num_retrieved,
            num_relevant=num_relevant,
            num_relevant_retrieved=num_relevant_retrieved
        )

    def _calculate_mrr(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1 / rank of first relevant document
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str]
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        DCG = sum(rel_i / log2(i + 1))
        NDCG = DCG / IDCG (ideal DCG)
        """
        if not relevant_set:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate ideal DCG (all relevant docs at top)
        idcg = sum(
            1.0 / math.log2(i + 2)
            for i in range(min(len(relevant_set), len(retrieved_ids)))
        )

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_map(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str]
    ) -> float:
        """
        Calculate Mean Average Precision.

        MAP = mean of precision at each relevant document position
        """
        if not relevant_set:
            return 0.0

        precisions = []
        num_relevant_seen = 0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precision_at_i = num_relevant_seen / (i + 1)
                precisions.append(precision_at_i)

        return sum(precisions) / len(relevant_set) if precisions else 0.0

    def evaluate_batch(
        self,
        ground_truth: List[GroundTruthItem],
        retrieved_results: List[List[str]],
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate multiple queries and return average metrics.

        Args:
            ground_truth: List of ground truth items
            retrieved_results: List of retrieved ID lists (one per query)
            k: Evaluate at top-k

        Returns:
            Dictionary of average metrics
        """
        all_metrics = []

        for gt, retrieved in zip(ground_truth, retrieved_results):
            metrics = self.evaluate(retrieved, gt.relevant_chunk_ids, k)
            all_metrics.append(metrics)

        # Calculate averages
        avg_metrics = {
            "precision": sum(m.precision for m in all_metrics) / len(all_metrics),
            "recall": sum(m.recall for m in all_metrics) / len(all_metrics),
            "f1_score": sum(m.f1_score for m in all_metrics) / len(all_metrics),
            "mrr": sum(m.mrr for m in all_metrics) / len(all_metrics),
            "ndcg": sum(m.ndcg for m in all_metrics) / len(all_metrics),
            "map": sum(m.map_score for m in all_metrics) / len(all_metrics),
        }

        logger.info(f"Batch evaluation complete: {len(all_metrics)} queries")
        logger.info(f"Average Precision@{k or 'all'}: {avg_metrics['precision']:.3f}")
        logger.info(f"Average Recall@{k or 'all'}: {avg_metrics['recall']:.3f}")
        logger.info(f"Average MRR: {avg_metrics['mrr']:.3f}")

        return avg_metrics


class RAGASEvaluator:
    """
    Evaluates RAG quality using RAGAS framework metrics.

    RAGAS (RAG Assessment) metrics:
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevancy: Does the answer address the query?
    - Context Precision: Are the retrieved chunks relevant?
    - Context Recall: Are all relevant chunks retrieved?

    For POC, uses heuristic-based evaluation.
    In production, would use LLM-based evaluation.
    """

    def __init__(self):
        """Initialize RAGAS evaluator"""
        logger.info("Initialized RAGASEvaluator (heuristic mode)")

    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate if answer is grounded in context.

        Heuristic: Check if answer statements appear in context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not context:
            return 0.0

        # Split answer into sentences
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if not answer_sentences:
            return 0.0

        # Check how many answer sentences have support in context
        grounded_count = 0
        context_lower = context.lower()

        for sentence in answer_sentences:
            # Check if key words from sentence appear in context
            words = set(sentence.lower().split())
            # Remove common words
            words = words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}

            if len(words) == 0:
                continue

            # Check if majority of words appear in context
            words_in_context = sum(1 for w in words if w in context_lower)
            if words_in_context / len(words) >= 0.5:
                grounded_count += 1

        return grounded_count / len(answer_sentences)

    def evaluate_answer_relevancy(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Evaluate if answer addresses the query.

        Heuristic: Check keyword overlap and question type matching.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Relevancy score (0-1)
        """
        if not query or not answer:
            return 0.0

        query_lower = query.lower()
        answer_lower = answer.lower()

        # Extract query keywords (excluding common words)
        query_words = set(query_lower.split())
        query_words = query_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                                     'in', 'on', 'at', 'how', 'what', 'when', 'where', 
                                     'why', 'who', 'can', 'do', 'does'}

        if not query_words:
            return 0.5  # Neutral if no meaningful keywords

        # Check keyword overlap
        answer_words = set(answer_lower.split())
        overlap = len(query_words & answer_words)
        keyword_score = overlap / len(query_words)

        # Check if answer type matches query type
        type_score = 1.0
        if 'how' in query_lower and 'step' not in answer_lower and 'process' not in answer_lower:
            type_score = 0.7
        elif 'why' in query_lower and 'because' not in answer_lower and 'reason' not in answer_lower:
            type_score = 0.7

        return (keyword_score + type_score) / 2

    def evaluate_context_precision(
        self,
        retrieved_chunks: List[str],
        query: str
    ) -> float:
        """
        Evaluate if retrieved chunks are relevant to query.

        Heuristic: Check keyword overlap between query and chunks.

        Args:
            retrieved_chunks: Retrieved context chunks
            query: User query

        Returns:
            Context precision score (0-1)
        """
        if not retrieved_chunks or not query:
            return 0.0

        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_words = query_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                                     'in', 'on', 'at', 'how', 'what', 'when', 'where'}

        if not query_words:
            return 0.5

        # Calculate relevance for each chunk
        relevant_count = 0
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words & chunk_words)

            # Consider chunk relevant if it has >30% keyword overlap
            if overlap / len(query_words) >= 0.3:
                relevant_count += 1

        return relevant_count / len(retrieved_chunks)

    def evaluate_context_recall(
        self,
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str]
    ) -> float:
        """
        Evaluate if all relevant chunks were retrieved.

        Args:
            retrieved_chunk_ids: IDs of retrieved chunks
            relevant_chunk_ids: IDs of relevant chunks (ground truth)

        Returns:
            Context recall score (0-1)
        """
        if not relevant_chunk_ids:
            return 1.0  # No relevant chunks to retrieve

        retrieved_set = set(retrieved_chunk_ids)
        relevant_set = set(relevant_chunk_ids)

        retrieved_relevant = len(retrieved_set & relevant_set)
        return retrieved_relevant / len(relevant_set)

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_chunks: List[str],
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: Optional[List[str]] = None
    ) -> RAGASMetrics:
        """
        Evaluate all RAGAS metrics.

        Args:
            query: User query
            answer: Generated answer
            context: Assembled context
            retrieved_chunks: List of retrieved chunk texts
            retrieved_chunk_ids: IDs of retrieved chunks
            relevant_chunk_ids: Ground truth relevant chunk IDs (optional)

        Returns:
            RAGAS metrics
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        answer_relevancy = self.evaluate_answer_relevancy(query, answer)
        context_precision = self.evaluate_context_precision(retrieved_chunks, query)

        # Context recall requires ground truth
        if relevant_chunk_ids:
            context_recall = self.evaluate_context_recall(
                retrieved_chunk_ids, relevant_chunk_ids
            )
        else:
            context_recall = 0.0  # Unknown without ground truth

        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall
        )


class RAGEvaluationFramework:
    """
    Complete RAG evaluation framework.

    Combines retrieval metrics and RAGAS metrics for comprehensive evaluation.
    """

    def __init__(self):
        """Initialize evaluation framework"""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.ragas_evaluator = RAGASEvaluator()

        logger.info("Initialized RAGEvaluationFramework")

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        ground_truth: GroundTruthItem,
        k: Optional[int] = None
    ) -> Tuple[RetrievalMetrics, float]:
        """
        Evaluate retrieval quality.

        Args:
            query: User query
            retrieved_results: Retrieved results
            ground_truth: Ground truth item
            k: Evaluate at top-k

        Returns:
            Tuple of (metrics, latency_ms)
        """
        start_time = datetime.now()

        retrieved_ids = [r.chunk_id for r in retrieved_results]
        metrics = self.retrieval_evaluator.evaluate(
            retrieved_ids=retrieved_ids,
            relevant_ids=ground_truth.relevant_chunk_ids,
            k=k
        )

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        return metrics, latency_ms

    def evaluate_end_to_end(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_results: List[RetrievalResult],
        ground_truth: Optional[GroundTruthItem] = None
    ) -> EvaluationReport:
        """
        Evaluate complete RAG pipeline.

        Args:
            query: User query
            answer: Generated answer
            context: Assembled context
            retrieved_results: Retrieved results
            ground_truth: Optional ground truth for retrieval metrics

        Returns:
            Complete evaluation report
        """
        start_time = datetime.now()

        # Retrieval metrics (if ground truth available)
        if ground_truth:
            retrieval_metrics, _ = self.evaluate_retrieval(
                query, retrieved_results, ground_truth
            )
        else:
            # Create dummy metrics
            retrieval_metrics = RetrievalMetrics(
                precision=0.0, recall=0.0, f1_score=0.0,
                mrr=0.0, ndcg=0.0, map_score=0.0,
                num_retrieved=len(retrieved_results),
                num_relevant=0, num_relevant_retrieved=0
            )

        # RAGAS metrics
        retrieved_chunks = [r.content for r in retrieved_results]
        retrieved_ids = [r.chunk_id for r in retrieved_results]
        relevant_ids = ground_truth.relevant_chunk_ids if ground_truth else None

        ragas_metrics = self.ragas_evaluator.evaluate(
            query=query,
            answer=answer,
            context=context,
            retrieved_chunks=retrieved_chunks,
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=relevant_ids
        )

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        return EvaluationReport(
            query=query,
            retrieval_metrics=retrieval_metrics,
            ragas_metrics=ragas_metrics,
            latency_ms=latency_ms,
            timestamp=datetime.now()
        )
