"""
Knowledge Base RAG Evaluator

Evaluates the Knowledge Base RAG system (policies, procedures, historical tickets).
Tests retrieval quality and answer generation for grounding agent responses.

Metrics:
- Faithfulness: LLM judges if answer is grounded in context (no hallucinations)
- Answer Relevancy: LLM judges if answer addresses the question
- Context Precision: Traditional metric - relevant chunks / total retrieved chunks
- Context Recall: Traditional metric - retrieved relevant / all relevant chunks

Targets (from system design doc):
- Faithfulness: ≥0.85
- Answer Relevancy: ≥0.90
- Context Precision: ≥0.80
- Context Recall: ≥0.90
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from evals.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class KBRAGTestCase:
    """A single Knowledge Base RAG test case"""
    question: str
    expected_answer: Optional[str] = None  # Ground truth answer (if available)
    relevant_doc_ids: List[str] = field(default_factory=list)  # IDs of docs that should be retrieved
    category: str = "general"  # policy, procedure, historical, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KBRAGEvalResult:
    """Result of evaluating a single KB RAG test case"""
    question: str
    answer: str
    retrieved_contexts: List[Dict[str, Any]]
    
    # LLM Judge metrics
    faithfulness_score: float
    faithfulness_verdict: str
    faithfulness_reasoning: str
    
    answer_relevancy_score: float
    answer_relevancy_verdict: str
    answer_relevancy_reasoning: str
    
    # Traditional metrics
    context_precision: float
    context_recall: float
    
    # Metadata
    num_retrieved: int
    num_relevant_retrieved: int
    num_relevant_total: int
    category: str
    
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KBRAGEvalSummary:
    """Summary of KB RAG evaluation across all test cases"""
    total_cases: int
    passed_cases: int
    failed_cases: int
    
    # Average scores
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    
    # Pass rates (based on targets)
    faithfulness_pass_rate: float  # Target: ≥0.85
    answer_relevancy_pass_rate: float  # Target: ≥0.90
    context_precision_pass_rate: float  # Target: ≥0.80
    context_recall_pass_rate: float  # Target: ≥0.90
    
    # Category breakdown
    results_by_category: Dict[str, List[KBRAGEvalResult]]
    
    # Detailed results
    results: List[KBRAGEvalResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        category_stats = {}
        for category, cat_results in self.results_by_category.items():
            if cat_results:
                category_stats[category] = {
                    "count": len(cat_results),
                    "passed": sum(1 for r in cat_results if r.passed),
                    "avg_faithfulness": sum(r.faithfulness_score for r in cat_results) / len(cat_results),
                    "avg_answer_relevancy": sum(r.answer_relevancy_score for r in cat_results) / len(cat_results),
                    "avg_context_precision": sum(r.context_precision for r in cat_results) / len(cat_results),
                    "avg_context_recall": sum(r.context_recall for r in cat_results) / len(cat_results)
                }
        
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "pass_rate": self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0
            },
            "average_scores": {
                "faithfulness": self.avg_faithfulness,
                "answer_relevancy": self.avg_answer_relevancy,
                "context_precision": self.avg_context_precision,
                "context_recall": self.avg_context_recall
            },
            "pass_rates": {
                "faithfulness": self.faithfulness_pass_rate,
                "answer_relevancy": self.answer_relevancy_pass_rate,
                "context_precision": self.context_precision_pass_rate,
                "context_recall": self.context_recall_pass_rate
            },
            "targets": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.90,
                "context_precision": 0.80,
                "context_recall": 0.90
            },
            "meets_targets": {
                "faithfulness": self.avg_faithfulness >= 0.85,
                "answer_relevancy": self.avg_answer_relevancy >= 0.90,
                "context_precision": self.avg_context_precision >= 0.80,
                "context_recall": self.avg_context_recall >= 0.90
            },
            "category_breakdown": category_stats
        }


class KnowledgeBaseRAGEvaluator:
    """
    Evaluates Knowledge Base RAG system performance.
    
    Tests:
    - Policy retrieval and answer generation
    - Procedure retrieval and answer generation
    - Historical ticket retrieval
    - Multi-document reasoning
    """
    
    def __init__(
        self,
        llm_judge: Optional[LLMJudge] = None,
        faithfulness_threshold: float = 0.85,
        answer_relevancy_threshold: float = 0.90,
        context_precision_threshold: float = 0.80,
        context_recall_threshold: float = 0.90
    ):
        """
        Initialize KB RAG evaluator.
        
        Args:
            llm_judge: LLM judge for faithfulness and relevancy (creates default if None)
            faithfulness_threshold: Minimum score for faithfulness (default: 0.85)
            answer_relevancy_threshold: Minimum score for answer relevancy (default: 0.90)
            context_precision_threshold: Minimum score for context precision (default: 0.80)
            context_recall_threshold: Minimum score for context recall (default: 0.90)
        """
        self.llm_judge = llm_judge or LLMJudge()
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevancy_threshold = answer_relevancy_threshold
        self.context_precision_threshold = context_precision_threshold
        self.context_recall_threshold = context_recall_threshold
    
    def evaluate_single(
        self,
        test_case: KBRAGTestCase,
        answer: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> KBRAGEvalResult:
        """
        Evaluate a single KB RAG test case.
        
        Args:
            test_case: The test case with question and expected relevant docs
            answer: The generated answer
            retrieved_contexts: List of retrieved context chunks with metadata
                               Each should have: {content, doc_id, score, metadata}
        
        Returns:
            KBRAGEvalResult with all metrics
        """
        # Step 1: LLM Judge - Faithfulness
        context_text = self._format_contexts(retrieved_contexts)
        faithfulness_result = self.llm_judge.judge_faithfulness(
            question=test_case.question,
            answer=answer,
            context=context_text
        )
        
        # Step 2: LLM Judge - Answer Relevancy
        relevancy_result = self.llm_judge.judge_answer_relevancy(
            question=test_case.question,
            answer=answer
        )
        
        # Step 3: Traditional Metrics - Context Precision
        context_precision = self._calculate_context_precision(
            retrieved_contexts=retrieved_contexts,
            relevant_doc_ids=test_case.relevant_doc_ids
        )
        
        # Step 4: Traditional Metrics - Context Recall
        context_recall = self._calculate_context_recall(
            retrieved_contexts=retrieved_contexts,
            relevant_doc_ids=test_case.relevant_doc_ids
        )
        
        # Determine if test passed (all metrics meet thresholds)
        passed = (
            faithfulness_result.score >= self.faithfulness_threshold and
            relevancy_result.score >= self.answer_relevancy_threshold and
            context_precision >= self.context_precision_threshold and
            context_recall >= self.context_recall_threshold
        )
        
        # Count relevant docs
        retrieved_doc_ids = {ctx.get("doc_id") for ctx in retrieved_contexts if ctx.get("doc_id")}
        relevant_doc_ids_set = set(test_case.relevant_doc_ids)
        num_relevant_retrieved = len(retrieved_doc_ids & relevant_doc_ids_set)
        
        return KBRAGEvalResult(
            question=test_case.question,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            faithfulness_score=faithfulness_result.score,
            faithfulness_verdict=faithfulness_result.verdict,
            faithfulness_reasoning=faithfulness_result.reasoning,
            answer_relevancy_score=relevancy_result.score,
            answer_relevancy_verdict=relevancy_result.verdict,
            answer_relevancy_reasoning=relevancy_result.reasoning,
            context_precision=context_precision,
            context_recall=context_recall,
            num_retrieved=len(retrieved_contexts),
            num_relevant_retrieved=num_relevant_retrieved,
            num_relevant_total=len(test_case.relevant_doc_ids),
            category=test_case.category,
            passed=passed,
            metadata=test_case.metadata
        )
    
    def evaluate_batch(
        self,
        test_cases: List[KBRAGTestCase],
        rag_pipeline: Any  # RAGPipeline instance
    ) -> KBRAGEvalSummary:
        """
        Evaluate multiple KB RAG test cases.
        
        Args:
            test_cases: List of test cases
            rag_pipeline: RAG pipeline instance to test
        
        Returns:
            KBRAGEvalSummary with aggregate metrics
        """
        results = []
        
        for test_case in test_cases:
            try:
                # Run RAG pipeline
                rag_result = rag_pipeline.retrieve_and_generate(
                    query=test_case.question
                )
                
                # Evaluate result
                eval_result = self.evaluate_single(
                    test_case=test_case,
                    answer=rag_result.get("answer", ""),
                    retrieved_contexts=rag_result.get("contexts", [])
                )
                results.append(eval_result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case: {e}")
                # Create failed result
                results.append(KBRAGEvalResult(
                    question=test_case.question,
                    answer="",
                    retrieved_contexts=[],
                    faithfulness_score=0.0,
                    faithfulness_verdict="error",
                    faithfulness_reasoning=str(e),
                    answer_relevancy_score=0.0,
                    answer_relevancy_verdict="error",
                    answer_relevancy_reasoning=str(e),
                    context_precision=0.0,
                    context_recall=0.0,
                    num_retrieved=0,
                    num_relevant_retrieved=0,
                    num_relevant_total=len(test_case.relevant_doc_ids),
                    category=test_case.category,
                    passed=False,
                    metadata={"error": str(e)}
                ))
        
        # Calculate summary statistics
        return self._create_summary(results)
    
    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """Format retrieved contexts for LLM judge"""
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            content = ctx.get("content", "")
            doc_id = ctx.get("doc_id", "unknown")
            score = ctx.get("score", 0.0)
            doc_type = ctx.get("metadata", {}).get("doc_type", "unknown")
            formatted.append(f"[Context {i}] (doc_id: {doc_id}, type: {doc_type}, score: {score:.3f})\n{content}")
        return "\n\n".join(formatted)
    
    def _calculate_context_precision(
        self,
        retrieved_contexts: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate context precision: relevant retrieved / total retrieved
        
        Measures: How many of the retrieved chunks are actually relevant?
        
        Note: Matches based on source filename in metadata, not chunk IDs.
        E.g., relevant_doc_ids=["cancellation_policy"] matches source="cancellation_policy.md"
        """
        if not retrieved_contexts:
            logger.debug("Context Precision: No retrieved contexts")
            return 0.0
        
        if not relevant_doc_ids:
            logger.debug("Context Precision: No relevant doc IDs")
            return 0.0
        
        # Extract source filenames from retrieved contexts
        retrieved_sources = []
        for ctx in retrieved_contexts:
            metadata = ctx.get("metadata", {})
            source = metadata.get("source", "")
            if source:
                # Remove extension and normalize
                source_base = source.replace(".md", "").replace(".txt", "").replace(".json", "")
                retrieved_sources.append(source_base)
        
        logger.debug(f"Context Precision - Retrieved sources: {retrieved_sources}")
        logger.debug(f"Context Precision - Relevant doc IDs: {relevant_doc_ids}")
        
        # Check how many retrieved sources match relevant doc IDs
        relevant_retrieved = 0
        for source in retrieved_sources:
            for relevant_id in relevant_doc_ids:
                # Flexible matching: check if relevant_id is in source or vice versa
                if relevant_id in source or source in relevant_id:
                    logger.debug(f"Context Precision - MATCH: '{source}' matches '{relevant_id}'")
                    relevant_retrieved += 1
                    break
        
        precision = relevant_retrieved / len(retrieved_contexts)
        logger.debug(f"Context Precision: {relevant_retrieved}/{len(retrieved_contexts)} = {precision:.3f}")
        return precision
    
    def _calculate_context_recall(
        self,
        retrieved_contexts: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate context recall: relevant retrieved / total relevant
        
        Measures: How many of the relevant documents were actually retrieved?
        
        Note: Matches based on source filename in metadata, not chunk IDs.
        E.g., relevant_doc_ids=["cancellation_policy"] matches source="cancellation_policy.md"
        """
        if not relevant_doc_ids:
            logger.debug("Context Recall: No relevant doc IDs expected, returning 1.0")
            return 1.0  # No relevant docs expected, so perfect recall
        
        # Extract unique source filenames from retrieved contexts
        retrieved_sources = set()
        for ctx in retrieved_contexts:
            metadata = ctx.get("metadata", {})
            source = metadata.get("source", "")
            if source:
                # Remove extension and normalize
                source_base = source.replace(".md", "").replace(".txt", "").replace(".json", "")
                retrieved_sources.add(source_base)
        
        logger.debug(f"Context Recall - Retrieved sources (unique): {retrieved_sources}")
        logger.debug(f"Context Recall - Relevant doc IDs: {relevant_doc_ids}")
        
        # Check how many relevant docs were retrieved
        relevant_retrieved = 0
        for relevant_id in relevant_doc_ids:
            for source in retrieved_sources:
                # Flexible matching: check if relevant_id is in source or vice versa
                if relevant_id in source or source in relevant_id:
                    logger.debug(f"Context Recall - MATCH: '{relevant_id}' found in '{source}'")
                    relevant_retrieved += 1
                    break
        
        recall = relevant_retrieved / len(relevant_doc_ids)
        logger.debug(f"Context Recall: {relevant_retrieved}/{len(relevant_doc_ids)} = {recall:.3f}")
        return recall
    
    def _create_summary(self, results: List[KBRAGEvalResult]) -> KBRAGEvalSummary:
        """Create summary statistics from evaluation results"""
        if not results:
            return KBRAGEvalSummary(
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_precision=0.0,
                avg_context_recall=0.0,
                faithfulness_pass_rate=0.0,
                answer_relevancy_pass_rate=0.0,
                context_precision_pass_rate=0.0,
                context_recall_pass_rate=0.0,
                results_by_category={},
                results=[]
            )
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        # Average scores
        avg_faithfulness = sum(r.faithfulness_score for r in results) / total
        avg_answer_relevancy = sum(r.answer_relevancy_score for r in results) / total
        avg_context_precision = sum(r.context_precision for r in results) / total
        avg_context_recall = sum(r.context_recall for r in results) / total
        
        # Pass rates (percentage meeting threshold)
        faithfulness_pass_rate = sum(
            1 for r in results if r.faithfulness_score >= self.faithfulness_threshold
        ) / total
        
        answer_relevancy_pass_rate = sum(
            1 for r in results if r.answer_relevancy_score >= self.answer_relevancy_threshold
        ) / total
        
        context_precision_pass_rate = sum(
            1 for r in results if r.context_precision >= self.context_precision_threshold
        ) / total
        
        context_recall_pass_rate = sum(
            1 for r in results if r.context_recall >= self.context_recall_threshold
        ) / total
        
        # Group by category
        results_by_category: Dict[str, List[KBRAGEvalResult]] = {}
        for result in results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)
        
        return KBRAGEvalSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_precision=avg_context_precision,
            avg_context_recall=avg_context_recall,
            faithfulness_pass_rate=faithfulness_pass_rate,
            answer_relevancy_pass_rate=answer_relevancy_pass_rate,
            context_precision_pass_rate=context_precision_pass_rate,
            context_recall_pass_rate=context_recall_pass_rate,
            results_by_category=results_by_category,
            results=results
        )
    
    def save_results(self, summary: KBRAGEvalSummary, output_path: str) -> None:
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        logger.info(f"Saved KB RAG evaluation results to {output_path}")
    
    @staticmethod
    def load_test_cases_from_file(file_path: str) -> List[KBRAGTestCase]:
        """
        Load test cases from JSON file.
        
        Expected format:
        [
            {
                "question": "What is the cancellation policy?",
                "expected_answer": "...",  # optional
                "relevant_doc_ids": ["policy_001", "policy_002"],
                "category": "policy",  # policy, procedure, historical, etc.
                "metadata": {}  # optional
            },
            ...
        ]
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return [
            KBRAGTestCase(
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                category=item.get("category", "general"),
                metadata=item.get("metadata", {})
            )
            for item in data
        ]
