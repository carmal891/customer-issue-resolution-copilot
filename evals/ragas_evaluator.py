"""
RAGAS-based RAG Evaluator

Uses the industry-standard RAGAS framework for evaluating RAG systems.
Replaces custom LLM judge with peer-reviewed, research-backed metrics.

RAGAS Metrics:
- Faithfulness: Answer grounded in context (no hallucinations)
- Answer Relevancy: Answer addresses the question
- Context Precision: Retrieved chunks are relevant
- Context Recall: All relevant chunks retrieved

Reference: https://github.com/explodinggradients/ragas
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datasets import Dataset
import asyncio
import nest_asyncio

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig

# Langchain imports for explicit LLM/embeddings
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper

# Import our custom gpt-5.4-mini wrapper
from evals.gpt54_mini_wrapper import Gpt54MiniLLM

# Apply nest_asyncio to allow nested event loops
# This fixes the deadlock when RAGAS tries to create an event loop
# while one already exists from httpx/async operations
nest_asyncio.apply()

logger = logging.getLogger(__name__)


@dataclass
class RAGASTestCase:
    """Test case for RAGAS evaluation"""
    question: str
    answer: str
    contexts: List[str]  # Retrieved context chunks
    ground_truth: Optional[str] = None  # Expected answer (optional)
    relevant_doc_ids: List[str] = field(default_factory=list)
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGASEvalResult:
    """Results from RAGAS evaluation"""
    # Overall metrics
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    
    # Per-question results
    per_question_results: List[Dict[str, Any]]
    
    # Summary statistics
    total_cases: int
    passed_cases: int
    failed_cases: int
    
    # Category breakdown
    results_by_category: Dict[str, Dict[str, float]]
    
    # Thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.85,
        "answer_relevancy": 0.90,
        "context_precision": 0.80,
        "context_recall": 0.90
    })
    
    def meets_targets(self) -> bool:
        """Check if all metrics meet their targets"""
        return (
            self.faithfulness >= self.thresholds["faithfulness"] and
            self.answer_relevancy >= self.thresholds["answer_relevancy"] and
            self.context_precision >= self.thresholds["context_precision"] and
            self.context_recall >= self.thresholds["context_recall"]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "pass_rate": self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0
            },
            "average_scores": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall
            },
            "targets": self.thresholds,
            "meets_targets": {
                "faithfulness": self.faithfulness >= self.thresholds["faithfulness"],
                "answer_relevancy": self.answer_relevancy >= self.thresholds["answer_relevancy"],
                "context_precision": self.context_precision >= self.thresholds["context_precision"],
                "context_recall": self.context_recall >= self.thresholds["context_recall"],
                "overall": self.meets_targets()
            },
            "category_breakdown": self.results_by_category,
            "per_question_results": self.per_question_results,
            "evaluation_framework": "RAGAS v0.4.3"
        }


class RAGASEvaluator:
    """
    RAGAS-based RAG evaluator using industry-standard metrics.
    
    Replaces custom LLM judge with peer-reviewed, research-backed evaluation.
    """
    
    def __init__(
        self,
        faithfulness_threshold: float = 0.85,
        answer_relevancy_threshold: float = 0.90,
        context_precision_threshold: float = 0.80,
        context_recall_threshold: float = 0.90
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            faithfulness_threshold: Minimum score for faithfulness (default: 0.85)
            answer_relevancy_threshold: Minimum score for answer relevancy (default: 0.90)
            context_precision_threshold: Minimum score for context precision (default: 0.80)
            context_recall_threshold: Minimum score for context recall (default: 0.90)
        """
        self.thresholds = {
            "faithfulness": faithfulness_threshold,
            "answer_relevancy": answer_relevancy_threshold,
            "context_precision": context_precision_threshold,
            "context_recall": context_recall_threshold
        }
        
        logger.info("Initialized RAGAS Evaluator")
        logger.info(f"Thresholds: {self.thresholds}")
    
    def evaluate_batch(
        self,
        test_cases: List[RAGASTestCase]
    ) -> RAGASEvalResult:
        """
        Evaluate multiple test cases using RAGAS.
        
        Args:
            test_cases: List of test cases with questions, answers, and contexts
        
        Returns:
            RAGASEvalResult with all metrics
        """
        if not test_cases:
            raise ValueError("No test cases provided")
        
        logger.info(f"Evaluating {len(test_cases)} test cases with RAGAS...")
        
        # Convert to RAGAS dataset format
        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for tc in test_cases:
            dataset_dict["question"].append(tc.question)
            dataset_dict["answer"].append(tc.answer)
            dataset_dict["contexts"].append(tc.contexts)
            # Use ground_truth if available, otherwise use answer as fallback
            dataset_dict["ground_truth"].append(tc.ground_truth or tc.answer)
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Initialize custom gpt-5.4-mini LLM wrapper
        logger.info("Initializing gpt-5.4-mini with custom wrapper...")
        langchain_llm = Gpt54MiniLLM(model="gpt-5.4-mini", temperature=0)
        llm = LangchainLLMWrapper(langchain_llm)
        
        # Test LLM connectivity before running evaluation
        logger.info("Testing LLM connectivity...")
        try:
            test_response = langchain_llm.invoke("Say 'ok'")
            logger.info(f"✓ LLM test successful: {test_response.content}")
        except Exception as e:
            logger.error(f"✗ LLM test failed: {e}")
            raise
        
        # Run RAGAS evaluation with explicit LLM and embeddings
        logger.info("Running RAGAS evaluation (this may take a few minutes)...")
        logger.info("Using gpt-5.4-mini (custom wrapper) for LLM judgments")
        logger.info("Using text-embedding-3-small for embeddings")
        
        ragas_results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=llm,  # Use our custom wrapped LLM
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        
        # Extract overall scores (RAGAS 0.4.3 returns lists, compute means)
        import numpy as np
        overall_faithfulness = float(np.nanmean(ragas_results["faithfulness"]))
        overall_answer_relevancy = float(np.nanmean(ragas_results["answer_relevancy"]))
        overall_context_precision = float(np.nanmean(ragas_results["context_precision"]))
        overall_context_recall = float(np.nanmean(ragas_results["context_recall"]))
        
        logger.info(f"RAGAS Evaluation Complete:")
        logger.info(f"  Faithfulness: {overall_faithfulness:.3f}")
        logger.info(f"  Answer Relevancy: {overall_answer_relevancy:.3f}")
        logger.info(f"  Context Precision: {overall_context_precision:.3f}")
        logger.info(f"  Context Recall: {overall_context_recall:.3f}")
        
        # Process per-question results
        per_question_results = []
        passed_count = 0
        
        # Group by category
        results_by_category: Dict[str, List[Dict[str, float]]] = {}
        
        for i, tc in enumerate(test_cases):
            # Get per-question scores (RAGAS 0.4.3 returns dict of lists, index directly)
            question_result = {
                "question": tc.question,
                "answer": tc.answer,  # Add actual answer
                "contexts": tc.contexts,  # Add retrieved contexts
                "category": tc.category,
                "faithfulness": float(ragas_results["faithfulness"][i]) if ragas_results["faithfulness"][i] is not None else overall_faithfulness,
                "answer_relevancy": float(ragas_results["answer_relevancy"][i]) if ragas_results["answer_relevancy"][i] is not None else overall_answer_relevancy,
                "context_precision": float(ragas_results["context_precision"][i]) if ragas_results["context_precision"][i] is not None else overall_context_precision,
                "context_recall": float(ragas_results["context_recall"][i]) if ragas_results["context_recall"][i] is not None else overall_context_recall,
            }
            
            # Check if passed
            passed = (
                question_result["faithfulness"] >= self.thresholds["faithfulness"] and
                question_result["answer_relevancy"] >= self.thresholds["answer_relevancy"] and
                question_result["context_precision"] >= self.thresholds["context_precision"] and
                question_result["context_recall"] >= self.thresholds["context_recall"]
            )
            
            question_result["passed"] = passed
            if passed:
                passed_count += 1
            
            per_question_results.append(question_result)
            
            # Group by category
            if tc.category not in results_by_category:
                results_by_category[tc.category] = []
            results_by_category[tc.category].append(question_result)
        
        # Calculate category statistics
        category_stats = {}
        for category, cat_results in results_by_category.items():
            category_stats[category] = {
                "count": len(cat_results),
                "passed": sum(1 for r in cat_results if r["passed"]),
                "avg_faithfulness": sum(r["faithfulness"] for r in cat_results) / len(cat_results),
                "avg_answer_relevancy": sum(r["answer_relevancy"] for r in cat_results) / len(cat_results),
                "avg_context_precision": sum(r["context_precision"] for r in cat_results) / len(cat_results),
                "avg_context_recall": sum(r["context_recall"] for r in cat_results) / len(cat_results)
            }
        
        return RAGASEvalResult(
            faithfulness=overall_faithfulness,
            answer_relevancy=overall_answer_relevancy,
            context_precision=overall_context_precision,
            context_recall=overall_context_recall,
            per_question_results=per_question_results,
            total_cases=len(test_cases),
            passed_cases=passed_count,
            failed_cases=len(test_cases) - passed_count,
            results_by_category=category_stats,
            thresholds=self.thresholds
        )
    
    def save_results(self, results: RAGASEvalResult, output_path: str) -> None:
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"Saved RAGAS evaluation results to {output_path}")


