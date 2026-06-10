"""
System Evaluation Runner - Integrated with Real Components

This script evaluates the actual RAG pipeline and skill matcher with indexed data.

Architecture Decision:
- RAG pipeline returns retrieval results via query_with_reranking()
- Evaluator needs both contexts AND answers for complete evaluation
- We add a thin wrapper that generates answers using LLM + retrieved contexts
- This tests the complete RAG pipeline: retrieval → reranking → generation

Requires:
- Indexed knowledge base (policies, procedures, tickets)
- Indexed skill triggers
- OpenAI API key for embeddings and LLM

Usage:
    python evaluations/run_system_evaluations.py
"""

import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Ensure OPENAI_API_KEY is set.")

from evals.ragas_evaluator import RAGASEvaluator, RAGASTestCase
from evals.skill_matching_rag_evaluator import SkillMatchingEvaluator, SkillMatchTestCase
from src.domain.models.issue import Issue, IssueChannel, IssuePriority

# Real system components
from src.application.rag.rag_pipeline import RAGPipeline, RAGConfig
from src.application.skills.skill_matcher import SkillMatcher
from src.application.skills.skill_registry import SkillRegistry
from src.infrastructure.embeddings.embedding_service import EmbeddingServiceFactory, EmbeddingProvider
from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter, DistanceMetric
from src.infrastructure.llm.llm_service import LLMService
from src.application.rag.reranking import CrossEncoderReranker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f).get('test_cases', [])


def initialize_rag_pipeline() -> RAGPipeline:
    """
    Initialize RAG pipeline with indexed data.

    Note: Uses production vector store. For isolated evaluation,
    run indexing scripts first with eval-specific collections.
    """
    logger.info("Initializing RAG pipeline for evaluation...")

    config = RAGConfig(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        vector_store_path="./data/vector_store",
        collection_name="hotel_knowledge",
        enable_reranking=True,
        retrieval_top_k=10,  # Rolled back to baseline
        rerank_top_k=5       # Rolled back to baseline
    )

    pipeline = RAGPipeline(config)

    try:
        count = pipeline.vector_store.count()
        logger.info(f" RAG pipeline initialized ({count} indexed chunks in vector store)")
    except Exception as e:
        logger.warning(f"Could not get vector store count: {e}")
        logger.info(" RAG pipeline initialized (count unavailable)")
    
    # Initialize BM25 for hybrid search
    try:
        global _bm25_documents
        if '_bm25_documents' in globals() and _bm25_documents:
            logger.info(f"Initializing BM25 index with {len(_bm25_documents)} documents...")
            pipeline.retriever.initialize_bm25(_bm25_documents)
            logger.info(" BM25 index initialized successfully")
        else:
            logger.warning(" No BM25 documents available - hybrid search will use dense retrieval only")
    except Exception as e:
        logger.warning(f"Could not initialize BM25: {e}")
        logger.info(" Continuing with dense retrieval only")

    return pipeline


def initialize_skill_matcher() -> SkillMatcher:
    """
    Initialize skill matcher with indexed skills.

    Note: Uses production vector store. For isolated evaluation,
    run indexing scripts first with eval-specific collections.
    """
    logger.info("Initializing skill matcher for evaluation...")

    import os

    # Create embedding service and vector store FIRST
    # IMPORTANT: Must use SentenceTransformer to match skills indexing (384 dimensions)
    embedding_service = EmbeddingServiceFactory.create(
        provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
        config={
            "model": "all-MiniLM-L6-v2"  # 384 dimensions, matches skills indexing
        }
    )
    vector_store = ChromaDBAdapter(
        persist_directory="./data/vector_store",
        collection_name="hotel_skills",
        distance_metric=DistanceMetric.COSINE
    )

    # Pass embedding_service and vector_store to SkillRegistry
    skill_registry = SkillRegistry(
        skills_dir="./data/skills",
        embedding_service=embedding_service,
        vector_store=vector_store
    )
    from src.application.rag.reranking import RerankConfig, RerankStrategy
    reranker = CrossEncoderReranker(
        config=RerankConfig(
            strategy=RerankStrategy.CROSS_ENCODER,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    llm_service = LLMService(model="gpt-5.4-mini")

    matcher = SkillMatcher(
        skill_registry=skill_registry,
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranker=reranker,
        llm_service=llm_service,
        high_confidence_threshold=0.60,
        medium_confidence_threshold=0.45
    )

    skill_count = len(skill_registry.get_all_skills())
    logger.info(f"Skill matcher initialized ({skill_count} skills)")

    return matcher


class RAGPipelineAdapter:
    """
    Adapter: RAG Pipeline → Evaluator Interface

    Converts query_with_reranking() output to retrieve_and_generate() format.
    Adds LLM-based answer generation on top of retrieval.
    """

    def __init__(self, rag_pipeline: RAGPipeline, llm_service: LLMService):
        self.rag_pipeline = rag_pipeline
        self.llm_service = llm_service

    def retrieve_and_generate(self, query: str, metadata_filters=None, top_k=None):
        """
        Retrieve + Generate answer.

        Returns:
            {"answer": str, "contexts": List[dict]}
        """
        # Step 1: Retrieve with reranking
        reranked_results, _ = self.rag_pipeline.query_with_reranking(
            query=query,
            metadata_filters=metadata_filters,
            top_k=top_k or 5
        )

        logger.debug(f"Query: '{query}' | Retrieved {len(reranked_results)} results")

        # Step 2: Format contexts
        contexts = [
            {
                "doc_id": r.retrieval_result.chunk_id,
                "content": r.retrieval_result.content,
                "score": r.rerank_score,
                "metadata": r.retrieval_result.metadata
            }
            for r in reranked_results
        ]

        if contexts:
            logger.debug(f"First context source: {contexts[0]['metadata'].get('source', 'unknown')}")
        else:
            logger.warning(f"No contexts retrieved for query: '{query}'")

        # Step 3: Generate answer using LLM
        context_text = "\n\n".join([
            f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['content']}"
            for c in contexts
        ])

        system_prompt = "You are a hotel customer service assistant. Answer questions based ONLY on the provided context. Be concise and factual."

        user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        answer_response = self.llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_completion_tokens=300
        )
        answer = answer_response.content

        return {"answer": answer, "contexts": contexts}


def run_kb_rag_evaluation(rag_adapter: RAGPipelineAdapter, num_cases: Optional[int] = None):
    """Evaluate KB RAG using RAGAS ()"""
    logger.info("\n" + "="*80)
    logger.info("KB RAG EVALUATION - RAGAS ()")
    logger.info("="*80)

    # Initialize RAGAS evaluator
    evaluator = RAGASEvaluator()

    # Load test cases
    test_cases_path = Path(__file__).parent / "test_data/kb_rag_test_cases.json"
    test_cases_json = load_test_cases(str(test_cases_path))

    # Run RAG pipeline for each test case to get answers and contexts
    logger.info(f"Running RAG pipeline for {len(test_cases_json[:num_cases] if num_cases else test_cases_json)} test cases...")
    
    ragas_test_cases = []
    for case in (test_cases_json[:num_cases] if num_cases else test_cases_json):
        # Get RAG result
        rag_result = rag_adapter.retrieve_and_generate(query=case['question'])
        
        # Extract contexts as strings
        contexts = [ctx['content'] for ctx in rag_result['contexts']]
        
        # Create RAGAS test case with ground truth from expected_answer_contains
        # This is critical for context_recall to work properly
        ground_truth = " ".join(case.get('expected_answer_contains', []))
        if not ground_truth:
            # Fallback to answer if no expected_answer_contains
            ground_truth = rag_result['answer']
        
        ragas_test_cases.append(RAGASTestCase(
            question=case['question'],
            answer=rag_result['answer'],
            contexts=contexts,
            ground_truth=ground_truth,
            relevant_doc_ids=case.get('relevant_doc_ids', []),
            category=case.get('category', 'general')
        ))
    
    # Run RAGAS evaluation
    logger.info(f"Evaluating with RAGAS...")
    results = evaluator.evaluate_batch(ragas_test_cases)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Framework: RAGAS v0.1.x ()")
    logger.info(f"Total: {results.total_cases} | Passed: {results.passed_cases} | Failed: {results.failed_cases}")
    logger.info(f"\nFaithfulness: {results.faithfulness:.3f} (target: ≥0.85)")
    logger.info(f"Answer Relevancy: {results.answer_relevancy:.3f} (target: ≥0.90)")
    logger.info(f"Context Precision: {results.context_precision:.3f} (target: ≥0.80)")
    logger.info(f"Context Recall: {results.context_recall:.3f} (target: ≥0.90)")

    logger.info("\nBy Category:")
    for cat, cat_stats in results.results_by_category.items():
        logger.info(f"  {cat}: {cat_stats['passed']}/{cat_stats['count']}")

    return results


def run_skill_matching_evaluation(skill_matcher: SkillMatcher, num_cases: Optional[int] = None):
    """Evaluate skill matching on real system"""
    logger.info("\n" + "="*80)
    logger.info("SKILL MATCHING EVALUATION (Real System)")
    logger.info("="*80)

    evaluator = SkillMatchingEvaluator(confidence_threshold=0.45)

    test_cases_path = Path(__file__).parent / "test_data/skill_matching_test_cases.json"
    test_cases_json = load_test_cases(str(test_cases_path))

    test_cases = [
        SkillMatchTestCase(
            issue_description=case['issue_description'],
            expected_skill_id=case.get('expected_skill_id'),
            expected_skill_name=case.get('expected_skill_name'),
            should_match=case.get('test_type') == 'positive',
            category=case.get('category', 'general')
        )
        for case in (test_cases_json[:num_cases] if num_cases else test_cases_json)
    ]

    logger.info(f"Running {len(test_cases)} test cases...")
    results = evaluator.evaluate_batch(test_cases, skill_matcher)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Total: {results.total_cases} | Passed: {results.passed_cases} | Failed: {results.failed_cases}")
    logger.info(f"\nTop-1 Accuracy: {results.top_1_accuracy:.3f} (target: ≥0.90)")
    logger.info(f"Top-3 Accuracy: {results.top_3_accuracy:.3f} (target: ≥0.95)")
    logger.info(f"MRR: {results.mean_reciprocal_rank:.3f}")
    logger.info(f"False Positive Rate: {results.false_positive_rate:.3f}")
    logger.info(f"False Negative Rate: {results.false_negative_rate:.3f}")

    logger.info("\nBy Category:")
    for cat, cat_results in results.results_by_category.items():
        passed = sum(1 for r in cat_results if r.passed)
        logger.info(f"  {cat}: {passed}/{len(cat_results)}")

    return results

def generate_markdown_report(
    kb_results: Dict[str, Any],
    skill_results: Dict[str, Any],
    timestamp: str,
    output_dir: str = "evals/reports"
) -> str:
    """
    Generate a PM-friendly markdown report from evaluation results.

    Args:
        kb_results: Knowledge base RAG evaluation results
        skill_results: Skill matching evaluation results
        timestamp: Timestamp string for the report
        output_dir: Directory to save the report

    Returns:
        Path to the generated report file
    """
    from pathlib import Path
    from datetime import datetime

    # Create reports directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate report filename
    report_file = f"{output_dir}/evaluation_report_{timestamp}.md"

    # Extract key metrics
    kb_pass_rate = kb_results["summary"]["pass_rate"] * 100
    skill_pass_rate = skill_results["summary"]["pass_rate"] * 100

    kb_total = kb_results["summary"]["total_cases"]
    kb_passed = kb_results["summary"]["passed_cases"]

    skill_total = skill_results["summary"]["total_cases"]
    skill_passed = skill_results["summary"]["passed_cases"]

    # Build report content
    report = f"""# Customer Issue Resolution Copilot - Evaluation Report

**Date:** {datetime.now().strftime("%B %d, %Y")}  
**Evaluation Run:** {timestamp}  
**System Version:** v1.0 (POC)

---

## Executive Summary

This report evaluates the **Customer Issue Resolution Copilot** system across two critical capabilities:

1. **Knowledge Base RAG** - How well the system retrieves and uses hotel policies to answer questions
2. **Skill Matching** - How accurately the system matches customer issues to pre-defined resolution workflows

### Overall Results

| Component | Status | Pass Rate | Passed | Total | Target |
|-----------|--------|-----------|--------|-------|--------|
| **Knowledge Base RAG** | {' **PASS**' if kb_pass_rate >= 85 else '️ **NEEDS IMPROVEMENT**'} | {kb_pass_rate:.1f}% | {kb_passed}/{kb_total} | 85%+ |
| **Skill Matching** | {' **PASS**' if skill_pass_rate >= 90 else '️ **NEEDS IMPROVEMENT**'} | {skill_pass_rate:.1f}% | {skill_passed}/{skill_total} | 90%+ |

---

## 1. Knowledge Base RAG Evaluation

### What We're Testing

When a customer asks a question (e.g., "What's your cancellation policy?"), the system must:
- Find the right policy documents
- Extract relevant information  
- Generate an accurate, grounded answer

### Metrics

| Metric | Description | Target | Actual | Status |
|--------|-------------|--------|--------|--------|
| **Faithfulness** | Answer is based on retrieved documents (no hallucination) | ≥85% | {kb_results['average_scores']['faithfulness']:.1%} | {'' if kb_results['meets_targets']['faithfulness'] else ''} |
| **Answer Relevancy** | Answer directly addresses the question | ≥90% | {kb_results['average_scores']['answer_relevancy']:.1%} | {'' if kb_results['meets_targets']['answer_relevancy'] else ''} |
| **Context Precision** | Retrieved documents are relevant | ≥80% | {kb_results['average_scores']['context_precision']:.1%} | {'' if kb_results['meets_targets']['context_precision'] else ''} |
| **Context Recall** | All relevant documents were retrieved | ≥90% | {kb_results['average_scores']['context_recall']:.1%} | {'' if kb_results['meets_targets']['context_recall'] else ''} |

### Results by Category

"""

    # Add KB category breakdown
    for category, stats in kb_results["category_breakdown"].items():
        report += f"""
#### {category.title()} Questions ({stats['count']} tests)

- **Pass Rate:** {stats['passed']}/{stats['count']} ({stats['passed']/stats['count']*100:.0f}%)
- **Avg Faithfulness:** {stats['avg_faithfulness']:.1%}
- **Avg Answer Relevancy:** {stats['avg_answer_relevancy']:.1%}
- **Avg Context Precision:** {stats['avg_context_precision']:.1%}
- **Avg Context Recall:** {stats['avg_context_recall']:.1%}
"""

    report += f"""
---

## 2. Skill Matching Evaluation

### What We're Testing

When a customer issue comes in (e.g., "I need to checkout late"), the system must:
- Understand the intent
- Match it to the right pre-built workflow (skill)
- Propose the correct resolution steps

### Metrics

| Metric | Description | Target | Actual | Status |
|--------|-------------|--------|--------|--------|
| **Top-1 Accuracy** | Correct skill is the #1 match | ≥90% | {skill_results['accuracy_metrics']['top_1_accuracy']:.1%} | {'' if skill_results['meets_targets']['top_1_accuracy'] else ''} |
| **Top-3 Accuracy** | Correct skill is in top 3 matches | ≥95% | {skill_results['accuracy_metrics']['top_3_accuracy']:.1%} | {'' if skill_results['meets_targets']['top_3_accuracy'] else ''} |
| **Mean Reciprocal Rank** | Average position of correct skill | Higher is better | {skill_results['accuracy_metrics']['mean_reciprocal_rank']:.2f} | - |

### Error Analysis

- **False Negative Rate:** {skill_results['error_analysis']['false_negative_rate']:.1%} (missed correct skills)
- **False Positive Rate:** {skill_results['error_analysis']['false_positive_rate']:.1%} (wrong skills matched)
- **Avg Confidence (Correct):** {skill_results['error_analysis']['avg_confidence_when_correct']:.2f}
- **Avg Confidence (Incorrect):** {skill_results['error_analysis']['avg_confidence_when_incorrect']:.2f}

### Results by Category

"""

    # Add skill matching category breakdown
    for category, stats in skill_results["category_breakdown"].items():
        status_emoji = "" if stats['passed'] == stats['count'] else "️" if stats['passed'] > 0 else ""
        report += f"""
#### {category.title()} Issues ({stats['count']} tests) {status_emoji}

- **Pass Rate:** {stats['passed']}/{stats['count']} ({stats['passed']/stats['count']*100:.0f}%)
- **Top-1 Accuracy:** {stats['top_1_accuracy']:.1%}
- **Top-3 Accuracy:** {stats['top_3_accuracy']:.1%}
- **Avg Confidence:** {stats['avg_confidence']:.2f}
"""

    report += f"""
---

## 3. Key Findings

### What's Working Well

"""

    # Identify what's working
    working_well = []
    for category, stats in skill_results["category_breakdown"].items():
        if stats['top_1_accuracy'] >= 0.9:
            working_well.append(f"- **{category.title()} Skill Matching:** {stats['top_1_accuracy']:.0%} accuracy ({stats['passed']}/{stats['count']} tests passed)")

    if kb_results['average_scores']['faithfulness'] >= 0.85:
        working_well.append(f"- **RAG Faithfulness:** {kb_results['average_scores']['faithfulness']:.1%} (no hallucinations)")

    if working_well:
        report += "\n".join(working_well) + "\n"
    else:
        report += "- No components currently meeting target thresholds\n"

    report += """
### ️ What Needs Improvement

"""

    # Identify what needs improvement
    needs_improvement = []

    if kb_pass_rate < 85:
        needs_improvement.append(f"- **Knowledge Base RAG:** Only {kb_pass_rate:.0f}% pass rate (target: 85%+)")
        if kb_results['average_scores']['faithfulness'] < 0.85:
            needs_improvement.append(f"  - Faithfulness: {kb_results['average_scores']['faithfulness']:.1%} (hallucination risk)")
        if kb_results['average_scores']['answer_relevancy'] < 0.90:
            needs_improvement.append(f"  - Answer Relevancy: {kb_results['average_scores']['answer_relevancy']:.1%} (answers not addressing questions)")

    for category, stats in skill_results["category_breakdown"].items():
        if stats['top_1_accuracy'] < 0.9:
            needs_improvement.append(f"- **{category.title()} Skill Matching:** Only {stats['top_1_accuracy']:.0%} accuracy (target: 90%+)")

    if skill_results['error_analysis']['avg_confidence_when_correct'] < 0.5:
        needs_improvement.append(f"- **Confidence Calibration:** Low confidence ({skill_results['error_analysis']['avg_confidence_when_correct']:.2f}) even when correct")

    if needs_improvement:
        report += "\n".join(needs_improvement) + "\n"
    else:
        report += "- All components meeting target thresholds \n"

    report += f"""
---

## 4. Recommendations

### Priority Actions

"""

    # Generate recommendations based on results
    recommendations = []

    if kb_pass_rate < 50:
        recommendations.append("""
** CRITICAL: Fix Knowledge Base RAG**
- Current pass rate: {:.0f}% (target: 85%+)
- Actions:
  1. Debug LLM service integration and prompt templates
  2. Verify context assembly and retrieval pipeline
  3. Add error handling and logging
- Timeline: Immediate (blocks system usability)
""".format(kb_pass_rate))
    elif kb_pass_rate < 85:
        recommendations.append("""
** HIGH: Improve Knowledge Base RAG**
- Current pass rate: {:.0f}% (target: 85%+)
- Actions:
  1. Review and optimize chunking strategy
  2. Tune reranking parameters
  3. Improve prompt engineering
- Timeline: 1-2 weeks
""".format(kb_pass_rate))

    for category, stats in skill_results["category_breakdown"].items():
        if stats['top_1_accuracy'] < 0.5:
            recommendations.append(f"""
** HIGH: Fix {category.title()} Skill Matching**
- Current accuracy: {stats['top_1_accuracy']:.0%} (target: 90%+)
- Actions:
  1. Expand skill trigger variations
  2. Add domain-specific terminology
  3. Re-index skills with new triggers
- Timeline: 1-2 days
""")

    if skill_results['error_analysis']['avg_confidence_when_correct'] < 0.5:
        recommendations.append(f"""
** MEDIUM: Tune Confidence Scoring**
- Current confidence: {skill_results['error_analysis']['avg_confidence_when_correct']:.2f} (low even when correct)
- Actions:
  1. Review embedding similarity thresholds
  2. Add metadata-based confidence boosting
  3. Calibrate confidence scores to accuracy
- Timeline: 2-3 days
""")

    if recommendations:
        report += "\n".join(recommendations)
    else:
        report += "- All components meeting targets - focus on maintaining quality \n"

    report += f"""
---

## 5. Technical Details

### System Configuration

- **LLM Model:** gpt-5.4-mini
- **Embedding Model:** text-embedding-3-small (OpenAI)
- **Vector Store:** ChromaDB (local)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2

### Test Coverage

- **KB RAG Tests:** {kb_total} test cases across {len(kb_results['category_breakdown'])} categories
- **Skill Matching Tests:** {skill_total} test cases across {len(skill_results['category_breakdown'])} categories

---

## Appendix: Understanding the Metrics

### RAG Metrics

- **Faithfulness:** Measures if the answer is grounded in retrieved documents (prevents hallucination)
- **Answer Relevancy:** Measures if the answer addresses the user's question
- **Context Precision:** Measures if retrieved documents are relevant to the question
- **Context Recall:** Measures if all relevant documents were retrieved

### Skill Matching Metrics

- **Top-1 Accuracy:** Percentage of times the correct skill is ranked #1
- **Top-3 Accuracy:** Percentage of times the correct skill is in top 3
- **Mean Reciprocal Rank (MRR):** Average of 1/rank for correct skill (1.0 is perfect)
- **Confidence Calibration:** How well confidence scores match actual accuracy

---

**Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Evaluation Framework Version:** 1.0  
**Results Files:**
- `evals/results/kb_rag_{timestamp}.json`
- `evals/results/skill_matching_{timestamp}.json`
"""

    # Write report to file
    with open(report_file, 'w') as f:
        f.write(report)

        logger.info(f" Generated evaluation report: {report_file}")
    return report_file



def clear_and_reindex_vector_stores():
    """
    Clear evaluation vector stores and re-index from scratch.

    This ensures each evaluation run starts with fresh, consistent data.
    Uses direct indexing instead of subprocess to avoid venv issues.
    """
    import shutil
    import os

    logger.info("\n" + "="*80)
    logger.info("CLEARING AND RE-INDEXING VECTOR STORES")
    logger.info("="*80)

    # Clear vector store directories
    vector_store_path = Path("./data/vector_store")
    if vector_store_path.exists():
        logger.info("️ Clearing existing vector store...")
        shutil.rmtree(vector_store_path)
        logger.info(" Vector store cleared")

    # Re-index knowledge base directly
    logger.info("\n Re-indexing knowledge base...")
    try:
        # Initialize embedding service
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        from src.infrastructure.embeddings.embedding_service import OpenAIEmbeddingService
        embedding_service = OpenAIEmbeddingService(
            api_key=api_key,
            model='text-embedding-3-small'
        )

        # Initialize vector store
        kb_vector_store = ChromaDBAdapter(
            persist_directory="./data/vector_store",
            collection_name="hotel_knowledge",
            distance_metric=DistanceMetric.COSINE
        )

        # Load all policies
        policies_dir = Path("./data/mock/policies")
        policy_files = list(policies_dir.glob("*.md"))
        
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for policy_file in policy_files:
            with open(policy_file, 'r') as f:
                content = f.read()

            # Simple chunking - split by paragraphs
            file_chunks = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]

            for i, chunk in enumerate(file_chunks):
                chunk_id = f"policy_chunk_{len(all_chunks)}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": policy_file.name,
                    "source_name": policy_file.stem.replace('_', ' ').title(),
                    "chunk_index": i,
                    "doc_type": "policy",
                    "domain": "hotel_operations"
                })

        # Generate embeddings
        embedding_result = embedding_service.embed_texts(all_chunks)
        embeddings = embedding_result.embeddings

        # Clear and add to vector store
        kb_vector_store.clear()
        kb_vector_store.add_documents(
            chunk_ids=all_ids,
            contents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas
        )

        count = kb_vector_store.count()
        logger.info(f" Knowledge base indexed: {count} chunks (vector store)")
        
        # Also prepare BM25 documents for later initialization
        # Store them globally so initialize_rag_pipeline can access them
        global _bm25_documents
        _bm25_documents = [
            {
                'chunk_id': all_ids[i],
                'content': all_chunks[i],
                'metadata': all_metadatas[i]
            }
            for i in range(len(all_chunks))
        ]
        logger.info(f" Prepared {len(_bm25_documents)} documents for BM25 indexing")

    except Exception as e:
        logger.error(f" Knowledge base indexing failed: {e}")
        raise RuntimeError(f"Knowledge base indexing failed: {e}")

    # Re-index skills directly (no subprocess to avoid DB locks)
    logger.info("\n Re-indexing skills...")
    try:
        # Create fresh embedding service and vector store for skills
        skills_embedding_service = EmbeddingServiceFactory.create(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            config={"model": "all-MiniLM-L6-v2"}  # 384 dimensions
        )
        
        skills_vector_store = ChromaDBAdapter(
            persist_directory="./data/vector_store",
            collection_name="hotel_skills",
            distance_metric=DistanceMetric.COSINE
        )
        
        # Create skill registry and index skills with enriched embeddings
        skill_registry = SkillRegistry(
            skills_dir="./data/skills",
            embedding_service=skills_embedding_service,
            vector_store=skills_vector_store
        )
        
        # Index all skills
        skill_registry.index_all_skills()
        
        skill_count = len(skill_registry.get_all_skills())
        logger.info(f" Skills indexed: {skill_count} skills with enriched embeddings")
        
    except Exception as e:
        logger.error(f" Skills indexing failed: {e}")
        raise RuntimeError(f"Skills indexing failed: {e}")

    # Verify indexing worked
    logger.info("\n Verifying indexing...")
    try:
        kb_count = kb_vector_store.count()
        if kb_count > 0:
            logger.info(f" Knowledge base: {kb_count} documents indexed")
        else:
            raise RuntimeError("Knowledge base has 0 documents")
        
        # Skills verification will happen when skill matcher initializes
        logger.info(" Skills: Will verify during skill matcher initialization")
    except Exception as e:
        logger.error(f" Verification failed: {e}")
        raise RuntimeError(f"Indexing verification failed: {e}")

    logger.info("\n" + "="*80)
    logger.info("RE-INDEXING COMPLETE")
    logger.info("="*80 + "\n")


def verify_indexing():
    """
    Verify that vector stores were properly indexed.

    Checks:
    1. Knowledge base collection has documents
    2. Skills collection has documents
    """
    import os
    from src.infrastructure.embeddings.embedding_service import EmbeddingServiceFactory, EmbeddingProvider
    from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter, DistanceMetric

    try:
        # Check knowledge base
        logger.info("  Checking knowledge base collection...")
        kb_vector_store = ChromaDBAdapter(
            persist_directory="./data/vector_store",
            collection_name="hotel_knowledge",
            distance_metric=DistanceMetric.COSINE
        )
        kb_count = kb_vector_store.count()
        if kb_count > 0:
            logger.info(f" Knowledge base: {kb_count} documents indexed")
        else:
            logger.error(f" Knowledge base: 0 documents found!")
            raise RuntimeError("Knowledge base indexing verification failed")

        # Check skills
        logger.info("  Checking skills collection...")
        skills_vector_store = ChromaDBAdapter(
            persist_directory="./data/vector_store",
            collection_name="hotel_skills",
            distance_metric=DistanceMetric.COSINE
        )
        skills_count = skills_vector_store.count()
        if skills_count > 0:
            logger.info(f" Skills: {skills_count} documents indexed")
        else:
            logger.error(f" Skills: 0 documents found!")
            raise RuntimeError("Skills indexing verification failed")

    except Exception as e:
        logger.error(f" Verification failed: {e}")
        raise RuntimeError(f"Indexing verification failed: {e}")


def main():
    """Run system evaluations"""
    logger.info("="*80)
    logger.info("SYSTEM EVALUATION")
    logger.info("="*80)

    try:
        # Step 1: Clear and re-index vector stores for fresh evaluation
        clear_and_reindex_vector_stores()

        # Step 2: Initialize components
        logger.info("\nInitializing components...")
        rag_pipeline = initialize_rag_pipeline()
        llm_service = LLMService(model="gpt-5.4-mini")
        rag_adapter = RAGPipelineAdapter(rag_pipeline, llm_service)

        # Run KB RAG evaluation (15 test cases)
        kb_results = run_kb_rag_evaluation(rag_adapter, num_cases=15)
        
        # Run skill matching evaluation with enriched embeddings (15 test cases)
        skill_matcher = initialize_skill_matcher()
        skill_results = run_skill_matching_evaluation(skill_matcher, num_cases=15)

        # Export results
        results_dir = Path("./evals/results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kb_file = results_dir / f"kb_rag_{timestamp}.json"
        skill_file = results_dir / f"skill_matching_{timestamp}.json"

        with open(kb_file, 'w') as f:
            json.dump(kb_results.to_dict(), f, indent=2)
        
        # Only save skill results if we ran them
        if skill_results:
            with open(skill_file, 'w') as f:
                json.dump(skill_results.to_dict(), f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("COMPLETE")
        logger.info("="*80)
        logger.info(f"\nKB RAG Results: {kb_file}")
        if skill_results:
            logger.info(f"Skill Matching Results: {skill_file}")

        # Generate markdown report (skip if no skill results)
        if skill_results:
            logger.info("\n Generating evaluation report...")
            report_path = generate_markdown_report(
                kb_results=kb_results.to_dict(),
                skill_results=skill_results.to_dict(),
                timestamp=timestamp
            )
            logger.info(f" Report saved to: {report_path}")
        else:
            logger.info("\n Skipping full report generation")
    
        return 0

    except Exception as e:
        logger.error(f"\nFailed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
