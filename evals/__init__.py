"""
Evaluation Framework for Customer Issue Resolution Copilot

This module provides comprehensive evaluation capabilities for:
- RAG metrics (faithfulness, answer relevancy, context precision, context recall)
- Agent metrics (skill match accuracy, tool call correctness, plan quality)
- Approval flow metrics (approval rate, rejection reasons, time to approval)
- Guardrail effectiveness (catch rate on red-team cases)
- End-to-end workflow tests (known skill, novel task, skill reuse)

Collection Names:
- hotel_skills: Main collection for RAG retrieval and skill trigger embeddings
- hotel_knowledge_base: Default collection name (if not overridden)
"""

__version__ = "0.1.0"

from evals.llm_judge import LLMJudge
from evals.knowledge_base_rag_evaluator import KnowledgeBaseRAGEvaluator
from evals.skill_matching_rag_evaluator import SkillMatchingEvaluator

__all__ = [
    "LLMJudge",
    "KnowledgeBaseRAGEvaluator",
    "SkillMatchingEvaluator",
]
