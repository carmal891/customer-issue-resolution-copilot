"""
Confidence Checker for RAG Retrieval

Escalates to human when RAG retrieval confidence is too low or no relevant
policies are found. Prevents the system from hallucinating or making decisions
without proper grounding.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for retrieval results."""
    HIGH = "high"  # > 0.7
    MEDIUM = "medium"  # 0.3 - 0.7
    LOW = "low"  # < 0.3
    NONE = "none"  # No results


@dataclass
class ConfidenceCheckResult:
    """Result of confidence check."""
    confidence_level: ConfidenceLevel
    should_escalate: bool
    reason: str
    retrieval_score: Optional[float] = None
    num_results: int = 0
    recommendation: Optional[str] = None


class ConfidenceChecker:
    """
    Checks RAG retrieval confidence and escalates when necessary.

    Prevents the system from proceeding with low-quality or missing context.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.3
    MIN_RESULTS_REQUIRED = 1

    def __init__(
        self,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
        min_results: int = 1
    ):
        """
        Initialize confidence checker.

        Args:
            high_threshold: Score above which confidence is HIGH
            low_threshold: Score below which confidence is LOW
            min_results: Minimum number of results required
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_results = min_results
        self.logger = logging.getLogger(__name__)

    def check_retrieval_confidence(
        self,
        retrieval_results: List[Dict[str, Any]],
        query: str
    ) -> ConfidenceCheckResult:
        """
        Check confidence of RAG retrieval results.

        Args:
            retrieval_results: List of retrieved documents with scores
            query: Original query

        Returns:
            ConfidenceCheckResult with escalation decision
        """
        # No results found
        if not retrieval_results or len(retrieval_results) == 0:
            self.logger.warning(f"No retrieval results for query: {query}")
            return ConfidenceCheckResult(
                confidence_level=ConfidenceLevel.NONE,
                should_escalate=True,
                reason="No relevant policies or procedures found",
                num_results=0,
                recommendation="Escalate to human agent for manual policy lookup"
            )

        # Get top result score
        top_score = self._get_top_score(retrieval_results)
        num_results = len(retrieval_results)

        # Determine confidence level
        if top_score >= self.high_threshold:
            confidence_level = ConfidenceLevel.HIGH
            should_escalate = False
            reason = f"High confidence retrieval (score: {top_score:.2f})"
            recommendation = "Proceed with retrieved context"

        elif top_score >= self.low_threshold:
            confidence_level = ConfidenceLevel.MEDIUM
            should_escalate = False
            reason = f"Medium confidence retrieval (score: {top_score:.2f})"
            recommendation = "Proceed with caution, verify with human if action is consequential"

        else:
            confidence_level = ConfidenceLevel.LOW
            should_escalate = True
            reason = f"Low confidence retrieval (score: {top_score:.2f})"
            recommendation = "Escalate to human for verification"

        # Additional check: too few results
        if num_results < self.min_results:
            should_escalate = True
            reason += f" and insufficient results ({num_results} < {self.min_results})"

        self.logger.info(
            f"Confidence check: {confidence_level.value} "
            f"(score: {top_score:.2f}, results: {num_results})"
        )

        return ConfidenceCheckResult(
            confidence_level=confidence_level,
            should_escalate=should_escalate,
            reason=reason,
            retrieval_score=top_score,
            num_results=num_results,
            recommendation=recommendation
        )

    def check_conflicting_sources(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> ConfidenceCheckResult:
        """
        Check if retrieved sources contain conflicting information.

        Args:
            retrieval_results: List of retrieved documents

        Returns:
            ConfidenceCheckResult indicating if escalation is needed
        """
        if len(retrieval_results) < 2:
            return ConfidenceCheckResult(
                confidence_level=ConfidenceLevel.HIGH,
                should_escalate=False,
                reason="Single source, no conflicts possible",
                num_results=len(retrieval_results)
            )

        # Check for conflicting policy dates or versions
        # In a real system, this would use semantic similarity to detect conflicts
        # For POC, we check metadata

        sources = set()
        doc_types = set()

        for result in retrieval_results:
            metadata = result.get('metadata', {})
            sources.add(metadata.get('source', 'unknown'))
            doc_types.add(metadata.get('doc_type', 'unknown'))

        # If results come from very different sources, flag for review
        if len(sources) > 3:
            self.logger.warning(
                f"Retrieved results from {len(sources)} different sources"
            )
            return ConfidenceCheckResult(
                confidence_level=ConfidenceLevel.MEDIUM,
                should_escalate=True,
                reason=f"Multiple conflicting sources detected ({len(sources)} sources)",
                num_results=len(retrieval_results),
                recommendation="Human should verify which policy applies"
            )

        return ConfidenceCheckResult(
            confidence_level=ConfidenceLevel.HIGH,
            should_escalate=False,
            reason="No obvious conflicts detected",
            num_results=len(retrieval_results)
        )

    def should_proceed_with_action(
        self,
        confidence_result: ConfidenceCheckResult,
        action_risk_level: str
    ) -> bool:
        """
        Determine if system should proceed with an action given confidence level.

        Args:
            confidence_result: Result from confidence check
            action_risk_level: Risk level of the action (low, medium, high)

        Returns:
            True if system should proceed, False if should escalate
        """
        # Never proceed with high-risk actions on low confidence
        if action_risk_level == "high" and confidence_result.confidence_level == ConfidenceLevel.LOW:
            return False

        # Never proceed with no results
        if confidence_result.confidence_level == ConfidenceLevel.NONE:
            return False

        # Medium risk requires at least medium confidence
        if action_risk_level == "medium" and confidence_result.confidence_level == ConfidenceLevel.LOW:
            return False

        # Low risk can proceed with medium confidence
        return True

    def _get_top_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Get the highest score from retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Highest score, or 0.0 if no scores found
        """
        scores = []

        for result in results:
            # Try different possible score keys
            score = result.get('score') or result.get('relevance_score') or result.get('similarity')
            if score is not None:
                scores.append(float(score))

        return max(scores) if scores else 0.0

    def format_escalation_message(
        self,
        confidence_result: ConfidenceCheckResult,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        Format a human-readable escalation message.

        Args:
            confidence_result: Confidence check result
            query: Original query
            context: Additional context

        Returns:
            Formatted escalation message
        """
        message = f"""
️ LOW CONFIDENCE - HUMAN REVIEW REQUIRED

Query: {query}

Confidence Level: {confidence_result.confidence_level.value.upper()}
Reason: {confidence_result.reason}

Retrieved Results: {confidence_result.num_results}
Top Score: {confidence_result.retrieval_score:.2f if confidence_result.retrieval_score else 'N/A'}

Recommendation: {confidence_result.recommendation}

{f'Context: {context}' if context else ''}

Please review and provide guidance on how to proceed.
"""
        return message.strip()


# Singleton instance
_confidence_checker_instance: Optional[ConfidenceChecker] = None


def get_confidence_checker() -> ConfidenceChecker:
    """Get singleton confidence checker instance."""
    global _confidence_checker_instance
    if _confidence_checker_instance is None:
        _confidence_checker_instance = ConfidenceChecker()
    return _confidence_checker_instance
