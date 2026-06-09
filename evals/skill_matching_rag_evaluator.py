"""
Skill Matching Evaluator

Evaluates the Skill Matching system (semantic search over skill triggers).
Tests whether the system correctly matches customer issues to existing skills.

Metrics:
- Match Accuracy: Percentage of correct skill matches (top-1 accuracy)
- Top-K Accuracy: Percentage where correct skill is in top-K results
- Confidence Calibration: How well confidence scores align with correctness
- False Positive Rate: Incorrect high-confidence matches
- False Negative Rate: Missed matches (should have matched but didn't)
- Mean Reciprocal Rank (MRR): Average of 1/rank for correct matches

Targets:
- Match Accuracy (top-1): ≥90%
- Top-3 Accuracy: ≥95%
- Confidence Calibration: Well-calibrated (high confidence → high accuracy)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SkillMatchTestCase:
    """A single skill matching test case"""
    issue_description: str
    expected_skill_id: Optional[str]  # None if no skill should match
    expected_skill_name: Optional[str] = None
    should_match: bool = True  # False for negative test cases
    category: str = "general"  # booking, billing, amenity, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillMatchEvalResult:
    """Result of evaluating a single skill match test case"""
    issue_description: str
    expected_skill_id: Optional[str]
    expected_skill_name: Optional[str]
    
    # Matching results
    matched_skill_id: Optional[str]
    matched_skill_name: Optional[str]
    match_confidence: float
    match_score: float
    
    # Top-K results
    top_k_skill_ids: List[str]
    top_k_scores: List[float]
    
    # Evaluation metrics
    correct_match: bool  # Top-1 is correct
    correct_in_top_3: bool
    correct_in_top_5: bool
    rank_of_correct: Optional[int]  # None if not found
    
    # For negative cases (should not match)
    should_match: bool
    false_positive: bool  # Matched when shouldn't
    false_negative: bool  # Didn't match when should
    
    category: str
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillMatchEvalSummary:
    """Summary of skill matching evaluation"""
    total_cases: int
    passed_cases: int
    failed_cases: int
    
    # Accuracy metrics
    top_1_accuracy: float  # Target: ≥0.90
    top_3_accuracy: float  # Target: ≥0.95
    top_5_accuracy: float
    mean_reciprocal_rank: float  # MRR
    
    # Error analysis
    false_positive_rate: float
    false_negative_rate: float
    avg_confidence_when_correct: float
    avg_confidence_when_incorrect: float
    
    # Confidence calibration
    confidence_bins: Dict[str, Dict[str, float]]  # Binned confidence vs accuracy
    
    # Category breakdown
    results_by_category: Dict[str, List[SkillMatchEvalResult]]
    
    # Detailed results
    results: List[SkillMatchEvalResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        category_stats = {}
        for category, cat_results in self.results_by_category.items():
            if cat_results:
                category_stats[category] = {
                    "count": len(cat_results),
                    "passed": sum(1 for r in cat_results if r.passed),
                    "top_1_accuracy": sum(1 for r in cat_results if r.correct_match) / len(cat_results),
                    "top_3_accuracy": sum(1 for r in cat_results if r.correct_in_top_3) / len(cat_results),
                    "avg_confidence": sum(r.match_confidence for r in cat_results) / len(cat_results)
                }
        
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "pass_rate": self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0
            },
            "accuracy_metrics": {
                "top_1_accuracy": self.top_1_accuracy,
                "top_3_accuracy": self.top_3_accuracy,
                "top_5_accuracy": self.top_5_accuracy,
                "mean_reciprocal_rank": self.mean_reciprocal_rank
            },
            "error_analysis": {
                "false_positive_rate": self.false_positive_rate,
                "false_negative_rate": self.false_negative_rate,
                "avg_confidence_when_correct": self.avg_confidence_when_correct,
                "avg_confidence_when_incorrect": self.avg_confidence_when_incorrect
            },
            "targets": {
                "top_1_accuracy": 0.90,
                "top_3_accuracy": 0.95
            },
            "meets_targets": {
                "top_1_accuracy": self.top_1_accuracy >= 0.90,
                "top_3_accuracy": self.top_3_accuracy >= 0.95
            },
            "confidence_calibration": self.confidence_bins,
            "category_breakdown": category_stats
        }


class SkillMatchingEvaluator:
    """
    Evaluates Skill Matching system performance.
    
    Tests:
    - Correct skill identification for known patterns
    - Handling of paraphrased requests
    - Rejection of out-of-scope requests
    - Confidence calibration
    """
    
    def __init__(
        self,
        top_1_accuracy_threshold: float = 0.90,
        top_3_accuracy_threshold: float = 0.95,
        confidence_threshold: float = 0.45  # From SkillMatcher HIGH confidence
    ):
        """
        Initialize skill matching evaluator.
        
        Args:
            top_1_accuracy_threshold: Minimum top-1 accuracy (default: 0.90)
            top_3_accuracy_threshold: Minimum top-3 accuracy (default: 0.95)
            confidence_threshold: Confidence threshold for accepting matches (default: 0.45)
        """
        self.top_1_accuracy_threshold = top_1_accuracy_threshold
        self.top_3_accuracy_threshold = top_3_accuracy_threshold
        self.confidence_threshold = confidence_threshold
    
    def evaluate_single(
        self,
        test_case: SkillMatchTestCase,
        match_results: List[Dict[str, Any]]
    ) -> SkillMatchEvalResult:
        """
        Evaluate a single skill matching test case.
        
        Args:
            test_case: The test case with issue and expected skill
            match_results: List of skill matches from SkillMatcher
                          Each should have: {skill_id, skill_name, confidence, score}
        
        Returns:
            SkillMatchEvalResult with all metrics
        """
        # Extract top match
        top_match = match_results[0] if match_results else None
        matched_skill_id = top_match.get("skill_id") if top_match else None
        matched_skill_name = top_match.get("skill_name") if top_match else None
        # Use numeric match_confidence if available, otherwise convert string confidence
        match_confidence = top_match.get("match_confidence", 0.0) if top_match else 0.0
        if not isinstance(match_confidence, (int, float)):
            confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            match_confidence = confidence_map.get(str(match_confidence).lower(), 0.5)
        match_score = top_match.get("score", 0.0) if top_match else 0.0
        
        # Extract top-K (filter out None values)
        top_k_skill_ids = [m.get("skill_id") for m in match_results[:5] if m.get("skill_id")]
        top_k_scores = [m.get("score", 0.0) for m in match_results[:5]]
        
        # Evaluate correctness
        correct_match = (matched_skill_id == test_case.expected_skill_id)
        correct_in_top_3 = test_case.expected_skill_id in top_k_skill_ids[:3]
        correct_in_top_5 = test_case.expected_skill_id in top_k_skill_ids[:5]
        
        # Find rank of correct skill
        rank_of_correct = None
        if test_case.expected_skill_id:
            try:
                rank_of_correct = top_k_skill_ids.index(test_case.expected_skill_id) + 1
            except ValueError:
                rank_of_correct = None  # Not in top-5
        
        # Evaluate false positives/negatives
        false_positive = False
        false_negative = False
        
        if not test_case.should_match:
            # Negative case: should NOT match any skill
            # Check if confidence is high (string) or score is above threshold (float)
            high_confidence = (
                (isinstance(match_confidence, str) and match_confidence == "high") or
                (isinstance(match_confidence, (int, float)) and match_confidence >= self.confidence_threshold)
            )
            if matched_skill_id and high_confidence:
                false_positive = True
        else:
            # Positive case: should match expected skill
            if not correct_match:
                false_negative = True
        
        # Determine if test passed
        if test_case.should_match:
            # Positive case: must match correctly
            passed = correct_match
        else:
            # Negative case: must not match with high confidence
            passed = not false_positive
        
        return SkillMatchEvalResult(
            issue_description=test_case.issue_description,
            expected_skill_id=test_case.expected_skill_id,
            expected_skill_name=test_case.expected_skill_name,
            matched_skill_id=matched_skill_id,
            matched_skill_name=matched_skill_name,
            match_confidence=match_confidence,
            match_score=match_score,
            top_k_skill_ids=top_k_skill_ids,
            top_k_scores=top_k_scores,
            correct_match=correct_match,
            correct_in_top_3=correct_in_top_3,
            correct_in_top_5=correct_in_top_5,
            rank_of_correct=rank_of_correct,
            should_match=test_case.should_match,
            false_positive=false_positive,
            false_negative=false_negative,
            category=test_case.category,
            passed=passed,
            metadata=test_case.metadata
        )
    
    def evaluate_batch(
        self,
        test_cases: List[SkillMatchTestCase],
        skill_matcher: Any  # SkillMatcher instance
    ) -> SkillMatchEvalSummary:
        """
        Evaluate multiple skill matching test cases.
        
        Args:
            test_cases: List of test cases
            skill_matcher: SkillMatcher instance to test
        
        Returns:
            SkillMatchEvalSummary with aggregate metrics
        """
        results = []
        
        for test_case in test_cases:
            try:
                # Create mock issue for matching
                from src.domain.models.issue import Issue, IssueChannel, IssuePriority
                mock_issue = Issue(
                    issue_id=f"test_{len(results)}",
                    channel=IssueChannel.EMAIL,
                    body=test_case.issue_description,
                    priority=IssuePriority.MEDIUM,
                    guest_email="test@example.com",
                    booking_id=None,
                    subject="Test issue",
                    issue_type=None,
                    expected_skill=test_case.expected_skill_id,
                    expected_resolution=None
                )
                
                # Run skill matcher
                matches = skill_matcher.match_skill(
                    issue=mock_issue,
                    top_k=5
                )
                
                # Convert SkillMatch objects to dicts
                # Handle both string confidence ("high"/"medium"/"low") and numeric
                confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
                match_dicts = [
                    {
                        "skill_id": m.skill.skill_id,
                        "skill_name": m.skill.name,
                        "confidence": m.confidence,  # Keep original for display
                        "match_confidence": (
                            confidence_map.get(m.confidence.lower(), 0.5)
                            if isinstance(m.confidence, str)
                            else m.confidence
                        ),
                        "score": m.score
                    }
                    for m in matches
                ]
                
                # Evaluate result
                eval_result = self.evaluate_single(
                    test_case=test_case,
                    match_results=match_dicts
                )
                results.append(eval_result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case: {e}")
                # Create failed result
                results.append(SkillMatchEvalResult(
                    issue_description=test_case.issue_description,
                    expected_skill_id=test_case.expected_skill_id,
                    expected_skill_name=test_case.expected_skill_name,
                    matched_skill_id=None,
                    matched_skill_name=None,
                    match_confidence=0.0,
                    match_score=0.0,
                    top_k_skill_ids=[],
                    top_k_scores=[],
                    correct_match=False,
                    correct_in_top_3=False,
                    correct_in_top_5=False,
                    rank_of_correct=None,
                    should_match=test_case.should_match,
                    false_positive=False,
                    false_negative=test_case.should_match,
                    category=test_case.category,
                    passed=False,
                    metadata={"error": str(e)}
                ))
        
        # Calculate summary statistics
        return self._create_summary(results)
    
    def _create_summary(self, results: List[SkillMatchEvalResult]) -> SkillMatchEvalSummary:
        """Create summary statistics from evaluation results"""
        if not results:
            return SkillMatchEvalSummary(
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                top_1_accuracy=0.0,
                top_3_accuracy=0.0,
                top_5_accuracy=0.0,
                mean_reciprocal_rank=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                avg_confidence_when_correct=0.0,
                avg_confidence_when_incorrect=0.0,
                confidence_bins={},
                results_by_category={},
                results=[]
            )
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        # Accuracy metrics
        positive_cases = [r for r in results if r.should_match]
        if positive_cases:
            top_1_accuracy = sum(1 for r in positive_cases if r.correct_match) / len(positive_cases)
            top_3_accuracy = sum(1 for r in positive_cases if r.correct_in_top_3) / len(positive_cases)
            top_5_accuracy = sum(1 for r in positive_cases if r.correct_in_top_5) / len(positive_cases)
            
            # Mean Reciprocal Rank
            reciprocal_ranks = [
                1.0 / r.rank_of_correct if r.rank_of_correct else 0.0
                for r in positive_cases
            ]
            mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
        else:
            top_1_accuracy = 0.0
            top_3_accuracy = 0.0
            top_5_accuracy = 0.0
            mean_reciprocal_rank = 0.0
        
        # Error rates
        negative_cases = [r for r in results if not r.should_match]
        false_positive_rate = sum(1 for r in negative_cases if r.false_positive) / len(negative_cases) if negative_cases else 0.0
        false_negative_rate = sum(1 for r in positive_cases if r.false_negative) / len(positive_cases) if positive_cases else 0.0
        
        # Confidence analysis
        correct_results = [r for r in positive_cases if r.correct_match]
        incorrect_results = [r for r in positive_cases if not r.correct_match]
        
        avg_confidence_when_correct = (
            sum(r.match_confidence for r in correct_results) / len(correct_results)
            if correct_results else 0.0
        )
        avg_confidence_when_incorrect = (
            sum(r.match_confidence for r in incorrect_results) / len(incorrect_results)
            if incorrect_results else 0.0
        )
        
        # Confidence calibration bins
        confidence_bins = self._calculate_confidence_bins(positive_cases)
        
        # Group by category
        results_by_category: Dict[str, List[SkillMatchEvalResult]] = {}
        for result in results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)
        
        return SkillMatchEvalSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            top_1_accuracy=top_1_accuracy,
            top_3_accuracy=top_3_accuracy,
            top_5_accuracy=top_5_accuracy,
            mean_reciprocal_rank=mean_reciprocal_rank,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            avg_confidence_when_correct=avg_confidence_when_correct,
            avg_confidence_when_incorrect=avg_confidence_when_incorrect,
            confidence_bins=confidence_bins,
            results_by_category=results_by_category,
            results=results
        )
    
    def _calculate_confidence_bins(
        self,
        results: List[SkillMatchEvalResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence calibration bins.
        
        Groups results by confidence ranges and calculates accuracy within each bin.
        Well-calibrated system: 80% confidence → 80% accuracy
        """
        bins = {
            "0.0-0.2": [],
            "0.2-0.4": [],
            "0.4-0.6": [],
            "0.6-0.8": [],
            "0.8-1.0": []
        }
        
        for result in results:
            conf = result.match_confidence
            if conf < 0.2:
                bins["0.0-0.2"].append(result)
            elif conf < 0.4:
                bins["0.2-0.4"].append(result)
            elif conf < 0.6:
                bins["0.4-0.6"].append(result)
            elif conf < 0.8:
                bins["0.6-0.8"].append(result)
            else:
                bins["0.8-1.0"].append(result)
        
        calibration = {}
        for bin_name, bin_results in bins.items():
            if bin_results:
                accuracy = sum(1 for r in bin_results if r.correct_match) / len(bin_results)
                avg_confidence = sum(r.match_confidence for r in bin_results) / len(bin_results)
                calibration[bin_name] = {
                    "count": len(bin_results),
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                    "calibration_error": abs(accuracy - avg_confidence)
                }
        
        return calibration
    
    def save_results(self, summary: SkillMatchEvalSummary, output_path: str) -> None:
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        logger.info(f"Saved skill matching evaluation results to {output_path}")
    
    @staticmethod
    def load_test_cases_from_file(file_path: str) -> List[SkillMatchTestCase]:
        """
        Load test cases from JSON file.
        
        Expected format:
        [
            {
                "issue_description": "I need to checkout 2 hours late",
                "expected_skill_id": "late_checkout",
                "expected_skill_name": "Late Checkout Request",
                "should_match": true,
                "category": "booking",
                "metadata": {}
            },
            ...
        ]
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return [
            SkillMatchTestCase(
                issue_description=item["issue_description"],
                expected_skill_id=item.get("expected_skill_id"),
                expected_skill_name=item.get("expected_skill_name"),
                should_match=item.get("should_match", True),
                category=item.get("category", "general"),
                metadata=item.get("metadata", {})
            )
            for item in data
        ]
