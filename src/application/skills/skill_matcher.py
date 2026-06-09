"""
Skill Matcher

Matches customer issues to existing skills using semantic search, re-ranking, and structured matching.
Implements hybrid matching strategy: embedding similarity + re-ranking + metadata filtering.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.domain.models.skill import Skill
from src.domain.models.issue import Issue
from src.application.skills.skill_registry import SkillRegistry
from src.infrastructure.embeddings.embedding_service import IEmbeddingService
from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter
from src.application.rag.reranking import CrossEncoderReranker

# Query reformulation prompt for better semantic matching
QUERY_REFORMULATION_PROMPT = """You are a query reformulation assistant for a hotel customer service system.

Rephrase the user's query to make it more suitable for semantic search and skill matching.

Guidelines:
1. Extract core intent and action
2. Use standard hotel terminology
3. Be concise (1-2 sentences max)
4. Remove filler words
5. Preserve important details (times, dates, amounts)

Examples:

User Query: "i want to checkout late by 2hrs"
Reformulated: "Request late checkout extension for 2 hours"

User Query: "my room is too small, can i get a bigger one?"
Reformulated: "Request room upgrade to larger room type"

User Query: "Can you arrange a helicopter transfer to the hotel?"
Reformulated: "Request helicopter transfer service to hotel"

Now reformulate this query:

User Query: {user_query}
Reformulated:"""

logger = logging.getLogger(__name__)


class MatchConfidence(str, Enum):
    """Confidence levels for skill matching (adjusted for realistic embedding similarity)"""
    HIGH = "high"  # >=0.45 (realistic threshold for related concepts)
    MEDIUM = "medium"  # 0.35-0.45
    LOW = "low"  # 0.25-0.35
    NONE = "none"  # <0.25 or no match found


@dataclass
class SkillMatch:
    """Result of skill matching"""
    skill: Skill
    confidence: str
    score: float
    matched_trigger: str
    metadata_boost: float = 0.0


class SkillMatcher:
    """
    Matches issues to skills using hybrid approach with re-ranking.

    Matching Strategy:
    1. Semantic search: Embed issue description, search skill triggers
    2. Re-ranking: Use cross-encoder to re-rank candidates
    3. Metadata filtering: Domain, category, keywords
    4. Hybrid scoring: 60% rerank + 40% metadata
    5. Confidence thresholding: High/Medium/Low/None

    Responsibilities:
    - Embed issue descriptions
    - Search skill trigger embeddings
    - Re-rank candidates for better precision
    - Apply metadata filters
    - Calculate hybrid match scores
    - Return ranked matches with confidence
    """

    def __init__(
        self,
        skill_registry: SkillRegistry,
        embedding_service: IEmbeddingService,
        vector_store: ChromaDBAdapter,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_service: Optional[Any] = None,
        semantic_weight: float = 0.6,
        metadata_weight: float = 0.4,
        high_confidence_threshold: float = 0.60,
        medium_confidence_threshold: float = 0.45,
        enable_query_reformulation: bool = True
    ):
        """
        Initialize skill matcher.

        Args:
            skill_registry: Registry for accessing skills
            embedding_service: Service for generating embeddings
            vector_store: Vector store with indexed skill triggers
            reranker: Cross-encoder reranker for better precision (optional)
            llm_service: LLM service for query reformulation (optional)
            semantic_weight: Weight for semantic similarity (default 0.6)
            metadata_weight: Weight for metadata matching (default 0.4)
            high_confidence_threshold: Threshold for high confidence (default 0.45, realistic for embeddings)
            medium_confidence_threshold: Threshold for medium confidence (default 0.35)
            enable_query_reformulation: Enable LLM-based query reformulation (default True)
        """
        self.skill_registry = skill_registry
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker  # Optional reranker
        self.llm_service = llm_service  # Optional LLM for query reformulation
        self.semantic_weight = semantic_weight
        self.metadata_weight = metadata_weight
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.enable_query_reformulation = enable_query_reformulation

    def match_skill(
        self,
        issue: Issue,
        top_k: int = 5,
        min_confidence: Optional[MatchConfidence] = MatchConfidence.LOW
    ) -> List[SkillMatch]:
        """
        Find matching skills for an issue.

        Args:
            issue: Customer issue to match
            top_k: Number of top matches to return
            min_confidence: Minimum confidence level to include

        Returns:
            List of skill matches, ranked by confidence
        """
        try:
            # Step 1: Prepare issue text
            issue_text = self._prepare_issue_text(issue)

            # Step 1.5: Reformulate query for better semantic matching (optional)
            reformulated_text = self._reformulate_query(issue_text)

            # Fallback to original if reformulation returns None
            if reformulated_text is None or reformulated_text.strip() == "":
                reformulated_text = issue_text

            # Log reformulation for debugging
            if reformulated_text != issue_text:
                logger.info(f"Query reformulated from '{issue_text}' to '{reformulated_text}'")

            # Step 2: Hybrid Search - Semantic + Keyword
            # 2a. Semantic search using embeddings (use reformulated query)
            embedding_result = self.embedding_service.embed_texts([reformulated_text])
            issue_embedding = embedding_result.embeddings[0]

            semantic_results = self.vector_store.search(
                query_embedding=issue_embedding,
                n_results=top_k * 3,  # Get 3x candidates
                where={"$and": [
                    {"doc_type": {"$eq": "skill_trigger"}},
                    {"active": {"$eq": True}}
                ]}
            )

            # 2b. Keyword search - get all active skills and score by keyword overlap
            all_skills = self.skill_registry.get_all_skills(active_only=True)
            keyword_scores = self._keyword_search(issue_text, all_skills)

            # Step 3: Merge and deduplicate results from both searches
            merged_candidates = self._merge_search_results(
                semantic_results,
                keyword_scores,
                all_skills
            )

            if not merged_candidates:
                logger.info(f"No skill triggers found for issue {issue.issue_id}")
                return []

            # Step 4: Use merged candidates directly (already prepared)
            candidates = merged_candidates

            if not candidates:
                logger.info(f"No active skills found for issue {issue.issue_id}")
                return []

            # Step 4: Re-rank candidates using cross-encoder (if available)
            if self.reranker:
                try:
                    # Use cross-encoder to score pairs directly
                    rerank_pairs = [(issue_text, c['trigger_text']) for c in candidates]
                    self.reranker._load_model()
                    if self.reranker.model:
                        raw_scores = self.reranker.model.predict(rerank_pairs)
                        # Normalize scores to 0-1 range using min-max normalization
                        min_score = float(min(raw_scores))
                        max_score = float(max(raw_scores))
                        score_range = max_score - min_score
                        if score_range > 0:
                            rerank_scores = [(float(s) - min_score) / score_range for s in raw_scores]
                        else:
                            rerank_scores = [0.5 for _ in raw_scores]
                    else:
                        rerank_scores = [c['semantic_score'] for c in candidates]
                except Exception as e:
                    logger.warning(f"Reranking failed, using semantic scores: {e}")
                    rerank_scores = [c['semantic_score'] for c in candidates]
            else:
                # Fall back to semantic scores if no reranker
                rerank_scores = [c['semantic_score'] for c in candidates]

            # Step 5: Create matches using pre-calculated hybrid scores
            matches = []
            seen_skills = set()

            for i, candidate in enumerate(candidates):
                skill_id = candidate['skill_id']

                # Skip duplicate skills (keep highest scoring trigger)
                if skill_id in seen_skills:
                    continue
                seen_skills.add(skill_id)

                skill = candidate['skill']

                # Use the hybrid score already calculated in _merge_search_results
                # which combines semantic similarity (60%) and keyword overlap (40%)
                hybrid_score = candidate.get('semantic_score', 0.0)

                # Determine confidence level
                confidence = self._determine_confidence(hybrid_score)

                # Apply minimum confidence filter
                if min_confidence:
                    # Convert string confidence to enum for comparison
                    try:
                        confidence_enum = MatchConfidence(confidence)
                        if self._confidence_rank(confidence_enum) < self._confidence_rank(min_confidence):
                            continue
                    except ValueError:
                        # If confidence string is invalid, skip this match
                        continue

                match = SkillMatch(
                    skill=skill,
                    confidence=confidence,
                    score=hybrid_score,
                    matched_trigger=candidate['trigger_text'],
                    metadata_boost=0.0  # No metadata boost used
                )
                matches.append(match)

            # Step 6: Sort by score and return top_k
            matches.sort(key=lambda m: m.score, reverse=True)
            return matches[:top_k]

        except Exception as e:
            logger.error(f"Failed to match skills for issue {issue.issue_id}: {e}")
            return []

    def match_best_skill(
        self,
        issue: Issue,
        min_confidence: Optional[MatchConfidence] = MatchConfidence.MEDIUM
    ) -> Optional[SkillMatch]:
        """
        Find the single best matching skill.

        Args:
            issue: Customer issue to match
            min_confidence: Minimum confidence required

        Returns:
            Best skill match or None if no match meets threshold
        """
        matches = self.match_skill(issue, top_k=1, min_confidence=min_confidence)
        return matches[0] if matches else None

    def _prepare_issue_text(self, issue: Issue) -> str:
        """
        Prepare issue text for embedding.

        Combines subject, body, and relevant metadata.

        Args:
            issue: Customer issue

        Returns:
            Combined text for embedding
        """
        parts = []

        # Add subject if available
        if issue.subject:
            parts.append(issue.subject)

        # Add body (required field)
        if issue.body:
            parts.append(issue.body)

        # Add issue type if available
        if issue.issue_type:
            parts.append(f"Type: {issue.issue_type.value}")

        return " ".join(parts)

    def _reformulate_query(self, query_text: str) -> str:
        """
        Reformulate user query for better semantic matching using LLM.

        Args:
            query_text: Original user query

        Returns:
            Reformulated query optimized for semantic search, or original if reformulation fails
        """
        # Skip if reformulation is disabled or LLM service not available
        if not self.enable_query_reformulation or not self.llm_service:
            return query_text

        try:
            # Call LLM to reformulate query
            user_prompt = query_text
            response = self.llm_service.generate(
                system_prompt=QUERY_REFORMULATION_PROMPT.replace("{user_query}", ""),
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for consistent reformulation
                max_completion_tokens=100
            )

            reformulated = response.content.strip()

            # Validate reformulation (should be shorter and focused)
            if reformulated and len(reformulated) > 0 and len(reformulated) < len(query_text) * 2:
                logger.info(f"Query reformulated: '{query_text}' -> '{reformulated}'")
                return reformulated
            else:
                logger.warning(f"Invalid reformulation, using original query")
                return query_text

        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}, using original query")
            return query_text

    def _calculate_metadata_boost(self, issue: Issue, skill: Skill) -> float:
        """
        Calculate metadata matching boost.

        Considers:
        - Domain/category alignment
        - Keyword matches
        - Issue type alignment

        Args:
            issue: Customer issue
            skill: Skill to match

        Returns:
            Boost score between 0.0 and 1.0
        """
        boost = 0.0
        boost_count = 0

        # Check domain match
        issue_domain = issue.metadata.get('domain')
        skill_domain = skill.metadata.get('domain')
        if issue_domain and skill_domain and issue_domain == skill_domain:
            boost += 1.0
        boost_count += 1

        # Check category match
        issue_category = issue.metadata.get('category')
        skill_category = skill.metadata.get('category')
        if issue_category and skill_category and issue_category == skill_category:
            boost += 1.0
        boost_count += 1

        # Check issue type alignment
        if issue.issue_type:
            issue_type_str = issue.issue_type.value.lower()
            skill_name_lower = skill.name.lower()
            if issue_type_str in skill_name_lower or any(issue_type_str in t.lower() for t in skill.triggers):
                boost += 1.0
        boost_count += 1

        # Return average boost
        return boost / boost_count if boost_count > 0 else 0.0

    def _determine_confidence(self, score: float) -> str:
        """
        Determine confidence level from score.

        Args:
            score: Hybrid match score

        Returns:
            Confidence level
        """
        if score >= self.high_confidence_threshold:
            return MatchConfidence.HIGH.value
        elif score >= self.medium_confidence_threshold:
            return MatchConfidence.MEDIUM.value
        else:
            return MatchConfidence.LOW.value

    def _confidence_rank(self, confidence: MatchConfidence) -> int:
        """
        Get numeric rank for confidence level.

        Args:
            confidence: Confidence level

        Returns:
            Numeric rank (higher is better)
        """
        ranks = {
            MatchConfidence.NONE: 0,
            MatchConfidence.LOW: 1,
            MatchConfidence.MEDIUM: 2,
            MatchConfidence.HIGH: 3
        }
        return ranks.get(confidence, 0)
    def _keyword_search(self, query_text: str, skills: List[Skill]) -> Dict[str, float]:
        """
        Perform keyword-based search with phrase matching and BM25-like scoring.

        Args:
            query_text: Query text from issue
            skills: List of skills to search

        Returns:
            Dictionary mapping skill_id to keyword score (0-1)
        """
        query_lower = query_text.lower()
        query_words = set(query_lower.split())
        scores = {}

        for skill in skills:
            score = 0.0

            # Check for exact phrase matches in triggers (highest priority)
            for trigger in skill.triggers:
                trigger_lower = trigger.lower()
                # Exact phrase match - give perfect score
                if query_lower in trigger_lower or trigger_lower in query_lower:
                    score = max(score, 1.0)
                    break

            # Check for keyword matches in skill name
            skill_name_lower = skill.name.lower()
            name_words = set(skill_name_lower.split())
            name_intersection = query_words & name_words
            if name_intersection:
                name_score = len(name_intersection) / max(len(query_words), len(name_words))
                score = max(score, name_score * 0.7)

            # Check for keyword matches in triggers
            for trigger in skill.triggers:
                trigger_lower = trigger.lower()
                trigger_words = set(trigger_lower.split())
                trigger_intersection = query_words & trigger_words

                if trigger_intersection:
                    # Calculate overlap ratio
                    overlap_ratio = len(trigger_intersection) / max(len(query_words), len(trigger_words))
                    trigger_score = overlap_ratio * 0.8
                    score = max(score, trigger_score)

            # Boost for issue type alignment
            if hasattr(skill, 'metadata') and skill.metadata:
                skill_category = skill.metadata.get('category', '').lower()
                if any(word in skill_category for word in query_words):
                    score = min(1.0, score + 0.2)

            scores[skill.skill_id] = min(1.0, score)

        return scores

    def _merge_search_results(
        self,
        semantic_results: List[Any],
        keyword_scores: Dict[str, float],
        all_skills: List[Skill]
    ) -> List[Dict[str, Any]]:
        """
        Merge semantic and keyword search results, loading skills from vector DB metadata.

        Args:
            semantic_results: Results from vector search (with skill_data in metadata)
            keyword_scores: Scores from keyword search
            all_skills: All available skills (for keyword fallback only)

        Returns:
            Merged and deduplicated candidate list
        """
        from src.domain.models.skill import SkillStep, SkillStepType

        # Create skill lookup for keyword fallback
        skill_lookup = {s.skill_id: s for s in all_skills}

        # Track candidates by skill_id
        candidates_dict = {}

        # Add semantic search results - reconstruct skills from vector DB metadata
        for i, result in enumerate(semantic_results):
            skill_id = result.metadata.get('skill_id')

            if not skill_id:
                continue

            # Reconstruct skill from vector DB metadata (JSON string)
            skill_data_json = result.metadata.get('skill_data_json')
            if skill_data_json:
                import json
                skill_data = json.loads(skill_data_json)
                # Build skill from stored data
                steps = []
                for step_data in skill_data.get("steps", []):
                    step = SkillStep(
                        step_id=step_data["step_id"],
                        step_type=SkillStepType(step_data["step_type"]),
                        description=step_data["description"],
                        tool_name=step_data.get("tool_name"),
                        parameters=step_data.get("parameters", {}),
                        requires_approval=step_data.get("requires_approval", False),
                        approval_reason=step_data.get("approval_reason"),
                        expected_output=step_data.get("expected_output")
                    )
                    steps.append(step)

                skill = Skill(
                    skill_id=skill_data["skill_id"],
                    version=skill_data["version"],
                    name=skill_data["name"],
                    description=skill_data["description"],
                    triggers=skill_data["triggers"],
                    steps=steps,
                    metadata=skill_data.get("metadata", {}),
                    guardrails=skill_data.get("guardrails", {})
                )

                trigger_text = result.content  # Use the indexed trigger text
                semantic_score = 1.0 - result.distance

                if skill_id not in candidates_dict:
                    candidates_dict[skill_id] = {
                        'skill': skill,
                        'trigger_text': trigger_text,
                        'semantic_score': semantic_score,
                        'keyword_score': keyword_scores.get(skill_id, 0.0),
                        'skill_id': skill_id
                    }
                else:
                    # Keep highest semantic score if multiple triggers match
                    if semantic_score > candidates_dict[skill_id]['semantic_score']:
                        candidates_dict[skill_id]['semantic_score'] = semantic_score
                        candidates_dict[skill_id]['trigger_text'] = trigger_text

        # Add keyword matches that weren't in semantic results (fallback to all_skills)
        for skill_id, kw_score in keyword_scores.items():
            if kw_score > 0.1 and skill_id not in candidates_dict and skill_id in skill_lookup:
                skill = skill_lookup[skill_id]
                candidates_dict[skill_id] = {
                    'skill': skill,
                    'trigger_text': skill.triggers[0] if skill.triggers else skill.name,
                    'semantic_score': 0.0,  # No semantic match
                    'keyword_score': kw_score,
                    'skill_id': skill_id
                }

        # Calculate hybrid scores and sort
        candidates = []
        for candidate in candidates_dict.values():
            # Hybrid score: 60% semantic + 40% keyword (increased keyword weight for exact matches)
            hybrid_score = (candidate['semantic_score'] * 0.6 + candidate['keyword_score'] * 0.4)
            candidate['semantic_score'] = hybrid_score  # Store hybrid as semantic for now
            candidates.append(candidate)

        # Don't normalize - use raw hybrid scores to preserve true match quality
        # This ensures poor matches stay below the 0.60 threshold

        # Sort by hybrid score
        candidates.sort(key=lambda x: x['semantic_score'], reverse=True)

        return candidates


    def get_match_statistics(self, matches: List[SkillMatch]) -> Dict[str, Any]:
        """
        Calculate statistics for a set of matches.

        Args:
            matches: List of skill matches

        Returns:
            Dictionary with match statistics
        """
        if not matches:
            return {
                'total_matches': 0,
                'confidence_distribution': {},
                'best_score': 0.0,
                'average_score': 0.0
            }

        confidence_counts = {
            MatchConfidence.HIGH.value: 0,
            MatchConfidence.MEDIUM.value: 0,
            MatchConfidence.LOW.value: 0
        }

        for match in matches:
            if match.confidence in confidence_counts:
                confidence_counts[match.confidence] += 1

        return {
            'total_matches': len(matches),
            'confidence_distribution': confidence_counts,
            'best_score': matches[0].score if matches else 0.0,
            'average_score': sum(m.score for m in matches) / len(matches) if matches else 0.0
        }
