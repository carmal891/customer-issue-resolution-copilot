"""Skill Matcher interface for matching issues to skills."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.issue import Issue
    from ..models.skill import Skill


class SkillMatchResult:
    """Result of skill matching operation."""

    def __init__(self, skill: "Skill", confidence: float, reasoning: Optional[str] = None):
        """
        Initialize skill match result.

        Args:
            skill: Matched skill
            confidence: Confidence score (0.0 to 1.0)
            reasoning: Optional explanation of why skill matched
        """
        self.skill = skill
        self.confidence = confidence
        self.reasoning = reasoning

    def is_confident_match(self, threshold: float = 0.7) -> bool:
        """Check if match confidence exceeds threshold."""
        return self.confidence >= threshold


class ISkillMatcher(ABC):
    """
    Interface for skill matching system.
    
    This interface defines the contract for matching customer issues to
    existing skills using semantic search and pattern matching.
    """

    @abstractmethod
    async def match_skill(
        self,
        issue: "Issue",
        min_confidence: float = 0.7,
    ) -> Optional[SkillMatchResult]:
        """
        Match an issue to the best available skill.

        Args:
            issue: Customer issue to match
            min_confidence: Minimum confidence threshold for match

        Returns:
            SkillMatchResult if confident match found, None otherwise

        Raises:
            SkillMatchError: If matching process fails
        """
        pass

    @abstractmethod
    async def match_skills_ranked(
        self,
        issue: "Issue",
        top_k: int = 3,
    ) -> List[SkillMatchResult]:
        """
        Get top-k ranked skill matches for an issue.

        Args:
            issue: Customer issue to match
            top_k: Number of top matches to return

        Returns:
            List of skill match results ranked by confidence

        Raises:
            SkillMatchError: If matching process fails
        """
        pass

    @abstractmethod
    async def register_skill(self, skill: "Skill") -> None:
        """
        Register a new skill in the matcher.

        This adds the skill to both file storage and vector index.

        Args:
            skill: Skill to register

        Raises:
            SkillRegistrationError: If skill registration fails
        """
        pass

    @abstractmethod
    async def update_skill(self, skill: "Skill") -> None:
        """
        Update an existing skill.

        Args:
            skill: Updated skill

        Raises:
            SkillNotFoundError: If skill doesn't exist
            SkillUpdateError: If update fails
        """
        pass

    @abstractmethod
    async def get_skill(self, skill_id: str) -> Optional["Skill"]:
        """
        Retrieve a skill by ID.

        Args:
            skill_id: Skill identifier

        Returns:
            Skill if found, None otherwise

        Raises:
            SkillRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_active_skills(self) -> List["Skill"]:
        """
        List all active skills.

        Returns:
            List of active skills

        Raises:
            SkillRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def delete_skill(self, skill_id: str) -> None:
        """
        Delete a skill from the registry.

        Args:
            skill_id: Skill identifier to delete

        Raises:
            SkillNotFoundError: If skill doesn't exist
            SkillDeletionError: If deletion fails
        """
        pass

    @abstractmethod
    async def get_skill_count(self) -> int:
        """
        Get total number of registered skills.

        Returns:
            Number of skills

        Raises:
            SkillRetrievalError: If count retrieval fails
        """
        pass

    @abstractmethod
    async def reindex_skills(self) -> None:
        """
        Rebuild the skill index from file storage.

        This is useful after bulk updates or system recovery.

        Raises:
            SkillIndexError: If reindexing fails
        """
        pass
