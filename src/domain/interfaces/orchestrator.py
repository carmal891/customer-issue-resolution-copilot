"""Orchestrator interface for main coordination agent."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.issue import Issue
    from ..models.resolution import Resolution
    from ..models.skill import Skill
    from ..models.context import RAGContext


class IOrchestrator(ABC):
    """
    Interface for orchestrator agent.

    The orchestrator is the main coordination agent that:
    - Receives customer issues
    - Retrieves relevant context via RAG
    - Checks for existing skills
    - Routes to skill path or novel task path
    - Generates resolution plans
    - Prepares approval requests
    - Compiles new skills from successful resolutions
    """

    @abstractmethod
    async def handle_issue(
        self,
        issue: "Issue",
    ) -> "Resolution":
        """
        Main entry point for issue handling.

        This orchestrates the entire workflow:
        1. Classify issue
        2. Retrieve context via RAG
        3. Check for matching skill
        4. If skill exists: use skill path
        5. If no skill: use novel task path (ReAct/TPAO)
        6. Generate resolution plan
        7. Prepare approval requests
        8. Return resolution for human review

        Args:
            issue: Customer issue to handle

        Returns:
            Resolution plan with proposed actions

        Raises:
            OrchestrationError: If orchestration fails
        """
        pass

    @abstractmethod
    async def classify_issue(
        self,
        issue: "Issue",
    ) -> dict:
        """
        Classify issue type and extract key information.

        Returns:
            Dictionary with classification results:
            - issue_type: Detected issue type
            - priority: Calculated priority
            - entities: Extracted entities (booking_id, guest_name, etc.)
            - intent: User intent
            - confidence: Classification confidence

        Args:
            issue: Issue to classify

        Raises:
            ClassificationError: If classification fails
        """
        pass

    @abstractmethod
    async def retrieve_context(
        self,
        issue: "Issue",
    ) -> "RAGContext":
        """
        Retrieve relevant context for issue resolution.

        Uses RAG system to find:
        - Relevant policies
        - Similar past issues
        - Applicable procedures
        - Related documentation

        Args:
            issue: Issue to retrieve context for

        Returns:
            RAG context with retrieved information

        Raises:
            RetrievalError: If context retrieval fails
        """
        pass

    @abstractmethod
    async def match_skill(
        self,
        issue: "Issue",
        context: "RAGContext",
    ) -> Optional["Skill"]:
        """
        Check if existing skill matches the issue.

        Args:
            issue: Issue to match
            context: Retrieved context

        Returns:
            Matching skill if found, None otherwise

        Raises:
            SkillMatchError: If matching fails
        """
        pass

    @abstractmethod
    async def execute_skill_path(
        self,
        issue: "Issue",
        skill: "Skill",
        context: "RAGContext",
    ) -> "Resolution":
        """
        Execute resolution using existing skill.

        Args:
            issue: Issue to resolve
            skill: Matched skill to use
            context: Retrieved context

        Returns:
            Resolution plan based on skill

        Raises:
            SkillExecutionError: If skill execution fails
        """
        pass

    @abstractmethod
    async def execute_novel_task_path(
        self,
        issue: "Issue",
        context: "RAGContext",
    ) -> "Resolution":
        """
        Execute resolution for novel task using ReAct/TPAO loop.

        This is the reasoning loop for issues without matching skills:
        - Think: Analyze issue and context
        - Plan: Draft resolution steps
        - Act: Prepare actions
        - Observe: Validate plan

        Args:
            issue: Issue to resolve
            context: Retrieved context

        Returns:
            Resolution plan from reasoning loop

        Raises:
            ReasoningError: If reasoning loop fails
        """
        pass

    @abstractmethod
    async def compile_skill(
        self,
        resolution: "Resolution",
    ) -> Optional["Skill"]:
        """
        Compile successful resolution into reusable skill.

        This is the learning mechanism - converting novel task
        handling into reusable knowledge.

        Args:
            resolution: Successful resolution to compile

        Returns:
            Draft skill if compilation successful, None otherwise

        Raises:
            SkillCompilationError: If compilation fails
        """
        pass

    @abstractmethod
    async def validate_resolution(
        self,
        resolution: "Resolution",
    ) -> bool:
        """
        Validate resolution plan before approval.

        Checks:
        - All required information present
        - Actions are valid
        - No conflicting steps
        - Guardrails satisfied

        Args:
            resolution: Resolution to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    async def get_orchestration_metrics(self) -> dict:
        """
        Get orchestration metrics for monitoring.

        Returns:
            Dictionary with metrics:
            - Total issues handled
            - Skill path vs novel path ratio
            - Average resolution time
            - Success rate
            - Skills compiled

        Raises:
            MetricsError: If metrics retrieval fails
        """
        pass
