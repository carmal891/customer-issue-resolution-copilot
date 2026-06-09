"""Executor interface for action execution (Phase 1: Dummy implementation)."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.approval import ApprovalToken
    from ..models.resolution import Resolution, ResolutionStep


class IExecutor(ABC):
    """
    Interface for executor agent.

    Phase 1 (POC): Dummy implementation that delegates to Orchestrator.
    Human executes actions manually after approval.

    Phase 2 (Future): Autonomous execution of approved low-risk actions
    with error recovery and rollback capabilities.

    This interface is defined now to establish the contract and make
    Phase 2 migration straightforward.
    """

    @abstractmethod
    async def execute_resolution(
        self,
        resolution: "Resolution",
        approval_token: "ApprovalToken",
    ) -> "Resolution":
        """
        Execute a resolution plan with approved actions.

        Phase 1: Returns resolution with status updated to AWAITING_EXECUTION.
        Human performs actions manually and updates status.

        Phase 2: Autonomously executes approved actions and returns
        completed resolution with execution results.

        Args:
            resolution: Resolution plan to execute
            approval_token: Valid approval token

        Returns:
            Updated resolution with execution status

        Raises:
            UnauthorizedExecutionError: If token invalid or expired
            ExecutionError: If execution fails
        """
        pass

    @abstractmethod
    async def execute_step(
        self,
        step: "ResolutionStep",
        approval_token: "ApprovalToken",
    ) -> "ResolutionStep":
        """
        Execute a single resolution step.

        Phase 1: Returns step with status updated to AWAITING_EXECUTION.

        Phase 2: Executes the step and returns result.

        Args:
            step: Resolution step to execute
            approval_token: Valid approval token

        Returns:
            Updated step with execution status

        Raises:
            UnauthorizedExecutionError: If token invalid or expired
            ExecutionError: If execution fails
        """
        pass

    @abstractmethod
    async def validate_execution_readiness(
        self,
        resolution: "Resolution",
        approval_token: "ApprovalToken",
    ) -> bool:
        """
        Validate that resolution is ready for execution.

        Checks:
        - Approval token is valid and not expired
        - Token matches resolution actions
        - All required approvals are present
        - No conflicting state

        Args:
            resolution: Resolution to validate
            approval_token: Approval token to check

        Returns:
            True if ready for execution, False otherwise
        """
        pass

    @abstractmethod
    async def get_execution_status(
        self,
        resolution_id: str,
    ) -> dict:
        """
        Get current execution status of a resolution.

        Args:
            resolution_id: Resolution identifier

        Returns:
            Dictionary with execution status details

        Raises:
            ResolutionNotFoundError: If resolution doesn't exist
        """
        pass

    @abstractmethod
    async def cancel_execution(
        self,
        resolution_id: str,
        reason: str,
    ) -> None:
        """
        Cancel an in-progress execution.

        Phase 1: Updates status to CANCELLED.

        Phase 2: Attempts graceful cancellation and rollback.

        Args:
            resolution_id: Resolution identifier
            reason: Cancellation reason

        Raises:
            ResolutionNotFoundError: If resolution doesn't exist
            CancellationError: If cancellation fails
        """
        pass

    @abstractmethod
    async def record_manual_execution(
        self,
        resolution_id: str,
        step_id: str,
        result: dict,
        executor_id: str,
    ) -> None:
        """
        Record result of manual execution (Phase 1).

        This allows human operators to record what they did
        after manual execution.

        Args:
            resolution_id: Resolution identifier
            step_id: Step identifier
            result: Execution result details
            executor_id: ID of person who executed

        Raises:
            ResolutionNotFoundError: If resolution doesn't exist
            StepNotFoundError: If step doesn't exist
        """
        pass

    @abstractmethod
    async def get_execution_history(
        self,
        resolution_id: str,
    ) -> List[dict]:
        """
        Get execution history for a resolution.

        Args:
            resolution_id: Resolution identifier

        Returns:
            List of execution events with timestamps

        Raises:
            ResolutionNotFoundError: If resolution doesn't exist
        """
        pass

    @abstractmethod
    async def prepare_phase2_migration(self) -> dict:
        """
        Prepare for Phase 2 autonomous execution migration.

        Returns diagnostic information about:
        - Current execution patterns
        - Manual execution frequency
        - Candidate actions for automation
        - Risk assessment for autonomous execution

        Returns:
            Dictionary with migration readiness assessment
        """
        pass
