"""
Executor Agent (Phase 1 - Dummy Implementation)

This is a Phase 1 dummy implementation that:
1. Verifies approval tokens
2. Delegates execution back to the orchestrator
3. Provides monitoring hooks for Phase 2

Phase 2 Enhancement Plan:
- Autonomous execution of approved actions
- Error recovery and retry logic
- Parallel action execution
- Real-time progress tracking
- Rollback capabilities

For now, this serves as a placeholder that enforces approval gates
and provides the interface for future autonomous execution.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass

from src.domain.models.resolution import Resolution, ResolutionStep, ResolutionStatus
from src.domain.models.approval import ApprovalToken
from src.application.approval.approval_service import ApprovalService
from src.application.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of action execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAUTHORIZED = "unauthorized"


@dataclass
class ExecutionResult:
    """Result of executing approved actions."""
    resolution_id: str
    status: ExecutionStatus
    executed_actions: List[Dict[str, Any]]
    failed_actions: List[Dict[str, Any]]
    execution_time_ms: float
    error: Optional[str] = None


class UnauthorizedExecutionError(Exception):
    """Raised when execution is attempted without valid approval."""
    pass


class ExecutorAgent:
    """
    Executor Agent for approved action execution.

    Phase 1: Dummy implementation with approval verification
    Phase 2: Full autonomous execution with error recovery

    This agent:
    - Verifies approval tokens before execution
    - Enforces approval gates
    - Provides execution monitoring hooks
    - Maintains execution audit trail
    """

    def __init__(
        self,
        approval_service: ApprovalService,
        tool_registry: ToolRegistry,
        enable_autonomous_execution: bool = False
    ):
        """
        Initialize executor agent.

        Args:
            approval_service: Service for approval verification
            tool_registry: Registry of available tools
            enable_autonomous_execution: Enable Phase 2 autonomous execution
        """
        self.approval_service = approval_service
        self.tool_registry = tool_registry
        self.enable_autonomous_execution = enable_autonomous_execution
        self.execution_history: List[ExecutionResult] = []
        self.logger = logging.getLogger(__name__)

        if enable_autonomous_execution:
            self.logger.warning(
                "Autonomous execution is ENABLED. "
                "This should only be used in Phase 2 with proper safeguards."
            )
        else:
            self.logger.info(
                "Executor Agent initialized in Phase 1 mode (approval verification only)"
            )

    def execute_with_approval(
        self,
        resolution: Resolution,
        approval_token: str,
        actions: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute approved actions with token verification.

        Phase 1: Verifies token and returns execution plan
        Phase 2: Actually executes actions autonomously

        Args:
            resolution: Resolution containing actions
            approval_token: Approval token for authorization
            actions: Optional specific actions to execute

        Returns:
            ExecutionResult

        Raises:
            UnauthorizedExecutionError: If token is invalid
        """
        start_time = datetime.now()

        self.logger.info(
            f"Execution requested for resolution {resolution.resolution_id} "
            f"with token {approval_token[:8]}..."
        )

        # Step 1: Verify approval token
        if not self._verify_approval_token(approval_token):
            error_msg = "Invalid or expired approval token"
            self.logger.error(f"Execution denied: {error_msg}")

            result = ExecutionResult(
                resolution_id=resolution.resolution_id,
                status=ExecutionStatus.UNAUTHORIZED,
                executed_actions=[],
                failed_actions=actions or [],
                execution_time_ms=0,
                error=error_msg
            )
            self.execution_history.append(result)

            raise UnauthorizedExecutionError(error_msg)

        self.logger.info("Approval token verified successfully")

        # Step 2: Get actions to execute
        actions_to_execute = actions or resolution.metadata.get("proposed_actions", [])

        if not actions_to_execute:
            self.logger.warning("No actions to execute")
            return ExecutionResult(
                resolution_id=resolution.resolution_id,
                status=ExecutionStatus.COMPLETED,
                executed_actions=[],
                failed_actions=[],
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

        # Step 3: Execute actions (Phase 1 vs Phase 2)
        if self.enable_autonomous_execution:
            # Phase 2: Autonomous execution
            result = self._execute_actions_autonomous(
                resolution, actions_to_execute, approval_token
            )
        else:
            # Phase 1: Dummy execution (just verification)
            result = self._execute_actions_dummy(
                resolution, actions_to_execute, approval_token
            )

        # Calculate execution time
        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Store in history
        self.execution_history.append(result)

        self.logger.info(
            f"Execution completed for {resolution.resolution_id}: "
            f"{result.status.value}, {len(result.executed_actions)} actions"
        )

        return result

    def execute_without_approval(
        self,
        resolution: Resolution,
        actions: List[Dict[str, Any]]
    ) -> ExecutionResult:
        """
        Attempt to execute actions without approval.

        This should always fail with UnauthorizedExecutionError.
        Used for testing approval gate enforcement.

        Args:
            resolution: Resolution
            actions: Actions to execute

        Returns:
            ExecutionResult with UNAUTHORIZED status

        Raises:
            UnauthorizedExecutionError: Always raised
        """
        self.logger.error(
            f"Unauthorized execution attempt for resolution {resolution.resolution_id}"
        )

        result = ExecutionResult(
            resolution_id=resolution.resolution_id,
            status=ExecutionStatus.UNAUTHORIZED,
            executed_actions=[],
            failed_actions=actions,
            execution_time_ms=0,
            error="Execution attempted without valid approval token"
        )

        self.execution_history.append(result)

        raise UnauthorizedExecutionError(
            "Cannot execute actions without valid approval token"
        )

    def get_execution_status(self, resolution_id: str) -> Optional[ExecutionResult]:
        """
        Get execution status for a resolution.

        Args:
            resolution_id: Resolution ID

        Returns:
            Latest ExecutionResult or None
        """
        for result in reversed(self.execution_history):
            if result.resolution_id == resolution_id:
                return result
        return None

    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution metrics
        """
        total = len(self.execution_history)
        completed = sum(1 for r in self.execution_history if r.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for r in self.execution_history if r.status == ExecutionStatus.FAILED)
        unauthorized = sum(1 for r in self.execution_history if r.status == ExecutionStatus.UNAUTHORIZED)

        total_actions = sum(len(r.executed_actions) for r in self.execution_history)
        failed_actions = sum(len(r.failed_actions) for r in self.execution_history)

        return {
            "total_executions": total,
            "completed": completed,
            "failed": failed,
            "unauthorized": unauthorized,
            "success_rate": completed / total if total > 0 else 0,
            "total_actions_executed": total_actions,
            "total_actions_failed": failed_actions,
            "phase": "2_autonomous" if self.enable_autonomous_execution else "1_dummy"
        }

    def _verify_approval_token(self, token: str) -> bool:
        """
        Verify approval token is valid.

        Args:
            token: Token string

        Returns:
            True if valid
        """
        return self.approval_service.validate_token(token)

    def _execute_actions_dummy(
        self,
        resolution: Resolution,
        actions: List[Dict[str, Any]],
        approval_token: str
    ) -> ExecutionResult:
        """
        Phase 1: Dummy execution (verification only).

        This doesn't actually execute actions, just verifies they could be executed.

        Args:
            resolution: Resolution
            actions: Actions to execute
            approval_token: Approval token

        Returns:
            ExecutionResult with simulated success
        """
        self.logger.info(
            f"[Phase 1 Dummy] Simulating execution of {len(actions)} actions"
        )

        # Simulate successful execution
        executed_actions = []

        for i, action in enumerate(actions):
            # Add execution step to resolution
            step = ResolutionStep(
                step_number=len(resolution.steps) + 1,
                step_type="execution_simulated",
                description=f"[SIMULATED] {action.get('description', 'Action')}",
                tool_used=action.get("tool_name"),
                input_data=action.get("parameters", {}),
                output_data={"simulated": True, "status": "would_execute"},
                success=True
            )
            resolution.add_step(step)

            executed_actions.append({
                **action,
                "executed": True,
                "simulated": True,
                "step_number": step.step_number
            })

        self.logger.info(
            f"[Phase 1 Dummy] Simulated {len(executed_actions)} actions successfully"
        )

        return ExecutionResult(
            resolution_id=resolution.resolution_id,
            status=ExecutionStatus.COMPLETED,
            executed_actions=executed_actions,
            failed_actions=[],
            execution_time_ms=0  # Will be set by caller
        )

    def _execute_actions_autonomous(
        self,
        resolution: Resolution,
        actions: List[Dict[str, Any]],
        approval_token: str
    ) -> ExecutionResult:
        """
        Phase 2: Autonomous execution (FUTURE IMPLEMENTATION).

        This will actually execute actions using the tool registry.

        Args:
            resolution: Resolution
            actions: Actions to execute
            approval_token: Approval token

        Returns:
            ExecutionResult with actual execution results
        """
        self.logger.info(
            f"[Phase 2 Autonomous] Executing {len(actions)} actions"
        )

        executed_actions = []
        failed_actions = []

        for action in actions:
            try:
                # Get tool
                tool_name = action.get("tool_name")
                if not tool_name:
                    raise ValueError("Action missing tool_name")

                tool = self.tool_registry.get(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")

                # Execute tool
                parameters = action.get("parameters", {})
                result: ToolResult = tool.execute(**parameters)

                # Add execution step to resolution
                step = ResolutionStep(
                    step_number=len(resolution.steps) + 1,
                    step_type="execution",
                    description=action.get("description", f"Executed {tool_name}"),
                    tool_used=tool_name,
                    input_data=parameters,
                    output_data=result.data,
                    success=result.is_success(),
                    error_message=result.error if not result.is_success() else None
                )
                resolution.add_step(step)

                if result.is_success():
                    executed_actions.append({
                        **action,
                        "executed": True,
                        "result": result.data,
                        "step_number": step.step_number
                    })
                else:
                    failed_actions.append({
                        **action,
                        "executed": False,
                        "error": result.error,
                        "step_number": step.step_number
                    })

            except Exception as e:
                self.logger.error(f"Action execution failed: {e}")
                failed_actions.append({
                    **action,
                    "executed": False,
                    "error": str(e)
                })

        status = ExecutionStatus.COMPLETED if not failed_actions else ExecutionStatus.FAILED

        self.logger.info(
            f"[Phase 2 Autonomous] Executed {len(executed_actions)} actions, "
            f"{len(failed_actions)} failed"
        )

        return ExecutionResult(
            resolution_id=resolution.resolution_id,
            status=status,
            executed_actions=executed_actions,
            failed_actions=failed_actions,
            execution_time_ms=0,  # Will be set by caller
            error=f"{len(failed_actions)} actions failed" if failed_actions else None
        )


# Phase 2 Implementation Plan
"""
PHASE 2 ENHANCEMENT PLAN

1. Autonomous Execution
   - Implement parallel action execution
   - Add execution queue management
   - Support action dependencies
   - Implement retry logic with exponential backoff

2. Error Recovery
   - Automatic retry for transient failures
   - Rollback capabilities for failed transactions
   - Partial success handling
   - Error classification (transient vs permanent)

3. Progress Tracking
   - Real-time execution progress updates
   - Streaming execution logs
   - Webhook notifications for completion
   - Progress percentage calculation

4. Advanced Features
   - Conditional action execution
   - Action batching and optimization
   - Resource pooling for tool execution
   - Execution timeout management

5. Safety & Monitoring
   - Circuit breaker pattern for failing tools
   - Rate limiting per tool
   - Execution metrics and alerting
   - Audit trail with detailed logs

6. Integration Points
   - Event bus for execution events
   - Metrics export (Prometheus/StatsD)
   - Distributed tracing support
   - External system webhooks

Implementation Timeline:
- Week 1: Basic autonomous execution
- Week 2: Error recovery and retry logic
- Week 3: Progress tracking and monitoring
- Week 4: Advanced features and optimization
"""
