"""Tool Executor interface for executing operational actions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..models.approval import ApprovalToken


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    FAILED = "failed"
    REQUIRES_APPROVAL = "requires_approval"
    UNAUTHORIZED = "unauthorized"
    TIMEOUT = "timeout"


class ToolExecutionResult:
    """Result of tool execution."""

    def __init__(
        self,
        status: ToolExecutionStatus,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize tool execution result.

        Args:
            status: Execution status
            output: Tool output if successful
            error: Error message if failed
            metadata: Additional execution metadata
        """
        self.status = status
        self.output = output
        self.error = error
        self.metadata = metadata or {}

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolExecutionStatus.SUCCESS

    def requires_approval(self) -> bool:
        """Check if execution requires approval."""
        return self.status == ToolExecutionStatus.REQUIRES_APPROVAL


class IToolExecutor(ABC):
    """
    Interface for tool execution system.

    This interface defines the contract for executing operational tools
    with approval gating and safety checks.
    """

    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        approval_token: Optional["ApprovalToken"] = None,
    ) -> ToolExecutionResult:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            approval_token: Optional approval token for authorized execution

        Returns:
            ToolExecutionResult with status and output

        Raises:
            ToolNotFoundError: If tool doesn't exist
            UnauthorizedExecutionError: If approval required but not provided
            ToolExecutionError: If execution fails
        """
        pass

    @abstractmethod
    async def validate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> bool:
        """
        Validate tool call parameters without executing.

        Args:
            tool_name: Name of tool to validate
            parameters: Tool parameters to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        pass

    @abstractmethod
    async def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get JSON schema for a tool's parameters.

        Args:
            tool_name: Name of tool

        Returns:
            JSON schema for tool parameters

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        pass

    @abstractmethod
    async def list_available_tools(self) -> List[str]:
        """
        List all available tool names.

        Returns:
            List of tool names

        Raises:
            ToolRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def check_approval_required(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> bool:
        """
        Check if tool execution requires approval.

        Args:
            tool_name: Name of tool
            parameters: Tool parameters

        Returns:
            True if approval required, False otherwise

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        pass

    @abstractmethod
    async def get_tool_description(self, tool_name: str) -> str:
        """
        Get human-readable description of a tool.

        Args:
            tool_name: Name of tool

        Returns:
            Tool description

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        pass

    @abstractmethod
    async def execute_tool_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        approval_token: Optional["ApprovalToken"] = None,
    ) -> List[ToolExecutionResult]:
        """
        Execute multiple tools in sequence.

        Args:
            tool_calls: List of tool calls with name and parameters
            approval_token: Optional approval token for authorized execution

        Returns:
            List of execution results

        Raises:
            ToolExecutionError: If batch execution fails
        """
        pass

    @abstractmethod
    async def dry_run_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulate tool execution without making changes.

        Args:
            tool_name: Name of tool
            parameters: Tool parameters

        Returns:
            Simulated execution result

        Raises:
            ToolNotFoundError: If tool doesn't exist
            ToolExecutionError: If dry run fails
        """
        pass
