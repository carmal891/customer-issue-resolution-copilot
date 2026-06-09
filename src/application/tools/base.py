"""
Base Tool Classes

Defines the abstract interface for tools and common utilities.
Tools are the primary way agents interact with external systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    """Tool execution status"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Contains the output data, status, and any metadata about the execution.
    """
    status: ToolStatus
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    requires_approval: bool = False
    approval_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if execution was successful"""
        return self.status == ToolStatus.SUCCESS

    def is_failure(self) -> bool:
        """Check if execution failed"""
        return self.status == ToolStatus.FAILURE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "requires_approval": self.requires_approval,
            "approval_reason": self.approval_reason,
            "metadata": self.metadata
        }


class ToolExecutionError(Exception):
    """Raised when tool execution fails"""

    def __init__(
        self,
        message: str,
        tool_name: str,
        inputs: Dict[str, Any],
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.inputs = inputs
        self.original_error = original_error


class Tool(ABC):
    """
    Abstract base class for all tools.

    Tools are the interface between the agent and external systems.
    Each tool should:
    - Have a clear, descriptive name
    - Define its input schema
    - Validate inputs
    - Execute the action
    - Return a structured result
    - Handle errors gracefully
    """

    def __init__(self, name: str, description: str):
        """
        Initialize tool.

        Args:
            name: Unique tool identifier
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_execution_time_ms = 0.0

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for tool inputs.

        This is used for:
        - Input validation
        - LLM function calling
        - Documentation generation

        Returns:
            JSON schema dict
        """
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool's core logic.

        This is the method subclasses should implement.

        Args:
            **kwargs: Tool-specific inputs

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate inputs against schema.

        Args:
            inputs: Input parameters

        Raises:
            ValueError: If inputs are invalid
        """
        schema = self.get_input_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field in required:
            if field not in inputs:
                raise ValueError(f"Missing required field: {field}")

        # Check field types (basic validation)
        for field, value in inputs.items():
            if field not in properties:
                logger.warning(f"Unknown field '{field}' for tool '{self.name}'")

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with input validation and error handling.

        This is the public interface that should be called.

        Args:
            **kwargs: Tool-specific inputs

        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            self.validate_inputs(kwargs)

            # Execute
            logger.info(f"Executing tool '{self.name}' with inputs: {kwargs}")
            result = self._execute(**kwargs)

            # Update metrics
            self._execution_count += 1
            if result.is_success():
                self._success_count += 1
            else:
                self._failure_count += 1

            # Calculate execution time
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time_ms
            self._total_execution_time_ms += execution_time_ms

            logger.info(
                f"Tool '{self.name}' completed with status {result.status} "
                f"in {execution_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            # Handle execution errors
            self._execution_count += 1
            self._failure_count += 1

            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            self._total_execution_time_ms += execution_time_ms

            logger.error(
                f"Tool '{self.name}' failed: {str(e)}",
                exc_info=True
            )

            # Return failure result
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"exception_type": type(e).__name__}
            )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get tool execution metrics.

        Returns:
            Dict with execution statistics
        """
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": (
                self._success_count / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "avg_execution_time_ms": (
                self._total_execution_time_ms / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "total_execution_time_ms": self._total_execution_time_ms
        }

    def to_function_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling schema.

        Returns:
            Function schema dict
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_input_schema()
        }


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides:
    - Tool registration
    - Tool lookup by name
    - Tool listing
    - Metrics aggregation
    """

    def __init__(self):
        """Initialize empty registry"""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.

        Returns:
            List of Tool instances
        """
        return list(self._tools.values())

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function schemas for all tools.

        Returns:
            List of function schema dicts
        """
        return [tool.to_function_schema() for tool in self._tools.values()]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for all tools.

        Returns:
            Dict with tool metrics
        """
        return {
            "total_tools": len(self._tools),
            "tools": {
                name: tool.get_metrics()
                for name, tool in self._tools.items()
            }
        }

    def clear(self) -> None:
        """Clear all registered tools"""
        self._tools.clear()
        logger.info("Cleared tool registry")
