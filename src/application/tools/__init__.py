"""
Tool Layer

Provides abstract interfaces and concrete implementations for tools
that the agent can use to interact with external systems.
"""

from src.application.tools.base import (
    Tool,
    ToolResult,
    ToolExecutionError,
    ToolRegistry
)

from src.application.tools.mock_tools import (
    LookupBookingTool,
    CheckPolicyTool,
    ProcessRefundTool,
    SendEmailTool,
    UpdatePMSTool,
    NotifyTeamTool,
    CheckRoomAvailabilityTool,
    UpdateBookingTool
)

__all__ = [
    # Base classes
    "Tool",
    "ToolResult",
    "ToolExecutionError",
    "ToolRegistry",

    # Mock tools
    "LookupBookingTool",
    "CheckPolicyTool",
    "ProcessRefundTool",
    "SendEmailTool",
    "UpdatePMSTool",
    "NotifyTeamTool",
    "CheckRoomAvailabilityTool",
    "UpdateBookingTool",
]
