"""Domain interfaces for the hotel issue resolution system."""

from .rag_retriever import IRAGRetriever
from .skill_matcher import ISkillMatcher, SkillMatchResult
from .tool_executor import IToolExecutor, ToolExecutionResult, ToolExecutionStatus
from .approval_gateway import IApprovalGateway
from .executor import IExecutor
from .orchestrator import IOrchestrator

__all__ = [
    # RAG System
    "IRAGRetriever",

    # Skills System
    "ISkillMatcher",
    "SkillMatchResult",

    # Tool System
    "IToolExecutor",
    "ToolExecutionResult",
    "ToolExecutionStatus",

    # Approval System
    "IApprovalGateway",

    # Execution System
    "IExecutor",

    # Orchestration
    "IOrchestrator",
]
