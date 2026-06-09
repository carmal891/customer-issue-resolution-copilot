"""Domain exceptions for the hotel issue resolution system."""

# Base exceptions
class DomainException(Exception):
    """Base exception for all domain errors."""
    pass


class ValidationError(DomainException):
    """Raised when domain validation fails."""
    pass


# Booking exceptions
class BookingError(DomainException):
    """Base exception for booking-related errors."""
    pass


class BookingNotFoundError(BookingError):
    """Raised when booking cannot be found."""
    pass


class InvalidBookingStateError(BookingError):
    """Raised when booking is in invalid state for operation."""
    pass


class BookingConflictError(BookingError):
    """Raised when booking conflicts with another booking."""
    pass


# Issue exceptions
class IssueError(DomainException):
    """Base exception for issue-related errors."""
    pass


class IssueNotFoundError(IssueError):
    """Raised when issue cannot be found."""
    pass


class InvalidIssueStateError(IssueError):
    """Raised when issue is in invalid state for operation."""
    pass


# Skill exceptions
class SkillError(DomainException):
    """Base exception for skill-related errors."""
    pass


class SkillNotFoundError(SkillError):
    """Raised when skill cannot be found."""
    pass


class SkillMatchError(SkillError):
    """Raised when skill matching fails."""
    pass


class SkillRegistrationError(SkillError):
    """Raised when skill registration fails."""
    pass


class SkillUpdateError(SkillError):
    """Raised when skill update fails."""
    pass


class SkillDeletionError(SkillError):
    """Raised when skill deletion fails."""
    pass


class SkillRetrievalError(SkillError):
    """Raised when skill retrieval fails."""
    pass


class SkillIndexError(SkillError):
    """Raised when skill indexing fails."""
    pass


class SkillCompilationError(SkillError):
    """Raised when skill compilation from resolution fails."""
    pass


class SkillExecutionError(SkillError):
    """Raised when skill execution fails."""
    pass


# RAG exceptions
class RAGError(DomainException):
    """Base exception for RAG-related errors."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


class IngestionError(RAGError):
    """Raised when document ingestion fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class RerankingError(RAGError):
    """Raised when reranking fails."""
    pass


class ChunkingError(RAGError):
    """Raised when document chunking fails."""
    pass


# Tool exceptions
class ToolError(DomainException):
    """Base exception for tool-related errors."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when tool cannot be found."""
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass


class ToolValidationError(ToolError):
    """Raised when tool parameter validation fails."""
    pass


class ToolRetrievalError(ToolError):
    """Raised when tool retrieval fails."""
    pass


class UnauthorizedExecutionError(ToolError):
    """Raised when execution attempted without valid approval."""
    pass


# Approval exceptions
class ApprovalError(DomainException):
    """Base exception for approval-related errors."""
    pass


class ApprovalNotFoundError(ApprovalError):
    """Raised when approval request cannot be found."""
    pass


class ApprovalRequestError(ApprovalError):
    """Raised when approval request submission fails."""
    pass


class InvalidApprovalStateError(ApprovalError):
    """Raised when approval is in invalid state for operation."""
    pass


class ApprovalRetrievalError(ApprovalError):
    """Raised when approval retrieval fails."""
    pass


class TokenNotFoundError(ApprovalError):
    """Raised when approval token cannot be found."""
    pass


class TokenRevocationError(ApprovalError):
    """Raised when token revocation fails."""
    pass


class TokenExpiredError(ApprovalError):
    """Raised when approval token has expired."""
    pass


class TokenInvalidError(ApprovalError):
    """Raised when approval token is invalid."""
    pass


# Resolution exceptions
class ResolutionError(DomainException):
    """Base exception for resolution-related errors."""
    pass


class ResolutionNotFoundError(ResolutionError):
    """Raised when resolution cannot be found."""
    pass


class InvalidResolutionStateError(ResolutionError):
    """Raised when resolution is in invalid state for operation."""
    pass


class StepNotFoundError(ResolutionError):
    """Raised when resolution step cannot be found."""
    pass


# Execution exceptions
class ExecutionError(DomainException):
    """Base exception for execution-related errors."""
    pass


class CancellationError(ExecutionError):
    """Raised when execution cancellation fails."""
    pass


# Orchestration exceptions
class OrchestrationError(DomainException):
    """Base exception for orchestration-related errors."""
    pass


class ClassificationError(OrchestrationError):
    """Raised when issue classification fails."""
    pass


class ReasoningError(OrchestrationError):
    """Raised when reasoning loop fails."""
    pass


# Guardrail exceptions
class GuardrailError(DomainException):
    """Base exception for guardrail violations."""
    pass


class PIIDetectedError(GuardrailError):
    """Raised when PII is detected in request."""
    pass


class PromptInjectionError(GuardrailError):
    """Raised when prompt injection is detected."""
    pass


class LowConfidenceError(GuardrailError):
    """Raised when confidence is too low to proceed."""
    pass


class RateLimitError(GuardrailError):
    """Raised when rate limit is exceeded."""
    pass


# Notification exceptions
class NotificationError(DomainException):
    """Base exception for notification-related errors."""
    pass


# Metrics exceptions
class MetricsError(DomainException):
    """Base exception for metrics-related errors."""
    pass


class MetricsRetrievalError(MetricsError):
    """Raised when metrics retrieval fails."""
    pass


__all__ = [
    # Base
    "DomainException",
    "ValidationError",

    # Booking
    "BookingError",
    "BookingNotFoundError",
    "InvalidBookingStateError",
    "BookingConflictError",

    # Issue
    "IssueError",
    "IssueNotFoundError",
    "InvalidIssueStateError",

    # Skill
    "SkillError",
    "SkillNotFoundError",
    "SkillMatchError",
    "SkillRegistrationError",
    "SkillUpdateError",
    "SkillDeletionError",
    "SkillRetrievalError",
    "SkillIndexError",
    "SkillCompilationError",
    "SkillExecutionError",

    # RAG
    "RAGError",
    "RetrievalError",
    "IngestionError",
    "EmbeddingError",
    "RerankingError",
    "ChunkingError",

    # Tool
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolValidationError",
    "ToolRetrievalError",
    "UnauthorizedExecutionError",

    # Approval
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalRequestError",
    "InvalidApprovalStateError",
    "ApprovalRetrievalError",
    "TokenNotFoundError",
    "TokenRevocationError",
    "TokenExpiredError",
    "TokenInvalidError",

    # Resolution
    "ResolutionError",
    "ResolutionNotFoundError",
    "InvalidResolutionStateError",
    "StepNotFoundError",

    # Execution
    "ExecutionError",
    "CancellationError",

    # Orchestration
    "OrchestrationError",
    "ClassificationError",
    "ReasoningError",

    # Guardrails
    "GuardrailError",
    "PIIDetectedError",
    "PromptInjectionError",
    "LowConfidenceError",
    "RateLimitError",

    # Notification
    "NotificationError",

    # Metrics
    "MetricsError",
    "MetricsRetrievalError",
]
