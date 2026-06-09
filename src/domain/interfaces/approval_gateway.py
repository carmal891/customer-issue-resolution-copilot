"""Approval Gateway interface for human-in-the-loop approval workflow."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.approval import ApprovalRequest, ApprovalToken, ApprovalStatus


class IApprovalGateway(ABC):
    """
    Interface for approval gateway system.
    
    This interface defines the contract for managing human approval
    workflow for risky or consequential actions.
    """

    @abstractmethod
    async def request_approval(
        self,
        request: "ApprovalRequest",
    ) -> str:
        """
        Submit an approval request.

        Args:
            request: Approval request with action details

        Returns:
            Request ID for tracking

        Raises:
            ApprovalRequestError: If request submission fails
        """
        pass

    @abstractmethod
    async def get_approval_status(
        self,
        request_id: str,
    ) -> "ApprovalStatus":
        """
        Get current status of an approval request.

        Args:
            request_id: Request identifier

        Returns:
            Current approval status

        Raises:
            ApprovalNotFoundError: If request doesn't exist
        """
        pass

    @abstractmethod
    async def approve_request(
        self,
        request_id: str,
        approver_id: str,
        notes: Optional[str] = None,
    ) -> "ApprovalToken":
        """
        Approve a pending request.

        Args:
            request_id: Request identifier
            approver_id: ID of person approving
            notes: Optional approval notes

        Returns:
            Approval token for authorized execution

        Raises:
            ApprovalNotFoundError: If request doesn't exist
            InvalidApprovalStateError: If request not in pending state
            ApprovalError: If approval fails
        """
        pass

    @abstractmethod
    async def reject_request(
        self,
        request_id: str,
        approver_id: str,
        reason: str,
    ) -> None:
        """
        Reject a pending request.

        Args:
            request_id: Request identifier
            approver_id: ID of person rejecting
            reason: Rejection reason

        Raises:
            ApprovalNotFoundError: If request doesn't exist
            InvalidApprovalStateError: If request not in pending state
            ApprovalError: If rejection fails
        """
        pass

    @abstractmethod
    async def validate_token(
        self,
        token: "ApprovalToken",
    ) -> bool:
        """
        Validate an approval token.

        Args:
            token: Approval token to validate

        Returns:
            True if valid and not expired, False otherwise
        """
        pass

    @abstractmethod
    async def revoke_token(
        self,
        token_id: str,
    ) -> None:
        """
        Revoke an approval token.

        Args:
            token_id: Token identifier to revoke

        Raises:
            TokenNotFoundError: If token doesn't exist
            TokenRevocationError: If revocation fails
        """
        pass

    @abstractmethod
    async def get_pending_requests(
        self,
        approver_id: Optional[str] = None,
    ) -> List["ApprovalRequest"]:
        """
        Get all pending approval requests.

        Args:
            approver_id: Optional filter by approver

        Returns:
            List of pending requests

        Raises:
            ApprovalRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_request_history(
        self,
        issue_id: str,
    ) -> List["ApprovalRequest"]:
        """
        Get approval history for an issue.

        Args:
            issue_id: Issue identifier

        Returns:
            List of approval requests for the issue

        Raises:
            ApprovalRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_approval_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """
        Get approval metrics for analysis.

        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)

        Returns:
            Dictionary with metrics (approval rate, avg time, etc.)

        Raises:
            MetricsRetrievalError: If metrics retrieval fails
        """
        pass

    @abstractmethod
    async def notify_approver(
        self,
        request_id: str,
        approver_id: str,
    ) -> None:
        """
        Send notification to approver about pending request.

        Args:
            request_id: Request identifier
            approver_id: Approver to notify

        Raises:
            NotificationError: If notification fails
        """
        pass
