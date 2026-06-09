"""
Human Approval Service

This service handles the human-in-the-loop approval workflow:
1. Formats approval requests with risk assessment
2. Generates approval tokens for authorized execution
3. Enforces approval gates for different action types
4. Maintains audit trail of all approvals/rejections
5. Handles rejection feedback loop

Approval Gates:
- Financial actions (refunds, charges)
- Booking modifications (room changes, date changes)
- Operational actions (access grants, escalations)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import secrets
import hashlib
import logging
from dataclasses import dataclass

from src.domain.models.approval import ApprovalRequest, ApprovalToken, ApprovalStatus, RiskLevel
from src.domain.models.resolution import Resolution

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of actions requiring approval."""

    FINANCIAL = "financial"
    BOOKING_MODIFICATION = "booking_modification"
    OPERATIONAL = "operational"
    COMMUNICATION = "communication"
    DATA_ACCESS = "data_access"


@dataclass
class ApprovalGate:
    """Configuration for an approval gate."""

    action_type: ActionType
    requires_approval: bool
    risk_level: RiskLevel
    approval_timeout_minutes: int
    auto_reject_after_timeout: bool


class ApprovalService:
    """
    Service for managing human approval workflow.

    This service:
    - Assesses risk level of proposed actions
    - Generates approval requests
    - Creates secure approval tokens
    - Validates approvals
    - Maintains audit trail
    """

    # Default approval gates configuration
    DEFAULT_GATES = {
        ActionType.FINANCIAL: ApprovalGate(
            action_type=ActionType.FINANCIAL,
            requires_approval=True,
            risk_level=RiskLevel.HIGH,
            approval_timeout_minutes=60,
            auto_reject_after_timeout=False,
        ),
        ActionType.BOOKING_MODIFICATION: ApprovalGate(
            action_type=ActionType.BOOKING_MODIFICATION,
            requires_approval=True,
            risk_level=RiskLevel.MEDIUM,
            approval_timeout_minutes=30,
            auto_reject_after_timeout=False,
        ),
        ActionType.OPERATIONAL: ApprovalGate(
            action_type=ActionType.OPERATIONAL,
            requires_approval=True,
            risk_level=RiskLevel.MEDIUM,
            approval_timeout_minutes=120,
            auto_reject_after_timeout=False,
        ),
        ActionType.COMMUNICATION: ApprovalGate(
            action_type=ActionType.COMMUNICATION,
            requires_approval=True,
            risk_level=RiskLevel.LOW,
            approval_timeout_minutes=15,
            auto_reject_after_timeout=False,
        ),
        ActionType.DATA_ACCESS: ApprovalGate(
            action_type=ActionType.DATA_ACCESS,
            requires_approval=True,
            risk_level=RiskLevel.HIGH,
            approval_timeout_minutes=30,
            auto_reject_after_timeout=True,
        ),
    }

    def __init__(self, approval_gates: Optional[Dict[ActionType, ApprovalGate]] = None):
        """
        Initialize approval service.

        Args:
            approval_gates: Custom approval gate configuration
        """
        self.approval_gates = approval_gates or self.DEFAULT_GATES
        self.approval_history: List[ApprovalRequest] = []
        self.active_tokens: Dict[str, ApprovalToken] = {}
        self.logger = logging.getLogger(__name__)

    def create_approval_request(
        self,
        resolution: Resolution,
        proposed_actions: List[Dict[str, Any]],
        requester: str = "system",
    ) -> ApprovalRequest:
        """
        Create an approval request for proposed actions.

        Args:
            resolution: Resolution containing the actions
            proposed_actions: List of actions requiring approval
            requester: Who is requesting approval

        Returns:
            ApprovalRequest with risk assessment
        """
        self.logger.info(f"Creating approval request for resolution {resolution.resolution_id}")

        # Assess risk level
        risk_level, risk_factors = self._assess_risk(proposed_actions)

        # Determine action types
        action_types = self._classify_actions(proposed_actions)

        # Create action description
        action_description = (
            f"Execute {len(proposed_actions)} action(s) for resolution {resolution.resolution_id}"
        )
        if proposed_actions:
            first_action = proposed_actions[0]
            action_description = first_action.get("description", action_description)

        # Determine primary action type
        primary_action_type = (
            "multiple_actions"
            if len(proposed_actions) > 1
            else (proposed_actions[0].get("action", "unknown") if proposed_actions else "unknown")
        )

        # Create approval request
        approval_request = ApprovalRequest(
            request_id=f"apr_{secrets.token_hex(8)}",
            issue_id=resolution.issue_id,
            action_type=primary_action_type,
            action_description=action_description,
            risk_level=risk_level,
            proposed_changes={
                "resolution_id": resolution.resolution_id,
                "actions": proposed_actions,
                "action_types": [at.value for at in action_types],
                "num_actions": len(proposed_actions),
                "resolution_steps": len(resolution.steps),
            },
            supporting_context={"requester": requester, "risk_factors": risk_factors},
        )

        # Store in history
        self.approval_history.append(approval_request)

        self.logger.info(
            f"Approval request {approval_request.request_id} created "
            f"with risk level {risk_level.value}"
        )

        return approval_request

    def approve_request(
        self, request_id: str, approver: str, comments: Optional[str] = None
    ) -> ApprovalToken:
        """
        Approve an approval request and generate execution token.

        Args:
            request_id: Approval request ID
            approver: Who approved the request
            comments: Optional approval comments

        Returns:
            ApprovalToken for authorized execution

        Raises:
            ValueError: If request not found or already processed
        """
        self.logger.info(f"Processing approval for request {request_id}")

        # Find request
        request = self._find_request(request_id)
        if not request:
            raise ValueError(f"Approval request {request_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Request {request_id} already processed with status {request.status.value}"
            )

        # Update request
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver
        request.approved_at = datetime.utcnow()
        request.notes = comments

        # Generate approval token
        token = self._generate_approval_token(request, approver)

        # Store active token
        self.active_tokens[token.token] = token

        self.logger.info(
            f"Request {request_id} approved by {approver}, " f"token {token.token[:8]}... generated"
        )

        return token

    def reject_request(self, request_id: str, approver: str, reason: str) -> ApprovalRequest:
        """
        Reject an approval request.

        Args:
            request_id: Approval request ID
            approver: Who rejected the request
            reason: Rejection reason

        Returns:
            Updated ApprovalRequest

        Raises:
            ValueError: If request not found or already processed
        """
        self.logger.info(f"Processing rejection for request {request_id}")

        # Find request
        request = self._find_request(request_id)
        if not request:
            raise ValueError(f"Approval request {request_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Request {request_id} already processed with status {request.status.value}"
            )

        # Update request
        request.status = ApprovalStatus.REJECTED
        request.approved_by = approver
        request.approved_at = datetime.utcnow()
        request.rejection_reason = reason

        self.logger.info(f"Request {request_id} rejected by {approver}: {reason}")

        return request

    def validate_token(self, token_str: str) -> bool:
        """
        Validate an approval token.

        Args:
            token_str: Token string to validate

        Returns:
            True if token is valid and not expired
        """
        token = self.active_tokens.get(token_str)

        if not token:
            self.logger.warning(f"Token validation failed: token not found")
            return False

        if not token.is_valid():
            self.logger.warning(f"Token validation failed: token invalid (used or expired)")
            return False

        return True

    def revoke_token(self, token_str: str, reason: str) -> bool:
        """
        Revoke an approval token.

        Args:
            token_str: Token to revoke
            reason: Revocation reason

        Returns:
            True if token was revoked
        """
        token = self.active_tokens.get(token_str)

        if not token:
            self.logger.warning(f"Cannot revoke token: not found")
            return False

        # Mark token as used to effectively revoke it
        token.mark_used()

        self.logger.info(f"Token {token_str[:8]}... revoked: {reason}")

        return True

    def get_approval_statistics(self) -> Dict[str, Any]:
        """
        Get approval statistics.

        Returns:
            Dictionary with approval metrics
        """
        total = len(self.approval_history)
        approved = sum(1 for r in self.approval_history if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in self.approval_history if r.status == ApprovalStatus.REJECTED)
        pending = sum(1 for r in self.approval_history if r.status == ApprovalStatus.PENDING)

        # Calculate average approval time
        approval_times = [
            (r.approved_at - r.created_at).total_seconds()
            for r in self.approval_history
            if r.approved_at and r.status == ApprovalStatus.APPROVED
        ]
        avg_approval_time = sum(approval_times) / len(approval_times) if approval_times else 0

        return {
            "total_requests": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "approval_rate": approved / total if total > 0 else 0,
            "avg_approval_time_seconds": avg_approval_time,
            "active_tokens": len([t for t in self.active_tokens.values() if not t.is_expired()]),
        }

    def _assess_risk(self, actions: List[Dict[str, Any]]) -> tuple[RiskLevel, List[str]]:
        """
        Assess risk level of proposed actions.

        Args:
            actions: List of proposed actions

        Returns:
            Tuple of (risk_level, risk_factors)
        """
        risk_factors = []
        max_risk = RiskLevel.LOW

        for action in actions:
            tool_name = action.get("tool_name", "")
            parameters = action.get("parameters", {})

            # Financial actions are high risk
            if tool_name == "process_refund":
                risk_factors.append("Financial transaction: refund")
                max_risk = RiskLevel.HIGH

                # Check refund amount
                amount = parameters.get("amount")
                if amount and (amount == "full" or float(amount) > 100):
                    risk_factors.append(f"High value refund: {amount}")

            # Booking modifications are medium risk
            elif tool_name == "update_pms":
                risk_factors.append("Booking modification")
                if max_risk == RiskLevel.LOW:
                    max_risk = RiskLevel.MEDIUM

            # Data access is high risk
            elif "lookup" in tool_name.lower() and "sensitive" in str(parameters):
                risk_factors.append("Sensitive data access")
                max_risk = RiskLevel.HIGH

            # Multiple actions increase risk
            if len(actions) > 3:
                risk_factors.append(f"Multiple actions: {len(actions)}")
                if max_risk == RiskLevel.LOW:
                    max_risk = RiskLevel.MEDIUM

        return max_risk, risk_factors

    def _classify_actions(self, actions: List[Dict[str, Any]]) -> List[ActionType]:
        """
        Classify actions by type.

        Args:
            actions: List of proposed actions

        Returns:
            List of ActionType enums
        """
        action_types = set()

        for action in actions:
            tool_name = action.get("tool_name", "")

            if tool_name in ["process_refund", "charge_card"]:
                action_types.add(ActionType.FINANCIAL)
            elif tool_name in ["update_pms", "modify_booking"]:
                action_types.add(ActionType.BOOKING_MODIFICATION)
            elif tool_name in ["send_email", "send_sms"]:
                action_types.add(ActionType.COMMUNICATION)
            elif tool_name in ["lookup_booking", "get_guest_data"]:
                action_types.add(ActionType.DATA_ACCESS)
            else:
                action_types.add(ActionType.OPERATIONAL)

        return list(action_types)

    def _generate_approval_token(self, request: ApprovalRequest, approver: str) -> ApprovalToken:
        """
        Generate a secure approval token.

        Args:
            request: Approved request
            approver: Who approved

        Returns:
            ApprovalToken
        """
        # Generate secure random token
        token_str = secrets.token_urlsafe(32)

        # Calculate expiration based on risk level
        if request.risk_level == RiskLevel.HIGH:
            expires_in_minutes = 15
        elif request.risk_level == RiskLevel.MEDIUM:
            expires_in_minutes = 30
        else:
            expires_in_minutes = 60

        expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)

        # Create token with required fields
        token = ApprovalToken(
            token=token_str,
            request_id=request.request_id,
            issue_id=request.issue_id,
            action_type=request.action_type,
            approved_by=approver,
            expires_at=expires_at,
        )

        return token

    def _find_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Find approval request by ID."""
        for request in self.approval_history:
            if request.request_id == request_id:
                return request
        return None
