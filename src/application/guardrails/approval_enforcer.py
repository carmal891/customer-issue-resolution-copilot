"""
Approval Gating Enforcement

Ensures no consequential actions are executed without valid approval tokens.
This is a critical safety mechanism for the hotel operations system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.domain.models.approval import ApprovalToken, ApprovalStatus

logger = logging.getLogger(__name__)


class UnauthorizedExecutionError(Exception):
    """Raised when execution is attempted without valid approval."""
    pass


@dataclass
class EnforcementResult:
    """Result of approval enforcement check."""
    is_authorized: bool
    reason: Optional[str] = None
    token_id: Optional[str] = None
    checked_at: datetime = None

    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now()


class ApprovalEnforcer:
    """
    Enforces approval gating for all consequential actions.

    No action that modifies state, processes payments, or accesses
    sensitive data can execute without a valid approval token.
    """

    # Actions that ALWAYS require approval
    ALWAYS_REQUIRE_APPROVAL = {
        'process_refund',
        'charge_card',
        'update_booking',
        'cancel_booking',
        'upgrade_room',
        'grant_access',
        'modify_reservation',
        'apply_discount',
        'waive_fee',
    }

    # Actions that require approval above certain thresholds
    CONDITIONAL_APPROVAL = {
        'process_refund': {'threshold_key': 'amount', 'threshold_value': 0},
        'apply_discount': {'threshold_key': 'percentage', 'threshold_value': 10},
        'waive_fee': {'threshold_key': 'amount', 'threshold_value': 0},
    }

    def __init__(self, token_expiry_minutes: int = 30):
        """
        Initialize approval enforcer.

        Args:
            token_expiry_minutes: How long approval tokens remain valid
        """
        self.token_expiry_minutes = token_expiry_minutes
        self.logger = logging.getLogger(__name__)
        self._used_tokens: set = set()  # Track used tokens to prevent replay

    def requires_approval(
        self, 
        action_name: str, 
        action_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if an action requires approval.

        Args:
            action_name: Name of the action/tool
            action_params: Parameters for the action

        Returns:
            True if approval is required
        """
        # Always require approval for certain actions
        if action_name in self.ALWAYS_REQUIRE_APPROVAL:
            return True

        # Check conditional approval requirements
        if action_name in self.CONDITIONAL_APPROVAL and action_params:
            condition = self.CONDITIONAL_APPROVAL[action_name]
            threshold_key = condition['threshold_key']
            threshold_value = condition['threshold_value']

            if threshold_key in action_params:
                param_value = action_params[threshold_key]
                if param_value > threshold_value:
                    return True

        return False

    def enforce(
        self,
        action_name: str,
        action_params: Dict[str, Any],
        approval_token: Optional[ApprovalToken] = None
    ) -> EnforcementResult:
        """
        Enforce approval requirement for an action.

        Args:
            action_name: Name of the action
            action_params: Action parameters
            approval_token: Approval token (if provided)

        Returns:
            EnforcementResult indicating if action is authorized

        Raises:
            UnauthorizedExecutionError: If action requires approval but token is invalid
        """
        # Check if approval is required
        if not self.requires_approval(action_name, action_params):
            return EnforcementResult(
                is_authorized=True,
                reason="Action does not require approval"
            )

        # Approval is required - validate token
        if approval_token is None:
            self.logger.error(
                f"Attempted execution of {action_name} without approval token"
            )
            raise UnauthorizedExecutionError(
                f"Action '{action_name}' requires approval but no token provided"
            )

        # Validate token
        validation_result = self._validate_token(approval_token, action_name, action_params)

        if not validation_result.is_authorized:
            self.logger.error(
                f"Invalid approval token for {action_name}: {validation_result.reason}"
            )
            raise UnauthorizedExecutionError(
                f"Invalid approval token: {validation_result.reason}"
            )

        # Mark token as used
        self._used_tokens.add(approval_token.token_id)

        self.logger.info(
            f"Approved execution of {action_name} with token {approval_token.token_id}"
        )

        return validation_result

    def _validate_token(
        self,
        token: ApprovalToken,
        action_name: str,
        action_params: Dict[str, Any]
    ) -> EnforcementResult:
        """
        Validate an approval token.

        Checks:
        1. Token status is APPROVED
        2. Token hasn't expired
        3. Token hasn't been used before (prevent replay)
        4. Token matches the action being executed

        Args:
            token: Approval token to validate
            action_name: Action being executed
            action_params: Action parameters

        Returns:
            EnforcementResult
        """
        # Check token status
        if token.status != ApprovalStatus.APPROVED:
            return EnforcementResult(
                is_authorized=False,
                reason=f"Token status is {token.status.value}, not APPROVED",
                token_id=token.token_id
            )

        # Check expiry
        if token.expires_at and datetime.now() > token.expires_at:
            return EnforcementResult(
                is_authorized=False,
                reason=f"Token expired at {token.expires_at}",
                token_id=token.token_id
            )

        # Check if token was already used (prevent replay attacks)
        if token.token_id in self._used_tokens:
            return EnforcementResult(
                is_authorized=False,
                reason="Token has already been used",
                token_id=token.token_id
            )

        # Validate token matches the action
        # In a real system, token would contain action details to verify
        # For POC, we trust that the token was issued for this action

        return EnforcementResult(
            is_authorized=True,
            reason="Token is valid",
            token_id=token.token_id
        )

    def get_approval_requirements(
        self, 
        action_name: str, 
        action_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get approval requirements for an action.

        Useful for displaying to users what needs approval and why.

        Args:
            action_name: Name of the action
            action_params: Action parameters

        Returns:
            Dictionary describing approval requirements
        """
        requires_approval = self.requires_approval(action_name, action_params)

        if not requires_approval:
            return {
                'requires_approval': False,
                'reason': 'Action is safe and does not require approval'
            }

        # Determine reason
        reason = "Unknown"
        risk_level = "medium"

        if action_name in self.ALWAYS_REQUIRE_APPROVAL:
            if 'refund' in action_name or 'charge' in action_name:
                reason = "Financial transaction"
                risk_level = "high"
            elif 'booking' in action_name or 'reservation' in action_name:
                reason = "Booking modification"
                risk_level = "high"
            elif 'access' in action_name:
                reason = "Access control change"
                risk_level = "medium"
            else:
                reason = "Consequential action"
                risk_level = "medium"

        elif action_name in self.CONDITIONAL_APPROVAL and action_params:
            condition = self.CONDITIONAL_APPROVAL[action_name]
            threshold_key = condition['threshold_key']
            threshold_value = condition['threshold_value']
            param_value = action_params.get(threshold_key, 0)

            reason = f"{threshold_key.title()} ({param_value}) exceeds threshold ({threshold_value})"
            risk_level = "high" if param_value > threshold_value * 2 else "medium"

        return {
            'requires_approval': True,
            'reason': reason,
            'risk_level': risk_level,
            'action': action_name,
            'parameters': action_params or {}
        }


# Singleton instance
_enforcer_instance: Optional[ApprovalEnforcer] = None


def get_approval_enforcer() -> ApprovalEnforcer:
    """Get singleton approval enforcer instance."""
    global _enforcer_instance
    if _enforcer_instance is None:
        _enforcer_instance = ApprovalEnforcer()
    return _enforcer_instance
