"""Approval-related domain models."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from decimal import Decimal
from pydantic import BaseModel, Field
import secrets


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskLevel(str, Enum):
    """Risk level of an action requiring approval."""

    LOW = "low"  # Informational, no state change
    MEDIUM = "medium"  # Minor state change, reversible
    HIGH = "high"  # Significant state change, financial impact
    CRITICAL = "critical"  # Major financial or operational impact


class ApprovalRequest(BaseModel):
    """Request for human approval of an action."""

    request_id: str = Field(
        default_factory=lambda: f"APR_{secrets.token_hex(8)}",
        description="Unique approval request identifier",
    )
    issue_id: str = Field(..., description="Related issue identifier")
    action_type: str = Field(..., description="Type of action requiring approval")
    action_description: str = Field(..., description="Human-readable description of action")
    risk_level: RiskLevel = Field(..., description="Risk level of the action")
    proposed_changes: Dict[str, Any] = Field(
        ..., description="Proposed changes or actions"
    )
    financial_impact: Optional[Decimal] = Field(
        None, description="Financial impact in USD if applicable"
    )
    affected_booking_id: Optional[str] = Field(
        None, description="Booking ID affected by this action"
    )
    supporting_context: Dict[str, Any] = Field(
        default_factory=dict, description="Supporting context for decision"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When request was created"
    )
    expires_at: Optional[datetime] = Field(
        None, description="When request expires if not acted upon"
    )
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Current status"
    )
    approved_by: Optional[str] = Field(None, description="Who approved the request")
    approved_at: Optional[datetime] = Field(None, description="When request was approved")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection")
    notes: Optional[str] = Field(None, description="Additional notes from approver")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "request_id": "APR_a1b2c3d4e5f6g7h8",
                "issue_id": "ISS_001",
                "action_type": "process_refund",
                "action_description": "Process $85 refund for booking BK12345 due to cancellation 7 days in advance",
                "risk_level": "medium",
                "proposed_changes": {
                    "booking_id": "BK12345",
                    "refund_amount": 85.00,
                    "refund_reason": "Cancellation per policy",
                    "refund_method": "original_payment",
                },
                "financial_impact": 85.00,
                "affected_booking_id": "BK12345",
                "supporting_context": {
                    "policy": "100% refund for cancellations 7+ days before check-in",
                    "guest_loyalty_tier": "gold",
                },
                "status": "pending",
            }
        }

    def is_pending(self) -> bool:
        """Check if request is pending."""
        return self.status == ApprovalStatus.PENDING

    def is_approved(self) -> bool:
        """Check if request is approved."""
        return self.status == ApprovalStatus.APPROVED

    def is_high_risk(self) -> bool:
        """Check if request is high or critical risk."""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def requires_immediate_attention(self) -> bool:
        """Check if request requires immediate attention."""
        return self.risk_level == RiskLevel.CRITICAL or (
            self.financial_impact is not None and self.financial_impact > 500
        )

    def approve(self, approved_by: str, notes: Optional[str] = None) -> None:
        """Approve the request."""
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()
        if notes:
            self.notes = notes

    def reject(self, rejected_by: str, reason: str, notes: Optional[str] = None) -> None:
        """Reject the request."""
        self.status = ApprovalStatus.REJECTED
        self.approved_by = rejected_by  # Track who rejected
        self.approved_at = datetime.utcnow()
        self.rejection_reason = reason
        if notes:
            self.notes = notes


class ApprovalToken(BaseModel):
    """Token authorizing execution of an approved action."""

    token: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secure approval token",
    )
    request_id: str = Field(..., description="Related approval request ID")
    issue_id: str = Field(..., description="Related issue ID")
    action_type: str = Field(..., description="Type of action authorized")
    approved_by: str = Field(..., description="Who approved the action")
    issued_at: datetime = Field(
        default_factory=datetime.utcnow, description="When token was issued"
    )
    expires_at: datetime = Field(..., description="When token expires")
    used: bool = Field(default=False, description="Whether token has been used")
    used_at: Optional[datetime] = Field(None, description="When token was used")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
                "request_id": "APR_a1b2c3d4e5f6g7h8",
                "issue_id": "ISS_001",
                "action_type": "process_refund",
                "approved_by": "manager@hotel.com",
                "issued_at": "2024-06-01T10:30:00Z",
                "expires_at": "2024-06-01T11:30:00Z",
                "used": False,
            }
        }

    def is_valid(self) -> bool:
        """Check if token is valid (not used and not expired)."""
        return not self.used and datetime.utcnow() < self.expires_at

    def mark_used(self) -> None:
        """Mark token as used."""
        self.used = True
        self.used_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() >= self.expires_at
