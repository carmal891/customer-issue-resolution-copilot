"""Resolution-related domain models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ResolutionStatus(str, Enum):
    """Status of issue resolution."""

    IN_PROGRESS = "in_progress"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class ResolutionStep(BaseModel):
    """Individual step taken during resolution."""

    step_number: int = Field(..., ge=1, description="Step sequence number")
    step_type: str = Field(..., description="Type of step (retrieval, tool_call, reasoning)")
    description: str = Field(..., description="Description of what was done")
    tool_used: Optional[str] = Field(default=None, description="Tool used if applicable")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input to the step")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output from the step")
    success: bool = Field(..., description="Whether step was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if step failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When step was executed"
    )
    duration_ms: Optional[float] = Field(None, ge=0, description="Step duration in milliseconds")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "step_number": 1,
                "step_type": "tool_call",
                "description": "Looked up booking details",
                "tool_used": "lookup_booking",
                "input_data": {"booking_id": "BK12345"},
                "output_data": {
                    "booking": {
                        "booking_id": "BK12345",
                        "guest_name": "John Smith",
                        "check_in": "2024-06-15",
                    }
                },
                "success": True,
                "timestamp": "2024-06-01T10:30:00Z",
                "duration_ms": 45.2,
            }
        }


class Resolution(BaseModel):
    """Resolution entity representing the resolution of an issue."""

    resolution_id: str = Field(..., description="Unique resolution identifier")
    issue_id: str = Field(..., description="Related issue identifier")
    status: ResolutionStatus = Field(..., description="Current resolution status")
    skill_used: Optional[str] = Field(
        None, description="Skill ID if existing skill was used"
    )
    skill_matched: bool = Field(
        default=False, description="Whether an existing skill was matched"
    )
    novel_task: bool = Field(
        default=False, description="Whether this was handled as a novel task"
    )
    steps: List[ResolutionStep] = Field(
        default_factory=list, description="Steps taken during resolution"
    )
    approval_request_id: Optional[str] = Field(
        None, description="Approval request ID if approval was required"
    )
    approval_granted: Optional[bool] = Field(
        None, description="Whether approval was granted"
    )
    outcome: Optional[str] = Field(None, description="Final outcome description")
    guest_satisfaction: Optional[int] = Field(
        None, ge=1, le=5, description="Guest satisfaction rating (1-5)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When resolution started"
    )
    completed_at: Optional[datetime] = Field(
        None, description="When resolution completed"
    )
    total_duration_ms: Optional[float] = Field(
        None, ge=0, description="Total resolution time in milliseconds"
    )
    compiled_skill_id: Optional[str] = Field(
        None, description="ID of skill compiled from this resolution if novel task"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "resolution_id": "RES_001",
                "issue_id": "ISS_001",
                "status": "completed",
                "skill_used": "late_checkout_request",
                "skill_matched": True,
                "novel_task": False,
                "steps": [
                    {
                        "step_number": 1,
                        "step_type": "tool_call",
                        "description": "Looked up booking",
                        "tool_used": "lookup_booking",
                        "input_data": {"booking_id": "BK12345"},
                        "output_data": {"booking": {}},
                        "success": True,
                        "timestamp": "2024-06-01T10:30:00Z",
                    }
                ],
                "approval_request_id": "APR_a1b2c3d4",
                "approval_granted": True,
                "outcome": "Late checkout approved until 2pm (Gold member benefit)",
                "guest_satisfaction": 5,
                "created_at": "2024-06-01T10:30:00Z",
                "completed_at": "2024-06-01T10:35:00Z",
                "total_duration_ms": 300000,
            }
        }

    def add_step(self, step: ResolutionStep) -> None:
        """Add a resolution step."""
        self.steps.append(step)

    def is_complete(self) -> bool:
        """Check if resolution is complete."""
        return self.status in [
            ResolutionStatus.COMPLETED,
            ResolutionStatus.FAILED,
            ResolutionStatus.ESCALATED,
        ]

    def is_successful(self) -> bool:
        """Check if resolution was successful."""
        return self.status == ResolutionStatus.COMPLETED and all(
            step.success for step in self.steps
        )

    def requires_approval(self) -> bool:
        """Check if resolution requires approval."""
        return self.status == ResolutionStatus.PENDING_APPROVAL

    def mark_completed(self, outcome: str) -> None:
        """Mark resolution as completed."""
        self.status = ResolutionStatus.COMPLETED
        self.outcome = outcome
        self.completed_at = datetime.utcnow()
        if self.created_at:
            self.total_duration_ms = (
                self.completed_at - self.created_at
            ).total_seconds() * 1000

    def mark_failed(self, error_message: str) -> None:
        """Mark resolution as failed."""
        self.status = ResolutionStatus.FAILED
        self.outcome = f"Failed: {error_message}"
        self.completed_at = datetime.utcnow()
        if self.created_at:
            self.total_duration_ms = (
                self.completed_at - self.created_at
            ).total_seconds() * 1000

    def mark_escalated(self, reason: str) -> None:
        """Mark resolution as escalated."""
        self.status = ResolutionStatus.ESCALATED
        self.outcome = f"Escalated: {reason}"
        self.completed_at = datetime.utcnow()
        if self.created_at:
            self.total_duration_ms = (
                self.completed_at - self.created_at
            ).total_seconds() * 1000

    @property
    def step_count(self) -> int:
        """Get number of steps taken."""
        return len(self.steps)

    @property
    def successful_steps(self) -> int:
        """Get number of successful steps."""
        return sum(1 for step in self.steps if step.success)

    @property
    def failed_steps(self) -> int:
        """Get number of failed steps."""
        return sum(1 for step in self.steps if not step.success)
