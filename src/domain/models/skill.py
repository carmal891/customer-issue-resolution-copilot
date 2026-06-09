"""Skill-related domain models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SkillStatus(str, Enum):
    """Status of a skill in the registry."""

    DRAFT = "draft"  # Newly compiled, needs review
    ACTIVE = "active"  # Approved and ready to use
    DEPRECATED = "deprecated"  # No longer recommended
    ARCHIVED = "archived"  # Removed from active use


class SkillStepType(str, Enum):
    """Types of steps in a skill."""

    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    GENERATE_PLAN = "generate_plan"
    HUMAN_APPROVAL = "human_approval"


class SkillStep(BaseModel):
    """Individual step in a skill execution."""

    step_id: str = Field(..., description="Unique step identifier within skill")
    step_type: SkillStepType = Field(..., description="Type of step")
    description: str = Field(..., description="Human-readable description of step")
    tool_name: Optional[str] = Field(None, description="Tool to call if step_type is tool_call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the step"
    )
    requires_approval: bool = Field(
        default=False, description="Whether this step requires human approval"
    )
    approval_reason: Optional[str] = Field(
        None, description="Reason why approval is required"
    )
    expected_output: Optional[str] = Field(
        None, description="Description of expected output"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "step_id": "lookup_booking",
                "step_type": "tool_call",
                "description": "Lookup booking details from PMS",
                "tool_name": "lookup_booking",
                "parameters": {"booking_id": "{booking_id}"},
                "requires_approval": False,
                "expected_output": "Booking details including guest info and dates",
            }
        }


class Skill(BaseModel):
    """Skill entity representing a reusable workflow."""

    skill_id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    version: str = Field(default="1.0", description="Skill version")
    status: SkillStatus = Field(default=SkillStatus.DRAFT, description="Skill status")
    description: str = Field(..., description="Detailed description of what skill does")
    triggers: List[str] = Field(
        ..., description="Trigger phrases or patterns that match this skill"
    )
    steps: List[SkillStep] = Field(..., description="Ordered list of execution steps")
    guardrails: Dict[str, Any] = Field(
        default_factory=dict, description="Guardrails and constraints"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (created_from, success_rate, avg_time)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When skill was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When skill was last updated"
    )
    created_by: str = Field(default="system", description="Who created the skill")
    usage_count: int = Field(default=0, ge=0, description="Number of times skill was used")
    success_count: int = Field(default=0, ge=0, description="Number of successful executions")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "skill_id": "late_checkout_request",
                "name": "Late Checkout Request Handler",
                "version": "1.0",
                "status": "active",
                "description": "Handle guest requests for late checkout based on loyalty tier and availability",
                "triggers": [
                    "late checkout",
                    "extend checkout",
                    "check out later",
                    "stay longer",
                ],
                "steps": [
                    {
                        "step_id": "lookup_booking",
                        "step_type": "tool_call",
                        "description": "Lookup booking details",
                        "tool_name": "lookup_booking",
                        "parameters": {"booking_id": "{booking_id}"},
                        "requires_approval": False,
                    },
                    {
                        "step_id": "check_availability",
                        "step_type": "tool_call",
                        "description": "Check room availability for extended hours",
                        "tool_name": "check_room_availability",
                        "parameters": {"room_number": "{room_number}", "date": "{date}"},
                        "requires_approval": False,
                    },
                    {
                        "step_id": "determine_eligibility",
                        "step_type": "reasoning",
                        "description": "Determine late checkout eligibility based on loyalty tier",
                        "parameters": {},
                        "requires_approval": False,
                    },
                    {
                        "step_id": "update_checkout",
                        "step_type": "tool_call",
                        "description": "Update checkout time in PMS",
                        "tool_name": "update_pms",
                        "parameters": {"booking_id": "{booking_id}", "new_checkout": "{new_time}"},
                        "requires_approval": True,
                        "approval_reason": "Booking modification",
                    },
                ],
                "guardrails": {
                    "max_late_checkout_hours": 4,
                    "requires_availability": True,
                },
                "metadata": {
                    "created_from": "manual",
                    "avg_resolution_time_minutes": 5,
                },
                "usage_count": 42,
                "success_count": 40,
            }
        }

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def is_active(self) -> bool:
        """Check if skill is active and ready to use."""
        return self.status == SkillStatus.ACTIVE

    def requires_human_approval(self) -> bool:
        """Check if any step requires human approval."""
        return any(step.requires_approval for step in self.steps)

    def increment_usage(self, success: bool = True) -> None:
        """Increment usage counters."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.updated_at = datetime.utcnow()
