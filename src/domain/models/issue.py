"""Issue-related domain models."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Types of customer issues."""

    CANCELLATION = "cancellation"
    REFUND = "refund"
    ROOM_UPGRADE = "room_upgrade"
    LATE_CHECKOUT = "late_checkout"
    BILLING = "billing"
    COMPLAINT = "complaint"
    ACCESSIBILITY = "accessibility"
    AMENITY_REQUEST = "amenity_request"
    BOOKING_MODIFICATION = "booking_modification"
    SPECIAL_REQUEST = "special_request"
    OTHER = "other"


class IssueChannel(str, Enum):
    """Channels through which issues are received."""

    EMAIL = "email"
    CHAT = "chat"
    PHONE = "phone"
    BOOKING_COM = "booking_com"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    IN_PERSON = "in_person"
    MOBILE_APP = "mobile_app"


class IssuePriority(str, Enum):
    """Priority levels for issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Issue(BaseModel):
    """Issue entity representing a customer issue or request."""

    issue_id: str = Field(..., description="Unique issue identifier")
    channel: IssueChannel = Field(..., description="Channel through which issue was received")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When issue was received"
    )
    guest_email: Optional[str] = Field(None, description="Guest email if available")
    booking_id: Optional[str] = Field(None, description="Related booking ID if applicable")
    subject: Optional[str] = Field(None, description="Issue subject line")
    body: str = Field(..., description="Issue description or message body")
    issue_type: Optional[IssueType] = Field(None, description="Classified issue type")
    priority: IssuePriority = Field(
        default=IssuePriority.MEDIUM, description="Issue priority"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (e.g., from_user, thread_id)"
    )
    expected_skill: Optional[str] = Field(
        None, description="Expected skill for evaluation (ground truth)"
    )
    expected_resolution: Optional[str] = Field(
        None, description="Expected resolution for evaluation (ground truth)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "issue_id": "ISS_001",
                "channel": "email",
                "timestamp": "2024-06-01T10:30:00Z",
                "guest_email": "john.smith@email.com",
                "booking_id": "BK12345",
                "subject": "Late checkout request",
                "body": "Hi, I have a booking for tomorrow. My flight is at 6pm, can I get a late checkout until 2pm? I'm a Gold member. Thanks!",
                "issue_type": "late_checkout",
                "priority": "medium",
                "metadata": {},
                "expected_skill": "late_checkout_request",
                "expected_resolution": "Approve late checkout until 2pm (Gold member benefit)",
            }
        }

    def is_high_priority(self) -> bool:
        """Check if issue is high priority."""
        return self.priority in [IssuePriority.HIGH, IssuePriority.URGENT]

    def has_booking_context(self) -> bool:
        """Check if issue has associated booking."""
        return self.booking_id is not None

    def requires_immediate_attention(self) -> bool:
        """Check if issue requires immediate attention."""
        return self.priority == IssuePriority.URGENT or self.channel == IssueChannel.TWITTER
