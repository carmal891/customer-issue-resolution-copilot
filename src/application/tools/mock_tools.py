"""
Mock Tool Implementations

These are mock implementations of tools for demonstration and testing.
In production, these would be replaced with real API integrations using
the adapter pattern.
"""

from typing import Any, Dict, Optional
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.application.tools.base import Tool, ToolResult, ToolStatus


class LookupBookingTool(Tool):
    """
    Look up booking details by booking ID.

    In production, this would query the PMS (Property Management System).
    """

    def __init__(self):
        super().__init__(
            name="lookup_booking",
            description="Retrieve booking details including guest info, room, dates, and charges"
        )
        # Load mock bookings data
        self._bookings = self._load_mock_bookings()

    def _load_mock_bookings(self) -> Dict[str, Any]:
        """Load mock bookings from JSON file"""
        try:
            bookings_file = Path("data/mock/bookings/bookings.json")
            if bookings_file.exists():
                with open(bookings_file, 'r') as f:
                    data = json.load(f)
                    # Index by booking_id for fast lookup
                    return {b["booking_id"]: b for b in data.get("bookings", [])}
            return {}
        except Exception as e:
            print(f"Warning: Could not load mock bookings: {e}")
            return {}

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "Unique booking identifier"
                },
                "include_billing": {
                    "type": "boolean",
                    "description": "Include detailed billing information",
                    "default": False
                },
                "get_original_quote": {
                    "type": "boolean",
                    "description": "Include original booking quote",
                    "default": False
                }
            },
            "required": ["booking_id"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        booking_id = kwargs["booking_id"]
        include_billing = kwargs.get("include_billing", False)
        get_original_quote = kwargs.get("get_original_quote", False)

        # Look up booking
        booking = self._bookings.get(booking_id)

        if not booking:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error=f"Booking {booking_id} not found"
            )

        # Build response
        result_data = {
            "booking_id": booking["booking_id"],
            "guest_name": booking["guest_name"],
            "guest_email": booking["guest_email"],
            "room_number": booking["room_number"],
            "room_type": booking["room_type"],
            "check_in_date": booking["check_in_date"],
            "checkout_date": booking["checkout_date"],
            "status": booking["status"],
            "loyalty_tier": booking.get("loyalty_tier", "none"),
            "total_amount": booking["total_amount"]
        }

        if include_billing:
            result_data["billing_details"] = {
                "room_charges": booking["total_amount"] * 0.85,
                "taxes": booking["total_amount"] * 0.15,
                "additional_charges": [],
                "payment_method": "credit_card",
                "payment_status": "paid"
            }

        if get_original_quote:
            result_data["original_quote"] = {
                "room_rate": booking["total_amount"] * 0.85,
                "tax_rate": 0.15,
                "subtotal": booking["total_amount"] * 0.85,
                "included_items": ["room", "wifi", "breakfast"]
            }

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=result_data
        )


class CheckPolicyTool(Tool):
    """
    Check hotel policies by type or query.

    In production, this would query a policy management system or use RAG.
    """

    def __init__(self):
        super().__init__(
            name="check_policy",
            description="Retrieve hotel policies for cancellation, refunds, upgrades, etc."
        )
        self._policies = self._load_mock_policies()

    def _load_mock_policies(self) -> Dict[str, str]:
        """Load mock policies from markdown files"""
        policies = {}
        try:
            policies_dir = Path("data/mock/policies")
            if policies_dir.exists():
                for policy_file in policies_dir.glob("*.md"):
                    policy_name = policy_file.stem
                    with open(policy_file, 'r') as f:
                        policies[policy_name] = f.read()
            return policies
        except Exception as e:
            print(f"Warning: Could not load mock policies: {e}")
            return {}

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "policy_type": {
                    "type": "string",
                    "description": "Type of policy to retrieve",
                    "enum": ["cancellation", "refund", "upgrade", "billing", "accessibility", "loyalty", "complaint"]
                },
                "query": {
                    "type": "string",
                    "description": "Specific query about the policy"
                }
            },
            "required": ["policy_type"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        policy_type = kwargs["policy_type"]
        query = kwargs.get("query", "")

        # Map policy types to file names
        policy_map = {
            "cancellation": "cancellation_refund_policy",
            "refund": "cancellation_refund_policy",
            "upgrade": "room_upgrade_policy",
            "billing": "cancellation_refund_policy",
            "accessibility": "accessibility_policy",
            "loyalty": "loyalty_program_policy",
            "complaint": "complaint_handling_policy"
        }

        policy_file = policy_map.get(policy_type)
        if not policy_file or policy_file not in self._policies:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error=f"Policy type '{policy_type}' not found"
            )

        policy_content = self._policies[policy_file]

        # Extract relevant section if query provided
        if query:
            # Simple keyword matching (in production, use RAG)
            lines = policy_content.split('\n')
            relevant_lines = [
                line for line in lines
                if query.lower() in line.lower()
            ]
            if relevant_lines:
                policy_content = '\n'.join(relevant_lines[:10])  # First 10 matches

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "policy_type": policy_type,
                "content": policy_content[:1000],  # Truncate for demo
                "full_policy_available": True
            }
        )


class ProcessRefundTool(Tool):
    """
    Process refund to guest's payment method.

    In production, this would integrate with payment gateway.
    """

    def __init__(self):
        super().__init__(
            name="process_refund",
            description="Process refund to guest's original payment method"
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "Booking ID for the refund"
                },
                "amount": {
                    "type": "number",
                    "description": "Refund amount in dollars"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for refund"
                },
                "payment_method": {
                    "type": "string",
                    "description": "Payment method to refund to",
                    "default": "original"
                },
                "adjustment_type": {
                    "type": "string",
                    "description": "Type of adjustment",
                    "enum": ["refund", "billing_correction", "compensation"],
                    "default": "refund"
                }
            },
            "required": ["booking_id", "amount", "reason"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        booking_id = kwargs["booking_id"]
        amount = kwargs["amount"]
        reason = kwargs["reason"]

        # Validate amount
        if amount < 0:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error="Refund amount must be positive"
            )

        # Simulate refund processing
        transaction_id = f"REF-{booking_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Simulate 95% success rate
        if random.random() < 0.95:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "transaction_id": transaction_id,
                    "booking_id": booking_id,
                    "amount": amount,
                    "status": "processed",
                    "estimated_arrival": "5-7 business days",
                    "reason": reason
                },
                requires_approval=amount > 100,  # Require approval for large refunds
                approval_reason=f"Refund of ${amount} requires manager approval"
            )
        else:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error="Payment gateway temporarily unavailable"
            )


class SendEmailTool(Tool):
    """
    Send email to guest.

    In production, this would integrate with email service (SendGrid, SES, etc.).
    """

    def __init__(self):
        super().__init__(
            name="send_email",
            description="Send email notification to guest"
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "template": {
                    "type": "string",
                    "description": "Email template name"
                },
                "data": {
                    "type": "object",
                    "description": "Template data"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject (optional, derived from template)"
                }
            },
            "required": ["to", "template", "data"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        to = kwargs["to"]
        template = kwargs["template"]
        data = kwargs["data"]

        # Validate email
        if "@" not in to:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error="Invalid email address"
            )

        # Simulate email sending
        message_id = f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "message_id": message_id,
                "to": to,
                "template": template,
                "status": "sent",
                "sent_at": datetime.now().isoformat()
            }
        )


class UpdatePMSTool(Tool):
    """
    Update Property Management System.

    In production, this would integrate with PMS API (Opera, Mews, etc.).
    """

    def __init__(self):
        super().__init__(
            name="update_pms",
            description="Update property management system with booking changes or service requests"
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["create_service_request", "release_room", "update_room_status", "add_note"]
                },
                "booking_id": {
                    "type": "string",
                    "description": "Booking ID (if applicable)"
                },
                "room_number": {
                    "type": "string",
                    "description": "Room number (if applicable)"
                },
                "data": {
                    "type": "object",
                    "description": "Action-specific data"
                }
            },
            "required": ["action"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]

        # Simulate PMS update
        ticket_id = f"PMS-{action.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "ticket_id": ticket_id,
                "action": action,
                "status": "completed",
                "updated_at": datetime.now().isoformat()
            }
        )


class NotifyTeamTool(Tool):
    """
    Notify internal team (housekeeping, maintenance, front desk, etc.).

    In production, this would integrate with team communication system.
    """

    def __init__(self):
        super().__init__(
            name="notify_team",
            description="Send notification to internal team"
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team to notify",
                    "enum": ["housekeeping", "maintenance", "front_desk", "management", "concierge"]
                },
                "notification_type": {
                    "type": "string",
                    "description": "Type of notification",
                    "enum": ["service_request", "urgent", "info", "escalation"]
                },
                "data": {
                    "type": "object",
                    "description": "Notification data"
                }
            },
            "required": ["team", "notification_type", "data"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        team = kwargs["team"]
        notification_type = kwargs["notification_type"]
        data = kwargs["data"]

        # Simulate team notification
        notification_id = f"NOTIF-{team.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "notification_id": notification_id,
                "team": team,
                "type": notification_type,
                "status": "delivered",
                "delivered_at": datetime.now().isoformat()
            }
        )


class CheckRoomAvailabilityTool(Tool):
    """
    Check room availability for specific dates or times.

    In production, this would query PMS availability system.
    """

    def __init__(self):
        super().__init__(
            name="check_room_availability",
            description="Check room availability for dates, times, or upgrades"
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "room_number": {
                    "type": "string",
                    "description": "Specific room number (optional)"
                },
                "room_type": {
                    "type": "string",
                    "description": "Room type/category"
                },
                "check_in_date": {
                    "type": "string",
                    "description": "Check-in date (YYYY-MM-DD)"
                },
                "check_out_date": {
                    "type": "string",
                    "description": "Check-out date (YYYY-MM-DD)"
                },
                "date": {
                    "type": "string",
                    "description": "Specific date to check"
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range (e.g., '12:00-16:00')"
                },
                "room_category": {
                    "type": "string",
                    "description": "Category filter (e.g., 'upgrade')"
                },
                "current_room_type": {
                    "type": "string",
                    "description": "Current room type for upgrade checks"
                }
            }
        }

    def _execute(self, **kwargs) -> ToolResult:
        # Simulate availability check
        # In production, this would query real availability data

        # Simulate 70% availability
        is_available = random.random() < 0.7

        if is_available:
            # Generate mock available rooms/times
            if kwargs.get("room_category") == "upgrade":
                available_upgrades = [
                    {
                        "type": "deluxe",
                        "number": "301",
                        "rate": 250,
                        "amenities": ["king_bed", "city_view", "balcony"]
                    },
                    {
                        "type": "suite",
                        "number": "401",
                        "rate": 350,
                        "amenities": ["king_bed", "ocean_view", "balcony", "living_room"]
                    }
                ]
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "available": True,
                        "upgrades": available_upgrades
                    }
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "available": True,
                        "next_booking_time": "15:00",
                        "has_conflict": False
                    }
                )
        else:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "available": False,
                    "next_booking_time": "11:00",
                    "has_conflict": True,
                    "reason": "Room booked for incoming guest"
                }
            )


class UpdateBookingTool(Tool):
    """
    Update booking details (room, dates, charges, status, etc.).

    In production, this would update PMS booking records.
    """

    def __init__(self):
        super().__init__(
            name="update_booking",
            description="Update booking details including room, dates, charges, and status"
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "Booking ID to update"
                },
                "status": {
                    "type": "string",
                    "description": "New booking status",
                    "enum": ["confirmed", "cancelled", "checked_in", "checked_out", "no_show"]
                },
                "checkout_time": {
                    "type": "string",
                    "description": "Updated checkout time"
                },
                "additional_charges": {
                    "type": "number",
                    "description": "Additional charges to add"
                },
                "new_room_type": {
                    "type": "string",
                    "description": "New room type for upgrades"
                },
                "new_room_number": {
                    "type": "string",
                    "description": "New room number"
                },
                "notes": {
                    "type": "string",
                    "description": "Notes about the update"
                },
                "cancellation_date": {
                    "type": "string",
                    "description": "Cancellation date"
                },
                "cancellation_reason": {
                    "type": "string",
                    "description": "Reason for cancellation"
                },
                "refund_amount": {
                    "type": "number",
                    "description": "Refund amount"
                },
                "billing_adjustments": {
                    "type": "array",
                    "description": "Billing adjustments"
                },
                "final_amount": {
                    "type": "number",
                    "description": "Final billing amount"
                },
                "add_charge": {
                    "type": "object",
                    "description": "Add a new charge"
                },
                "remove_charge": {
                    "type": "object",
                    "description": "Remove a charge"
                }
            },
            "required": ["booking_id"]
        }

    def _execute(self, **kwargs) -> ToolResult:
        booking_id = kwargs["booking_id"]

        # Simulate booking update
        update_id = f"UPD-{booking_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Build update summary
        updates = []
        for key, value in kwargs.items():
            if key != "booking_id" and value is not None:
                updates.append(f"{key}: {value}")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "update_id": update_id,
                "booking_id": booking_id,
                "status": "success",
                "updates_applied": updates,
                "updated_at": datetime.now().isoformat()
            },
            requires_approval=True,
            approval_reason="Booking modification requires approval"
        )


# Factory function to create all tools
def create_default_tools() -> list[Tool]:
    """
    Create all default mock tools.

    Returns:
        List of Tool instances
    """
    return [
        LookupBookingTool(),
        CheckPolicyTool(),
        ProcessRefundTool(),
        SendEmailTool(),
        UpdatePMSTool(),
        NotifyTeamTool(),
        CheckRoomAvailabilityTool(),
        UpdateBookingTool()
    ]
