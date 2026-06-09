"""
Generate mock historical resolutions for the Customer Issue Resolution Copilot.
This script creates realistic resolution records showing patterns and successful outcomes.
"""

import json
from datetime import datetime, timedelta
import random

# Configuration
NUM_RESOLUTIONS = 35
BASE_DATE = datetime(2024, 3, 22)

# Resolution templates by issue type
RESOLUTION_TEMPLATES = {
    "LATE_CHECKOUT": {
        "steps": [
            "Verified guest loyalty status and current occupancy",
            "Checked room availability for extended stay",
            "Approved late checkout until {time}",
            "Updated PMS with late checkout flag",
            "Sent confirmation email to guest"
        ],
        "tools": ["lookup_booking", "check_policy", "update_pms", "send_email"],
        "outcome": "Late checkout approved until {time}. Guest notified via email.",
        "approval_required": False
    },
    "ROOM_UPGRADE": {
        "steps": [
            "Verified guest loyalty tier and benefits",
            "Checked availability of upgrade rooms",
            "Calculated upgrade cost differential",
            "Processed complimentary upgrade for {loyalty} member",
            "Updated room assignment in PMS",
            "Sent upgrade confirmation to guest"
        ],
        "tools": ["lookup_booking", "check_policy", "update_pms", "send_email"],
        "outcome": "Complimentary upgrade to {room_type} approved. Guest moved to room {new_room}.",
        "approval_required": False
    },
    "CANCELLATION_REFUND": {
        "steps": [
            "Retrieved booking details and cancellation policy",
            "Verified cancellation timeline against policy",
            "Calculated refund amount based on policy",
            "Requested approval for ${amount} refund",
            "Processed refund to original payment method",
            "Sent cancellation confirmation email"
        ],
        "tools": ["lookup_booking", "check_policy", "process_refund", "send_email"],
        "outcome": "Booking cancelled. Full refund of ${amount} processed within policy guidelines.",
        "approval_required": True
    },
    "BILLING_DISPUTE": {
        "steps": [
            "Retrieved detailed billing records",
            "Compared charges against booking confirmation",
            "Identified billing error: {error_type}",
            "Requested approval for ${amount} adjustment",
            "Processed credit to guest account",
            "Sent corrected invoice to guest"
        ],
        "tools": ["lookup_booking", "check_policy", "process_refund", "send_email"],
        "outcome": "Billing error corrected. ${amount} credited to guest account.",
        "approval_required": True
    },
    "ROOM_ISSUE": {
        "steps": [
            "Logged maintenance request for room {room}",
            "Dispatched maintenance team immediately",
            "Verified issue resolution with guest",
            "Offered ${compensation} service recovery credit",
            "Updated room maintenance log"
        ],
        "tools": ["update_pms", "notify_team", "send_email"],
        "outcome": "{issue_type} resolved. Guest satisfied. ${compensation} credit applied.",
        "approval_required": False
    },
    "AMENITY_REQUEST": {
        "steps": [
            "Verified amenity availability",
            "Dispatched housekeeping to room {room}",
            "Confirmed delivery with guest",
            "Updated guest preferences in system"
        ],
        "tools": ["update_pms", "notify_team"],
        "outcome": "Amenity request fulfilled. Items delivered to room {room}.",
        "approval_required": False
    },
    "ACCESSIBILITY": {
        "steps": [
            "Reviewed accessibility requirements",
            "Verified accessible room availability",
            "Processed room change to accessible room {new_room}",
            "Confirmed accessibility features meet guest needs",
            "Updated booking with accessibility notes"
        ],
        "tools": ["lookup_booking", "check_policy", "update_pms", "send_email"],
        "outcome": "Guest moved to accessible room {new_room} with required features.",
        "approval_required": False
    },
    "COMPLAINT": {
        "steps": [
            "Documented complaint details thoroughly",
            "Escalated to {department} manager",
            "Conducted service recovery discussion with guest",
            "Offered ${compensation} compensation and {perk}",
            "Followed up to ensure guest satisfaction"
        ],
        "tools": ["update_pms", "send_email", "notify_team"],
        "outcome": "Complaint resolved. Guest accepted ${compensation} compensation plus {perk}.",
        "approval_required": True
    },
    "BOOKING_MODIFICATION": {
        "steps": [
            "Retrieved current booking details",
            "Checked availability for requested dates",
            "Calculated any rate differences",
            "Updated booking dates in PMS",
            "Sent modified confirmation to guest"
        ],
        "tools": ["lookup_booking", "check_policy", "update_pms", "send_email"],
        "outcome": "Booking modified successfully. New dates: {new_dates}.",
        "approval_required": False
    },
    "LOST_ITEM": {
        "steps": [
            "Checked lost and found database",
            "Contacted housekeeping for room {room}",
            "Located item: {item}",
            "Arranged shipping to guest address",
            "Sent tracking information to guest"
        ],
        "tools": ["update_pms", "send_email", "notify_team"],
        "outcome": "Item located and shipped to guest. Tracking: {tracking}.",
        "approval_required": False
    }
}

# Sample data for filling templates
ROOM_TYPES = ["Deluxe Ocean View", "Executive Suite", "Premium King"]
DEPARTMENTS = ["Guest Services", "Front Office", "Operations"]
PERKS = ["complimentary breakfast", "spa credit", "late checkout", "room upgrade"]
ERROR_TYPES = ["duplicate charge", "incorrect rate applied", "unauthorized minibar charge"]
ISSUE_TYPES = ["AC malfunction", "WiFi connectivity", "noise disturbance"]
ITEMS = ["phone charger", "laptop", "jacket", "book"]


def generate_resolution_id(index: int) -> str:
    """Generate a unique resolution ID."""
    return f"RES-2024-{2000 + index}"


def generate_resolution(index: int) -> dict:
    """Generate a single resolution record."""
    # Select issue type
    issue_type = random.choice(list(RESOLUTION_TEMPLATES.keys()))
    template = RESOLUTION_TEMPLATES[issue_type]

    # Generate timestamps
    days_ago = random.randint(5, 60)
    created_at = BASE_DATE - timedelta(days=days_ago)
    resolution_time = random.randint(30, 480)  # 30 min to 8 hours
    resolved_at = created_at + timedelta(minutes=resolution_time)

    # Generate booking reference
    booking_id = f"BK-2024-{random.randint(1, 55):03d}"
    room_number = f"{random.randint(2, 9)}{random.randint(1, 25):02d}"

    # Fill template variables
    variables = {
        "time": random.choice(["2:00 PM", "3:00 PM", "1:00 PM"]),
        "loyalty": random.choice(["GOLD", "PLATINUM", "SILVER"]),
        "room_type": random.choice(ROOM_TYPES),
        "new_room": f"{random.randint(6, 9)}{random.randint(1, 25):02d}",
        "amount": random.choice([150.00, 250.00, 350.00, 500.00]),
        "error_type": random.choice(ERROR_TYPES),
        "room": room_number,
        "issue_type": random.choice(ISSUE_TYPES),
        "compensation": random.choice([50, 75, 100]),
        "department": random.choice(DEPARTMENTS),
        "perk": random.choice(PERKS),
        "new_dates": f"{(created_at + timedelta(days=10)).strftime('%Y-%m-%d')} to {(created_at + timedelta(days=13)).strftime('%Y-%m-%d')}",
        "item": random.choice(ITEMS),
        "tracking": f"1Z{random.randint(100000, 999999)}"
    }

    # Build resolution steps
    steps = []
    for i, step_template in enumerate(template["steps"]):
        step = {
            "step_number": i + 1,
            "description": step_template.format(**variables),
            "status": "COMPLETED",
            "completed_at": (created_at + timedelta(minutes=i * (resolution_time // len(template["steps"])))).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        steps.append(step)

    # Build resolution
    resolution = {
        "resolution_id": generate_resolution_id(index),
        "issue_id": f"ISS-2024-{1000 + random.randint(1, 500)}",
        "booking_id": booking_id,
        "issue_type": issue_type,
        "priority": random.choice(["LOW", "MEDIUM", "HIGH"]),
        "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolved_at": resolved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolution_time_minutes": resolution_time,
        "assigned_agent": random.choice([
            "agent_sarah_m",
            "agent_john_d",
            "agent_maria_g",
            "agent_david_l"
        ]),
        "steps": steps,
        "tools_used": template["tools"],
        "outcome": template["outcome"].format(**variables),
        "approval_required": template["approval_required"],
        "guest_satisfaction": random.choice(["SATISFIED", "VERY_SATISFIED", "SATISFIED", "VERY_SATISFIED", "NEUTRAL"]),
        "metadata": {
            "room_number": room_number,
            "booking_id": booking_id,
            "channel": random.choice(["EMAIL", "CHAT", "PHONE", "SLACK"])
        }
    }

    # Add approval info if required
    if template["approval_required"]:
        approval_time = random.randint(10, 60)
        resolution["approval_details"] = {
            "requested_at": (created_at + timedelta(minutes=approval_time)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approved_by": random.choice(["manager_alice_j", "manager_robert_k", "manager_susan_l"]),
            "approved_at": (created_at + timedelta(minutes=approval_time + 5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approval_notes": "Approved within policy guidelines"
        }

    # Add follow-up if guest was very satisfied
    if resolution["guest_satisfaction"] == "VERY_SATISFIED":
        resolution["follow_up"] = {
            "conducted_at": (resolved_at + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": "EMAIL",
            "notes": "Guest confirmed satisfaction with resolution"
        }

    return resolution


def main():
    """Generate all resolutions and save to JSON file."""
    print(f"Generating {NUM_RESOLUTIONS} mock historical resolutions...")

    resolutions = []
    for i in range(NUM_RESOLUTIONS):
        resolution = generate_resolution(i)
        resolutions.append(resolution)

    # Save to file
    output_path = "data/mock/resolutions/historical_resolutions.json"
    with open(output_path, "w") as f:
        json.dump(resolutions, f, indent=2)

    print(f" Generated {len(resolutions)} resolutions")
    print(f" Saved to {output_path}")

    # Print statistics
    type_counts = {}
    satisfaction_counts = {}
    approval_count = 0

    total_time = 0
    for resolution in resolutions:
        issue_type = resolution["issue_type"]
        satisfaction = resolution["guest_satisfaction"]

        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        satisfaction_counts[satisfaction] = satisfaction_counts.get(satisfaction, 0) + 1

        if resolution["approval_required"]:
            approval_count += 1

        total_time += resolution["resolution_time_minutes"]

    avg_time = total_time / len(resolutions)

    print("\nStatistics:")
    print(f"  Type distribution: {type_counts}")
    print(f"  Satisfaction distribution: {satisfaction_counts}")
    print(f"  Resolutions requiring approval: {approval_count}/{len(resolutions)}")
    print(f"  Average resolution time: {avg_time:.1f} minutes")


if __name__ == "__main__":
    main()
