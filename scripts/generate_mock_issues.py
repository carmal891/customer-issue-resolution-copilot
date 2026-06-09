"""
Generate mock customer issues for the Customer Issue Resolution Copilot.
This script creates realistic customer issues across various channels.
"""

import json
from datetime import datetime, timedelta
import random

# Configuration
NUM_ISSUES = 25
BASE_DATE = datetime(2024, 3, 22)  # "Today" for consistency

# Issue types and their typical channels
ISSUE_TEMPLATES = [
    {
        "type": "LATE_CHECKOUT",
        "channels": ["EMAIL", "CHAT", "PHONE"],
        "templates": [
            "Hi, I'm checking out tomorrow but my flight isn't until 6 PM. Is it possible to get a late checkout until 2 PM? I'm in room {room}. Booking {booking}.",
            "Hello, I need to request late checkout for tomorrow. My meeting runs until 1 PM and I'd like to checkout around 3 PM if possible. Room {room}, confirmation {booking}.",
            "Can I please have late checkout? I'm a {loyalty} member staying in room {room}. Would really appreciate staying until 2 PM tomorrow.",
        ]
    },
    {
        "type": "ROOM_UPGRADE",
        "channels": ["EMAIL", "CHAT", "DIRECT"],
        "templates": [
            "I'm a {loyalty} member and was wondering if there are any complimentary upgrades available for my stay? Booking {booking}, room {room}.",
            "Hi! I'm celebrating my anniversary and would love to upgrade to an ocean view room if possible. Currently in {room}. Confirmation {booking}.",
            "Hello, I booked a standard room but would like to upgrade to a suite if available. What are my options? Booking {booking}.",
        ]
    },
    {
        "type": "CANCELLATION_REFUND",
        "channels": ["EMAIL", "BOOKING_COM", "PHONE"],
        "templates": [
            "I need to cancel my reservation {booking} for {date}. Can I get a full refund? I booked 3 weeks ago.",
            "Unfortunately I have to cancel my booking {booking} due to a family emergency. What is your cancellation policy? Check-in was supposed to be {date}.",
            "Hi, I want to cancel booking {booking}. I'm within the cancellation window, right? Please process my refund.",
        ]
    },
    {
        "type": "BILLING_DISPUTE",
        "channels": ["EMAIL", "PHONE", "CHAT"],
        "templates": [
            "I was charged ${amount} but my booking confirmation says ${correct_amount}. Can you please explain this discrepancy? Booking {booking}.",
            "There's an error on my bill. I see charges for minibar items I never used. Room {room}, total shows ${amount}. Please review.",
            "My credit card was charged twice for booking {booking}. I only authorized one payment of ${correct_amount}. Please refund the duplicate charge.",
        ]
    },
    {
        "type": "ROOM_ISSUE",
        "channels": ["CHAT", "PHONE", "SLACK"],
        "templates": [
            "The AC in room {room} isn't working and it's very hot. Can someone come fix this ASAP?",
            "Hi, I'm in room {room} and the WiFi password isn't working. I've tried multiple times. Can you help?",
            "There's a lot of noise from the room above me (I'm in {room}). It's past midnight and I can't sleep. Can you do something?",
        ]
    },
    {
        "type": "AMENITY_REQUEST",
        "channels": ["CHAT", "PHONE", "EMAIL"],
        "templates": [
            "Can I get extra towels and pillows sent to room {room}? Thanks!",
            "Hi, I need hypoallergenic bedding for room {room}. I have allergies and the current bedding is causing issues.",
            "Could you send up a crib to room {room}? We have an infant with us and forgot to request it at booking.",
        ]
    },
    {
        "type": "ACCESSIBILITY",
        "channels": ["EMAIL", "PHONE", "BOOKING_COM"],
        "templates": [
            "I booked room {room} but need to confirm it has wheelchair accessibility. I require a roll-in shower. Booking {booking}.",
            "My booking {booking} is for an accessible room but I want to make sure it has visual fire alarms. I'm hearing impaired.",
            "I need to change my room to an accessible one. I didn't realize I booked a standard room. Can you help? Booking {booking}.",
        ]
    },
    {
        "type": "COMPLAINT",
        "channels": ["EMAIL", "TWITTER", "BOOKING_COM"],
        "templates": [
            "Very disappointed with the service at check-in. The staff was rude and unhelpful. Booking {booking}. I expect better from a hotel of this caliber.",
            "Room {room} was not clean when I arrived. There were towels on the floor and the bathroom wasn't cleaned. This is unacceptable.",
            "I've been waiting 45 minutes for room service. This is ridiculous. Room {room}. Where is my order?",
        ]
    },
    {
        "type": "BOOKING_MODIFICATION",
        "channels": ["EMAIL", "CHAT", "PHONE"],
        "templates": [
            "I need to change my check-in date from {date} to {new_date}. Is this possible? Booking {booking}.",
            "Can I extend my stay by 2 more nights? Currently checking out {date} but would like to stay until {new_date}. Room {room}.",
            "I need to add another guest to my reservation {booking}. Do I need to pay extra?",
        ]
    },
    {
        "type": "LOST_ITEM",
        "channels": ["EMAIL", "PHONE", "CHAT"],
        "templates": [
            "I checked out yesterday from room {room} and left my phone charger. Can you check if housekeeping found it?",
            "I think I left my laptop in the business center. I was there this morning. Can someone check? I'm {name}, room {room}.",
            "I left a jacket in room {room} after checkout. Black leather jacket. Has it been turned in to lost and found?",
        ]
    },
]

# Sample booking data to reference
SAMPLE_BOOKINGS = [
    {"booking_id": "BK-2024-002", "room": "815", "confirmation": "GPH-45783-B", "guest_name": "James Rodriguez", "loyalty": "PLATINUM"},
    {"booking_id": "BK-2024-005", "room": "318", "confirmation": "EXP-334521", "guest_name": "Lisa Anderson", "loyalty": "NONE"},
    {"booking_id": "BK-2024-006", "room": "612", "confirmation": "GPH-45785-D", "guest_name": "David Kim", "loyalty": "GOLD"},
    {"booking_id": "BK-2024-009", "room": "714", "confirmation": "GPH-45787-F", "guest_name": "Amanda Taylor", "loyalty": "SILVER"},
    {"booking_id": "BK-2024-010", "room": "520", "confirmation": "GPH-45788-G", "guest_name": "Christopher Brown", "loyalty": "GOLD"},
    {"booking_id": "BK-2024-014", "room": "405", "confirmation": "EXP-334522", "guest_name": "Daniel Harris", "loyalty": "NONE"},
    {"booking_id": "BK-2024-020", "room": "525", "confirmation": "GPH-45794-M", "guest_name": "Paul Young", "loyalty": "SILVER"},
]

# Priority levels
PRIORITIES = {
    "LATE_CHECKOUT": "MEDIUM",
    "ROOM_UPGRADE": "LOW",
    "CANCELLATION_REFUND": "MEDIUM",
    "BILLING_DISPUTE": "HIGH",
    "ROOM_ISSUE": "HIGH",
    "AMENITY_REQUEST": "LOW",
    "ACCESSIBILITY": "HIGH",
    "COMPLAINT": "HIGH",
    "BOOKING_MODIFICATION": "MEDIUM",
    "LOST_ITEM": "MEDIUM",
}


def generate_issue_id(index: int) -> str:
    """Generate a unique issue ID."""
    return f"ISS-2024-{1000 + index}"


def generate_issue(index: int) -> dict:
    """Generate a single customer issue."""
    # Select issue template
    template_group = random.choice(ISSUE_TEMPLATES)
    issue_type = template_group["type"]
    channel = random.choice(template_group["channels"])
    template = random.choice(template_group["templates"])
    
    # Select booking reference
    booking_ref = random.choice(SAMPLE_BOOKINGS)
    
    # Generate dates
    hours_ago = random.randint(1, 72)
    created_at = BASE_DATE - timedelta(hours=hours_ago)
    
    # Fill template with data
    check_in_date = (BASE_DATE + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d")
    new_date = (BASE_DATE + timedelta(days=random.randint(8, 14))).strftime("%Y-%m-%d")
    
    description = template.format(
        room=booking_ref["room"],
        booking=booking_ref["confirmation"],
        loyalty=booking_ref["loyalty"],
        date=check_in_date,
        new_date=new_date,
        amount=random.choice([567.00, 1047.00, 298.00, 627.00]),
        correct_amount=random.choice([567.00, 1047.00, 298.00, 627.00]),
        name=booking_ref["guest_name"]
    )
    
    # Build issue
    issue = {
        "issue_id": generate_issue_id(index),
        "booking_id": booking_ref["booking_id"],
        "guest_name": booking_ref["guest_name"],
        "issue_type": issue_type,
        "channel": channel,
        "priority": PRIORITIES[issue_type],
        "status": random.choices(
            ["OPEN", "IN_PROGRESS", "RESOLVED"],
            weights=[40, 30, 30],
            k=1
        )[0],
        "description": description,
        "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": {
            "room_number": booking_ref["room"],
            "confirmation_number": booking_ref["confirmation"],
            "loyalty_tier": booking_ref["loyalty"]
        }
    }
    
    # Add channel-specific metadata
    if channel == "EMAIL":
        issue["metadata"]["email_subject"] = f"Re: Booking {booking_ref['confirmation']} - {issue_type.replace('_', ' ').title()}"
        issue["metadata"]["from_email"] = f"{booking_ref['guest_name'].lower().replace(' ', '.')}@email.com"
    elif channel == "TWITTER":
        issue["metadata"]["tweet_id"] = f"tw_{1234567890 + index}"
        issue["metadata"]["handle"] = f"@{booking_ref['guest_name'].split()[0].lower()}{random.randint(100, 999)}"
    elif channel == "BOOKING_COM":
        issue["metadata"]["platform_message_id"] = f"BKNG-MSG-{5000 + index}"
    elif channel == "SLACK":
        issue["metadata"]["slack_channel"] = "#guest-services"
        issue["metadata"]["slack_thread_ts"] = f"1710{random.randint(100000, 999999)}.{random.randint(100000, 999999)}"
    
    # Add resolution info if resolved
    if issue["status"] == "RESOLVED":
        resolved_hours = random.randint(1, hours_ago - 1)
        resolved_at = created_at + timedelta(hours=resolved_hours)
        issue["resolved_at"] = resolved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        issue["resolution_summary"] = f"Issue resolved via {channel.lower()}. Guest satisfied with outcome."
    
    # Add assignment if in progress or resolved
    if issue["status"] in ["IN_PROGRESS", "RESOLVED"]:
        issue["assigned_to"] = random.choice([
            "agent_sarah_m",
            "agent_john_d",
            "agent_maria_g",
            "agent_david_l"
        ])
    
    return issue


def main():
    """Generate all issues and save to JSON file."""
    print(f"Generating {NUM_ISSUES} mock customer issues...")
    
    issues = []
    for i in range(NUM_ISSUES):
        issue = generate_issue(i)
        issues.append(issue)
    
    # Save to file
    output_path = "data/mock/issues/customer_issues.json"
    with open(output_path, "w") as f:
        json.dump(issues, f, indent=2)
    
print(f" Generated {len(issues)} issues")
print(f" Saved to {output_path}")
    
    # Print statistics
    type_counts = {}
    channel_counts = {}
    status_counts = {}
    priority_counts = {}
    
    for issue in issues:
        issue_type = issue["issue_type"]
        channel = issue["channel"]
        status = issue["status"]
        priority = issue["priority"]
        
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print("\nStatistics:")
    print(f"  Type distribution: {type_counts}")
    print(f"  Channel distribution: {channel_counts}")
    print(f"  Status distribution: {status_counts}")
    print(f"  Priority distribution: {priority_counts}")


if __name__ == "__main__":
    main()
