"""
Generate mock booking data for the Customer Issue Resolution Copilot.
This script creates realistic hotel booking records with various statuses and scenarios.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

# Configuration
NUM_BOOKINGS = 55
START_DATE = datetime(2024, 3, 10)
END_DATE = datetime(2024, 4, 10)

# Room types and rates
ROOM_TYPES = {
    "STANDARD_QUEEN": {"rate": 149.00, "view": "City View"},
    "STANDARD_DOUBLE": {"rate": 129.00, "view": "Courtyard View"},
    "DELUXE_KING": {"rate": 189.00, "view": "City View"},
    "DELUXE_DOUBLE": {"rate": 209.00, "view": "Ocean View"},
    "EXECUTIVE_SUITE": {"rate": 349.00, "view": "Ocean View"},
    "PRESIDENTIAL_SUITE": {"rate": 599.00, "view": "Panoramic Ocean View"},
    "ACCESSIBLE_KING": {"rate": 169.00, "view": "City View"},
}

# Booking channels
CHANNELS = ["DIRECT", "BOOKING_COM", "EXPEDIA", "HOTELS_COM"]

# Booking statuses
STATUSES = ["CONFIRMED", "CHECKED_IN", "CHECKED_OUT", "CANCELLED", "NO_SHOW"]

# Loyalty tiers
LOYALTY_TIERS = ["NONE", "SILVER", "GOLD", "PLATINUM"]

# Sample names
FIRST_NAMES = [
    "Sarah", "James", "Emily", "Michael", "Lisa", "David", "Jennifer", "Robert",
    "Amanda", "Christopher", "Maria", "Thomas", "Patricia", "Daniel", "Nancy",
    "Paul", "Dorothy", "Kevin", "Barbara", "Mark", "Sandra", "Donald", "Ashley",
    "Kenneth", "Donna", "Joshua", "Carol", "Brian", "Michelle", "George", "Betty",
    "Edward", "Laura", "Jason", "Helen", "Ryan", "Deborah", "Gary", "Sharon",
    "Timothy", "Cynthia", "Jose", "Kathleen", "Larry", "Amy", "Jeffrey", "Angela",
    "Frank", "Melissa", "Scott", "Brenda", "Eric", "Rebecca", "Stephen", "Virginia"
]

LAST_NAMES = [
    "Mitchell", "Rodriguez", "Chen", "Thompson", "Anderson", "Kim", "Martinez",
    "Wilson", "Taylor", "Brown", "Garcia", "Lee", "White", "Harris", "Clark",
    "Lewis", "Walker", "Hall", "Allen", "Young", "King", "Wright", "Lopez",
    "Hill", "Scott", "Green", "Adams", "Baker", "Nelson", "Carter", "Perez",
    "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards",
    "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan",
    "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson", "Cox", "Howard"
]

# Special requests pool
SPECIAL_REQUESTS = [
    "High floor, quiet room",
    "Late checkout requested",
    "Ground floor preferred",
    "Early check-in if possible",
    "Non-smoking room",
    "Hypoallergenic bedding",
    "Extra workspace, printer access needed",
    "Two beds, connecting room if available",
    "Wheelchair accessible room required, roll-in shower",
    "Champagne and flowers in room",
    "Quiet room away from elevators",
    "VIP arrival, airport transfer arranged",
    "Extra towels",
    "Business center access, meeting room needed",
    "Pet-friendly room - small dog",
    "Express checkout",
    "Anniversary celebration - flowers requested",
    "Crib needed for infant",
    "Foam pillows preferred",
    "Extra coffee pods",
    "Medical conference attendee, quiet floor preferred",
    None,
]


def generate_guest_id(index: int) -> str:
    """Generate a unique guest ID."""
    return f"G-{10234 + index}"


def generate_confirmation_number(index: int, channel: str) -> str:
    """Generate a confirmation number based on channel."""
    if channel == "DIRECT":
        return f"GPH-{45782 + index}-{chr(65 + (index % 26))}"
    elif channel == "BOOKING_COM":
        return f"BKNG-{789456 + index}"
    elif channel == "EXPEDIA":
        return f"EXP-{334521 + index}"
    else:  # HOTELS_COM
        return f"HOTELS-{556789 + index}"


def generate_loyalty_info(tier: str, index: int) -> Dict[str, Any]:
    """Generate loyalty program information."""
    if tier == "NONE":
        return {"loyalty_tier": "NONE", "loyalty_number": None}

    tier_prefix = {
        "SILVER": "SILV",
        "GOLD": "GOLD",
        "PLATINUM": "PLAT"
    }

    return {
        "loyalty_tier": tier,
        "loyalty_number": f"GP-{tier_prefix[tier]}-{1000 + index * 123 % 9999}"
    }


def generate_booking(index: int) -> Dict[str, Any]:
    """Generate a single booking record."""
    # Random selections
    channel = random.choice(CHANNELS)
    room_type = random.choice(list(ROOM_TYPES.keys()))
    room_info = ROOM_TYPES[room_type]
    loyalty_tier = random.choices(
        LOYALTY_TIERS,
        weights=[30, 25, 25, 20],  # More non-loyalty, fewer platinum
        k=1
    )[0]

    # Date generation
    days_from_start = random.randint(0, 30)
    check_in = START_DATE + timedelta(days=days_from_start)
    nights = random.randint(2, 5)
    check_out = check_in + timedelta(days=nights)

    # Booking created 1-20 days before check-in
    created_days_before = random.randint(1, 20)
    created_at = check_in - timedelta(days=created_days_before)

    # Status determination based on dates
    today = datetime(2024, 3, 22)  # Fixed "today" for consistency
    if check_out < today:
        # Past bookings
        status = random.choices(
            ["CHECKED_OUT", "CANCELLED", "NO_SHOW"],
            weights=[85, 10, 5],
            k=1
        )[0]
    elif check_in <= today < check_out:
        # Current stays
        status = "CHECKED_IN"
    else:
        # Future bookings
        status = random.choices(
            ["CONFIRMED", "CANCELLED"],
            weights=[90, 10],
            k=1
        )[0]

    # Guest information
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    email_domains = ["email.com", "gmail.com", "company.com", "tech.com", 
                     "consulting.com", "startup.io", "lawfirm.com", "finance.com",
                     "healthcare.org", "marketing.com", "university.edu"]
    email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(email_domains)}"

    # Room assignment
    floor = random.randint(2, 9)
    room_number = f"{floor}{random.randint(1, 25):02d}"

    # Build booking
    booking = {
        "booking_id": f"BK-2024-{index + 1:03d}",
        "confirmation_number": generate_confirmation_number(index, channel),
        "guest": {
            "guest_id": generate_guest_id(index),
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": f"+1-555-{123 + index:04d}",
            **generate_loyalty_info(loyalty_tier, index)
        },
        "room": {
            "room_number": room_number,
            "room_type": room_type,
            "floor": floor,
            "rate_per_night": room_info["rate"],
            "view": room_info["view"]
        },
        "check_in_date": check_in.strftime("%Y-%m-%d"),
        "check_out_date": check_out.strftime("%Y-%m-%d"),
        "nights": nights,
        "total_amount": round(room_info["rate"] * nights, 2),
        "status": status,
        "booking_channel": channel,
        "special_requests": random.choice(SPECIAL_REQUESTS),
        "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    # Add status-specific fields
    if status == "CHECKED_IN":
        check_in_time = check_in.replace(
            hour=random.randint(14, 18),
            minute=random.randint(0, 59)
        )
        booking["checked_in_at"] = check_in_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        booking["checked_out_at"] = None
    elif status == "CHECKED_OUT":
        check_in_time = check_in.replace(
            hour=random.randint(14, 18),
            minute=random.randint(0, 59)
        )
        check_out_time = check_out.replace(
            hour=random.randint(9, 13),
            minute=random.randint(0, 59)
        )
        booking["checked_in_at"] = check_in_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        booking["checked_out_at"] = check_out_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif status == "CANCELLED":
        cancel_days_before = random.randint(1, 10)
        cancelled_at = check_in - timedelta(days=cancel_days_before)
        booking["checked_in_at"] = None
        booking["checked_out_at"] = None
        booking["cancelled_at"] = cancelled_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        booking["cancellation_reason"] = random.choice([
            "Guest cancelled - schedule change",
            "Guest cancelled - personal reasons",
            "Hotel cancelled - overbooking",
            "Guest cancelled - found alternative"
        ])
    elif status == "NO_SHOW":
        no_show_time = check_in.replace(hour=18, minute=0)
        booking["checked_in_at"] = None
        booking["checked_out_at"] = None
        booking["no_show_date"] = no_show_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:  # CONFIRMED
        booking["checked_in_at"] = None
        booking["checked_out_at"] = None

    # Add special notes for loyalty members
    if loyalty_tier in ["GOLD", "PLATINUM"] and status in ["CHECKED_OUT", "CHECKED_IN"]:
        if random.random() < 0.3:  # 30% chance
            perks = []
            if loyalty_tier == "PLATINUM":
                perks.extend(["Complimentary upgrade applied", "Complimentary breakfast included"])
            if random.random() < 0.5:
                perks.append("Late checkout granted" if status == "CHECKED_OUT" else "Late checkout approved")
            if perks:
                booking["special_requests"] = ", ".join(perks)

    return booking


def main():
    """Generate all bookings and save to JSON file."""
    print(f"Generating {NUM_BOOKINGS} mock bookings...")

    bookings = []
    for i in range(NUM_BOOKINGS):
        booking = generate_booking(i)
        bookings.append(booking)

    # Save to file
    output_path = "data/mock/bookings/bookings.json"
    with open(output_path, "w") as f:
        json.dump(bookings, f, indent=2)

    print(f"Generated {len(bookings)} bookings")
    print(f"Saved to {output_path}")

    # Print statistics
    status_counts = {}
    channel_counts = {}
    loyalty_counts = {}

    for booking in bookings:
        status = booking["status"]
        channel = booking["booking_channel"]
        loyalty = booking["guest"]["loyalty_tier"]

        status_counts[status] = status_counts.get(status, 0) + 1
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        loyalty_counts[loyalty] = loyalty_counts.get(loyalty, 0) + 1

    print("\nStatistics:")
    print(f"  Status distribution: {status_counts}")
    print(f"  Channel distribution: {channel_counts}")
    print(f"  Loyalty distribution: {loyalty_counts}")


if __name__ == "__main__":
    main()
