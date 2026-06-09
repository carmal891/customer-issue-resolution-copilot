"""Booking-related domain models."""

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


class LoyaltyTier(str, Enum):
    """Guest loyalty tier levels."""

    STANDARD = "standard"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class RoomType(str, Enum):
    """Hotel room types."""

    STANDARD = "standard"
    DELUXE = "deluxe"
    SUITE = "suite"
    ACCESSIBLE = "accessible"


class BookingStatus(str, Enum):
    """Booking status values."""

    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class PaymentMethod(str, Enum):
    """Payment method types."""

    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    CASH = "cash"
    BANK_TRANSFER = "bank_transfer"


class Guest(BaseModel):
    """Guest entity representing a hotel guest."""

    guest_id: str = Field(..., description="Unique guest identifier")
    name: str = Field(..., description="Guest full name")
    email: EmailStr = Field(..., description="Guest email address")
    phone: str = Field(..., description="Guest phone number")
    loyalty_tier: LoyaltyTier = Field(
        default=LoyaltyTier.STANDARD, description="Guest loyalty tier"
    )
    loyalty_points: int = Field(default=0, ge=0, description="Accumulated loyalty points")
    total_stays: int = Field(default=0, ge=0, description="Total number of stays")
    preferences: dict = Field(default_factory=dict, description="Guest preferences")
    notes: Optional[str] = Field(None, description="Additional notes about the guest")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "guest_id": "G1001",
                "name": "John Smith",
                "email": "john.smith@email.com",
                "phone": "+1-555-0123",
                "loyalty_tier": "gold",
                "loyalty_points": 7500,
                "total_stays": 15,
                "preferences": {"room_type": "deluxe", "floor_preference": "high"},
                "notes": "Prefers quiet rooms",
            }
        }


class Room(BaseModel):
    """Room entity representing a hotel room."""

    room_number: str = Field(..., description="Room number")
    room_type: RoomType = Field(..., description="Type of room")
    floor: int = Field(..., ge=0, description="Floor number")
    rate_per_night: Decimal = Field(..., gt=0, description="Rate per night in USD")
    max_occupancy: int = Field(default=2, ge=1, description="Maximum occupancy")
    amenities: List[str] = Field(default_factory=list, description="Room amenities")
    is_accessible: bool = Field(default=False, description="ADA accessible room")
    near_elevator: bool = Field(default=False, description="Near elevator")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "room_number": "305",
                "room_type": "deluxe",
                "floor": 3,
                "rate_per_night": "200.00",
                "max_occupancy": 2,
                "amenities": ["wifi", "tv", "minibar", "coffee_maker"],
                "is_accessible": False,
                "near_elevator": False,
            }
        }


class Booking(BaseModel):
    """Booking entity representing a hotel reservation."""

    booking_id: str = Field(..., description="Unique booking identifier")
    guest_id: str = Field(..., description="Guest identifier")
    guest_name: str = Field(..., description="Guest name")
    guest_email: EmailStr = Field(..., description="Guest email")
    guest_phone: str = Field(..., description="Guest phone")
    loyalty_tier: LoyaltyTier = Field(..., description="Guest loyalty tier at booking time")
    check_in_date: date = Field(..., description="Check-in date")
    check_out_date: date = Field(..., description="Check-out date")
    room_number: Optional[str] = Field(None, description="Assigned room number")
    room_type: RoomType = Field(..., description="Booked room type")
    rate_per_night: Decimal = Field(..., gt=0, description="Rate per night")
    total_amount: Decimal = Field(..., gt=0, description="Total booking amount")
    booking_status: BookingStatus = Field(..., description="Current booking status")
    booking_date: date = Field(..., description="Date when booking was made")
    special_requests: List[str] = Field(
        default_factory=list, description="Special requests from guest"
    )
    payment_method: PaymentMethod = Field(..., description="Payment method")
    payment_last_four: str = Field(..., description="Last 4 digits of payment card")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "booking_id": "BK12345",
                "guest_id": "G1001",
                "guest_name": "John Smith",
                "guest_email": "john.smith@email.com",
                "guest_phone": "+1-555-0123",
                "loyalty_tier": "gold",
                "check_in_date": "2024-06-15",
                "check_out_date": "2024-06-18",
                "room_number": "305",
                "room_type": "deluxe",
                "rate_per_night": "200.00",
                "total_amount": "600.00",
                "booking_status": "confirmed",
                "booking_date": "2024-06-01",
                "special_requests": ["late_checkout", "high_floor"],
                "payment_method": "credit_card",
                "payment_last_four": "1234",
            }
        }

    @property
    def nights(self) -> int:
        """Calculate number of nights."""
        return (self.check_out_date - self.check_in_date).days

    def is_active(self) -> bool:
        """Check if booking is active (not cancelled or checked out)."""
        return self.booking_status in [BookingStatus.CONFIRMED, BookingStatus.CHECKED_IN]

    def can_be_modified(self) -> bool:
        """Check if booking can be modified."""
        return self.booking_status == BookingStatus.CONFIRMED

    def can_be_cancelled(self) -> bool:
        """Check if booking can be cancelled."""
        return self.booking_status == BookingStatus.CONFIRMED
