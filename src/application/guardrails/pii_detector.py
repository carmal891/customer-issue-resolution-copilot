"""
Regex-Based PII Detection for Hospitality Sector

Lightweight, dependency-free PII detection using regex patterns
tailored for hotel and hospitality operations.
"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII relevant to hospitality sector."""
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    SSN = "ssn"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    ROOM_NUMBER = "room_number"  # Hotel-specific
    BOOKING_REFERENCE = "booking_reference"  # Hotel-specific


@dataclass
class PIIMatch:
    """A detected PII entity."""
    pii_type: PIIType
    original_value: str
    masked_value: str
    start_pos: int
    end_pos: int
    confidence: float


class PIIDetector:
    """
    Regex-based PII detector for hospitality operations.

    Features:
    - No external dependencies
    - Fast pattern matching
    - Hotel-specific patterns (room numbers, booking refs)
    - Configurable masking strategies
    """

    # Regex patterns for different PII types
    PATTERNS = {
        PIIType.CREDIT_CARD: [
            # Visa, Mastercard, Amex, Discover
            (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b', 0.95),
            # With spaces or dashes
            (r'\b(?:4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}|5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4})\b', 0.90),
        ],
        PIIType.EMAIL: [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.99),
        ],
        PIIType.PHONE: [
            # US/International formats
            (r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', 0.95),
            (r'\b\+?[1-9]\d{1,14}\b', 0.85),  # E.164 format
        ],
        PIIType.SSN: [
            # US Social Security Number
            (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', 0.90),
        ],
        PIIType.PASSPORT: [
            # US Passport (9 digits)
            (r'\b[0-9]{9}\b', 0.70),
            # Generic passport patterns
            (r'\b[A-Z]{1,2}[0-9]{6,9}\b', 0.75),
        ],
        PIIType.DRIVERS_LICENSE: [
            # US state patterns (varies by state)
            (r'\b[A-Z]{1,2}[0-9]{5,8}\b', 0.65),
        ],
        PIIType.DATE_OF_BIRTH: [
            # MM/DD/YYYY, DD/MM/YYYY
            (r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b', 0.85),
            # YYYY-MM-DD
            (r'\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])\b', 0.85),
        ],
        PIIType.ADDRESS: [
            # Street address patterns
            (r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir)\b', 0.80),
        ],
        PIIType.ROOM_NUMBER: [
            # Hotel room numbers (e.g., 305, Room 412, Rm 203)
            (r'\b(?:Room|Rm|Suite|Ste)[\s#]?([0-9]{3,4})\b', 0.95),
            (r'\b([0-9]{3,4})\s*(?:room|rm)\b', 0.90),
        ],
        PIIType.BOOKING_REFERENCE: [
            # Booking reference patterns (e.g., BK-2024-001, CONF123456)
            (r'\b(?:BK|CONF|RES|BOOKING)[-\s]?[0-9]{4,10}\b', 0.95),
        ],
    }

    # Sensitive PII types that should trigger blocking
    SENSITIVE_TYPES = {
        PIIType.CREDIT_CARD,
        PIIType.SSN,
        PIIType.PASSPORT,
    }

    # Masking strategies per PII type
    MASKING_STRATEGIES = {
        PIIType.CREDIT_CARD: lambda v: f"****-****-****-{v[-4:]}" if len(v) >= 4 else "****",
        PIIType.EMAIL: lambda v: f"{v[0]}***@{v.split('@')[1]}" if '@' in v else "***@***.com",
        PIIType.PHONE: lambda v: f"***-***-{v[-4:]}" if len(v) >= 4 else "***-****",
        PIIType.SSN: lambda v: "***-**-****",
        PIIType.PASSPORT: lambda v: f"{v[:2]}*****" if len(v) >= 2 else "*****",
        PIIType.DRIVERS_LICENSE: lambda v: f"{v[0]}*****" if len(v) >= 1 else "*****",
        PIIType.DATE_OF_BIRTH: lambda v: "**/**/****",
        PIIType.ADDRESS: lambda v: "*** [ADDRESS REDACTED] ***",
        PIIType.ROOM_NUMBER: lambda v: "Room ***",
        PIIType.BOOKING_REFERENCE: lambda v: f"{v.split('-')[0] if '-' in v else v[:2]}-****",
    }

    def __init__(self):
        """Initialize regex PII detector."""
        # Compile all patterns for performance
        self.compiled_patterns: Dict[PIIType, List[Tuple[re.Pattern, float]]] = {}
        for pii_type, patterns in self.PATTERNS.items():
            self.compiled_patterns[pii_type] = [
                (re.compile(pattern, re.IGNORECASE), confidence)
                for pattern, confidence in patterns
            ]
        logger.info("Regex PII Detector initialized with hospitality-specific patterns")

    def detect_and_mask(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """
        Detect and mask PII in text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (masked_text, list of PIIMatch objects)
        """
        if not text:
            return text, []

        matches: List[PIIMatch] = []
        masked_text = text
        offset = 0  # Track position changes due to masking

        # Detect all PII types
        for pii_type, patterns in self.compiled_patterns.items():
            for pattern, confidence in patterns:
                for match in pattern.finditer(text):
                    original_value = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                    # Apply masking strategy
                    masking_func = self.MASKING_STRATEGIES.get(
                        pii_type,
                        lambda v: "*" * len(v)
                    )
                    masked_value = masking_func(original_value)

                    # Create PIIMatch
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        original_value=original_value,
                        masked_value=masked_value,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence
                    )
                    matches.append(pii_match)

                    # Replace in masked text (accounting for offset)
                    adjusted_start = start_pos + offset
                    adjusted_end = end_pos + offset
                    masked_text = (
                        masked_text[:adjusted_start] +
                        masked_value +
                        masked_text[adjusted_end:]
                    )
                    offset += len(masked_value) - len(original_value)

        # Sort matches by position and remove duplicates
        matches = self._deduplicate_matches(matches)

        logger.info(f"Detected {len(matches)} PII entities in text")
        return masked_text, matches

    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return matches

        # Sort by start position, then by confidence (descending)
        sorted_matches = sorted(
            matches,
            key=lambda m: (m.start_pos, -m.confidence)
        )

        deduplicated = []
        last_end = -1

        for match in sorted_matches:
            # Skip if overlaps with previous match
            if match.start_pos < last_end:
                continue
            deduplicated.append(match)
            last_end = match.end_pos

        return deduplicated

    def should_block_request(self, text: str) -> Tuple[bool, str]:
        """
        Determine if request should be blocked due to excessive PII.

        Args:
            text: Input text to check

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        _, pii_matches = self.detect_and_mask(text)

        if not pii_matches:
            return False, ""

        # Count sensitive PII types
        sensitive_matches = [
            match for match in pii_matches
            if match.pii_type in self.SENSITIVE_TYPES
        ]

        # Block if multiple sensitive PII items found
        if len(sensitive_matches) >= 2:
            return True, (
                f"Request contains {len(sensitive_matches)} highly sensitive PII items "
                f"({', '.join(m.pii_type.value for m in sensitive_matches)}). "
                "Please remove sensitive information and try again."
            )

        # Block if credit card detected (high risk)
        credit_card_matches = [
            match for match in pii_matches
            if match.pii_type == PIIType.CREDIT_CARD
        ]
        if credit_card_matches:
            return True, (
                "Request contains credit card information. "
                "Please do not share credit card details in messages. "
                "Use secure payment channels instead."
            )

        return False, ""

    def validate_credit_card(self, card_number: str) -> bool:
        """
        Validate credit card using Luhn algorithm.

        Args:
            card_number: Credit card number (digits only)

        Returns:
            True if valid, False otherwise
        """
        # Remove spaces and dashes
        digits = re.sub(r'[\s\-]', '', card_number)

        if not digits.isdigit() or len(digits) < 13 or len(digits) > 19:
            return False

        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0


# Singleton instance
_detector_instance = None


def get_pii_detector() -> PIIDetector:
    """Get or create singleton PII detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PIIDetector()
    return _detector_instance
