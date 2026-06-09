"""
Tool Access Validator with Input Validation and Rate Limiting

Ensures tools are called with valid inputs and enforces rate limits
to prevent abuse or runaway execution in the hotel operations system.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when tool input validation fails."""
    pass


@dataclass
class ValidationRule:
    """A validation rule for a tool parameter."""
    param_name: str
    required: bool = True
    param_type: Optional[type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    error_message: Optional[str] = None


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a tool."""
    max_calls_per_minute: int = 60
    max_calls_per_hour: int = 1000
    max_calls_per_day: int = 10000


@dataclass
class ToolCallRecord:
    """Record of a tool call for rate limiting."""
    tool_name: str
    timestamp: datetime
    params: Dict[str, Any]
    success: bool


class ToolValidator:
    """
    Validates tool inputs and enforces rate limits.

    Prevents invalid tool calls and protects against abuse
    in the hotel operations system.
    """

    # Tool validation rules for hotel domain
    TOOL_VALIDATION_RULES = {
        'lookup_booking': [
            ValidationRule(
                param_name='booking_id',
                required=True,
                param_type=str,
                pattern=r'^BK[0-9]{5}$',
                error_message="Booking ID must be in format BK12345"
            ),
        ],
        'process_refund': [
            ValidationRule(
                param_name='booking_id',
                required=True,
                param_type=str,
                pattern=r'^BK[0-9]{5}$',
                error_message="Booking ID must be in format BK12345"
            ),
            ValidationRule(
                param_name='amount',
                required=True,
                param_type=(int, float),
                min_value=0.01,
                max_value=10000.00,
                error_message="Refund amount must be between $0.01 and $10,000"
            ),
            ValidationRule(
                param_name='reason',
                required=True,
                param_type=str,
                min_length=10,
                max_length=500,
                error_message="Refund reason must be 10-500 characters"
            ),
        ],
        'update_booking': [
            ValidationRule(
                param_name='booking_id',
                required=True,
                param_type=str,
                pattern=r'^BK[0-9]{5}$',
            ),
            ValidationRule(
                param_name='updates',
                required=True,
                param_type=dict,
                error_message="Updates must be a dictionary"
            ),
        ],
        'send_email': [
            ValidationRule(
                param_name='to',
                required=True,
                param_type=str,
                pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                error_message="Invalid email address"
            ),
            ValidationRule(
                param_name='subject',
                required=True,
                param_type=str,
                min_length=1,
                max_length=200,
            ),
            ValidationRule(
                param_name='body',
                required=True,
                param_type=str,
                min_length=1,
                max_length=10000,
            ),
        ],
        'upgrade_room': [
            ValidationRule(
                param_name='booking_id',
                required=True,
                param_type=str,
                pattern=r'^BK[0-9]{5}$',
            ),
            ValidationRule(
                param_name='new_room_type',
                required=True,
                param_type=str,
                allowed_values=['standard', 'deluxe', 'suite', 'penthouse'],
                error_message="Invalid room type"
            ),
        ],
        'apply_discount': [
            ValidationRule(
                param_name='booking_id',
                required=True,
                param_type=str,
                pattern=r'^BK[0-9]{5}$',
            ),
            ValidationRule(
                param_name='percentage',
                required=True,
                param_type=(int, float),
                min_value=0,
                max_value=100,
                error_message="Discount percentage must be 0-100"
            ),
        ],
    }

    # Rate limits for each tool
    TOOL_RATE_LIMITS = {
        'process_refund': RateLimitConfig(
            max_calls_per_minute=10,
            max_calls_per_hour=100,
            max_calls_per_day=500
        ),
        'update_booking': RateLimitConfig(
            max_calls_per_minute=30,
            max_calls_per_hour=500,
            max_calls_per_day=2000
        ),
        'send_email': RateLimitConfig(
            max_calls_per_minute=20,
            max_calls_per_hour=200,
            max_calls_per_day=1000
        ),
        'lookup_booking': RateLimitConfig(
            max_calls_per_minute=100,
            max_calls_per_hour=1000,
            max_calls_per_day=10000
        ),
    }

    # Default rate limit for tools without specific limits
    DEFAULT_RATE_LIMIT = RateLimitConfig(
        max_calls_per_minute=60,
        max_calls_per_hour=1000,
        max_calls_per_day=5000
    )

    def __init__(self):
        """Initialize tool validator."""
        self.logger = logging.getLogger(__name__)
        self.call_history: Dict[str, List[ToolCallRecord]] = defaultdict(list)

    def validate_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Validate a tool call before execution.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Raises:
            ValidationError: If validation fails
        """
        # Check if tool has validation rules
        if tool_name not in self.TOOL_VALIDATION_RULES:
            self.logger.warning(f"No validation rules defined for tool: {tool_name}")
            return

        rules = self.TOOL_VALIDATION_RULES[tool_name]

        # Validate each rule
        for rule in rules:
            self._validate_parameter(params, rule)

        self.logger.debug(f"Validation passed for {tool_name}")

    def _validate_parameter(
        self,
        params: Dict[str, Any],
        rule: ValidationRule
    ) -> None:
        """
        Validate a single parameter against a rule.

        Args:
            params: Tool parameters
            rule: Validation rule

        Raises:
            ValidationError: If validation fails
        """
        param_name = rule.param_name
        value = params.get(param_name)

        # Check required
        if rule.required and value is None:
            raise ValidationError(
                rule.error_message or f"Required parameter '{param_name}' is missing"
            )

        # Skip further validation if value is None and not required
        if value is None:
            return

        # Check type
        if rule.param_type is not None:
            if not isinstance(value, rule.param_type):
                raise ValidationError(
                    rule.error_message or 
                    f"Parameter '{param_name}' must be of type {rule.param_type.__name__}"
                )

        # Check numeric bounds
        if rule.min_value is not None and isinstance(value, (int, float)):
            if value < rule.min_value:
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' must be >= {rule.min_value}"
                )

        if rule.max_value is not None and isinstance(value, (int, float)):
            if value > rule.max_value:
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' must be <= {rule.max_value}"
                )

        # Check string length
        if rule.min_length is not None and isinstance(value, str):
            if len(value) < rule.min_length:
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' must be at least {rule.min_length} characters"
                )

        if rule.max_length is not None and isinstance(value, str):
            if len(value) > rule.max_length:
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' must be at most {rule.max_length} characters"
                )

        # Check pattern
        if rule.pattern is not None and isinstance(value, str):
            if not re.match(rule.pattern, value):
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' does not match required pattern"
                )

        # Check allowed values
        if rule.allowed_values is not None:
            if value not in rule.allowed_values:
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' must be one of: {', '.join(map(str, rule.allowed_values))}"
                )

        # Custom validator
        if rule.custom_validator is not None:
            if not rule.custom_validator(value):
                raise ValidationError(
                    rule.error_message or
                    f"Parameter '{param_name}' failed custom validation"
                )

    def check_rate_limit(self, tool_name: str) -> None:
        """
        Check if tool call is within rate limits.

        Args:
            tool_name: Name of the tool

        Raises:
            ValidationError: If rate limit exceeded
        """
        # Get rate limit config
        rate_limit = self.TOOL_RATE_LIMITS.get(tool_name, self.DEFAULT_RATE_LIMIT)

        # Get call history for this tool
        history = self.call_history[tool_name]

        now = datetime.now()

        # Clean up old records
        self._cleanup_old_records(tool_name, now)

        # Check minute limit
        minute_ago = now - timedelta(minutes=1)
        calls_last_minute = sum(1 for record in history if record.timestamp > minute_ago)

        if calls_last_minute >= rate_limit.max_calls_per_minute:
            raise ValidationError(
                f"Rate limit exceeded for {tool_name}: "
                f"{calls_last_minute} calls in last minute "
                f"(limit: {rate_limit.max_calls_per_minute}/min)"
            )

        # Check hour limit
        hour_ago = now - timedelta(hours=1)
        calls_last_hour = sum(1 for record in history if record.timestamp > hour_ago)

        if calls_last_hour >= rate_limit.max_calls_per_hour:
            raise ValidationError(
                f"Rate limit exceeded for {tool_name}: "
                f"{calls_last_hour} calls in last hour "
                f"(limit: {rate_limit.max_calls_per_hour}/hour)"
            )

        # Check day limit
        day_ago = now - timedelta(days=1)
        calls_last_day = sum(1 for record in history if record.timestamp > day_ago)

        if calls_last_day >= rate_limit.max_calls_per_day:
            raise ValidationError(
                f"Rate limit exceeded for {tool_name}: "
                f"{calls_last_day} calls in last day "
                f"(limit: {rate_limit.max_calls_per_day}/day)"
            )

        self.logger.debug(
            f"Rate limit check passed for {tool_name} "
            f"(minute: {calls_last_minute}, hour: {calls_last_hour}, day: {calls_last_day})"
        )

    def record_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        success: bool = True
    ) -> None:
        """
        Record a tool call for rate limiting.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            success: Whether the call succeeded
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            timestamp=datetime.now(),
            params=params,
            success=success
        )

        self.call_history[tool_name].append(record)

        self.logger.debug(f"Recorded tool call: {tool_name} (success: {success})")

    def _cleanup_old_records(self, tool_name: str, now: datetime) -> None:
        """
        Remove records older than 24 hours.

        Args:
            tool_name: Name of the tool
            now: Current timestamp
        """
        cutoff = now - timedelta(days=1)

        history = self.call_history[tool_name]
        self.call_history[tool_name] = [
            record for record in history
            if record.timestamp > cutoff
        ]

    def get_tool_usage_stats(self, tool_name: str) -> Dict[str, Any]:
        """
        Get usage statistics for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with usage stats
        """
        history = self.call_history[tool_name]
        now = datetime.now()

        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        calls_last_minute = sum(1 for r in history if r.timestamp > minute_ago)
        calls_last_hour = sum(1 for r in history if r.timestamp > hour_ago)
        calls_last_day = sum(1 for r in history if r.timestamp > day_ago)

        successful_calls = sum(1 for r in history if r.success)
        failed_calls = len(history) - successful_calls

        rate_limit = self.TOOL_RATE_LIMITS.get(tool_name, self.DEFAULT_RATE_LIMIT)

        return {
            'tool_name': tool_name,
            'total_calls': len(history),
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'calls_last_minute': calls_last_minute,
            'calls_last_hour': calls_last_hour,
            'calls_last_day': calls_last_day,
            'rate_limits': {
                'per_minute': rate_limit.max_calls_per_minute,
                'per_hour': rate_limit.max_calls_per_hour,
                'per_day': rate_limit.max_calls_per_day,
            },
            'utilization': {
                'minute': f"{calls_last_minute}/{rate_limit.max_calls_per_minute}",
                'hour': f"{calls_last_hour}/{rate_limit.max_calls_per_hour}",
                'day': f"{calls_last_day}/{rate_limit.max_calls_per_day}",
            }
        }


# Singleton instance
_tool_validator_instance: Optional[ToolValidator] = None


def get_tool_validator() -> ToolValidator:
    """Get singleton tool validator instance."""
    global _tool_validator_instance
    if _tool_validator_instance is None:
        _tool_validator_instance = ToolValidator()
    return _tool_validator_instance
