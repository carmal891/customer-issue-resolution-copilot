"""
Guardrails module for Customer Issue Resolution Copilot.

This module provides security guardrails using:
- Regex-based PII detection and masking (hospitality-specific)
- Guardrails AI for prompt injection detection
- Approval enforcement for risky actions
- Confidence checking for low-quality retrievals
- Tool validation for bounded access control
"""

import logging
from typing import Optional

from .pii_detector import PIIDetector
from .guardrails_ai_injection_detector import GuardrailsAIInjectionDetector
from .approval_enforcer import ApprovalEnforcer
from .confidence_checker import ConfidenceChecker
from .tool_validator import ToolValidator

logger = logging.getLogger(__name__)

# ============================================================
# SINGLETON INSTANCES - Guardrails
# ============================================================

_pii_detector: Optional[PIIDetector] = None
_injection_detector: Optional[GuardrailsAIInjectionDetector] = None
_approval_enforcer: Optional[ApprovalEnforcer] = None
_confidence_checker: Optional[ConfidenceChecker] = None
_tool_validator: Optional[ToolValidator] = None


def get_pii_detector() -> PIIDetector:
    """
    Get singleton instance of regex-based PII detector.
    
    Returns:
        PIIDetector: Regex-based PII detection and masking for hospitality sector
    """
    global _pii_detector
    if _pii_detector is None:
        logger.info(" Initializing regex-based PII detector (hospitality-specific)")
        _pii_detector = PIIDetector()
    return _pii_detector


def get_injection_detector() -> GuardrailsAIInjectionDetector:
    """
    Get singleton instance of injection detector (Guardrails AI).
    
    Returns:
        GuardrailsAIInjectionDetector: Production-grade prompt injection detection
    """
    global _injection_detector
    if _injection_detector is None:
        logger.info("️ Initializing injection detector (Guardrails AI)")
        _injection_detector = GuardrailsAIInjectionDetector()
    return _injection_detector


def get_approval_enforcer() -> ApprovalEnforcer:
    """
    Get singleton instance of approval enforcer.
    
    Returns:
        ApprovalEnforcer: Enforces human approval for risky actions
    """
    global _approval_enforcer
    if _approval_enforcer is None:
        logger.info(" Initializing approval enforcer")
        _approval_enforcer = ApprovalEnforcer()
    return _approval_enforcer


def get_confidence_checker() -> ConfidenceChecker:
    """
    Get singleton instance of confidence checker.
    
    Returns:
        ConfidenceChecker: Checks RAG retrieval confidence and escalates low-quality results
    """
    global _confidence_checker
    if _confidence_checker is None:
        logger.info(" Initializing confidence checker")
        _confidence_checker = ConfidenceChecker()
    return _confidence_checker


def get_tool_validator() -> ToolValidator:
    """
    Get singleton instance of tool validator.
    
    Returns:
        ToolValidator: Validates tool inputs and enforces bounded access
    """
    global _tool_validator
    if _tool_validator is None:
        logger.info(" Initializing tool validator")
        _tool_validator = ToolValidator()
    return _tool_validator


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "get_pii_detector",
    "get_injection_detector",
    "get_approval_enforcer",
    "get_confidence_checker",
    "get_tool_validator",
    "PIIDetector",
    "GuardrailsAIInjectionDetector",
    "ApprovalEnforcer",
    "ConfidenceChecker",
    "ToolValidator",
]
