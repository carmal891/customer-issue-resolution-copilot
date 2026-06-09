"""
Self-contained prompt injection detector using regex patterns.

Zero external dependencies - uses only Python standard library.
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of injection threats."""
    INSTRUCTION_OVERRIDE = "INSTRUCTION_OVERRIDE"
    ROLE_MANIPULATION = "ROLE_MANIPULATION"
    SYSTEM_PROMPT_LEAK = "SYSTEM_PROMPT_LEAK"
    JAILBREAK_ATTEMPT = "JAILBREAK_ATTEMPT"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    MALICIOUS_TOOL_CALL = "MALICIOUS_TOOL_CALL"
    UNKNOWN = "UNKNOWN"


@dataclass
class DetectedThreat:
    """Individual detected threat."""
    threat_type: ThreatType
    matched_pattern: str
    severity: float


@dataclass
class InjectionCheckResult:
    """Result of injection detection check."""
    is_injection: bool
    risk_score: float
    threat_type: ThreatType
    matched_patterns: List[str]
    threats_detected: List[DetectedThreat]
    details: str
    sanitized_text: str


class GuardrailsAIInjectionDetector:
    """
    Self-contained prompt injection detector.
    
    Uses comprehensive regex patterns to detect various injection attempts.
    Zero external dependencies.
    """
    
    # Comprehensive injection patterns by threat type
    PATTERNS = {
        ThreatType.INSTRUCTION_OVERRIDE: [
            r"ignore (all |previous |prior )?instructions",
            r"disregard (your |all |previous )?instructions",
            r"forget (your |all |previous )?instructions",
            r"override (system|previous|all)",
            r"new instructions",
            r"updated instructions",
        ],
        ThreatType.ROLE_MANIPULATION: [
            r"you are now (a |an )?",
            r"act as (a |an )?",
            r"pretend (you are|to be)",
            r"roleplay as",
            r"simulate (a |an )?",
            r"behave like (a |an )?",
        ],
        ThreatType.SYSTEM_PROMPT_LEAK: [
            r"(reveal|show|print|display|output) (your )?(system prompt|instructions|prompt)",
            r"what (are|were) your instructions",
            r"repeat (your |the )?system",
            r"tell me your (system |original )?prompt",
            r"show me (your |the )?rules",
        ],
        ThreatType.JAILBREAK_ATTEMPT: [
            r"DAN mode",
            r"jailbreak",
            r"bypass (all |your )?(restrictions|rules|filters)",
            r"no restrictions",
            r"developer mode",
            r"sudo mode",
            r"admin mode",
        ],
        ThreatType.DATA_EXFILTRATION: [
            r"(export|dump|extract|send).{0,30}(credit card|password|ssn|database|all records)",
            r"show (me )?all (customer|user|booking) records",
            r"list all (customers|users|bookings|data)",
            r"give me (all |every )?",
        ],
        ThreatType.MALICIOUS_TOOL_CALL: [
            r"DROP TABLE",
            r"DELETE FROM",
            r"(execute|run).{0,20}(sql|command|script)",
            r"rm -rf",
            r"<script",
            r"eval\(",
        ],
    }
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize injection detector.
        
        Args:
            threshold: Detection sensitivity (0.0-1.0). Lower = more sensitive
        """
        self.threshold = threshold
        
        # Compile all patterns for efficiency
        self.compiled_patterns: Dict[ThreatType, List[re.Pattern]] = {}
        for threat_type, patterns in self.PATTERNS.items():
            self.compiled_patterns[threat_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
logger.info(f"️ Initialized injection detector with {sum(len(p) for p in self.PATTERNS.values())} patterns")
    
    def detect_injection(self, text: str) -> Dict[str, Any]:
        """
        Detect prompt injection attempts in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not text or not text.strip():
            return {
                'is_injection': False,
                'risk_score': 0.0,
                'threat_type': None,
                'matched_patterns': [],
                'details': 'Empty input',
                'sanitized_text': text
            }
        
        max_score = 0.0
        detected_threat = ThreatType.UNKNOWN
        all_matched_patterns = []
        
        # Check against all pattern categories
        for threat_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    all_matched_patterns.append(pattern.pattern)
                    # Higher score for more severe threats
                    threat_score = self._get_threat_score(threat_type)
                    if threat_score > max_score:
                        max_score = threat_score
                        detected_threat = threat_type
        
        is_injection = max_score >= self.threshold
        
        details = (
            f"Detected {len(all_matched_patterns)} suspicious patterns. "
            f"Threat type: {detected_threat.value}"
            if is_injection
            else "No injection detected"
        )
        
        return {
            'is_injection': is_injection,
            'risk_score': max_score,
            'threat_type': detected_threat.value if is_injection else None,
            'matched_patterns': all_matched_patterns,
            'details': details,
            'sanitized_text': text if not is_injection else "[BLOCKED: Potential injection attempt]"
        }
    
    def _get_threat_score(self, threat_type: ThreatType) -> float:
        """Get severity score for threat type."""
        severity_scores = {
            ThreatType.MALICIOUS_TOOL_CALL: 1.0,  # Most severe
            ThreatType.DATA_EXFILTRATION: 0.95,
            ThreatType.INSTRUCTION_OVERRIDE: 0.9,
            ThreatType.SYSTEM_PROMPT_LEAK: 0.85,
            ThreatType.JAILBREAK_ATTEMPT: 0.8,
            ThreatType.ROLE_MANIPULATION: 0.7,
            ThreatType.UNKNOWN: 0.5,
        }
        return severity_scores.get(threat_type, 0.5)
    
    def check_content(self, text: str) -> InjectionCheckResult:
        """
        Check content for injection attempts (app.py interface).
        
        Args:
            text: Text to check
            
        Returns:
            InjectionCheckResult with detection details
        """
        result = self.detect_injection(text)
        
        # Build threats_detected list for app.py compatibility
        threats_detected = []
        if result['is_injection']:
            threat_type = ThreatType(result['threat_type']) if result.get('threat_type') else ThreatType.UNKNOWN
            severity = result['risk_score']
            
            for pattern in result.get('matched_patterns', []):
                threats_detected.append(DetectedThreat(
                    threat_type=threat_type,
                    matched_pattern=pattern,
                    severity=severity
                ))
        
        return InjectionCheckResult(
            is_injection=result['is_injection'],
            risk_score=result['risk_score'],
            threat_type=ThreatType(result['threat_type']) if result.get('threat_type') else ThreatType.UNKNOWN,
            matched_patterns=result.get('matched_patterns', []),
            threats_detected=threats_detected,
            details=result['details'],
            sanitized_text=result['sanitized_text']
        )
    
    def should_block_content(self, check_result: InjectionCheckResult) -> Tuple[bool, str]:
        """
        Determine if content should be blocked (app.py interface).
        
        Args:
            check_result: Result from check_content
            
        Returns:
            Tuple of (should_block, reason)
        """
        if check_result.is_injection:
            reason = (
f" Potential {check_result.threat_type.value} detected "
                f"(risk score: {check_result.risk_score:.2f}). "
                f"Matched {len(check_result.matched_patterns)} suspicious patterns."
            )
            return True, reason
        
        return False, ""
