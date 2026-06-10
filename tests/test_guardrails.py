"""
Guardrails Test Suite

Tests PII detection, prompt injection detection, and other safety mechanisms
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import re
from typing import Dict, List, Tuple, Any
from src.application.guardrails.pii_detector import PIIDetector
from src.application.guardrails.guardrails_ai_injection_detector import GuardrailsAIInjectionDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_test_file() -> List[Dict[str, Any]]:
    """Parse guardrails_test_cases.txt"""
    test_file = Path(__file__).parent / "guardrails_test_cases.txt"
    tests = []
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Split by test sections
    test_sections = re.split(r'### Test \d+:', content)
    
    for section in test_sections[1:]:  # Skip first empty section
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        test = {
            'name': lines[0].strip(),
            'input': '',
            'expected': '',
            'type': ''
        }
        
        for line in lines[1:]:
            if line.startswith('Input:'):
                test['input'] = line.replace('Input:', '').strip().strip('"')
            elif line.startswith('Expected:'):
                test['expected'] = line.replace('Expected:', '').strip()
            elif line.startswith('PII Type:') or line.startswith('Threat Type:'):
                test['type'] = line.split(':', 1)[1].strip()
        
        if test['input']:
            tests.append(test)
    
    return tests


def run_pii_tests(pii_detector: PIIDetector, tests: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict]]:
    """Run PII detection tests"""
    passed = 0
    failed = 0
    failures = []
    
    for idx, test in enumerate(tests, 1):
        if 'PII' not in test['expected'].upper() and 'DETECT' not in test['expected'].upper():
            continue
        
        try:
            masked_text, pii_found = pii_detector.detect_and_mask(test['input'])
            should_block, reason = pii_detector.should_block_request(pii_found)
            
            # Check if test expectation matches
            if 'BLOCK' in test['expected'].upper():
                if should_block:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        'test_num': idx,
                        'name': test['name'],
                        'input': test['input'][:50] + '...',
                        'expected': 'BLOCK',
                        'actual': 'NOT BLOCKED',
                        'pii_found': [p['type'] for p in pii_found]
                    })
            elif 'DETECT' in test['expected'].upper() or 'PASS' in test['expected'].upper():
                if len(pii_found) > 0 or not should_block:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        'test_num': idx,
                        'name': test['name'],
                        'input': test['input'][:50] + '...',
                        'expected': 'DETECT (not block)',
                        'actual': 'BLOCKED' if should_block else 'NOT DETECTED',
                        'pii_found': [p['type'] for p in pii_found]
                    })
            else:
                passed += 1
                
        except Exception as e:
            failed += 1
            failures.append({
                'test_num': idx,
                'name': test['name'],
                'input': test['input'][:50] + '...',
                'expected': test['expected'],
                'actual': f'ERROR: {str(e)}',
                'pii_found': []
            })
    
    return passed, failed, failures


def run_injection_tests(injection_detector: GuardrailsAIInjectionDetector, tests: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict]]:
    """Run prompt injection detection tests"""
    passed = 0
    failed = 0
    failures = []
    
    for idx, test in enumerate(tests, 1):
        if 'INJECTION' not in test['expected'].upper() and 'OVERRIDE' not in test['expected'].upper() and 'JAILBREAK' not in test['expected'].upper():
            continue
        
        try:
            result = injection_detector.scan(test['input'])
            
            # Check if test expectation matches
            if 'BLOCK' in test['expected'].upper():
                if not result.is_safe:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        'test_num': idx,
                        'name': test['name'],
                        'input': test['input'][:50] + '...',
                        'expected': 'BLOCK (injection detected)',
                        'actual': 'NOT BLOCKED',
                        'threats': result.threats_found
                    })
            elif 'PASS' in test['expected'].upper():
                if result.is_safe:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        'test_num': idx,
                        'name': test['name'],
                        'input': test['input'][:50] + '...',
                        'expected': 'PASS (no injection)',
                        'actual': 'BLOCKED',
                        'threats': result.threats_found
                    })
            else:
                passed += 1
                
        except Exception as e:
            failed += 1
            failures.append({
                'test_num': idx,
                'name': test['name'],
                'input': test['input'][:50] + '...',
                'expected': test['expected'],
                'actual': f'ERROR: {str(e)}',
                'threats': []
            })
    
    return passed, failed, failures


def run_guardrails_tests() -> Dict[str, Any]:
    """Run all guardrails tests"""
    print("\n📋 Loading test cases...")
    tests = parse_test_file()
    print(f"✅ Loaded {len(tests)} test cases\n")
    
    print("🔧 Initializing guardrails...")
    pii_detector = PIIDetector()
    injection_detector = GuardrailsAIInjectionDetector()
    print("✅ Guardrails initialized\n")
    
    print("🧪 Running PII detection tests...")
    pii_passed, pii_failed, pii_failures = run_pii_tests(pii_detector, tests)
    print(f"   PII Tests: {pii_passed} passed, {pii_failed} failed\n")
    
    print("🧪 Running injection detection tests...")
    inj_passed, inj_failed, inj_failures = run_injection_tests(injection_detector, tests)
    print(f"   Injection Tests: {inj_passed} passed, {inj_failed} failed\n")
    
    total_passed = pii_passed + inj_passed
    total_failed = pii_failed + inj_failed
    total_tests = total_passed + total_failed
    all_failures = pii_failures + inj_failures
    
    # Print results
    print(f"{'='*80}")
    print("GUARDRAILS TEST RESULTS")
    print(f"{'='*80}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    
    if all_failures:
        print(f"\n{'='*80}")
        print("FAILED TESTS")
        print(f"{'='*80}")
        for failure in all_failures[:10]:  # Show first 10 failures
            print(f"\nTest #{failure['test_num']}: {failure['name']}")
            print(f"  Input: {failure['input']}")
            print(f"  Expected: {failure['expected']}")
            print(f"  Actual: {failure['actual']}")
        
        if len(all_failures) > 10:
            print(f"\n... and {len(all_failures) - 10} more failures")
    
    return {
        'passed': total_passed,
        'failed': total_failed,
        'total': total_tests,
        'pass_rate': total_passed / total_tests * 100 if total_tests > 0 else 0,
        'failures': all_failures,
        'details': f"PII: {pii_passed}/{pii_passed+pii_failed}, Injection: {inj_passed}/{inj_passed+inj_failed}"
    }


if __name__ == "__main__":
    results = run_guardrails_tests()
    sys.exit(0 if results['failed'] == 0 else 1)

# Made with Bob
