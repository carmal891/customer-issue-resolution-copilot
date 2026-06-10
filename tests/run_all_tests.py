"""
Comprehensive Test Runner for Customer Issue Resolution Copilot

Runs all test suites:
1. Skill Matching Tests
2. TPAO RAG Tests  
3. Guardrails Tests

Usage:
    python tests/run_all_tests.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime
from typing import Dict, Any

# Import test modules
from tests.test_skill_matching import run_skill_matching_tests
from tests.test_tpao_rag import run_tpao_rag_tests
from tests.test_guardrails import run_guardrails_tests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_summary(results: Dict[str, Any]):
    """Print test summary"""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_passed = 0
    total_failed = 0
    
    for suite_name, suite_results in results.items():
        passed = suite_results['passed']
        failed = suite_results['failed']
        total = passed + failed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        total_passed += passed
        total_failed += failed
        
        status_emoji = "✅" if failed == 0 else "⚠️" if pass_rate >= 70 else "❌"
        
        print(f"\n{status_emoji} {suite_name}")
        print(f"   Passed: {passed}/{total} ({pass_rate:.1f}%)")
        print(f"   Failed: {failed}/{total}")
        
        if suite_results.get('details'):
            print(f"   Details: {suite_results['details']}")
    
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    
    total_tests = total_passed + total_failed
    overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({overall_pass_rate:.1f}%)")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    elif overall_pass_rate >= 80:
        print("\n✅ GOOD - Most tests passing")
    elif overall_pass_rate >= 60:
        print("\n⚠️  NEEDS IMPROVEMENT - Some tests failing")
    else:
        print("\n❌ CRITICAL - Many tests failing")
    
    print("="*80 + "\n")
    
    return total_failed == 0


def main():
    """Run all test suites"""
    print("\n" + "="*80)
    print("CUSTOMER ISSUE RESOLUTION COPILOT - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    results = {}
    
    try:
        # 1. Skill Matching Tests
        print("\n" + "="*80)
        print("1. SKILL MATCHING TESTS")
        print("="*80)
        skill_results = run_skill_matching_tests()
        results['Skill Matching'] = skill_results
        
        # 2. TPAO RAG Tests
        print("\n" + "="*80)
        print("2. TPAO RAG TESTS")
        print("="*80)
        rag_results = run_tpao_rag_tests()
        results['TPAO RAG'] = rag_results
        
        # 3. Guardrails Tests
        print("\n" + "="*80)
        print("3. GUARDRAILS TESTS")
        print("="*80)
        guardrails_results = run_guardrails_tests()
        results['Guardrails'] = guardrails_results
        
        # Print summary
        all_passed = print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
