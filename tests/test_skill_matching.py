"""
Skill Matching Test Suite

Tests semantic skill matching accuracy based on test_queries.txt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import Dict, List, Tuple
from src.domain.models.issue import Issue, IssueChannel, IssuePriority
from src.application.skills.skill_matcher import SkillMatcher
from src.application.skills.skill_registry import SkillRegistry
from src.application.rag.rag_pipeline import RAGPipeline
from src.infrastructure.llm.llm_service import LLMService

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_test_file() -> List[Tuple[str, str, str, str]]:
    """Parse skill_matching_test_queries.txt"""
    test_file = Path(__file__).parent / "skill_matching_test_queries.txt"
    tests = []
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '|' not in line:
                continue
            
            # Skip header-like lines
            if line.startswith('1.') or 'Query |' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    # Extract query (remove number prefix)
                    query = parts[0].strip()
                    if query[0].isdigit() and '.' in query[:3]:
                        query = query.split('.', 1)[1].strip()
                    
                    expected_skill_id = parts[1].strip()
                    expected_skill_name = parts[2].strip()
                    category = parts[3].strip()
                    
                    tests.append((query, expected_skill_id, expected_skill_name, category))
    
    return tests


def initialize_skill_matcher() -> SkillMatcher:
    """Initialize skill matcher with indexed skills"""
    # Initialize RAG pipeline for embeddings
    rag_pipeline = RAGPipeline()
    
    # Initialize skill registry
    skill_registry = SkillRegistry(
        embedding_service=rag_pipeline.embedding_service,
        vector_store=rag_pipeline.vector_store
    )
    
    # Initialize LLM service
    llm_service = LLMService(model="gpt-5.4-mini")
    
    # Create skill matcher
    skill_matcher = SkillMatcher(
        skill_registry=skill_registry,
        embedding_service=rag_pipeline.embedding_service,
        vector_store=rag_pipeline.vector_store,
        llm_service=llm_service,
        enable_query_reformulation=True
    )
    
    return skill_matcher


def run_skill_matching_tests() -> Dict[str, any]:
    """Run all skill matching tests"""
    print("\n📋 Loading test cases...")
    tests = parse_test_file()
    print(f"✅ Loaded {len(tests)} test cases\n")
    
    print("🔧 Initializing skill matcher...")
    skill_matcher = initialize_skill_matcher()
    print("✅ Skill matcher initialized\n")
    
    passed = 0
    failed = 0
    failures = []
    
    print("🧪 Running tests...\n")
    
    for idx, (query, expected_skill_id, expected_skill_name, category) in enumerate(tests, 1):
        try:
            # Create mock issue
            issue = Issue(
                issue_id=f"test_{idx}",
                channel=IssueChannel.EMAIL,
                body=query,
                priority=IssuePriority.MEDIUM,
                guest_email="test@example.com",
                booking_id=None,
                subject="Test issue",
                issue_type=None,
                expected_skill=expected_skill_id,
                expected_resolution=None
            )
            
            # Match skill
            matches = skill_matcher.match_skill(issue, top_k=3)
            
            # Check if correct skill is top match
            if matches and matches[0].skill.skill_id == expected_skill_id:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"
                actual_skill = matches[0].skill.skill_id if matches else "NO_MATCH"
                actual_name = matches[0].skill.name if matches else "NO_MATCH"
                actual_score = matches[0].score if matches else 0.0
                
                failures.append({
                    'test_num': idx,
                    'query': query,
                    'expected': f"{expected_skill_id} ({expected_skill_name})",
                    'actual': f"{actual_skill} ({actual_name})",
                    'score': actual_score,
                    'category': category
                })
            
            # Print progress every 5 tests
            if idx % 5 == 0:
                print(f"Progress: {idx}/{len(tests)} tests completed")
                
        except Exception as e:
            failed += 1
            status = "❌ ERROR"
            failures.append({
                'test_num': idx,
                'query': query,
                'expected': f"{expected_skill_id} ({expected_skill_name})",
                'actual': f"ERROR: {str(e)}",
                'score': 0.0,
                'category': category
            })
    
    # Print results
    print(f"\n{'='*80}")
    print("SKILL MATCHING TEST RESULTS")
    print(f"{'='*80}")
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    
    if failures:
        print(f"\n{'='*80}")
        print("FAILED TESTS")
        print(f"{'='*80}")
        for failure in failures:
            print(f"\nTest #{failure['test_num']} ({failure['category']})")
            print(f"  Query: {failure['query']}")
            print(f"  Expected: {failure['expected']}")
            print(f"  Actual: {failure['actual']}")
            print(f"  Score: {failure['score']:.4f}")
    
    return {
        'passed': passed,
        'failed': failed,
        'total': len(tests),
        'pass_rate': passed / len(tests) * 100 if len(tests) > 0 else 0,
        'failures': failures
    }


if __name__ == "__main__":
    results = run_skill_matching_tests()
    sys.exit(0 if results['failed'] == 0 else 1)

# Made with Bob
