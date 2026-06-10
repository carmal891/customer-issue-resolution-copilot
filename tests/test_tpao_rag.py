"""
TPAO RAG Test Suite

Tests Knowledge Base RAG retrieval quality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import Dict, List, Tuple, Any
from src.application.rag.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_test_file() -> List[Tuple[str, str, str, str]]:
    """Parse tpao_rag_test_queries.txt"""
    test_file = Path(__file__).parent / "tpao_rag_test_queries.txt"
    tests = []
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '|' not in line:
                continue
            
            # Parse line: Query | Expected Doc | Category | Complexity
            parts = line.split('|')
            if len(parts) >= 4:
                query = parts[0].strip()
                if query[0].isdigit() and '.' in query[:3]:
                    query = query.split('.', 1)[1].strip()
                
                expected_doc = parts[1].strip()
                category = parts[2].strip()
                complexity = parts[3].strip()
                
                tests.append((query, expected_doc, category, complexity))
    
    return tests


def initialize_rag_pipeline() -> RAGPipeline:
    """Initialize RAG pipeline"""
    return RAGPipeline()


def check_document_retrieved(retrieved_contexts: List[Dict[str, Any]], expected_doc: str) -> bool:
    """Check if expected document was retrieved"""
    for context in retrieved_contexts:
        source = context.get('metadata', {}).get('source', '')
        # Flexible matching - check if expected doc name is in source
        if expected_doc.replace('.md', '') in source.replace('.md', ''):
            return True
    return False


def run_tpao_rag_tests() -> Dict[str, Any]:
    """Run all TPAO RAG tests"""
    print("\n📋 Loading test cases...")
    tests = parse_test_file()
    print(f"✅ Loaded {len(tests)} test cases\n")
    
    print("🔧 Initializing RAG pipeline...")
    rag_pipeline = initialize_rag_pipeline()
    print("✅ RAG pipeline initialized\n")
    
    passed = 0
    failed = 0
    failures = []
    
    print("🧪 Running tests...\n")
    
    for idx, (query, expected_doc, category, complexity) in enumerate(tests, 1):
        try:
            # Query RAG pipeline
            reranked_results, _ = rag_pipeline.query_with_reranking(
                query=query,
                top_k=5
            )
            
            # Convert to contexts format
            contexts = [
                {
                    'content': r.retrieval_result.content,
                    'metadata': r.retrieval_result.metadata,
                    'score': r.rerank_score
                }
                for r in reranked_results
            ]
            
            # Check if expected document was retrieved
            if check_document_retrieved(contexts, expected_doc):
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"
                retrieved_docs = [
                    c.get('metadata', {}).get('source', 'unknown')
                    for c in contexts[:3]
                ]
                
                failures.append({
                    'test_num': idx,
                    'query': query,
                    'expected_doc': expected_doc,
                    'retrieved_docs': retrieved_docs,
                    'category': category,
                    'complexity': complexity
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
                'expected_doc': expected_doc,
                'retrieved_docs': [f"ERROR: {str(e)}"],
                'category': category,
                'complexity': complexity
            })
    
    # Print results
    print(f"\n{'='*80}")
    print("TPAO RAG TEST RESULTS")
    print(f"{'='*80}")
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    
    if failures:
        print(f"\n{'='*80}")
        print("FAILED TESTS")
        print(f"{'='*80}")
        for failure in failures[:10]:  # Show first 10 failures
            print(f"\nTest #{failure['test_num']} ({failure['category']}, {failure['complexity']})")
            print(f"  Query: {failure['query']}")
            print(f"  Expected: {failure['expected_doc']}")
            print(f"  Retrieved: {', '.join(failure['retrieved_docs'][:3])}")
        
        if len(failures) > 10:
            print(f"\n... and {len(failures) - 10} more failures")
    
    return {
        'passed': passed,
        'failed': failed,
        'total': len(tests),
        'pass_rate': passed / len(tests) * 100 if len(tests) > 0 else 0,
        'failures': failures,
        'details': f"Context Recall: {passed}/{len(tests)} queries retrieved expected documents"
    }


if __name__ == "__main__":
    results = run_tpao_rag_tests()
    sys.exit(0 if results['failed'] == 0 else 1)

# Made with Bob
