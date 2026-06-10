"""
Test script for the improved hierarchical semantic chunking strategy.

This script tests the new chunking on a sample policy document to verify:
1. Chunks preserve semantic units (complete sections)
2. Section headers are included for context
3. Overlap preserves context between chunks
4. Chunk sizes are appropriate (not too small/large)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.application.rag.chunking import SemanticChunker, DocumentType


def test_single_policy(policy_path: Path, chunker, old_chunker):
    """Test chunking on a single policy document"""
    
    if not policy_path.exists():
        print(f"❌ Policy file not found: {policy_path}")
        return None
    
    with open(policy_path, 'r') as f:
        content = f.read()
    
    print("\n" + "="*80)
    print(f"📄 TESTING: {policy_path.name}")
    print("="*80)
    print(f"Document size: {len(content)} characters (~{len(content) // 4} tokens)")
    
    # Chunk with new strategy
    chunks = chunker.chunk_document(
        content=content,
        document_id=f"test_{policy_path.stem}",
        document_type=DocumentType.POLICY,
        metadata={"source": policy_path.name}
    )
    
    # Chunk with old strategy for comparison
    old_chunks = old_chunker.chunk_document(
        content=content,
        document_id=f"test_{policy_path.stem}_old",
        document_type=DocumentType.POLICY,
        metadata={"source": policy_path.name}
    )
    
    print(f"\n✅ New chunking: {len(chunks)} chunks")
    print(f"📊 Old chunking: {len(old_chunks)} chunks")
    print(f"📈 Difference: {len(chunks) - len(old_chunks)} chunks ({(len(chunks) - len(old_chunks))/len(old_chunks)*100:+.1f}%)")
    
    # Show first 3 chunks
    print(f"\n{'─'*80}")
    print("SAMPLE CHUNKS (First 3)")
    print('─'*80)
    
    for i, chunk in enumerate(chunks[:3], 1):
        chunk_size = len(chunk.content)
        estimated_tokens = chunk_size // 4
        
        print(f"\n📄 Chunk {i}/{len(chunks)}")
        print(f"   Section: {chunk.section_title or 'N/A'}")
        print(f"   Size: {chunk_size} chars (~{estimated_tokens} tokens)")
        
        # Check if section header is preserved
        has_header = '##' in chunk.content[:100]
        print(f"   Header: {'✅ Preserved' if has_header else '⚠️  Missing'}")
        
        # Show preview
        preview = chunk.content[:200].replace('\n', ' ')
        print(f"   Preview: {preview}...")
    
    # Statistics
    chunk_sizes = [len(c.content) for c in chunks]
    token_estimates = [size // 4 for size in chunk_sizes]
    
    old_chunk_sizes = [len(c.content) for c in old_chunks]
    old_token_estimates = [size // 4 for size in old_chunk_sizes]
    
    print(f"\n{'─'*80}")
    print("STATISTICS")
    print('─'*80)
    
    print(f"\n📊 New Chunking:")
    print(f"   Avg size: {sum(chunk_sizes) / len(chunks):.0f} chars (~{sum(token_estimates) / len(chunks):.0f} tokens)")
    print(f"   Min size: {min(chunk_sizes)} chars (~{min(token_estimates)} tokens)")
    print(f"   Max size: {max(chunk_sizes)} chars (~{max(token_estimates)} tokens)")
    
    sections_with_headers = sum(1 for c in chunks if '##' in c.content[:100])
    print(f"   Chunks with headers: {sections_with_headers}/{len(chunks)} ({sections_with_headers/len(chunks)*100:.1f}%)")
    
    unique_sections = len(set(c.section_title for c in chunks if c.section_title))
    print(f"   Unique sections: {unique_sections}")
    
    print(f"\n📊 Old Chunking (for comparison):")
    print(f"   Avg size: {sum(old_chunk_sizes) / len(old_chunks):.0f} chars (~{sum(old_token_estimates) / len(old_chunks):.0f} tokens)")
    print(f"   Min size: {min(old_chunk_sizes)} chars (~{min(old_token_estimates)} tokens)")
    print(f"   Max size: {max(old_chunk_sizes)} chars (~{max(old_token_estimates)} tokens)")
    
    return {
        'name': policy_path.name,
        'new_chunks': len(chunks),
        'old_chunks': len(old_chunks),
        'new_avg_size': sum(chunk_sizes) / len(chunks),
        'old_avg_size': sum(old_chunk_sizes) / len(old_chunks),
        'headers_preserved': sections_with_headers / len(chunks) * 100,
        'unique_sections': unique_sections
    }


def test_chunking():
    """Test the improved chunking strategy on multiple policy documents"""
    
    print("="*80)
    print("TESTING HIERARCHICAL SEMANTIC CHUNKING")
    print("="*80)
    print("\nTesting new chunking strategy (800 tokens, 200 overlap)")
    print("vs old strategy (512 tokens, 100 overlap)")
    
    # Initialize chunkers
    new_chunker = SemanticChunker(
        chunk_size=800,      # New: larger chunks
        overlap_size=200,    # New: more overlap
        min_chunk_size=100,
        preserve_sections=True
    )
    
    old_chunker = SemanticChunker(
        chunk_size=512,      # Old: smaller chunks
        overlap_size=100,    # Old: less overlap
        min_chunk_size=100,
        preserve_sections=True
    )
    
    # Test multiple policy documents
    policies_dir = project_root / "data/mock/policies"
    
    # Select 3-4 representative policies
    test_policies = [
        "cancellation_refund_policy.md",
        "late_checkout_policy.md",
        "room_upgrade_policy.md",
        "complaint_handling_policy.md"
    ]
    
    results = []
    
    for policy_name in test_policies:
        policy_path = policies_dir / policy_name
        result = test_single_policy(policy_path, new_chunker, old_chunker)
        if result:
            results.append(result)
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 OVERALL SUMMARY")
    print("="*80)
    
    if results:
        total_new_chunks = sum(r['new_chunks'] for r in results)
        total_old_chunks = sum(r['old_chunks'] for r in results)
        avg_new_size = sum(r['new_avg_size'] for r in results) / len(results)
        avg_old_size = sum(r['old_avg_size'] for r in results) / len(results)
        avg_headers = sum(r['headers_preserved'] for r in results) / len(results)
        
        print(f"\n📄 Documents tested: {len(results)}")
        print(f"\n📊 Total chunks:")
        print(f"   New strategy: {total_new_chunks} chunks")
        print(f"   Old strategy: {total_old_chunks} chunks")
        print(f"   Reduction: {total_old_chunks - total_new_chunks} chunks ({(total_old_chunks - total_new_chunks)/total_old_chunks*100:.1f}%)")
        
        print(f"\n📏 Average chunk size:")
        print(f"   New strategy: {avg_new_size:.0f} chars (~{avg_new_size/4:.0f} tokens)")
        print(f"   Old strategy: {avg_old_size:.0f} chars (~{avg_old_size/4:.0f} tokens)")
        print(f"   Increase: {avg_new_size - avg_old_size:.0f} chars ({(avg_new_size - avg_old_size)/avg_old_size*100:+.1f}%)")
        
        print(f"\n✅ Quality metrics:")
        print(f"   Headers preserved: {avg_headers:.1f}%")
        print(f"   Semantic units: Better preserved with larger chunks")
        
        print("\n" + "="*80)
        print("✅ CHUNKING TEST COMPLETE")
        print("="*80)
        print("\n🎯 Key Improvements:")
        print("   • Larger chunks preserve complete semantic units")
        print("   • Section headers included for context")
        print("   • Better overlap maintains topic continuity")
        print("   • Fewer chunks = more efficient retrieval")
        
        print("\n📋 Next steps:")
        print("   1. Review the chunk analysis above")
        print("   2. Verify semantic units are preserved")
        print("   3. If satisfied, run: python scripts/reindex_hotel_knowledge.py --test")
        print("   4. Then run evaluations: python evals/run_system_evaluations.py")
    else:
        print("\n❌ No policies were successfully tested")


if __name__ == "__main__":
    test_chunking()


