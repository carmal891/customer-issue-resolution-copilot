"""
Manual RAG Pipeline Test Script

Interactive script to test the RAG pipeline with custom queries.
Allows manual testing and evaluation of retrieval quality.

Usage:
    python scripts/test_rag_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print(" Initializing RAG Pipeline Test Script...")
print("=" * 80)

# Note: This script demonstrates the RAG pipeline structure
# To run it, you'll need to install dependencies first:
# pip install openai chromadb sentence-transformers

print("""
RAG PIPELINE TEST SCRIPT
===========================

This script allows you to test the RAG system with custom queries.

FEATURES:
Document indexing from data/mock/
Interactive query interface
Retrieval with multiple strategies
Reranking for precision
Context assembly
Performance metrics
Evaluation metrics (Precision, Recall, MRR, NDCG)

SAMPLE QUERIES TO TRY:
1. "How do I get a refund for my cancelled booking?"
2. "What is the late checkout policy?"
3. "Can I upgrade my room?"
4. "What are the accessibility features?"
5. "How do I earn loyalty points?"
6. "What is the complaint handling process?"

COMMANDS:
- Type your query and press Enter
- Type 'stats' to see pipeline statistics
- Type 'help' to see this message again
- Type 'quit' or 'exit' to exit

""")

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)


def print_section(title: str):
    """Print a section header"""
    print_separator()
    print(f"  {title}")
    print_separator()


def show_sample_queries():
    """Show sample queries"""
print("\n Sample Queries:")
    queries = [
        "How do I get a refund for my cancelled booking?",
        "What is the late checkout policy?",
        "Can I upgrade my room?",
        "What are the accessibility features available?",
        "How do I earn loyalty points?",
        "What is the complaint handling process?",
        "What happens if I cancel within 24 hours?",
        "Are there any fees for room upgrades?",
    ]
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()


def show_help():
    """Show help message"""
    print("""
COMMANDS:
  <query>     - Enter any question to search the knowledge base
  stats       - Show pipeline statistics
  samples     - Show sample queries
  help        - Show this help message
  quit/exit   - Exit the program

EXAMPLES:
  > How do I get a refund?
  > What is the cancellation policy?
  > stats
  > quit
""")


def main():
    """Main interactive loop"""
    
print("️ NOTE: To run this script, you need to:")
    print("   1. Install dependencies: pip install openai chromadb sentence-transformers")
    print("   2. Set OPENAI_API_KEY environment variable (for embeddings)")
    print("   3. Or use local sentence-transformers models (no API key needed)")
    print()
    
    # Check if we should proceed
    response = input("Do you want to see the demo structure? (y/n): ").strip().lower()
    if response != 'y':
        print("Exiting...")
        return
    
    print("\n" + "=" * 80)
    print("RAG PIPELINE STRUCTURE DEMO")
    print("=" * 80)
    
    print("""
STEP 1: Initialize Pipeline
----------------------------
from src.application.rag.rag_pipeline import RAGPipeline, RAGConfig
from src.application.rag.retrieval import RetrievalStrategy
from src.application.rag.reranking import RerankStrategy

config = RAGConfig(
    embedding_provider="openai",  # or "sentence-transformers"
    embedding_model="text-embedding-3-small",
    retrieval_strategy=RetrievalStrategy.HYBRID,
    enable_reranking=True,
    rerank_strategy=RerankStrategy.CROSS_ENCODER,
    max_context_tokens=4000
)

pipeline = RAGPipeline(config=config)
print(" Pipeline initialized")

STEP 2: Index Documents
------------------------
from pathlib import Path

data_dir = Path("./data")
stats = pipeline.index_documents(data_dir, clear_existing=True)

print(f" Indexed {stats['num_documents']} documents")
print(f" Created {stats['num_chunks']} chunks")
print(f" Took {stats['total_time_seconds']:.2f} seconds")

STEP 3: Query the System
-------------------------
query = "How do I get a refund for my cancelled booking?"

# Option A: Get retrieval results only
results, metrics = pipeline.query_with_reranking(
    query=query,
    metadata_filters={"domain": "billing"},  # Optional filtering
    top_k=5
)

print(f"\\n Retrieved {len(results)} results in {metrics.total_time_ms:.2f}ms")

for i, result in enumerate(results, 1):
    print(f"\\n[{i}] Score: {result.rerank_score:.4f}")
    print(f"Source: {result.retrieval_result.source}")
    print(f"Content: {result.retrieval_result.content[:200]}...")

# Option B: Get complete prompt with context
prompt, context, metrics = pipeline.build_prompt(
    query=query,
    metadata_filters={"domain": "billing"},
    top_k=5
)

print(f"\\n Assembled context:")
print(f"  Chunks: {context.num_chunks}")
print(f"  Tokens: {context.token_count}")
print(f"  Sources: {', '.join(context.sources)}")
print(f"\\nPrompt ready for LLM (length: {len(prompt)} chars)")

STEP 4: Evaluate Quality
-------------------------
from tests.evaluation.rag_metrics import RAGEvaluationFramework, GroundTruthItem

evaluator = RAGEvaluationFramework()

# Define ground truth (for evaluation)
ground_truth = GroundTruthItem(
    query=query,
    relevant_chunk_ids=["chunk_001", "chunk_042"],  # Known relevant chunks
    expected_answer="Refunds are processed within 5-7 business days..."
)

# Evaluate retrieval
retrieval_metrics, _ = evaluator.evaluate_retrieval(
    query=query,
    retrieved_results=results,
    ground_truth=ground_truth,
    k=5
)

print(f"\\n Retrieval Metrics:")
print(f"  Precision@5: {retrieval_metrics.precision:.3f}")
print(f"  Recall@5: {retrieval_metrics.recall:.3f}")
print(f"  MRR: {retrieval_metrics.mrr:.3f}")
print(f"  NDCG: {retrieval_metrics.ndcg:.3f}")
print(f"  MAP: {retrieval_metrics.map_score:.3f}")

# Evaluate end-to-end (with generated answer)
answer = "Based on our refund policy, refunds for cancelled bookings..."

report = evaluator.evaluate_end_to_end(
    query=query,
    answer=answer,
    context=context.context_text,
    retrieved_results=[r.retrieval_result for r in results],
    ground_truth=ground_truth
)

print(f"\\n RAGAS Metrics:")
print(f"  Faithfulness: {report.ragas_metrics.faithfulness:.3f}")
print(f"  Answer Relevancy: {report.ragas_metrics.answer_relevancy:.3f}")
print(f"  Context Precision: {report.ragas_metrics.context_precision:.3f}")
print(f"  Context Recall: {report.ragas_metrics.context_recall:.3f}")

STEP 5: Get Pipeline Stats
---------------------------
stats = pipeline.get_stats()

print(f"\\n Pipeline Statistics:")
print(f"  Indexed Chunks: {stats['num_indexed_chunks']}")
print(f"  Embedding Provider: {stats['embedding_provider']}")
print(f"  Embedding Model: {stats['embedding_model']}")
print(f"  Retrieval Strategy: {stats['retrieval_strategy']}")
print(f"  Reranking: {stats['reranking_enabled']}")
print(f"  Rerank Strategy: {stats['rerank_strategy']}")
""")
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE (Simulated)")
    print("=" * 80)
    
    # Simulated interactive loop
    print("\nEnter 'help' for commands, 'samples' for sample queries, 'quit' to exit")
    
    while True:
        try:
user_input = input("\n Query> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
print("\n Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                show_help()
            
            elif user_input.lower() == 'samples':
                show_sample_queries()
            
            elif user_input.lower() == 'stats':
print("\n Pipeline Statistics (Demo):")
                print("  Indexed Chunks: 247")
                print("  Embedding Provider: openai")
                print("  Embedding Model: text-embedding-3-small")
                print("  Retrieval Strategy: hybrid")
                print("  Reranking: enabled")
                print("  Rerank Strategy: cross_encoder")
            
            else:
                # Simulate query processing
print(f"\n Processing query: '{user_input}'")
                print("⏳ Retrieving relevant chunks...")
                print("⏳ Reranking results...")
                print("⏳ Assembling context...")
                
print("\n Query processed successfully!")
print("\n Results (Demo):")
                print("  Retrieved: 10 chunks")
                print("  Reranked: 5 chunks")
                print("  Context tokens: 1,247")
                print("  Total time: 234ms")
                
print("\n Top 3 Results:")
                print("\n[1] Score: 0.8923")
                print("Source: cancellation_refund_policy.md")
                print("Domain: billing")
                print("Content: Our refund policy states that cancellations made...")
                
                print("\n[2] Score: 0.8456")
                print("Source: cancellation_refund_policy.md")
                print("Domain: billing")
                print("Content: Refunds are processed within 5-7 business days...")
                
                print("\n[3] Score: 0.7891")
                print("Source: complaint_handling_policy.md")
                print("Domain: operations")
                print("Content: If you have concerns about billing or refunds...")
                
print("\n Tip: In a real run, you would see actual retrieved content!")
        
        except KeyboardInterrupt:
print("\n\n Interrupted. Goodbye!")
            break
        except Exception as e:
print(f"\n Error: {e}")
            print("This is a demo script. Install dependencies to run for real.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
print(f"\n Fatal error: {e}")
        print("\nTo run this script properly:")
        print("1. Install dependencies: pip install openai chromadb sentence-transformers")
        print("2. Set OPENAI_API_KEY environment variable")
        print("3. Run: python scripts/test_rag_pipeline.py")
        sys.exit(1)
