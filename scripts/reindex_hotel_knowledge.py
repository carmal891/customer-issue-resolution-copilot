#!/usr/bin/env python3
"""
Reindex all policies into the hotel_knowledge collection used by TPAO.
This script uses the same simple approach as test_rag_simple.py but targets the production collection.

Usage:
    python reindex_hotel_knowledge.py           # Use production DB (data/vector_db)
    python reindex_hotel_knowledge.py --test    # Use test DB (./data/vector_store)
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Reindex hotel knowledge base')
parser.add_argument('--test', action='store_true', help='Use test database path (./data/vector_store)')
args = parser.parse_args()

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.embeddings.embedding_service import (
    OpenAIEmbeddingService
)
from src.infrastructure.vector_store.chromadb_adapter import (
    ChromaDBAdapter,
    DistanceMetric
)

print(" Reindexing hotel_knowledge collection...")
print("=" * 80)

# Step 1: Initialize OpenAI embedding service
print("\n Step 1: Initializing OpenAI Embedding Service...")
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print(" OPENAI_API_KEY not found in environment")
        sys.exit(1)

    embedding_service = OpenAIEmbeddingService(
        api_key=api_key,
        model='text-embedding-3-small'
    )
    print(f" Embedding service initialized: text-embedding-3-small")
    print(f"   Dimensions: {embedding_service.get_dimensions()}")
except Exception as e:
    print(f" Error initializing embedding service: {e}")
    sys.exit(1)

# Step 2: Initialize vector store
print("\n Step 2: Initializing Vector Store...")
try:
    # Use test or production path based on --test flag
    persist_dir = "./data/vector_store" if args.test else "data/vector_db"
    db_type = "TEST" if args.test else "PRODUCTION"

    vector_store = ChromaDBAdapter(
        persist_directory=persist_dir,
        collection_name="hotel_knowledge",  # Correct collection name for knowledge base
        distance_metric=DistanceMetric.COSINE
    )
    print(" Vector store initialized")
    print(f"   Collection: hotel_knowledge ({db_type})")
    print(f"   Persist directory: {persist_dir}")
    print(f"   Distance metric: COSINE")
except Exception as e:
    print(f" Error initializing vector store: {e}")
    sys.exit(1)

# Step 3: Load ALL policies
print("\n Step 3: Loading All Policy Documents...")
try:
    policies_dir = Path("./data/mock/policies")

    if not policies_dir.exists():
        print(f" Policies directory not found: {policies_dir}")
        sys.exit(1)

    # Get all markdown files
    policy_files = list(policies_dir.glob("*.md"))

    if not policy_files:
        print(f" No policy files found in {policies_dir}")
        sys.exit(1)

    print(f" Found {len(policy_files)} policy files:")
    for pf in sorted(policy_files):
        print(f"   - {pf.name}")

    # Load and chunk all policies
    all_chunks = []
    all_metadatas = []

    for policy_file in policy_files:
        with open(policy_file, 'r') as f:
            content = f.read()

        # Simple chunking - split by paragraphs
        file_chunks = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]

        # Create metadata for each chunk
        for i, chunk in enumerate(file_chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": policy_file.name,
                "source_name": policy_file.stem.replace('_', ' ').title(),
                "chunk_index": i,
                "doc_type": "policy",
                "domain": "hotel_operations"
            })

        print(f" {policy_file.name}: {len(file_chunks)} chunks")

    print(f"\n   Total: {len(all_chunks)} chunks from {len(policy_files)} policies")

except Exception as e:
    print(f" Error loading policies: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Generate embeddings
print("\n Step 4: Generating Embeddings...")
try:
    embedding_result = embedding_service.embed_texts(all_chunks)
    embeddings = embedding_result.embeddings
    print(f" Generated {len(embeddings)} embeddings")
    print(f"   Model: {embedding_result.model}")
    print(f"   Dimensions: {embedding_result.dimensions}")
    if embedding_result.token_count:
        print(f"   Tokens used: {embedding_result.token_count}")
except Exception as e:
    print(f" Error generating embeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Clear and reindex
print("\n Step 5: Clearing and Reindexing Vector Store...")
try:
    # Clear existing data
    print("   Clearing existing data...")
    vector_store.clear()

    # Add chunks with embeddings
    ids = [f"policy_chunk_{i}" for i in range(len(all_chunks))]

    print(f"   Adding {len(all_chunks)} chunks...")
    vector_store.add_documents(
        chunk_ids=ids,
        contents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )

    count = vector_store.count()
    print(f" Successfully indexed {count} chunks in hotel_knowledge collection")
except Exception as e:
    print(f" Error indexing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Verify with test query
print("\n Step 6: Verifying with Test Query...")
try:
    test_query = "suspicious person loitering in hallway"
    print(f"   Query: '{test_query}'")

    query_embedding = embedding_service.embed_query(test_query)
    results = vector_store.search(
        query_embedding=query_embedding,
        n_results=3
    )

    if results:
        print(f" Found {len(results)} results:")
        for i, result in enumerate(results):
            source = result.metadata.get('source', 'Unknown')
            print(f"\n   Result {i+1} (score: {result.score:.4f}):")
            print(f"   Source: {source}")
            print(f"   Content: {result.content[:150]}...")
    else:
        print(" ️ No results found - this might be a problem")

except Exception as e:
    print(f" Error during verification: {e}")

print("\n" + "=" * 80)
print(" Reindexing Complete!")
db_type = "TEST" if args.test else "PRODUCTION"
persist_dir = "./data/vector_store" if args.test else "data/vector_db"
print(f"\nThe hotel_knowledge collection ({db_type}) now contains all policy documents.")
print(f"Location: {persist_dir}")
print("TPAO should now retrieve policies instead of using LLM fallback.")
print("\nTest it by submitting: 'There's a suspicious person loitering in the hallway'")
