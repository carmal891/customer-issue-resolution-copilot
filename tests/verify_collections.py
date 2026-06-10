#!/usr/bin/env python3
"""
Verify vector database collections and their contents.
Shows what's in each collection to avoid confusion.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter, DistanceMetric

print("="*80)
print("VECTOR DATABASE VERIFICATION")
print("="*80)

persist_dir = "./data/vector_store"

print(f"\nDatabase Directory: {persist_dir}")
print("\n" + "-"*80)

# Check hotel_knowledge collection
print("\n📚 Collection: hotel_knowledge (Policy Documents)")
print("-"*80)
try:
    kb_store = ChromaDBAdapter(
        persist_directory=persist_dir,
        collection_name="hotel_knowledge",
        distance_metric=DistanceMetric.COSINE
    )
    count = kb_store.count()
    print(f"✅ Status: Exists")
    print(f"📊 Documents: {count} chunks")
    
    if count == 0:
        print("⚠️  Collection is empty - needs indexing")
    else:
        print(f"✅ Collection has data - ready for queries")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("⚠️  Collection may not exist yet")

# Check hotel_skills collection
print("\n" + "-"*80)
print("\n🎯 Collection: hotel_skills (Skill Triggers)")
print("-"*80)
try:
    skills_store = ChromaDBAdapter(
        persist_directory=persist_dir,
        collection_name="hotel_skills",
        distance_metric=DistanceMetric.COSINE
    )
    count = skills_store.count()
    print(f"✅ Status: Exists")
    print(f"📊 Documents: {count} skills")
    
    if count == 0:
        print("⚠️  Collection is empty - needs indexing")
    else:
        print(f"✅ Collection has data - ready for queries")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("⚠️  Collection may not exist yet")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n✅ Both collections use the SAME database directory:")
print(f"   {persist_dir}")
print("\n📚 hotel_knowledge = Policy documents for RAG")
print("🎯 hotel_skills = Skill triggers for matching")
print("\n💡 Think of collections like SQL tables in the same database!")
print("="*80)


