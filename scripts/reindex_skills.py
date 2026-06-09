#!/usr/bin/env python3
"""
Re-index all skills in vector DB with complete skill data.

This script loads skills from YAML files and stores complete skill data
(including all steps) in ChromaDB metadata for fast runtime retrieval.

Usage:
    python scripts/reindex_skills.py           # Use production DB (data/vector_db)
    python scripts/reindex_skills.py --test    # Use test DB (./data/vector_store)
"""

import sys
import argparse
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Reindex skills')
parser.add_argument('--test', action='store_true', help='Use test database path (./data/vector_store)')
args = parser.parse_args()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.application.skills.skill_registry import SkillRegistry
from src.infrastructure.embeddings.embedding_service import SentenceTransformerEmbeddingService
from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter


def main():
    """Re-index all skills with complete data."""
    # Use test or production path based on --test flag
    persist_dir = "./data/vector_store" if args.test else "data/vector_db"
    db_type = "TEST" if args.test else "PRODUCTION"
    
print(f" Re-indexing skills with complete data in vector DB ({db_type})...")
    print(f"   Persist directory: {persist_dir}")
    
    # Initialize components
print(" Initializing embedding service and vector store...")
    embedding_service = SentenceTransformerEmbeddingService()
    vector_store = ChromaDBAdapter(
        collection_name="hotel_skills",
        persist_directory=persist_dir
    )
    
    # Initialize skill registry
print(" Loading skill registry...")
    registry = SkillRegistry(
        skills_dir="data/skills",
        registry_file="data/skills/registry.yaml",
        embedding_service=embedding_service,
        vector_store=vector_store
    )
    
    # Get all active skills
    active_skills = registry.list_skills(active_only=True)
print(f" Found {len(active_skills)} active skills")
    
    # Re-index each skill
print("\n Re-indexing skills...")
    success_count = 0
    for metadata in active_skills:
        skill_id = metadata.skill_id
        print(f"  • {skill_id} ({metadata.name})...", end=" ")
        
        if registry.index_skill_triggers(skill_id):
            success_count += 1
print("")
        else:
print(" FAILED")
    
print(f"\n Successfully re-indexed {success_count}/{len(active_skills)} skills")
print(f" Skills now contain complete data in vector DB ({db_type})!")
    print(f"   Location: {persist_dir}")
print("\n Skills will now be loaded directly from vector DB at runtime (no file I/O)")


if __name__ == "__main__":
    main()
