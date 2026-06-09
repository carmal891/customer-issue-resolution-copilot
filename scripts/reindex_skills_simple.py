#!/usr/bin/env python3
"""Simple script to re-index skills with enriched embeddings."""

import sys
import os

# Set environment to avoid OpenMP issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("SKILL RE-INDEXING SCRIPT")
print("=" * 80)

# Step 1: Delete old vector store
print("\n[1/3] Deleting old vector store...")
import shutil
from pathlib import Path

vector_store_path = Path("data/vector_store")
if vector_store_path.exists():
    shutil.rmtree(vector_store_path)
    print(" Deleted old vector store")
else:
    print("   ℹ️  No existing vector store found")

# Step 2: Initialize components
print("\n[2/3] Initializing components...")
from src.infrastructure.embeddings.embedding_service import OpenAIEmbeddingService
from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter
from src.application.skills.skill_registry import SkillRegistry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(" Error: OPENAI_API_KEY not found in environment variables")
    print("   Please set it in your .env file or export it:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

embedding_service = OpenAIEmbeddingService(api_key=api_key)
vector_store = ChromaDBAdapter(
    collection_name="hotel_skills", persist_directory="./data/vector_store"
)
skill_registry = SkillRegistry(embedding_service=embedding_service, vector_store=vector_store)
print(" Components initialized")

# Step 3: Load and index skills from YAML files
print("\n[3/3] Loading and indexing skills from YAML files...")

# Get list of skill metadata (loads from registry.yaml)
skill_metadata_list = skill_registry.list_skills(active_only=True)
print(f"   Found {len(skill_metadata_list)} skills in registry")

if len(skill_metadata_list) == 0:
    print("\n ️ No skills found in registry!")
    print("   Make sure data/skills/registry.yaml exists and contains skill entries")
    sys.exit(1)

indexed_count = 0
for metadata in skill_metadata_list:
    print(f"\n   Loading: {metadata.skill_id}")
    print(f"   Name: {metadata.name}")
    print(f"   File: {metadata.file_path}")

    try:
        # Load skill from YAML file
        skill = skill_registry._load_skill_from_file(metadata.file_path)
        print(f"   Triggers: {len(skill.triggers)} patterns")

        # Index the skill triggers
        result = skill_registry.index_skill_triggers(skill.skill_id)
        if result:
            indexed_count += 1
            print(f" Successfully indexed")
        else:
            print(f" Failed to index")
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print(f"COMPLETE: Indexed {indexed_count}/{len(skill_metadata_list)} skills")
print("=" * 80)
