#!/usr/bin/env python3
"""
Index all policy documents into the vector store using RAGPipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from src.application.rag.rag_pipeline import RAGPipeline, RAGConfig

# Load environment variables
load_dotenv()

def main():
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
print(" Error: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file")
        return
    
    print("Initializing RAG Pipeline...")
    
    # Create config - use 'hotel_knowledge' collection for policies
    config = RAGConfig(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        vector_store_path="./data/vector_store",
        collection_name="hotel_knowledge",  # Same collection used by TPAO
        chunk_size=512,
        chunk_overlap=100
    )
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Index documents - pass parent directory since pipeline adds /mock
    data_dir = Path("data")
    
    print(f"\nIndexing documents from {data_dir}/mock...")
    print("This will index:")
    print("  - Policy documents (*.md)")
    print("  - Booking data (bookings.json)")
    print("  - Customer issues (customer_issues.json)")
    print("  - Historical resolutions (historical_resolutions.json)")
    
    # Index with clear_existing=True to start fresh
    stats = pipeline.index_documents(data_dir, clear_existing=True)
    
print("\n Indexing complete!")
    print(f"\nStatistics:")
    print(f"  Documents indexed: {stats.get('num_documents', 0)}")
    print(f"  Chunks created: {stats.get('num_chunks', 0)}")
    print(f"  Time taken: {stats.get('total_time_seconds', 0):.2f}s")
    
    # Verify security policy was indexed
    print("\nVerifying security policy...")
    try:
        from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter
        vector_store = ChromaDBAdapter(
            persist_directory='data/vector_store',
            collection_name='hotel_knowledge'
        )
        collection = vector_store.client.get_collection('hotel_knowledge')
        results = collection.get()
        
        security_found = any('security_safety_policy' in str(metadata.get('source', ''))
                            for metadata in results['metadatas'])
        
        if security_found:
print(" Security policy successfully indexed")
        else:
print(" Security policy NOT found in index")
            
    except Exception as e:
        print(f"Could not verify: {e}")

if __name__ == '__main__':
    main()
