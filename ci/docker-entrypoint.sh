#!/bin/bash
set -e

echo "Starting Customer Issue Resolution Copilot..."

# Check if ChromaDB database exists and is corrupted
if [ -d "/app/data/vector_db" ]; then
    echo "Checking ChromaDB database integrity..."
    
    # Try to check if the database has the correct schema
    # If it fails, we'll clean it up
    python3 -c "
import chromadb
import os
import shutil
try:
    client = chromadb.PersistentClient(path='/app/data/vector_db')
    # Try to list collections - this will fail if schema is wrong
    client.list_collections()
    print('ChromaDB database is healthy')
except Exception as e:
    print(f'ChromaDB database is corrupted: {e}')
    print('Cleaning up corrupted database...')
    if os.path.exists('/app/data/vector_db'):
        shutil.rmtree('/app/data/vector_db')
        os.makedirs('/app/data/vector_db', exist_ok=True)
    print('Database cleaned successfully')
" || true
fi

# Ensure data directories exist
mkdir -p /app/data/vector_db
mkdir -p /app/data/skills
mkdir -p /app/data/policies
mkdir -p /app/logs

echo "Starting Streamlit application..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
