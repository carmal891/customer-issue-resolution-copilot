.PHONY: help setup install reindex run clean test eval shell

# Default target
help:
	@echo "Customer Issue Resolution Copilot - Makefile Commands"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies with Poetry (creates venv)"
	@echo "  make setup      - Alias for install"
	@echo "  make shell      - Activate Poetry virtual environment"
	@echo "  make reindex    - Reindex skills and knowledge base"
	@echo "  make run        - Start the Streamlit application"
	@echo "  make start      - Reindex and run (full startup)"
	@echo "  make clean      - Clean vector stores and cache"
	@echo "  make test       - Run tests"
	@echo "  make eval       - Run evaluations"
	@echo ""
	@echo "Note: All commands run inside Poetry's virtual environment"
	@echo ""

# Install dependencies with Poetry (creates venv automatically)
install:
	@echo "📦 Installing dependencies with Poetry..."
	@echo "Poetry will create and use a virtual environment automatically"
	poetry install
	@echo "✅ Dependencies installed in Poetry venv"
	@echo ""
	@echo "To activate the venv manually, run: poetry shell"


# Activate Poetry shell
shell:
	@echo "🐚 Activating Poetry virtual environment..."
	poetry shell

# Reindex skills and knowledge base (runs in Poetry venv)
reindex:
	@echo "🧹 Cleaning old vector stores first..."
	rm -rf data/vector_store data/vector_db
	@echo ""
	@echo "🔄 Reindexing skills (in Poetry venv)..."
	poetry run python reindex_skills_simple.py
	@echo ""
	@echo "🔄 Reindexing hotel knowledge base (in Poetry venv)..."
	poetry run python reindex_hotel_knowledge.py
	@echo ""
	@echo "✅ Reindexing complete!"

# Run Streamlit app (runs in Poetry venv)
run:
	@echo "🚀 Starting Streamlit application (in Poetry venv)..."
	poetry run streamlit run app.py

# Full startup: reindex then run
start: reindex run

# Clean vector stores and cache
clean:
	@echo "🧹 Cleaning vector stores and cache..."
	rm -rf data/vector_store
	rm -rf data/vector_db
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

# Run tests (runs in Poetry venv)
test:
	@echo "🧪 Running tests (in Poetry venv)..."
	poetry run pytest tests/ -v
	@echo "✅ Tests complete!"

# Run evaluations (runs in Poetry venv)
eval:
	@echo "📊 Running evaluations (in Poetry venv)..."
	poetry run python evals/run_system_evaluations.py
	@echo "✅ Evaluations complete!"
