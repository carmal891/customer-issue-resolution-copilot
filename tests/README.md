# Tests Directory

no This directory contains all tests for the Customer Issue Resolution Copilot system.

## Directory Structure

```
tests/
├── README.md                    # This file
├── test_agentic_workflow.py     # End-to-end agentic workflow tests
├── test_simple_workflow.py      # Simple workflow tests
├── unit/                        # Unit tests for individual components
├── integration/                 # Integration tests
│   └── test_rag_pipeline.py    # RAG pipeline integration tests
└── evaluation/                  # Evaluation metrics and test harnesses
    └── rag_metrics.py          # RAG evaluation metrics
```

## Test Categories

### Unit Tests (`unit/`)
Tests for individual components in isolation:
- Domain models
- Service classes
- Utility functions
- Individual tools

### Integration Tests (`integration/`)
Tests for component interactions:
- RAG pipeline (retrieval + reranking)
- Skill matching system
- Orchestrator + TPAO loop
- Tool execution flow

### Evaluation Tests (`evaluation/`)
Evaluation frameworks and metrics:
- RAG quality metrics (faithfulness, relevancy, precision, recall)
- Agent behavior metrics (skill match accuracy, plan quality)
- Guardrail effectiveness
- End-to-end workflow validation

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test category
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Evaluation tests only
python -m pytest tests/evaluation/
```

### Run specific test file
```bash
python -m pytest tests/test_agentic_workflow.py -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Data

Test data is located in:
- `data/mock/` - Mock bookings, issues, policies, resolutions
- `evals/test_data/` - Evaluation test cases

## Evaluation System

The evaluation system is located in the `evals/` directory (separate from tests):
- `evals/run_system_evaluations.py` - Main evaluation runner
- `evals/knowledge_base_rag_evaluator.py` - KB RAG evaluator
- `evals/skill_matching_rag_evaluator.py` - Skill matching evaluator
- `evals/test_data/` - Evaluation test cases
- `evals/results/` - Evaluation results (JSON)
- `evals/reports/` - Evaluation reports (Markdown)

Run evaluations:
```bash
python evals/run_system_evaluations.py
```

## Writing New Tests

### Unit Test Template
```python
import pytest
from src.domain.models.issue import Issue

def test_issue_creation():
    issue = Issue(
        issue_id="test_001",
        channel="email",
        body="Test issue"
    )
    assert issue.issue_id == "test_001"
```

### Integration Test Template
```python
import pytest
from src.application.rag.rag_pipeline import RAGPipeline

@pytest.fixture
def rag_pipeline():
    # Setup
    pipeline = RAGPipeline(...)
    yield pipeline
    # Teardown

def test_rag_retrieval(rag_pipeline):
    results = rag_pipeline.retrieve("test query")
    assert len(results) > 0
```

## Test Guidelines

1. **Isolation**: Unit tests should not depend on external services
2. **Fixtures**: Use pytest fixtures for common setup
3. **Mocking**: Mock external dependencies (LLM, vector DB) in unit tests
4. **Coverage**: Aim for >80% code coverage
5. **Documentation**: Add docstrings to test functions
6. **Naming**: Use descriptive test names (test_<what>_<condition>_<expected>)

## CI/CD Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

## Troubleshooting

### Common Issues

**Import errors**: Ensure you're running from project root
```bash
cd /path/to/customer-issue-resolution-copilot
python -m pytest tests/
```

**Missing dependencies**: Install test dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

**Vector DB errors**: Clear test vector store
```bash
rm -rf data/vector_store_test/
```

## Related Documentation

- [Testing Guide](../docs/TESTING_GUIDE.md) - Comprehensive testing documentation
- [Evaluation Framework](../docs/evaluation-framework.md) - Evaluation system design
- [System Design](../docs/system-design-document.md) - Overall system architecture