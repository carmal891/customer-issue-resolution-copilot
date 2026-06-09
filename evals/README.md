# Evaluation Framework

This directory contains the automated evaluation framework for the Customer Issue Resolution Copilot system.

## Overview

The evaluation framework tests two critical components:
1. **RAG Pipeline** - Knowledge base retrieval quality
2. **Skill Matching** - Semantic skill selection accuracy

## Quick Start

Run the complete evaluation suite:

```bash
python evals/run_system_evaluations.py
```

This will:
1. Clear and re-index vector stores with test data
2. Run RAG quality evaluations
3. Run skill matching evaluations
4. Generate a PM-friendly markdown report in `evals/evals/reports/`

## Database Configuration

### Test vs Production Isolation

The system uses **separate vector databases** for testing and production to ensure evaluation runs don't interfere with production data.

#### Database Paths

| Environment | Knowledge Base Path | Skills Path | Collection Names |
|-------------|-------------------|-------------|------------------|
| **Production** | `data/vector_db` | `data/vector_db` | `hotel_knowledge`, `hotel_skills` |
| **Test/Eval** | `./data/vector_store` | `./data/vector_store` | `hotel_knowledge`, `hotel_skills` |

### How It Works

The indexing scripts accept a `--test` flag to switch between environments:

**Production Indexing:**
```bash
python reindex_hotel_knowledge.py           # Uses data/vector_db
python scripts/reindex_skills.py            # Uses data/vector_db
```

**Test/Evaluation Indexing:**
```bash
python reindex_hotel_knowledge.py --test    # Uses ./data/vector_store
python scripts/reindex_skills.py --test     # Uses ./data/vector_store
```

**Automated Evaluation** (uses `--test` automatically):
```bash
python evals/run_system_evaluations.py
```

### Why This Approach?

1. **Isolation:** Test runs won't corrupt production embeddings
2. **Reproducibility:** Each evaluation starts with a clean slate
3. **Flexibility:** Same scripts work for both environments
4. **Safety:** Production data remains untouched during testing

## Evaluation Workflow

### 1. Automatic Re-indexing

Before each evaluation run, the system:

```
🗑️  Clear ./data/vector_store/
📚 Re-index knowledge base → hotel_knowledge collection
🎯 Re-index skills → hotel_skills collection
🔍 Verify indexing succeeded (checks document counts)
```

**Verification Step:**
- Ensures knowledge base has documents (expected: ~300+)
- Ensures skills collection has documents (expected: ~13)
- Fails fast if indexing didn't work

### 2. RAG Evaluation

Tests knowledge base retrieval quality using LLM-as-judge:

**Metrics:**
- **Faithfulness** (≥0.85): Answer grounded in retrieved docs
- **Answer Relevancy** (≥0.90): Answer addresses the question
- **Context Precision** (≥0.80): Retrieved docs are relevant
- **Context Recall** (≥0.90): All relevant docs retrieved

**Test Categories:**
- Policy questions (cancellation, refund, upgrade)
- Procedural questions (check-in, complaints, security)
- Edge cases (conflicting info, missing policies)

### 3. Skill Matching Evaluation

Tests semantic skill selection accuracy:

**Metrics:**
- **Top-1 Accuracy** (≥90%): Correct skill ranked #1
- **Top-3 Accuracy** (≥95%): Correct skill in top 3
- **Mean Reciprocal Rank (MRR)**: Average 1/rank

**Test Categories:**
- Exact trigger matches
- Paraphrased requests
- Multi-intent queries
- Ambiguous cases

### 4. Report Generation

Generates a markdown report with:
- Executive summary
- Pass/fail status for each metric
- Category-level breakdowns
- Detailed findings and recommendations
- Timestamp and result file references

**Report Location:** `evals/evals/reports/evaluation_report_YYYYMMDD_HHMMSS.md`

## Test Data

### Knowledge Base Test Cases

Located in: `evals/test_cases/kb_rag_test_cases.json`

Example structure:
```json
{
  "question": "What is the cancellation policy?",
  "expected_answer": "Guests can cancel...",
  "category": "policy_cancellation",
  "ground_truth_sources": ["cancellation_policy.md"]
}
```

### Skill Matching Test Cases

Located in: `evals/test_cases/skill_matching_test_cases.json`

Example structure:
```json
{
  "query": "I need to check out later than usual",
  "expected_skill_id": "late_checkout_request",
  "category": "checkout",
  "difficulty": "easy"
}
```

## Evaluation Results

Results are saved in JSON format:

- `evals/results/kb_rag_YYYYMMDD_HHMMSS.json`
- `evals/results/skill_matching_YYYYMMDD_HHMMSS.json`

Each result includes:
- Individual test case results
- Aggregate metrics
- Category breakdowns
- Timestamp and configuration

## Troubleshooting

### "Vector store not configured" Error

**Cause:** SkillRegistry not receiving embedding service and vector store

**Fix:** Ensure `initialize_skill_matcher()` creates and passes these components

### "Knowledge base: 0 documents found"

**Cause:** Indexing script didn't populate the collection

**Possible reasons:**
1. Wrong database path (check `--test` flag usage)
2. Wrong collection name (should be `hotel_knowledge`, not `hotel_skills`)
3. Indexing script failed silently

**Fix:** Check indexing script output for errors

### "Skills: 0 documents found"

**Cause:** Skills indexing failed

**Possible reasons:**
1. Wrong database path
2. Skills directory not found
3. No active skills in registry

**Fix:** Verify skills exist in `data/skills/` and are marked as active

### Model Access Errors

**Cause:** Using unavailable model (e.g., gpt-4o-mini)

**Fix:** Use `gpt-5.4-mini` or `gpt-5.4-nano` for all LLM calls

## Configuration

### LLM Models

- **RAG Evaluation (LLM-as-judge):** `gpt-5.4-mini`
- **Skill Matching:** Sentence transformers (local)
- **Knowledge Base Embeddings:** OpenAI `text-embedding-3-small`

### Thresholds

Defined in `run_system_evaluations.py`:

```python
RAG_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.80,
    "context_recall": 0.90
}

SKILL_MATCHING_THRESHOLDS = {
    "top_1_accuracy": 0.90,
    "top_3_accuracy": 0.95,
    "mrr": 0.85
}
```

## Adding New Test Cases

### RAG Test Cases

1. Edit `evals/test_cases/kb_rag_test_cases.json`
2. Add new test case with required fields:
   - `question`: User's question
   - `expected_answer`: Ground truth answer
   - `category`: Test category
   - `ground_truth_sources`: Expected source documents

### Skill Matching Test Cases

1. Edit `evals/test_cases/skill_matching_test_cases.json`
2. Add new test case with required fields:
   - `query`: User's request
   - `expected_skill_id`: Correct skill ID
   - `category`: Test category
   - `difficulty`: easy/medium/hard

## CI/CD Integration

To integrate with CI/CD:

```bash
# Run evaluations and check exit code
python evals/run_system_evaluations.py

# Exit code 0 = all tests passed
# Exit code 1 = some tests failed
```

## Performance Considerations

- **Indexing time:** ~10-30 seconds (depends on corpus size)
- **RAG evaluation:** ~2-5 minutes (depends on test case count and LLM latency)
- **Skill matching:** ~10-30 seconds (local embeddings, fast)
- **Total runtime:** ~5-10 minutes for full suite


## Evaluation FLow Diagram

````mermaid
flowchart TD
    Start[Run Evaluations] --> Init[Initialize System]
    Init --> Clear[Clear & Reindex Vector Stores]
    
    Clear --> LoadKB[Load KB Test Cases]
    Clear --> LoadSM[Load Skill Matching Test Cases]
    
    LoadKB --> KB[Knowledge Base RAG Evaluator]
    LoadSM --> SM[Skill Matching RAG Evaluator]
    
    KB --> KBLoop{For Each Test Case}
    KBLoop --> Query[Execute RAG Query]
    Query --> Retrieve[Get Retrieved Contexts]
    Retrieve --> Answer[Generate Answer]
    
    Answer --> Faith[LLM Judge: Faithfulness]
    Answer --> Rel[LLM Judge: Answer Relevancy]
    Retrieve --> Prec[Calculate Context Precision]
    Retrieve --> Rec[Calculate Context Recall]
    
    Faith --> KBMetrics[Aggregate KB Metrics]
    Rel --> KBMetrics
    Prec --> KBMetrics
    Rec --> KBMetrics
    
    KBMetrics --> KBPass{Pass Thresholds?}
    KBPass -->|Faithfulness ≥ 0.85| KBPass2
    KBPass2{Answer Relevancy ≥ 0.90} --> KBPass3
    KBPass3{Context Precision ≥ 0.80} --> KBPass4
    KBPass4{Context Recall ≥ 0.90} --> KBResult[KB Results]
    
    SM --> SMLoop{For Each Test Case}
    SMLoop --> Match[Execute Skill Matching]
    Match --> Rank[Get Ranked Skills]
    
    Rank --> Top1[Check Top-1 Match]
    Rank --> Top3[Check Top-3 Match]
    Rank --> MRR[Calculate MRR]
    
    Top1 --> SMMetrics[Aggregate SM Metrics]
    Top3 --> SMMetrics
    MRR --> SMMetrics
    
    SMMetrics --> SMPass{Pass Thresholds?}
    SMPass -->|Top-1 ≥ 0.90| SMPass2
    SMPass2{Top-3 ≥ 0.95} --> SMResult[SM Results]
    
    KBResult --> Save[Save Results to JSON]
    SMResult --> Save
    
    Save --> Report[Generate Markdown Report]
    
    Report --> Summary[Executive Summary]
    Report --> Details[Detailed Metrics]
    Report --> Cases[Individual Test Cases]
    Report --> Recs[Recommendations]
    
    Summary --> Output[evals/reports/evaluation_report_TIMESTAMP.md]
    Details --> Output
    Cases --> Output
    Recs --> Output
    
    Output --> End[Evaluation Complete]
    
    style Faith fill:#e1f5ff
    style Rel fill:#e1f5ff
    style Prec fill:#fff4e1
    style Rec fill:#fff4e1
    style KBPass fill:#d4edda
    style SMPass fill:#d4edda
    style Output fill:#f8d7da
````