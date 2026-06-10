# TPAO Loop - RAG Retrieval Flow

This document explains how the TPAO (Think-Plan-Act-Observe) loop integrates with the RAG (Retrieval-Augmented Generation) pipeline for intelligent query processing and context retrieval.

## Overview

The TPAO loop is our implementation of the ReAct reasoning pattern for handling novel customer service tasks. During the **Think Phase**, it uses an advanced RAG pipeline with hybrid search to retrieve relevant hotel policies and procedures.

## Complete RAG Retrieval Flow

```
User Query: "I want to book a cab for tomorrow evening"
    ↓
[TPAO Loop - Think Phase] Query Reformulation
    ↓
LLM Rephrasing: "Guest needs transportation service to arrange a cab for tomorrow evening"
    ↓
[RAG Pipeline] query_with_reranking()
    ↓
[Dense Retriever] _hybrid_retrieve()
    ├─→ Vector Search (OpenAI embeddings) → 10 results
    ├─→ BM25 Search (keyword matching) → 10 results
    └─→ RRF Fusion (70% vector + 30% BM25) → 10 combined results
    ↓
[Cross-Encoder Reranker] ms-marco-MiniLM-L-6-v2
    ↓
Final 5 results with high relevance scores
    ↓
[TPAO Loop] Context stored with citations
```

## Key Components

### 1. Query Reformulation (TPAO Loop)

**Location**: `tpao_loop.py` - `_think_phase()` method (lines 216-238)

**Purpose**: Transform user queries into clear, focused search queries that preserve context and intent.

**Process**:
```python
# Original query
"I want to book a cab for tomorrow evening trip"

# LLM reformulation prompt
"Rephrase this customer issue in 1-2 clear, concise sentences..."

# Rephrased query
"Guest needs transportation service to arrange a cab for tomorrow evening."
```

**Why This Matters**:
- Prevents keyword confusion (e.g., "book a cab" ≠ "hotel booking")
- Preserves semantic meaning and context
- Improves retrieval precision

**Logging**:
```
[Think] Query reformulation prompt: Rephrase this customer issue...
[Think] LLM rephrased query: Guest needs transportation service...
[Think] Final query with subject: Action: AI-generated resolution. Guest needs...
```

### 2. RAG Pipeline Entry Point

**Location**: `rag_pipeline.py` - `query_with_reranking()` method

**Purpose**: Orchestrate the complete retrieval and reranking process.

**Process**:
```python
reranked_results, metrics = rag_pipeline.query_with_reranking(
    query=rephrased_query,
    metadata_filters={"doc_type": "policy"},
    top_k=5
)
```

**What It Does**:
1. Calls the retriever to get initial candidates (2x the final count)
2. Applies cross-encoder reranking for precision
3. Returns top-k results with metrics

### 3. Hybrid Retrieval Strategy

**Location**: `retrieval.py` - `_hybrid_retrieve()` method (lines 269-339)

**Purpose**: Combine semantic understanding (vector search) with exact keyword matching (BM25) for optimal retrieval.

**Process**:

#### Step 1: Dense Retrieval (Vector Similarity)
```python
# Generate query embedding using OpenAI
query_embedding = embedding_service.embed_texts([query])

# Search vector store
dense_results = vector_store.search(
    query_embedding=query_embedding,
    n_results=top_k * 2,  # Get 20 candidates
    where=metadata_filters
)
```

**Technology**: OpenAI `text-embedding-3-small` (1536 dimensions)

#### Step 2: Sparse Retrieval (BM25 Keyword Matching)
```python
# BM25 search for exact keyword matches
sparse_results = bm25_retriever.search(
    query=query,
    top_k=top_k * 2,  # Get 20 candidates
    metadata_filters=metadata_filters
)
```

**Technology**: BM25 algorithm with TF-IDF scoring (k1=1.5, b=0.75)

#### Step 3: Reciprocal Rank Fusion (RRF)
```python
# Combine both result sets intelligently
fused_results = reciprocal_rank_fusion(
    dense_results=dense_results,
    sparse_results=sparse_results,
    k=60,  # RRF constant
    weights=(0.7, 0.3)  # 70% dense, 30% sparse
)
```

**Formula**: 
```
RRF_score = 0.7 * (1 / (k + rank_dense)) + 0.3 * (1 / (k + rank_sparse))
```

**Why This Works**:
- Vector search captures semantic similarity ("cab" ≈ "transportation")
- BM25 captures exact keyword matches ("cancellation policy")
- RRF combines both without score normalization issues

**Logging**:
```
[Retrieval] Using HYBRID retrieval strategy (vector + BM25 + RRF)
Hybrid search: 10 dense + 10 sparse results
Retrieved 10 results in 45.23ms, avg_score=0.856
```

### 4. Cross-Encoder Reranking

**Location**: `reranking.py` - `CrossEncoderReranker`

**Purpose**: Final precision boost by scoring query-document pairs directly.

**Process**:
```python
# Rerank the fused results
reranked_results = reranker.rerank_with_metrics(
    query=query,
    results=retrieval_results,
    top_k=5  # Final count
)
```

**Technology**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Why This Matters**:
- Cross-encoders see both query and document together
- More accurate than separate embeddings
- Final quality gate before returning results

### 5. Context Assembly with Citations

**Location**: `tpao_loop.py` - `_think_phase()` method (lines 318-362)

**Purpose**: Format retrieved results with proper citations for transparency.

**Process**:
```python
for i, result in enumerate(reranked_results[:3], 1):
    citation = {
        "source_id": i,
        "source_name": result.metadata.get('source'),
        "doc_type": result.metadata.get('doc_type'),
        "relevance_score": round(result.rerank_score, 3),
        "content_preview": result.content[:200]
    }
    citations.append(citation)
```

**Output Example**:
```json
{
  "citations": [
    {
      "source_id": 1,
      "source_name": "late_checkout_policy.md",
      "doc_type": "policy",
      "relevance_score": 0.923,
      "content_preview": "Late checkout requests are subject to availability..."
    }
  ]
}
```

## Performance Metrics

### Evaluation Results (RAGAS)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Faithfulness** | ≥85% | **99.3%** | ✅ Excellent |
| **Answer Relevancy** | ≥90% | **93.2%** | ✅ Pass |
| **Context Precision** | ≥80% | **86.7%** | ✅ Pass |
| **Context Recall** | ≥90% | **91.5%** | ✅ Pass |

### Timing Breakdown

```
Query Reformulation:  ~200ms  (LLM call)
Dense Retrieval:      ~15ms   (Vector search)
Sparse Retrieval:     ~8ms    (BM25 search)
RRF Fusion:           ~2ms    (Computation)
Cross-Encoder:        ~20ms   (Reranking)
─────────────────────────────────────────
Total:                ~245ms  (End-to-end)
```

## Guardrails

### 1. No Policy Match Detection
```python
if len(reranked_results) == 0:
    logger.warning("No relevant policies found - using LLM general knowledge")
    # Add warning to resolution
    state.resolution.metadata["no_policy_match"] = True
    state.needs_approval = True  # Force human review
```

### 2. Low Confidence Warning
```python
avg_score = sum(r.rerank_score for r in reranked_results) / len(reranked_results)
if avg_score < 0.3:
    logger.warning(f"Low confidence retrieval (avg score: {avg_score:.3f})")
    state.resolution.metadata["low_confidence_warning"] = True
```

## Configuration

### RAG Pipeline Config
```python
RAGConfig(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    vector_store_path="./data/vector_store",
    collection_name="hotel_knowledge",
    enable_reranking=True,
    retrieval_top_k=10,  # Initial retrieval
    rerank_top_k=5       # Final results
)
```

### Retrieval Config
```python
RetrievalConfig(
    strategy=RetrievalStrategy.HYBRID,
    top_k=10,
    min_score=0.0,
    distance_metric=DistanceMetric.COSINE
)
```

### BM25 Config
```python
BM25Retriever(
    k1=1.5,  # Term frequency saturation
    b=0.75   # Length normalization
)
```

## Debugging Tips

### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Retrieval Strategy
Look for this log line:
```
[Retrieval] Using HYBRID retrieval strategy (vector + BM25 + RRF)
```

### Inspect Query Reformulation
```
[Think] Query reformulation prompt: ...
[Think] LLM rephrased query: ...
```

### Verify Results Quality
```
Retrieved 5 results in 45.23ms, avg_score=0.856
```

## Common Issues and Solutions

### Issue: Irrelevant Results
**Symptom**: Retrieved policies don't match the query intent

**Solution**: Check query reformulation
```
# Bad: "booking cab tomorrow"
# Good: "Guest needs transportation service to arrange a cab"
```

### Issue: Low Confidence Scores
**Symptom**: `avg_score < 0.3`

**Solution**: 
1. Check if query is too vague
2. Verify knowledge base has relevant content
3. Consider expanding indexed documents

### Issue: Slow Retrieval
**Symptom**: Retrieval takes >500ms

**Solution**:
1. Reduce `retrieval_top_k` (currently 10)
2. Disable reranking for faster results (lower quality)
3. Check vector store index health

## Related Documentation

- **RAG System**: `docs/rag-system-complete.md`
- **Chunking Strategy**: `docs/chunking-strategy.md`
- **Evaluation Framework**: `evals/README.md`
- **System Design**: `docs/system-design-document.md`

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | OpenAI text-embedding-3-small | Dense vector representations |
| **Vector Store** | ChromaDB | Similarity search |
| **Keyword Search** | BM25 (rank-bm25) | Exact keyword matching |
| **Fusion** | Reciprocal Rank Fusion | Combine dense + sparse |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Final precision boost |
| **LLM** | GPT-5.4-mini | Query reformulation & reasoning |

---

**Last Updated**: 2026-06-10  
**Maintained By**: AI Engineering Team  
**Version**: 1.0