# Customer Issue Resolution Copilot - Evaluation Report

**Date:** June 08, 2026  
**Evaluation Run:** 20260608_204057  
**System Version:** v1.0 (POC)

---

## Executive Summary

This report evaluates the **Customer Issue Resolution Copilot** system across two critical capabilities:

1. **Knowledge Base RAG** - How well the system retrieves and uses hotel policies to answer questions
2. **Skill Matching** - How accurately the system matches customer issues to pre-defined resolution workflows

### Overall Results

| Component | Status | Pass Rate | Passed | Total | Target |
|-----------|--------|-----------|--------|-------|--------|
| **Knowledge Base RAG** | ⚠️ **NEEDS IMPROVEMENT** | 60.0% | 3/5 | 85%+ |
| **Skill Matching** | ⚠️ **NEEDS IMPROVEMENT** | 70.0% | 7/10 | 90%+ |

---

## 1. Knowledge Base RAG Evaluation

### What We're Testing

When a customer asks a question (e.g., "What's your cancellation policy?"), the system must:
- Find the right policy documents
- Extract relevant information  
- Generate an accurate, grounded answer

### Metrics

| Metric | Description | Target | Actual | Status |
|--------|-------------|--------|--------|--------|
| **Faithfulness** | Answer is based on retrieved documents (no hallucination) | ≥85% | 98.0% | ✅ |
| **Answer Relevancy** | Answer directly addresses the question | ≥90% | 89.2% | ❌ |
| **Context Precision** | Retrieved documents are relevant | ≥80% | 100.0% | ✅ |
| **Context Recall** | All relevant documents were retrieved | ≥90% | 90.0% | ✅ |

### Results by Category


#### Policy Questions (3 tests)

- **Pass Rate:** 2/3 (67%)
- **Avg Faithfulness:** 96.7%
- **Avg Answer Relevancy:** 92.0%
- **Avg Context Precision:** 100.0%
- **Avg Context Recall:** 83.3%

#### Procedure Questions (2 tests)

- **Pass Rate:** 1/2 (50%)
- **Avg Faithfulness:** 100.0%
- **Avg Answer Relevancy:** 85.0%
- **Avg Context Precision:** 100.0%
- **Avg Context Recall:** 100.0%

---

## 2. Skill Matching Evaluation

### What We're Testing

When a customer issue comes in (e.g., "I need to checkout late"), the system must:
- Understand the intent
- Match it to the right pre-built workflow (skill)
- Propose the correct resolution steps

### Metrics

| Metric | Description | Target | Actual | Status |
|--------|-------------|--------|--------|--------|
| **Top-1 Accuracy** | Correct skill is the #1 match | ≥90% | 70.0% | ❌ |
| **Top-3 Accuracy** | Correct skill is in top 3 matches | ≥95% | 80.0% | ❌ |
| **Mean Reciprocal Rank** | Average position of correct skill | Higher is better | 0.75 | - |

### Error Analysis

- **False Negative Rate:** 30.0% (missed correct skills)
- **False Positive Rate:** 0.0% (wrong skills matched)
- **Avg Confidence (Correct):** 0.30
- **Avg Confidence (Incorrect):** 0.30

### Results by Category


#### Booking Issues (6 tests) ⚠️

- **Pass Rate:** 5/6 (83%)
- **Top-1 Accuracy:** 83.3%
- **Top-3 Accuracy:** 100.0%
- **Avg Confidence:** 0.30

#### Billing Issues (2 tests) ❌

- **Pass Rate:** 0/2 (0%)
- **Top-1 Accuracy:** 0.0%
- **Top-3 Accuracy:** 0.0%
- **Avg Confidence:** 0.30

#### Amenity Issues (2 tests) ✅

- **Pass Rate:** 2/2 (100%)
- **Top-1 Accuracy:** 100.0%
- **Top-3 Accuracy:** 100.0%
- **Avg Confidence:** 0.30

---

## 3. Key Findings

### ✅ What's Working Well

- **Amenity Skill Matching:** 100% accuracy (2/2 tests passed)
- **RAG Faithfulness:** 98.0% (no hallucinations)

### ⚠️ What Needs Improvement

- **Knowledge Base RAG:** Only 60% pass rate (target: 85%+)
  - Answer Relevancy: 89.2% (answers not addressing questions)
- **Booking Skill Matching:** Only 83% accuracy (target: 90%+)
- **Billing Skill Matching:** Only 0% accuracy (target: 90%+)
- **Confidence Calibration:** Low confidence (0.30) even when correct

---

## 4. Recommendations

### Priority Actions


**🟡 HIGH: Improve Knowledge Base RAG**
- Current pass rate: 60% (target: 85%+)
- Actions:
  1. Review and optimize chunking strategy
  2. Tune reranking parameters
  3. Improve prompt engineering
- Timeline: 1-2 weeks


**🟡 HIGH: Fix Billing Skill Matching**
- Current accuracy: 0% (target: 90%+)
- Actions:
  1. Expand skill trigger variations
  2. Add domain-specific terminology
  3. Re-index skills with new triggers
- Timeline: 1-2 days


**🟢 MEDIUM: Tune Confidence Scoring**
- Current confidence: 0.30 (low even when correct)
- Actions:
  1. Review embedding similarity thresholds
  2. Add metadata-based confidence boosting
  3. Calibrate confidence scores to accuracy
- Timeline: 2-3 days

---

## 5. Technical Details

### System Configuration

- **LLM Model:** gpt-5.4-mini
- **Embedding Model:** text-embedding-3-small (OpenAI)
- **Vector Store:** ChromaDB (local)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2

### Test Coverage

- **KB RAG Tests:** 5 test cases across 2 categories
- **Skill Matching Tests:** 10 test cases across 3 categories

---

## Appendix: Understanding the Metrics

### RAG Metrics

- **Faithfulness:** Measures if the answer is grounded in retrieved documents (prevents hallucination)
- **Answer Relevancy:** Measures if the answer addresses the user's question
- **Context Precision:** Measures if retrieved documents are relevant to the question
- **Context Recall:** Measures if all relevant documents were retrieved

### Skill Matching Metrics

- **Top-1 Accuracy:** Percentage of times the correct skill is ranked #1
- **Top-3 Accuracy:** Percentage of times the correct skill is in top 3
- **Mean Reciprocal Rank (MRR):** Average of 1/rank for correct skill (1.0 is perfect)
- **Confidence Calibration:** How well confidence scores match actual accuracy

---

**Report Generated:** 2026-06-08 20:40:57  
**Evaluation Framework Version:** 1.0  
**Results Files:**
- `evals/results/kb_rag_20260608_204057.json`
- `evals/results/skill_matching_20260608_204057.json`
