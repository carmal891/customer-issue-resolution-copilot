"""Context-related domain models for RAG system."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrievedContext(BaseModel):
    """Individual piece of retrieved context from knowledge base."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Retrieved text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata (doc_id, section, doc_type, source, etc.)",
    )
    source: str = Field(..., description="Source document or file")
    doc_type: str = Field(..., description="Type of document (policy, ticket, slack, etc.)")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "chunk_id": "POL-CANCEL-001_S2_SS1",
                "content": "Cancellations made 7+ days before check-in: 100% refund. No cancellation fee applied.",
                "score": 0.92,
                "metadata": {
                    "document_id": "POL-CANCEL-001",
                    "section": "2. STANDARD CANCELLATION TERMS",
                    "subsection": "2.1 Cancellation Windows",
                    "version": "3.2",
                    "last_updated": "2024-01-15",
                },
                "source": "cancellation_policy.txt",
                "doc_type": "policy",
            }
        }

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if retrieval confidence is high."""
        return self.score >= threshold

    def is_policy_document(self) -> bool:
        """Check if this is from a policy document."""
        return self.doc_type == "policy"


class RAGContext(BaseModel):
    """Aggregated RAG context for an issue."""

    query: str = Field(..., description="Original query used for retrieval")
    retrieved_contexts: List[RetrievedContext] = Field(
        ..., description="List of retrieved context chunks"
    )
    reranked: bool = Field(default=False, description="Whether contexts were reranked")
    total_retrieved: int = Field(..., ge=0, description="Total number of chunks retrieved")
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average relevance score")
    min_score: float = Field(..., ge=0.0, le=1.0, description="Minimum relevance score")
    max_score: float = Field(..., ge=0.0, le=1.0, description="Maximum relevance score")
    retrieval_time_ms: Optional[float] = Field(
        None, ge=0, description="Time taken for retrieval in milliseconds"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "query": "What is the cancellation policy for bookings made 7 days in advance?",
                "retrieved_contexts": [
                    {
                        "chunk_id": "POL-CANCEL-001_S2_SS1",
                        "content": "Cancellations made 7+ days before check-in: 100% refund",
                        "score": 0.92,
                        "metadata": {"document_id": "POL-CANCEL-001"},
                        "source": "cancellation_policy.txt",
                        "doc_type": "policy",
                    }
                ],
                "reranked": True,
                "total_retrieved": 5,
                "avg_score": 0.85,
                "min_score": 0.72,
                "max_score": 0.92,
                "retrieval_time_ms": 145.3,
            }
        }

    @classmethod
    def from_retrieved_contexts(
        cls, query: str, contexts: List[RetrievedContext], reranked: bool = False
    ) -> "RAGContext":
        """Create RAGContext from list of retrieved contexts."""
        if not contexts:
            return cls(
                query=query,
                retrieved_contexts=[],
                reranked=reranked,
                total_retrieved=0,
                avg_score=0.0,
                min_score=0.0,
                max_score=0.0,
            )

        scores = [ctx.score for ctx in contexts]
        return cls(
            query=query,
            retrieved_contexts=contexts,
            reranked=reranked,
            total_retrieved=len(contexts),
            avg_score=sum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
        )

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if overall retrieval confidence is high."""
        return self.avg_score >= threshold and self.max_score >= 0.8

    def has_conflicting_sources(self) -> bool:
        """Check if retrieved contexts might be conflicting."""
        # Simple heuristic: if we have contexts from different doc types with similar scores
        if len(self.retrieved_contexts) < 2:
            return False

        doc_types = set(ctx.doc_type for ctx in self.retrieved_contexts[:3])
        score_variance = max(ctx.score for ctx in self.retrieved_contexts[:3]) - min(
            ctx.score for ctx in self.retrieved_contexts[:3]
        )

        return len(doc_types) > 1 and score_variance < 0.1

    def get_top_k(self, k: int = 5) -> List[RetrievedContext]:
        """Get top k contexts by score."""
        return sorted(self.retrieved_contexts, key=lambda x: x.score, reverse=True)[:k]

    def get_policy_contexts(self) -> List[RetrievedContext]:
        """Get only policy document contexts."""
        return [ctx for ctx in self.retrieved_contexts if ctx.is_policy_document()]

    def format_for_prompt(self, max_contexts: int = 5) -> str:
        """Format contexts for inclusion in LLM prompt."""
        top_contexts = self.get_top_k(max_contexts)
        formatted = []

        for i, ctx in enumerate(top_contexts, 1):
            formatted.append(
                f"[Context {i}] (Source: {ctx.source}, Score: {ctx.score:.2f})\n{ctx.content}"
            )

        return "\n\n".join(formatted)
