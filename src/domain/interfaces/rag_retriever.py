"""RAG Retriever interface for knowledge base retrieval."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.context import RetrievedContext, RAGContext


class IRAGRetriever(ABC):
    """
    Interface for RAG (Retrieval-Augmented Generation) retrieval system.

    This interface defines the contract for retrieving relevant context from
    the knowledge base to ground agent responses and decisions.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Retrieve relevant contexts from knowledge base.

        Args:
            query: Search query string
            top_k: Number of top results to return
            filters: Optional metadata filters (e.g., doc_type, domain, date)

        Returns:
            List of retrieved context chunks with scores

        Raises:
            RetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGContext:
        """
        Retrieve and rerank contexts for better precision.

        Args:
            query: Search query string
            top_k_retrieval: Number of candidates to retrieve
            top_k_rerank: Number of top results after reranking
            filters: Optional metadata filters

        Returns:
            RAGContext with reranked results

        Raises:
            RetrievalError: If retrieval or reranking fails
        """
        pass

    @abstractmethod
    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: str,
    ) -> None:
        """
        Ingest a document into the knowledge base.

        Args:
            content: Document content
            metadata: Document metadata (doc_type, source, etc.)
            doc_id: Unique document identifier

        Raises:
            IngestionError: If document ingestion fails
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the knowledge base.

        Args:
            doc_id: Document identifier to delete

        Raises:
            DeletionError: If document deletion fails
        """
        pass

    @abstractmethod
    async def get_document_count(self) -> int:
        """
        Get total number of documents in knowledge base.

        Returns:
            Number of documents

        Raises:
            RetrievalError: If count retrieval fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if RAG system is healthy and operational.

        Returns:
            True if healthy, False otherwise
        """
        pass
