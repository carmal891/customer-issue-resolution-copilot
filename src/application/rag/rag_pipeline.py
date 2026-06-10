"""
RAG Pipeline - End-to-End Orchestration

Orchestrates the complete RAG workflow:
1. Document ingestion
2. Chunking
3. Embedding
4. Indexing
5. Retrieval
6. Reranking
7. Context assembly

This is the main entry point for RAG operations.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import json
from datetime import datetime

from src.application.rag.document_ingestion import (
    DocumentIngestionPipeline,
    Document,
    SourceType
)
from src.application.rag.chunking import SemanticChunker, Chunk, DocumentType
from src.infrastructure.embeddings.embedding_service import (
    IEmbeddingService,
    EmbeddingServiceFactory,
    EmbeddingProvider
)
from src.infrastructure.vector_store.chromadb_adapter import (
    ChromaDBAdapter,
    DistanceMetric
)
from src.application.rag.retrieval import (
    DenseRetriever,
    RetrievalConfig,
    RetrievalStrategy,
    RetrievalResult
)
from src.application.rag.reranking import (
    RerankingPipeline,
    RerankConfig,
    RerankStrategy,
    RerankResult
)
from src.application.rag.context_assembly import (
    ContextAssembler,
    PromptBuilder,
    AssemblyConfig,
    AssembledContext
)


logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Complete RAG pipeline configuration"""
    # Embedding config
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Vector store config
    vector_store_path: str = "./data/vector_store"
    collection_name: str = "hotel_knowledge"  # Knowledge base collection for policies
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    # Retrieval config
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    retrieval_top_k: int = 10
    retrieval_min_score: float = 0.0

    # Reranking config
    enable_reranking: bool = True
    rerank_strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5

    # Context assembly config
    max_context_tokens: int = 4000
    include_metadata: bool = True
    group_by_source: bool = False

    # Chunking config (optimized for semantic coherence)
    chunk_size: int = 800      # Larger chunks preserve complete semantic units
    chunk_overlap: int = 200   # More overlap preserves context between chunks


@dataclass
class RAGMetrics:
    """Metrics for RAG pipeline execution"""
    query: str
    total_time_ms: float
    ingestion_time_ms: float
    retrieval_time_ms: float
    rerank_time_ms: float
    assembly_time_ms: float
    num_documents_indexed: int
    num_chunks_created: int
    num_results_retrieved: int
    num_results_reranked: int
    final_context_tokens: int


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.

    Manages the full lifecycle:
    - Indexing: Ingest → Chunk → Embed → Store
    - Querying: Retrieve → Rerank → Assemble
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or RAGConfig()

        # Initialize components
        self._init_components()

        logger.info("Initialized RAGPipeline")

    def _init_components(self):
        """Initialize all pipeline components"""
        # Embedding service
        # Convert string provider to enum
        if self.config.embedding_provider.lower() in ["sentence-transformers", "sentence_transformer"]:
            provider_enum = EmbeddingProvider.SENTENCE_TRANSFORMER
            embedding_config = {
                'model_name': self.config.embedding_model,
                'device': 'cpu',
                'batch_size': 32
            }
        else:
            provider_enum = EmbeddingProvider.OPENAI
            embedding_config = {
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'model': self.config.embedding_model,
                'batch_size': 100
            }

        self.embedding_service = EmbeddingServiceFactory.create(
            provider=provider_enum,
            config=embedding_config
        )

        # Vector store
        self.vector_store = ChromaDBAdapter(
            persist_directory=self.config.vector_store_path,
            collection_name=self.config.collection_name,
            distance_metric=self.config.distance_metric
        )

        # Document ingestion
        self.ingestion_pipeline = DocumentIngestionPipeline()

        # Chunker
        self.chunker = SemanticChunker(
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.chunk_overlap
        )

        # Retriever
        retrieval_config = RetrievalConfig(
            strategy=self.config.retrieval_strategy,
            top_k=self.config.retrieval_top_k,
            min_score=self.config.retrieval_min_score,
            distance_metric=self.config.distance_metric
        )
        self.retriever = DenseRetriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            config=retrieval_config
        )

        # Reranker
        if self.config.enable_reranking:
            rerank_config = RerankConfig(
                strategy=self.config.rerank_strategy,
                model_name=self.config.rerank_model,
                top_k=self.config.rerank_top_k
            )
            self.reranker = RerankingPipeline(rerank_config)
        else:
            self.reranker = None

        # Context assembler
        assembly_config = AssemblyConfig(
            max_context_tokens=self.config.max_context_tokens,
            include_metadata=self.config.include_metadata,
            group_by_source=self.config.group_by_source
        )
        self.assembler = ContextAssembler(assembly_config)

        # Prompt builder
        self.prompt_builder = PromptBuilder(assembler=self.assembler)

        logger.info("All RAG components initialized")

    # ========== INDEXING METHODS ==========

    def index_documents(
        self,
        data_dir: Path,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Index all documents from data directory.

        Complete workflow:
        1. Ingest documents
        2. Chunk documents
        3. Generate embeddings
        4. Store in vector database

        Args:
            data_dir: Path to data directory
            clear_existing: Clear existing index

        Returns:
            Indexing statistics
        """
        start_time = datetime.now()

        logger.info(f"Starting document indexing from {data_dir}")

        if clear_existing:
            logger.info("Clearing existing index")
            self.vector_store.clear()

        # Step 1: Ingest documents
        logger.info("Step 1: Ingesting documents")
        documents = self._ingest_all_documents(data_dir)

        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents")
        all_chunks = self._chunk_documents(documents)

        # Step 3 & 4: Embed and index
        logger.info("Step 3-4: Embedding and indexing chunks")
        self._embed_and_index_chunks(all_chunks)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        stats = {
            "num_documents": len(documents),
            "num_chunks": len(all_chunks),
            "total_time_seconds": total_time,
            "chunks_per_second": len(all_chunks) / total_time if total_time > 0 else 0,
        }

        logger.info(
            f"Indexing complete: {stats['num_documents']} docs, "
            f"{stats['num_chunks']} chunks in {total_time:.2f}s"
        )

        return stats

    def _ingest_all_documents(self, data_dir: Path) -> List[Document]:
        """Ingest all documents from data directory"""
        # The ingestion pipeline already has base_path set, just call ingest_all()
        documents = self.ingestion_pipeline.ingest_all()
        logger.info(f"Ingested {len(documents)} total documents")
        return documents

    def _chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk all documents"""
        all_chunks = []

        for doc in documents:
            # Determine document type
            doc_type = self._map_source_to_doc_type(doc.source_type)

            # Chunk document with correct parameter names
            chunks = self.chunker.chunk_document(
                content=doc.content,
                document_id=doc.document_id,
                document_type=doc_type,
                metadata=doc.metadata
            )

            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        return all_chunks

    def _map_source_to_doc_type(self, source_type: SourceType) -> DocumentType:
        """Map source type to document type for chunking"""
        mapping = {
            SourceType.POLICY: DocumentType.POLICY,
            SourceType.BOOKING: DocumentType.TICKET,
            SourceType.ISSUE: DocumentType.CONVERSATION,
            SourceType.RESOLUTION: DocumentType.RESOLUTION,
            SourceType.RUNBOOK: DocumentType.RUNBOOK,
            SourceType.CONVERSATION: DocumentType.CONVERSATION,
        }
        return mapping.get(source_type, DocumentType.POLICY)

    def _embed_and_index_chunks(self, chunks: List[Chunk]):
        """Generate embeddings and index chunks"""
        # Filter out empty chunks
        valid_chunks = [chunk for chunk in chunks if chunk.content and chunk.content.strip()]

        if not valid_chunks:
            logger.warning("No valid chunks to index after filtering empty content")
            return

        if len(valid_chunks) < len(chunks):
            logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} empty chunks")

        # Sanitize metadata to comply with ChromaDB constraints (only str, int, float, bool)
        # ChromaDB does not accept lists or nested dictionaries in metadata
        sanitized_chunks = []
        for chunk in valid_chunks:
            # Check if metadata exists
            if chunk.metadata is None:
                sanitized_metadata = {}
            else:
                sanitized_metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (list, dict)):
                        # Convert complex types to JSON strings
                        sanitized_metadata[key] = json.dumps(value)
                    elif value is None:
                        sanitized_metadata[key] = ""
                    elif isinstance(value, bool):
                        sanitized_metadata[key] = value
                    elif isinstance(value, (int, float)):
                        sanitized_metadata[key] = value
                    else:
                        sanitized_metadata[key] = str(value)

            # Create a new chunk with sanitized metadata using correct constructor
            sanitized_chunk = Chunk(
                content=chunk.content,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_type=chunk.document_type,
                section_title=chunk.section_title,
                section_number=chunk.section_number,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=sanitized_metadata
            )
            sanitized_chunks.append(sanitized_chunk)

        # Prepare batch data
        texts = [chunk.content for chunk in sanitized_chunks]
        ids = [chunk.chunk_id for chunk in sanitized_chunks]

        # Generate embeddings in batch
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embedding_result = self.embedding_service.embed_texts(texts)
        embeddings = embedding_result.embeddings

        # Index in vector store
        logger.info(f"Indexing {len(embeddings)} chunks")
        self.vector_store.batch_add(
            chunks=sanitized_chunks,
            embeddings=embeddings
        )

        # Initialize BM25 index for hybrid search
        logger.info("Initializing BM25 index for hybrid search...")
        bm25_documents = [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in sanitized_chunks
        ]
        self.retriever.initialize_bm25(bm25_documents)

        logger.info("Indexing complete (vector + BM25)")

    # ========== QUERYING METHODS ==========

    def query(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], RAGMetrics]:
        """
        Query the RAG system (retrieve only, no reranking).

        Args:
            query: User query
            metadata_filters: Optional metadata filters
            top_k: Number of results

        Returns:
            Tuple of (results, metrics)
        """
        start_time = datetime.now()

        # Retrieve
        retrieval_start = datetime.now()
        results, retrieval_metrics = self.retriever.retrieve(
            query=query,
            metadata_filters=metadata_filters,
            top_k=top_k
        )
        retrieval_time = (datetime.now() - retrieval_start).total_seconds() * 1000

        # Build metrics
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        metrics = RAGMetrics(
            query=query,
            total_time_ms=total_time,
            ingestion_time_ms=0,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=0,
            assembly_time_ms=0,
            num_documents_indexed=self.vector_store.count(),
            num_chunks_created=0,
            num_results_retrieved=len(results),
            num_results_reranked=0,
            final_context_tokens=0
        )

        return results, metrics

    def query_with_reranking(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Tuple[List[RerankResult], RAGMetrics]:
        """
        Query with reranking for better precision.

        Args:
            query: User query
            metadata_filters: Optional metadata filters
            top_k: Number of final results

        Returns:
            Tuple of (reranked results, metrics)
        """
        start_time = datetime.now()

        # Retrieve (get more candidates for reranking)
        retrieval_start = datetime.now()
        retrieval_results, retrieval_metrics = self.retriever.retrieve(
            query=query,
            metadata_filters=metadata_filters,
            top_k=(top_k or self.config.rerank_top_k) * 2  # Get 2x for reranking
        )
        retrieval_time = (datetime.now() - retrieval_start).total_seconds() * 1000

        # Rerank
        rerank_start = datetime.now()
        if self.reranker and retrieval_results:
            reranked_results, rerank_metrics = self.reranker.rerank_with_metrics(
                query=query,
                results=retrieval_results,
                top_k=top_k
            )
            rerank_time = rerank_metrics["rerank_time_ms"]
        else:
            # No reranking - convert to RerankResult
            from src.application.rag.reranking import RerankResult
            reranked_results = [
                RerankResult(
                    retrieval_result=r,
                    rerank_score=r.score,
                    original_score=r.score,
                    rank=i + 1
                )
                for i, r in enumerate(retrieval_results[:top_k or self.config.rerank_top_k])
            ]
            rerank_time = 0

        # Build metrics
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        metrics = RAGMetrics(
            query=query,
            total_time_ms=total_time,
            ingestion_time_ms=0,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            assembly_time_ms=0,
            num_documents_indexed=self.vector_store.count(),
            num_chunks_created=0,
            num_results_retrieved=len(retrieval_results),
            num_results_reranked=len(reranked_results),
            final_context_tokens=0
        )

        return reranked_results, metrics

    def query_with_context(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Tuple[AssembledContext, RAGMetrics]:
        """
        Complete RAG query with context assembly.

        Args:
            query: User query
            metadata_filters: Optional metadata filters
            top_k: Number of results to use

        Returns:
            Tuple of (assembled context, metrics)
        """
        start_time = datetime.now()

        # Query with reranking
        reranked_results, partial_metrics = self.query_with_reranking(
            query=query,
            metadata_filters=metadata_filters,
            top_k=top_k
        )

        # Assemble context
        assembly_start = datetime.now()
        context = self.assembler.assemble_from_reranked(reranked_results, query)
        assembly_time = (datetime.now() - assembly_start).total_seconds() * 1000

        # Update metrics
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        metrics = RAGMetrics(
            query=query,
            total_time_ms=total_time,
            ingestion_time_ms=0,
            retrieval_time_ms=partial_metrics.retrieval_time_ms,
            rerank_time_ms=partial_metrics.rerank_time_ms,
            assembly_time_ms=assembly_time,
            num_documents_indexed=partial_metrics.num_documents_indexed,
            num_chunks_created=0,
            num_results_retrieved=partial_metrics.num_results_retrieved,
            num_results_reranked=partial_metrics.num_results_reranked,
            final_context_tokens=context.token_count
        )

        logger.info(
            f"RAG query complete: {metrics.num_results_reranked} results, "
            f"{metrics.final_context_tokens} tokens, {metrics.total_time_ms:.2f}ms"
        )

        return context, metrics

    def build_prompt(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        additional_instructions: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Tuple[str, AssembledContext, RAGMetrics]:
        """
        Build complete prompt with RAG context.

        This is the main method for agent integration.

        Args:
            query: User query/issue
            metadata_filters: Optional metadata filters
            additional_instructions: Optional additional instructions
            top_k: Number of results to use

        Returns:
            Tuple of (prompt, context, metrics)
        """
        # Get context
        context, metrics = self.query_with_context(
            query=query,
            metadata_filters=metadata_filters,
            top_k=top_k
        )

        # Build prompt
        prompt = self.prompt_builder.template.format(
            context=context.context_text,
            query=query,
            additional_instructions=additional_instructions
        )

        logger.info(f"Built complete prompt: {len(prompt)} characters")

        return prompt, context, metrics

    # ========== UTILITY METHODS ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "num_indexed_chunks": self.vector_store.count(),
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "retrieval_strategy": self.config.retrieval_strategy.value,
            "reranking_enabled": self.config.enable_reranking,
            "rerank_strategy": self.config.rerank_strategy.value if self.config.enable_reranking else None,
        }

    def clear_index(self):
        """Clear the vector store index"""
        logger.warning("Clearing vector store index")
        self.vector_store.clear()
