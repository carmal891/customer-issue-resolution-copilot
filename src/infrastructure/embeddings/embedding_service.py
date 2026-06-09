"""
Embedding service for RAG system.

Supports multiple embedding providers:
- OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
- Sentence Transformers (local models)

Key Features:
- Provider abstraction
- Batch processing
- Caching support
- Dimension normalization
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from enum import Enum


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    token_count: Optional[int] = None


class IEmbeddingService(ABC):
    """Abstract interface for embedding services."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass


class OpenAIEmbeddingService(IEmbeddingService):
    """
    OpenAI embedding service.
    
    Uses OpenAI's embedding API for high-quality embeddings.
    Recommended models:
    - text-embedding-3-small (1536 dims, fast, cost-effective)
    - text-embedding-3-large (3072 dims, highest quality)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",  # Using small model for compatibility
        batch_size: int = 100,
        timeout: float = 120.0  # 2 minutes timeout for large batches
    ):
        """
        Initialize OpenAI embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small for compatibility)
            batch_size: Batch size for API calls
            timeout: Request timeout in seconds (default: 120)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        
        # Log embedding model configuration
logger.info(f" Initializing OpenAI Embedding Service")
        logger.info(f"   Model: {model}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Timeout: {timeout}s")
        
        # Import OpenAI client
        try:
            from openai import OpenAI
            import httpx
            # Create client with custom timeout
            # Increase connect timeout to 60s for slow networks
            self.client = OpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(timeout, connect=60.0),
                max_retries=3
            )
logger.info(f" OpenAI client initialized successfully")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Set dimensions based on model
        # text-embedding-3-small: 1536 dims
        # text-embedding-3-large: 3072 dims (better quality, higher cost)
        self.dimensions = 1536 if "small" in model else 3072
        logger.info(f"   Embedding dimensions: {self.dimensions}")
        
        # Test connectivity
        self._test_connectivity(logger)
    
    def _test_connectivity(self, logger):
        """Test OpenAI API connectivity with a simple request."""
        try:
logger.info(f" Testing OpenAI API connectivity...")
            # Try a minimal embedding request
            test_response = self.client.embeddings.create(
                model=self.model,
                input=["test"]
            )
logger.info(f" API connectivity test successful")
            logger.info(f"   Response model: {test_response.model}")
            logger.info(f"   Embedding dimension: {len(test_response.data[0].embedding)}")
        except Exception as e:
logger.error(f" API connectivity test failed: {str(e)}")
            logger.error(f"   This may indicate network issues or invalid API key")
            logger.error(f"   The system will continue but may fail during embedding generation")
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts using OpenAI API.
        
        Processes in batches to respect API limits.
        """
        all_embeddings = []
        total_tokens = 0
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Track token usage
            total_tokens += response.usage.total_tokens
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model,
            dimensions=self.dimensions,
            token_count=total_tokens
        )
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return response.data[0].embedding
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions


class SentenceTransformerEmbeddingService(IEmbeddingService):
    """
    Sentence Transformer embedding service.
    
    Uses local sentence-transformers models for embeddings.
    Recommended models:
    - all-MiniLM-L6-v2 (384 dims, fast, good quality)
    - all-mpnet-base-v2 (768 dims, higher quality)
    - multi-qa-mpnet-base-dot-v1 (768 dims, optimized for Q&A)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize Sentence Transformer service.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Import and load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.dimensions = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.
        
        Uses sentence-transformers encode method with batching.
        """
        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResult(
            embeddings=embeddings_list,
            model=self.model_name,
            dimensions=self.dimensions,
            token_count=None  # Not applicable for local models
        )
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding[0].tolist()
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions


class EmbeddingServiceFactory:
    """
    Factory for creating embedding services.
    
    Simplifies service instantiation based on configuration.
    """
    
    @staticmethod
    def create(
        provider: EmbeddingProvider,
        config: Dict[str, Any]
    ) -> IEmbeddingService:
        """
        Create an embedding service based on provider and config.
        
        Args:
            provider: Embedding provider enum
            config: Configuration dictionary
            
        Returns:
            Embedding service instance
            
        Example config for OpenAI:
            {
                'api_key': 'sk-...',
                'model': 'text-embedding-3-small',
                'batch_size': 100
            }
            
        Example config for Sentence Transformer:
            {
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'batch_size': 32
            }
        """
        if provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingService(
                api_key=config['api_key'],
                model=config.get('model', 'text-embedding-3-small'),
                batch_size=config.get('batch_size', 100)
            )
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            return SentenceTransformerEmbeddingService(
                model_name=config.get('model_name', 'all-MiniLM-L6-v2'),
                device=config.get('device', 'cpu'),
                batch_size=config.get('batch_size', 32)
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class CachedEmbeddingService(IEmbeddingService):
    """
    Wrapper that adds caching to any embedding service.
    
    Caches embeddings to avoid redundant API calls or computations.
    Useful for development and testing.
    """
    
    def __init__(self, base_service: IEmbeddingService):
        """
        Initialize cached embedding service.
        
        Args:
            base_service: Underlying embedding service
        """
        self.base_service = base_service
        self.cache: Dict[str, List[float]] = {}
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings with caching.
        
        Checks cache first, only generates embeddings for cache misses.
        """
        # Separate cached and uncached texts
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append((i, self.cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            result = self.base_service.embed_texts(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, result.embeddings):
                self.cache[text] = embedding
            
            # Merge cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding
            for i, embedding in zip(uncached_indices, result.embeddings):
                all_embeddings[i] = embedding
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                model=result.model,
                dimensions=result.dimensions,
                token_count=result.token_count
            )
        else:
            # All texts were cached
            return EmbeddingResult(
                embeddings=[emb for _, emb in cached_embeddings],
                model=self.base_service.model if hasattr(self.base_service, 'model') else "cached",
                dimensions=self.base_service.get_dimensions(),
                token_count=0
            )
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query with caching."""
        if query not in self.cache:
            self.cache[query] = self.base_service.embed_query(query)
        return self.cache[query]
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.base_service.get_dimensions()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self.cache)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))
