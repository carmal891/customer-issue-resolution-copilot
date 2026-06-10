"""
BM25 Sparse Retriever for Keyword-Based Search

Implements BM25 (Best Matching 25) algorithm for keyword-based retrieval.
Used in hybrid search to complement dense vector retrieval.

BM25 is particularly effective for:
- Exact keyword matches
- Technical terms and proper nouns
- Policy numbers and specific identifiers
- Questions with unique terminology

Combined with dense retrieval via Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import math
from collections import Counter, defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 search"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int = 0


class BM25Retriever:
    """
    BM25 (Best Matching 25) retriever for keyword-based search.
    
    BM25 is a probabilistic ranking function that scores documents
    based on term frequency (TF) and inverse document frequency (IDF).
    
    Parameters:
    - k1: Controls term frequency saturation (default: 1.5)
    - b: Controls document length normalization (default: 0.75)
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0.75 typical)
            epsilon: Floor value for IDF (prevents negative scores)
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Index structures
        self.corpus: List[Dict[str, Any]] = []  # List of {chunk_id, content, metadata, tokens}
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # Term -> # docs containing term
        self.idf: Dict[str, float] = {}  # Term -> IDF score
        self.doc_len: List[int] = []  # Document lengths
        self.avgdl: float = 0.0  # Average document length
        
        logger.info(f"Initialized BM25Retriever (k1={k1}, b={b})")
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of dicts with keys: chunk_id, content, metadata
        """
        logger.info(f"Indexing {len(documents)} documents for BM25...")
        
        self.corpus = []
        self.doc_freqs = defaultdict(int)
        self.doc_len = []
        
        # Tokenize and index each document
        for doc in documents:
            tokens = self._tokenize(doc['content'])
            self.corpus.append({
                'chunk_id': doc['chunk_id'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'tokens': tokens
            })
            
            # Track document length
            self.doc_len.append(len(tokens))
            
            # Track document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate IDF scores
        self._calculate_idf()
        
        logger.info(
            f"BM25 index built: {len(self.corpus)} docs, "
            f"{len(self.doc_freqs)} unique terms, "
            f"avgdl={self.avgdl:.1f}"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Simple tokenization:
        - Lowercase
        - Split on whitespace and punctuation
        - Remove very short tokens
        """
        # Lowercase
        text = text.lower()
        
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter short tokens (< 2 chars)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    def _calculate_idf(self) -> None:
        """Calculate IDF (Inverse Document Frequency) for all terms."""
        num_docs = len(self.corpus)
        
        for term, doc_freq in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            # where N = total docs, df = docs containing term
            idf = math.log(
                (num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )
            
            # Apply epsilon floor to prevent negative scores
            self.idf[term] = max(idf, self.epsilon)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
        
        Returns:
            List of BM25 results sorted by score
        """
        if not self.corpus:
            logger.warning("BM25 index is empty")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Query produced no tokens")
            return []
        
        # Calculate BM25 scores for all documents
        scores = []
        for idx, doc in enumerate(self.corpus):
            # Apply metadata filters if provided
            if metadata_filters and not self._matches_filters(doc['metadata'], metadata_filters):
                continue
            
            score = self._calculate_bm25_score(
                query_tokens=query_tokens,
                doc_tokens=doc['tokens'],
                doc_len=self.doc_len[idx]
            )
            
            if score > 0:  # Only include docs with non-zero scores
                scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to BM25Result objects
        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            doc = self.corpus[idx]
            results.append(BM25Result(
                chunk_id=doc['chunk_id'],
                content=doc['content'],
                score=score,
                metadata=doc['metadata'],
                rank=rank
            ))
        
        top_score = results[0].score if results else 0.0
        logger.debug(
            f"BM25 search: query='{query[:50]}...', "
            f"found {len(results)} results, "
            f"top_score={top_score:.3f}"
        )
        
        return results
    
    def _calculate_bm25_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        doc_len: int
    ) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        BM25 formula:
        score = Σ(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))
        
        where:
        - qi: query term i
        - f(qi, D): frequency of qi in document D
        - |D|: length of document D
        - avgdl: average document length
        - k1, b: tuning parameters
        """
        score = 0.0
        
        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)
        
        # Calculate score for each query term
        for term in query_tokens:
            if term not in self.idf:
                continue  # Term not in corpus
            
            # Term frequency in document
            tf = doc_term_freqs.get(term, 0)
            
            if tf == 0:
                continue  # Term not in this document
            
            # IDF score for term
            idf = self.idf[term]
            
            # Length normalization
            norm = 1 - self.b + self.b * (doc_len / self.avgdl)
            
            # BM25 formula
            term_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            
            score += term_score
        
        return score
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Check if document metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_documents': len(self.corpus),
            'num_unique_terms': len(self.doc_freqs),
            'avg_doc_length': self.avgdl,
            'k1': self.k1,
            'b': self.b
        }


def reciprocal_rank_fusion(
    dense_results: List[Any],
    sparse_results: List[BM25Result],
    k: int = 60,
    weights: Tuple[float, float] = (0.7, 0.3)
) -> List[Dict[str, Any]]:
    """
    Combine dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score(d) = Σ(1 / (k + rank(d)))
    
    Args:
        dense_results: Results from dense (vector) retrieval
        sparse_results: Results from sparse (BM25) retrieval
        k: RRF constant (typically 60)
        weights: (dense_weight, sparse_weight) for combining scores
    
    Returns:
        Combined and re-ranked results
    """
    dense_weight, sparse_weight = weights
    
    # Build score maps
    scores = defaultdict(float)
    doc_map = {}  # chunk_id -> full document info
    
    # Add dense results
    for rank, result in enumerate(dense_results, 1):
        chunk_id = result.chunk_id if hasattr(result, 'chunk_id') else result.get('chunk_id')
        rrf_score = dense_weight / (k + rank)
        scores[chunk_id] += rrf_score
        
        # Store document info
        if chunk_id not in doc_map:
            doc_map[chunk_id] = {
                'chunk_id': chunk_id,
                'content': result.content if hasattr(result, 'content') else result.get('content'),
                'metadata': result.metadata if hasattr(result, 'metadata') else result.get('metadata', {}),
                'dense_score': result.score if hasattr(result, 'score') else result.get('score', 0),
                'sparse_score': 0.0,
                'dense_rank': rank,
                'sparse_rank': None
            }
    
    # Add sparse results
    for rank, result in enumerate(sparse_results, 1):
        chunk_id = result.chunk_id
        rrf_score = sparse_weight / (k + rank)
        scores[chunk_id] += rrf_score
        
        # Update or create document info
        if chunk_id in doc_map:
            doc_map[chunk_id]['sparse_score'] = result.score
            doc_map[chunk_id]['sparse_rank'] = rank
        else:
            doc_map[chunk_id] = {
                'chunk_id': chunk_id,
                'content': result.content,
                'metadata': result.metadata,
                'dense_score': 0.0,
                'sparse_score': result.score,
                'dense_rank': None,
                'sparse_rank': rank
            }
    
    # Sort by combined RRF score
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Build final results
    combined_results = []
    for rank, chunk_id in enumerate(ranked_ids, 1):
        doc = doc_map[chunk_id]
        doc['rrf_score'] = scores[chunk_id]
        doc['final_rank'] = rank
        combined_results.append(doc)
    
    logger.info(
        f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(combined_results)} combined results"
    )
    
    return combined_results


