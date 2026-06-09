"""
RAG Context Assembly Module

Assembles retrieved and reranked chunks into coherent context for agent prompts.

Handles:
- Chunk deduplication
- Context window management
- Source attribution
- Prompt construction
- Metadata preservation
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.application.rag.retrieval import RetrievalResult
from src.application.rag.reranking import RerankResult


logger = logging.getLogger(__name__)


@dataclass
class AssemblyConfig:
    """Configuration for context assembly"""
    max_context_tokens: int = 4000
    max_chunks: int = 10
    include_metadata: bool = True
    include_sources: bool = True
    deduplicate_content: bool = True
    similarity_threshold: float = 0.9  # For deduplication
    group_by_source: bool = False
    add_section_headers: bool = True
    token_buffer: int = 500  # Reserve for prompt template


@dataclass
class AssembledContext:
    """Assembled context ready for agent consumption"""
    context_text: str
    chunks_used: List[str]  # Chunk IDs
    sources: List[str]
    metadata: Dict[str, Any]
    token_count: int
    num_chunks: int
    assembly_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "context_text": self.context_text,
            "chunks_used": self.chunks_used,
            "sources": self.sources,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "num_chunks": self.num_chunks,
            "assembly_time_ms": self.assembly_time_ms,
        }


@dataclass
class PromptTemplate:
    """Template for constructing agent prompts"""
    system_prompt: str
    context_prefix: str = "## Relevant Context\n\n"
    context_suffix: str = "\n\n"
    query_prefix: str = "## Customer Issue\n\n"
    query_suffix: str = "\n\n"
    instruction: str = "## Instructions\n\nBased on the context above, help resolve the customer issue."
    
    def format(
        self,
        context: str,
        query: str,
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Format complete prompt.
        
        Args:
            context: Assembled context
            query: User query/issue
            additional_instructions: Optional additional instructions
        
        Returns:
            Formatted prompt
        """
        parts = [
            self.system_prompt,
            self.context_prefix,
            context,
            self.context_suffix,
            self.query_prefix,
            query,
            self.query_suffix,
            self.instruction,
        ]
        
        if additional_instructions:
            parts.append(f"\n\n{additional_instructions}")
        
        return "".join(parts)


class ContextAssembler:
    """
    Assembles retrieved chunks into coherent context.
    
    Handles token limits, deduplication, and formatting.
    """
    
    def __init__(self, config: Optional[AssemblyConfig] = None):
        """
        Initialize context assembler.
        
        Args:
            config: Assembly configuration
        """
        self.config = config or AssemblyConfig()
        logger.info(
            f"Initialized ContextAssembler with max_tokens={self.config.max_context_tokens}"
        )
    
    def assemble_from_retrieval(
        self,
        results: List[RetrievalResult],
        query: Optional[str] = None
    ) -> AssembledContext:
        """
        Assemble context from retrieval results.
        
        Args:
            results: Retrieval results
            query: Original query (for relevance-based ordering)
        
        Returns:
            Assembled context
        """
        start_time = datetime.now()
        
        # Deduplicate if configured
        if self.config.deduplicate_content:
            results = self._deduplicate_chunks(results)
        
        # Limit to max chunks
        results = results[:self.config.max_chunks]
        
        # Group by source if configured
        if self.config.group_by_source:
            context_text = self._assemble_grouped(results)
        else:
            context_text = self._assemble_sequential(results)
        
        # Calculate token count (approximate)
        token_count = self._estimate_tokens(context_text)
        
        # Truncate if exceeds limit
        if token_count > self.config.max_context_tokens:
            context_text, token_count = self._truncate_context(
                context_text,
                self.config.max_context_tokens
            )
        
        # Extract metadata
        chunks_used = [r.chunk_id for r in results]
        sources = list(set(r.source for r in results))
        
        metadata = {
            "num_sources": len(sources),
            "doc_types": list(set(r.doc_type for r in results)),
            "domains": list(set(r.domain for r in results if r.domain)),
            "avg_score": sum(r.score for r in results) / len(results) if results else 0.0,
        }
        
        end_time = datetime.now()
        assembly_time_ms = (end_time - start_time).total_seconds() * 1000
        
        assembled = AssembledContext(
            context_text=context_text,
            chunks_used=chunks_used,
            sources=sources,
            metadata=metadata,
            token_count=token_count,
            num_chunks=len(results),
            assembly_time_ms=assembly_time_ms,
        )
        
        logger.info(
            f"Assembled context: {len(results)} chunks, "
            f"{token_count} tokens, {assembly_time_ms:.2f}ms"
        )
        
        return assembled
    
    def assemble_from_reranked(
        self,
        results: List[RerankResult],
        query: Optional[str] = None
    ) -> AssembledContext:
        """
        Assemble context from reranked results.
        
        Args:
            results: Reranked results
            query: Original query
        
        Returns:
            Assembled context
        """
        # Extract retrieval results
        retrieval_results = [r.retrieval_result for r in results]
        return self.assemble_from_retrieval(retrieval_results, query)
    
    def _assemble_sequential(self, results: List[RetrievalResult]) -> str:
        """
        Assemble chunks sequentially with separators.
        
        Args:
            results: Retrieval results
        
        Returns:
            Assembled text
        """
        parts = []
        
        for i, result in enumerate(results):
            # Add section header if configured
            if self.config.add_section_headers:
                header = self._format_chunk_header(result, i + 1)
                parts.append(header)
            
            # Add content
            parts.append(result.content)
            
            # Add metadata if configured
            if self.config.include_metadata:
                metadata_str = self._format_metadata(result)
                if metadata_str:
                    parts.append(metadata_str)
            
            # Add separator
            parts.append("\n---\n")
        
        return "\n".join(parts)
    
    def _assemble_grouped(self, results: List[RetrievalResult]) -> str:
        """
        Assemble chunks grouped by source document.
        
        Args:
            results: Retrieval results
        
        Returns:
            Assembled text
        """
        # Group by source
        grouped: Dict[str, List[RetrievalResult]] = {}
        for result in results:
            source = result.source
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(result)
        
        # Assemble by group
        parts = []
        for source, group_results in grouped.items():
            # Source header
            parts.append(f"### Source: {source}\n")
            
            # Chunks from this source
            for i, result in enumerate(group_results):
                if self.config.add_section_headers and result.section:
                    parts.append(f"**Section: {result.section}**\n")
                
                parts.append(result.content)
                parts.append("\n")
            
            parts.append("\n---\n")
        
        return "\n".join(parts)
    
    def _format_chunk_header(self, result: RetrievalResult, rank: int) -> str:
        """Format header for a chunk"""
        parts = [f"### Chunk {rank}"]
        
        if result.section:
            parts.append(f" - {result.section}")
        
        if self.config.include_sources:
            parts.append(f" (Source: {result.source})")
        
        return "".join(parts) + "\n\n"
    
    def _format_metadata(self, result: RetrievalResult) -> str:
        """Format metadata for a chunk"""
        metadata_parts = []
        
        if result.doc_type:
            metadata_parts.append(f"Type: {result.doc_type}")
        
        if result.domain:
            metadata_parts.append(f"Domain: {result.domain}")
        
        if result.score:
            metadata_parts.append(f"Relevance: {result.score:.3f}")
        
        if metadata_parts:
            return f"\n*[{', '.join(metadata_parts)}]*\n"
        
        return ""
    
    def _deduplicate_chunks(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Remove near-duplicate chunks.
        
        Args:
            results: Retrieval results
        
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        deduplicated = [results[0]]  # Keep first result
        
        for result in results[1:]:
            # Check if similar to any kept result
            is_duplicate = False
            for kept in deduplicated:
                if self._is_similar(result.content, kept.content):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        logger.info(
            f"Deduplicated {len(results)} -> {len(deduplicated)} chunks"
        )
        
        return deduplicated
    
    def _is_similar(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are similar (simple heuristic).
        
        For POC, uses simple overlap ratio.
        In production, could use embedding similarity.
        """
        # Normalize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= self.config.similarity_threshold
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Uses simple heuristic: ~4 characters per token.
        For production, use tiktoken or similar.
        """
        return len(text) // 4
    
    def _truncate_context(
        self,
        text: str,
        max_tokens: int
    ) -> Tuple[str, int]:
        """
        Truncate context to fit token limit.
        
        Args:
            text: Context text
            max_tokens: Maximum tokens
        
        Returns:
            Tuple of (truncated text, actual token count)
        """
        # Estimate max characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text, self._estimate_tokens(text)
        
        # Truncate at sentence boundary
        truncated = text[:max_chars]
        
        # Find last sentence boundary
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        boundary = max(last_period, last_newline)
        
        if boundary > 0:
            truncated = truncated[:boundary + 1]
        
        # Add truncation notice
        truncated += "\n\n[Context truncated due to length...]"
        
        token_count = self._estimate_tokens(truncated)
        
        logger.warning(
            f"Context truncated from {self._estimate_tokens(text)} to {token_count} tokens"
        )
        
        return truncated, token_count


class PromptBuilder:
    """
    Builds complete prompts for agent consumption.
    
    Combines assembled context with query and instructions.
    """
    
    def __init__(
        self,
        assembler: Optional[ContextAssembler] = None,
        template: Optional[PromptTemplate] = None
    ):
        """
        Initialize prompt builder.
        
        Args:
            assembler: Context assembler
            template: Prompt template
        """
        self.assembler = assembler or ContextAssembler()
        self.template = template or self._default_template()
        
        logger.info("Initialized PromptBuilder")
    
    def _default_template(self) -> PromptTemplate:
        """Create default prompt template"""
        return PromptTemplate(
            system_prompt=(
                "You are a helpful hotel customer service assistant. "
                "Use the provided context to help resolve customer issues accurately. "
                "Always cite sources when making claims. "
                "If you're unsure, ask for clarification rather than guessing.\n\n"
            )
        )
    
    def build_prompt(
        self,
        query: str,
        results: List[RetrievalResult],
        additional_instructions: Optional[str] = None
    ) -> Tuple[str, AssembledContext]:
        """
        Build complete prompt from query and retrieval results.
        
        Args:
            query: User query/issue
            results: Retrieval results
            additional_instructions: Optional additional instructions
        
        Returns:
            Tuple of (formatted prompt, assembled context)
        """
        # Assemble context
        context = self.assembler.assemble_from_retrieval(results, query)
        
        # Format prompt
        prompt = self.template.format(
            context=context.context_text,
            query=query,
            additional_instructions=additional_instructions
        )
        
        logger.info(
            f"Built prompt: {self._estimate_tokens(prompt)} tokens total"
        )
        
        return prompt, context
    
    def build_prompt_from_reranked(
        self,
        query: str,
        results: List[RerankResult],
        additional_instructions: Optional[str] = None
    ) -> Tuple[str, AssembledContext]:
        """
        Build prompt from reranked results.
        
        Args:
            query: User query/issue
            results: Reranked results
            additional_instructions: Optional additional instructions
        
        Returns:
            Tuple of (formatted prompt, assembled context)
        """
        # Extract retrieval results
        retrieval_results = [r.retrieval_result for r in results]
        return self.build_prompt(query, retrieval_results, additional_instructions)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4


# Convenience function for quick context assembly
def assemble_context(
    results: List[RetrievalResult],
    max_tokens: int = 4000,
    include_metadata: bool = True
) -> str:
    """
    Quick context assembly with default settings.
    
    Args:
        results: Retrieval results
        max_tokens: Maximum context tokens
        include_metadata: Include chunk metadata
    
    Returns:
        Assembled context text
    """
    config = AssemblyConfig(
        max_context_tokens=max_tokens,
        include_metadata=include_metadata
    )
    assembler = ContextAssembler(config)
    context = assembler.assemble_from_retrieval(results)
    return context.context_text
