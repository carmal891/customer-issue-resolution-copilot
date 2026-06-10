"""
Advanced semantic chunking strategy for hotel policy documents.

This module implements section-aware semantic chunking with overlap,
optimized for policy documents and conversational threads.

Key Features:
- Section-based chunking (preserves document structure)
- Semantic boundary detection
- Configurable overlap for context preservation
- Metadata enrichment for better retrieval
- Support for multiple document types
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from enum import Enum


class DocumentType(Enum):
    """Document type classification for chunking strategy selection."""
    POLICY = "policy"
    RUNBOOK = "runbook"
    CONVERSATION = "conversation"
    TICKET = "ticket"
    RESOLUTION = "resolution"


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    content: str
    chunk_id: str
    document_id: str
    document_type: DocumentType
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticChunker:
    """
    Advanced semantic chunker with section awareness and overlap.

    Implements the chunking strategy described in docs/chunking-strategy.md:
    - Section-based chunking for policy documents
    - Semantic boundary detection
    - Configurable overlap (default 100 tokens)
    - Metadata enrichment
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
        preserve_sections: bool = True
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap_size: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size to avoid tiny fragments
            preserve_sections: Whether to respect section boundaries
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.preserve_sections = preserve_sections

        # Section header patterns (Markdown-style)
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][^.!?]*):$',  # Colon-terminated headers
            r'^\d+\.\s+(.+)$',  # Numbered sections
            r'^[A-Z\s]{3,}$',  # ALL CAPS headers
        ]

    def chunk_document(
        self,
        content: str,
        document_id: str,
        document_type: DocumentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk a document using semantic boundaries.

        Args:
            content: Document content
            document_id: Unique document identifier
            document_type: Type of document for strategy selection
            metadata: Additional metadata to attach to chunks

        Returns:
            List of chunks with metadata
        """
        if document_type == DocumentType.POLICY:
            return self._chunk_policy_document(content, document_id, metadata)
        elif document_type == DocumentType.CONVERSATION:
            return self._chunk_conversation(content, document_id, metadata)
        elif document_type == DocumentType.RESOLUTION:
            return self._chunk_resolution(content, document_id, metadata)
        else:
            return self._chunk_generic(content, document_id, document_type, metadata)

    def _chunk_policy_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk policy documents using hierarchical section-aware strategy.
        
        Strategy:
        1. Extract major sections (## headers)
        2. If section fits in chunk_size, keep as one chunk
        3. Otherwise, extract sub-sections (### headers)
        4. If sub-section fits, keep as one chunk
        5. Otherwise, split with overlap while preserving section context
        
        This preserves semantic units and improves context recall.
        """
        chunks = []
        major_sections = self._extract_major_sections(content)
        
        for major_idx, major_section in enumerate(major_sections):
            major_title = major_section['title']
            major_content = major_section['content']
            major_size = self._estimate_tokens(major_content)
            
            # If major section fits in one chunk, keep it together
            if major_size <= self.chunk_size:
                chunk_content = f"## {major_title}\n\n{major_content}"
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    document_type=DocumentType.POLICY,
                    section_title=major_title,
                    section_number=str(major_idx + 1),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
            else:
                # Split into sub-sections
                sub_sections = self._extract_sub_sections(major_content)
                
                for sub_idx, sub_section in enumerate(sub_sections):
                    sub_title = sub_section['title']
                    sub_content = sub_section['content']
                    sub_size = self._estimate_tokens(sub_content)
                    
                    # If sub-section fits in one chunk, keep it together
                    if sub_size <= self.chunk_size:
                        # Include major section context in chunk
                        full_title = f"{major_title} - {sub_title}"
                        chunk_content = f"## {major_title}\n### {sub_title}\n\n{sub_content}"
                        chunk = Chunk(
                            content=chunk_content,
                            chunk_id=f"{document_id}_chunk_{len(chunks)}",
                            document_id=document_id,
                            document_type=DocumentType.POLICY,
                            section_title=full_title,
                            section_number=f"{major_idx + 1}.{sub_idx + 1}",
                            metadata=metadata or {}
                        )
                        chunks.append(chunk)
                    else:
                        # Split with overlap, preserving section context
                        section_chunks = self._split_with_context_overlap(
                            content=sub_content,
                            document_id=document_id,
                            major_title=major_title,
                            sub_title=sub_title,
                            section_number=f"{major_idx + 1}.{sub_idx + 1}",
                            start_chunk_idx=len(chunks),
                            metadata=metadata
                        )
                        chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_major_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract major sections (## headers) from markdown content.
        
        Returns:
            List of dicts with 'title' and 'content' keys
        """
        sections = []
        current_section = None
        lines = content.split('\n')
        
        for line in lines:
            # Detect major section header (## but not ###)
            if line.strip().startswith('## ') and not line.strip().startswith('### '):
                # Save previous section
                if current_section and current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section)
                
                # Start new section
                title = line.strip().replace('## ', '').strip()
                current_section = {
                    'title': title,
                    'content': []
                }
            elif current_section is not None:
                # Add line to current section
                current_section['content'].append(line)
        
        # Add final section
        if current_section and current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
        
        # If no sections found, treat entire document as one section
        if not sections:
            sections = [{'title': 'Document', 'content': content}]
        
        return sections
    
    def _extract_sub_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sub-sections (### headers) from content.
        
        Returns:
            List of dicts with 'title' and 'content' keys
        """
        sub_sections = []
        current_sub = None
        lines = content.split('\n')
        
        for line in lines:
            # Detect sub-section header (###)
            if line.strip().startswith('### '):
                # Save previous sub-section
                if current_sub and current_sub['content']:
                    current_sub['content'] = '\n'.join(current_sub['content'])
                    sub_sections.append(current_sub)
                
                # Start new sub-section
                title = line.strip().replace('### ', '').strip()
                current_sub = {
                    'title': title,
                    'content': []
                }
            elif current_sub is not None:
                # Add line to current sub-section
                current_sub['content'].append(line)
        
        # Add final sub-section
        if current_sub and current_sub['content']:
            current_sub['content'] = '\n'.join(current_sub['content'])
            sub_sections.append(current_sub)
        
        # If no sub-sections found, treat entire content as one sub-section
        if not sub_sections:
            sub_sections = [{'title': 'Content', 'content': content}]
        
        return sub_sections
    
    def _split_with_context_overlap(
        self,
        content: str,
        document_id: str,
        major_title: str,
        sub_title: str,
        section_number: str,
        start_chunk_idx: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split content with overlap while preserving section context.
        
        Key improvement: Always include section headers in each chunk
        so context is preserved even when content is split.
        """
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self._estimate_tokens(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk with section context
                chunk_content = f"## {major_title}\n### {sub_title}\n\n" + ' '.join(current_chunk)
                
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{start_chunk_idx + len(chunks)}",
                    document_id=document_id,
                    document_type=DocumentType.POLICY,
                    section_title=f"{major_title} - {sub_title}",
                    section_number=section_number,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                
                # Calculate overlap (keep last N sentences)
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_size = self._estimate_tokens(s)
                    if overlap_size + s_size <= self.overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = self._estimate_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_content = f"## {major_title}\n### {sub_title}\n\n" + ' '.join(current_chunk)
            
            chunk = Chunk(
                content=chunk_content,
                chunk_id=f"{document_id}_chunk_{start_chunk_idx + len(chunks)}",
                document_id=document_id,
                document_type=DocumentType.POLICY,
                section_title=f"{major_title} - {sub_title}",
                section_number=section_number,
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        return chunks

    def _chunk_conversation(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk conversation threads by message groups.

        Conversations should be chunked by logical message groups
        to preserve context and flow.
        """
        # Split by message boundaries (assuming newline-separated messages)
        messages = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for msg in messages:
            msg_size = self._estimate_tokens(msg)

            if current_size + msg_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated messages
                chunk_content = '\n\n'.join(current_chunk)
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    document_type=DocumentType.CONVERSATION,
                    metadata=metadata or {}
                )
                chunks.append(chunk)

                # Start new chunk with overlap (last message)
                current_chunk = [current_chunk[-1], msg] if current_chunk else [msg]
                current_size = self._estimate_tokens('\n\n'.join(current_chunk))
            else:
                current_chunk.append(msg)
                current_size += msg_size

        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                document_type=DocumentType.CONVERSATION,
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_resolution(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk resolution documents by steps.

        Resolution documents contain step-by-step procedures
        that should be kept together when possible.
        """
        # Try to identify steps
        step_pattern = r'(?:^|\n)(?:Step \d+|[\d]+\.)\s*(.+?)(?=(?:\n(?:Step \d+|[\d]+\.)|$))'
        steps = re.findall(step_pattern, content, re.DOTALL | re.MULTILINE)

        if not steps:
            # No clear steps, use generic chunking
            return self._chunk_generic(content, document_id, DocumentType.RESOLUTION, metadata)

        chunks = []
        current_chunk = []
        current_size = 0

        for step_idx, step in enumerate(steps):
            step_size = self._estimate_tokens(step)

            if current_size + step_size > self.chunk_size and current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    document_type=DocumentType.RESOLUTION,
                    section_title=f"Steps {len(chunks) * 3 + 1}-{len(chunks) * 3 + len(current_chunk)}",
                    metadata=metadata or {}
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                current_chunk = [current_chunk[-1], step] if current_chunk else [step]
                current_size = self._estimate_tokens('\n\n'.join(current_chunk))
            else:
                current_chunk.append(step)
                current_size += step_size

        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                document_type=DocumentType.RESOLUTION,
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_generic(
        self,
        content: str,
        document_id: str,
        document_type: DocumentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Generic chunking strategy with overlap.

        Used for documents without clear structure.
        """
        return self._split_with_overlap(
            content,
            document_id,
            document_type,
            None,
            None,
            0,
            metadata
        )

    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extract sections from document based on headers.

        Returns list of dicts with 'title' and 'content' keys.
        """
        lines = content.split('\n')
        sections = []
        current_section = {'title': None, 'content': []}

        for line in lines:
            is_header = False

            # Check if line matches any section pattern
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section if it has content
                    if current_section['content']:
                        current_section['content'] = '\n'.join(current_section['content'])
                        sections.append(current_section)

                    # Start new section
                    current_section = {
                        'title': match.group(1) if match.lastindex else line.strip(),
                        'content': []
                    }
                    is_header = True
                    break

            if not is_header:
                current_section['content'].append(line)

        # Add final section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)

        # If no sections found, treat entire document as one section
        if not sections:
            sections = [{'title': None, 'content': content}]

        return sections

    def _split_with_overlap(
        self,
        content: str,
        document_id: str,
        document_type: DocumentType,
        section_title: Optional[str],
        section_number: Optional[str],
        start_chunk_idx: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split content into chunks with overlap.

        Uses sentence boundaries for cleaner splits.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self._estimate_tokens(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = ' '.join(current_chunk)
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{start_chunk_idx + len(chunks)}",
                    document_id=document_id,
                    document_type=document_type,
                    section_title=section_title,
                    section_number=section_number,
                    metadata=metadata or {}
                )
                chunks.append(chunk)

                # Calculate overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_size = self._estimate_tokens(s)
                    if overlap_size + s_size <= self.overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_size = self._estimate_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add final chunk
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_content = ' '.join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                chunk_id=f"{document_id}_chunk_{start_chunk_idx + len(chunks)}",
                document_id=document_id,
                document_type=document_type,
                section_title=section_title,
                section_number=section_number,
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple heuristic: ~4 characters per token.
        For production, use tiktoken or similar.
        """
        return len(text) // 4

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Calculate statistics for chunk quality assessment.

        Returns metrics useful for evaluating chunking strategy.
        """
        if not chunks:
            return {}

        chunk_sizes = [len(c.content) for c in chunks]
        token_estimates = [self._estimate_tokens(c.content) for c in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_chunk_size_chars': sum(chunk_sizes) / len(chunks),
            'min_chunk_size_chars': min(chunk_sizes),
            'max_chunk_size_chars': max(chunk_sizes),
            'avg_chunk_size_tokens': sum(token_estimates) / len(chunks),
            'min_chunk_size_tokens': min(token_estimates),
            'max_chunk_size_tokens': max(token_estimates),
            'chunks_with_sections': sum(1 for c in chunks if c.section_title),
            'unique_sections': len(set(c.section_title for c in chunks if c.section_title))
        }
