"""Text chunking utilities for RAG pipeline."""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 150,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separator: Primary separator to split on (falls back to sentences)
    
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    # If text is shorter than chunk size, return as-is
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # First, try to split by the separator
    sections = text.split(separator)
    
    current_chunk = ""
    for section in sections:
        # If adding this section would exceed chunk_size, save current chunk
        if current_chunk and len(current_chunk) + len(section) > chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + section
        else:
            current_chunk += (separator if current_chunk else "") + section
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Handle sections that are individually too long
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:  # If significantly over size
            # Split by sentences
            final_chunks.extend(_split_long_chunk(chunk, chunk_size, chunk_overlap))
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def _split_long_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split a long chunk by sentences."""
    # Try to split by sentences
    sentences = []
    for delimiter in ['. ', '.\n', '! ', '? ']:
        if delimiter in text:
            sentences = text.split(delimiter)
            # Add delimiter back
            sentences = [s + delimiter.strip() for s in sentences if s]
            break
    
    if not sentences:
        # Fallback: hard split by character count
        return _hard_split(text, chunk_size, chunk_overlap)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Hard split by character count (last resort)."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    
    return chunks


def chunk_with_metadata(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 150
) -> List[dict]:
    """
    Chunk text and return with metadata.
    
    Args:
        text: Text to chunk
        source: Source identifier (e.g., filename)
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of dicts with 'text', 'source', 'chunk_id'
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    return [
        {
            "text": chunk,
            "source": source,
            "chunk_id": i,
            "total_chunks": len(chunks)
        }
        for i, chunk in enumerate(chunks)
    ]


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (1 token ≈ 4 characters).
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4
