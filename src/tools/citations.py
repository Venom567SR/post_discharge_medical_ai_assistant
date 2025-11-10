from typing import List
from src.schemas import Citation, RetrievedChunk


def format_inline_citation(citation: Citation, index: int) -> str:
    """
    Format a citation for inline use in text.
    
    Args:
        citation: Citation object
        index: Citation number
    
    Returns:
        Formatted citation string (e.g., "[Ref p.14]" or "(Web Source)")
    """
    if citation.source_type == "reference":
        if citation.page:
            return f"[Ref p.{citation.page}]"
        elif citation.reference_id:
            return f"[Ref: {citation.reference_id}]"
        else:
            return f"[Ref {index}]"
    
    elif citation.source_type == "web":
        return "(Web Source)"
    
    elif citation.source_type == "web_stub":
        return "(Web Search Unavailable)"
    
    return f"[Source {index}]"


def format_citation_list(citations: List[Citation]) -> str:
    """
    Format a list of citations for display.
    
    Args:
        citations: List of Citation objects
    
    Returns:
        Formatted citations as a string
    """
    if not citations:
        return "No sources cited."
    
    lines = ["Sources:"]
    
    for i, citation in enumerate(citations, 1):
        if citation.source_type == "reference":
            source_info = citation.reference_id or "Reference"
            page_info = f", page {citation.page}" if citation.page else ""
            score_info = f" (relevance: {citation.score:.2f})" if citation.score else ""
            lines.append(f"• {source_info}{page_info}{score_info}")
        
        elif citation.source_type == "web":
            url = citation.url or "N/A"
            lines.append(f"• Web: {url}")
        
        elif citation.source_type == "web_stub":
            lines.append(f"• Web search unavailable (API key not configured)")
    
    return "\n".join(lines)


def extract_citation_tags(text: str) -> List[str]:
    """
    Extract citation tags from text.
    
    Args:
        text: Text containing citation tags like [Ref p.14]
    
    Returns:
        List of extracted citation tags
    """
    import re
    pattern = r'\[Ref[^\]]*\]|\(Web Source\)'
    return re.findall(pattern, text)


def add_inline_citations(text: str, citations: List[Citation]) -> str:
    """
    Add inline citations to text at appropriate points.
    
    Note: This is a simple implementation. In practice, the LLM should
    insert citations inline as it generates the response.
    
    Args:
        text: Text without citations
        citations: List of citations to add
    
    Returns:
        Text with inline citations
    """
    # This is a placeholder - in practice, citations should be added
    # by the LLM during generation using structured output
    if not citations:
        return text
    
    # Append citation list at the end
    citation_list = format_citation_list(citations)
    return f"{text}\n\n{citation_list}"


def chunks_to_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """
    Convert RetrievedChunk objects to Citation objects.
    
    Args:
        chunks: List of retrieved chunks
    
    Returns:
        List of Citation objects
    """
    citations = []
    
    for chunk in chunks:
        citation = Citation(
            source_type="reference",
            reference_id=chunk.source,
            page=chunk.page,
            url=None,
            score=chunk.score,
            snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        )
        citations.append(citation)
    
    return citations


def deduplicate_citations(citations: List[Citation]) -> List[Citation]:
    """
    Remove duplicate citations (same source and page).
    
    Args:
        citations: List of citations
    
    Returns:
        Deduplicated list
    """
    seen = set()
    unique_citations = []
    
    for citation in citations:
        # Create a unique key based on source type, reference_id, and page
        key = (
            citation.source_type,
            citation.reference_id,
            citation.page,
            citation.url
        )
        
        if key not in seen:
            seen.add(key)
            unique_citations.append(citation)
    
    return unique_citations


def validate_citation(citation: Citation) -> bool:
    """
    Validate that a citation has required fields.
    
    Args:
        citation: Citation to validate
    
    Returns:
        True if valid, False otherwise
    """
    if citation.source_type == "reference":
        return bool(citation.reference_id or citation.page)
    elif citation.source_type == "web":
        return bool(citation.url)
    elif citation.source_type == "web_stub":
        return True
    
    return False
