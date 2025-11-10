from typing import List
from loguru import logger

from src.config import TAVILY_API_KEY, WEB_SEARCH_MAX_RESULTS, has_tavily_key
from src.schemas import WebSearchResult, WebSearchResponse
from src.logging_setup import log_tool_call


def search_web(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS, user_id: str = "unknown") -> WebSearchResponse:
    """
    Search the web using Tavily API.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        user_id: User ID for logging
    
    Returns:
        WebSearchResponse with results or stub
    """
    # Check if API key is available
    if not has_tavily_key():
        logger.warning("Tavily API key not available, returning stub response")
        log_tool_call(
            user_id=user_id,
            tool="web_search",
            params={"query": query, "max_results": max_results},
            result="stub_response_no_key"
        )
        return _stub_response(query)
    
    # Attempt real search
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )
        
        results = []
        for item in response.get("results", []):
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source_type="web"
                )
            )
        
        log_tool_call(
            user_id=user_id,
            tool="web_search",
            params={"query": query, "max_results": max_results},
            result=f"found {len(results)} results"
        )
        
        return WebSearchResponse(
            results=results,
            query=query,
            is_stub=False
        )
    
    except ImportError:
        logger.warning("Tavily package not installed, returning stub response")
        return _stub_response(query)
    
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        log_tool_call(
            user_id=user_id,
            tool="web_search",
            params={"query": query, "max_results": max_results},
            result=f"error: {str(e)}"
        )
        return _stub_response(query)


def _stub_response(query: str) -> WebSearchResponse:
    """
    Generate a deterministic stub response when web search is unavailable.
    
    Args:
        query: Original search query
    
    Returns:
        WebSearchResponse with explanatory stub
    """
    stub_results = [
        WebSearchResult(
            title="Web Search Unavailable",
            url="",
            snippet=f"Web search for '{query}' could not be completed. "
                   f"The Tavily API key is not configured. To enable real-time web search, "
                   f"please add your TAVILY_API_KEY to the .env file. "
                   f"For now, please consult your healthcare provider or reliable medical "
                   f"websites for current information on this topic.",
            source_type="web_stub"
        )
    ]
    
    return WebSearchResponse(
        results=stub_results,
        query=query,
        is_stub=True
    )


def format_search_results(response: WebSearchResponse) -> str:
    """
    Format web search results for LLM context.
    
    Args:
        response: WebSearchResponse
    
    Returns:
        Formatted string
    """
    if response.is_stub:
        return f"Web search unavailable: {response.results[0].snippet}"
    
    if not response.results:
        return f"No web results found for query: {response.query}"
    
    formatted_parts = [f"Web search results for: {response.query}\n"]
    
    for i, result in enumerate(response.results, 1):
        formatted_parts.append(
            f"[Web Result {i}]:\n"
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Content: {result.snippet}\n"
        )
    
    return "\n".join(formatted_parts)


def should_use_web_search(query: str) -> bool:
    """
    Determine if a query should trigger web search.
    
    Triggers for:
    - Time-sensitive queries (latest, current, recent, new)
    - Guideline queries
    - Trending medical topics
    
    Args:
        query: User query
    
    Returns:
        True if web search is recommended
    """
    query_lower = query.lower()
    
    time_indicators = [
        "latest", "current", "recent", "new", "updated",
        "2024", "2025", "today", "this year"
    ]
    
    guideline_indicators = [
        "guideline", "recommendation", "protocol",
        "standard of care", "best practice"
    ]
    
    # Check for time-sensitive language
    for indicator in time_indicators:
        if indicator in query_lower:
            return True
    
    # Check for guideline queries
    for indicator in guideline_indicators:
        if indicator in query_lower:
            return True
    
    return False
