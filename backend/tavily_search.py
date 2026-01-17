"""Tavily web search integration for providing recent context to models."""

import httpx
from typing import Optional
from .config import TAVILY_API_KEY

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


async def search_web(query: str, max_results: int = 5) -> Optional[dict]:
    """
    Search the web using Tavily API.

    Returns search results or None if search fails or is not configured.
    """
    if not TAVILY_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "search_depth": "basic",
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return None


def format_search_results(results: dict) -> str:
    """Format Tavily search results into context for LLMs."""
    if not results:
        return ""

    parts = ["## Web Search Results\n"]

    # Include Tavily's generated answer if available
    if results.get("answer"):
        parts.append(f"**Summary:** {results['answer']}\n")

    # Include individual results
    if results.get("results"):
        parts.append("\n**Sources:**\n")
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")[:500]  # Limit content length
            parts.append(f"{i}. **{title}**\n   {url}\n   {content}\n")

    return "\n".join(parts)


def needs_web_search(query: str) -> bool:
    """
    Determine if a query likely needs web search for recent information.

    Looks for:
    - Time-sensitive keywords (latest, recent, current, today, 2025, 2026, etc.)
    - News/event-related terms
    - Specific company/product announcements
    - Questions about prices, stocks, weather
    """
    query_lower = query.lower()

    # Time-sensitive keywords
    time_keywords = [
        "latest", "recent", "current", "now", "today", "yesterday",
        "this week", "this month", "this year",
        "2025", "2026", "2027",  # Recent/future years
        "news", "announced", "announcement", "released", "launched",
        "update", "new version", "just",
    ]

    # Topics that typically need current info
    current_topics = [
        "stock price", "weather", "score", "election",
        "ceo of", "president of", "who is the",
        "how much does", "price of",
        "what happened", "did they",
    ]

    for keyword in time_keywords:
        if keyword in query_lower:
            return True

    for topic in current_topics:
        if topic in query_lower:
            return True

    return False


async def get_search_context(query: str) -> Optional[str]:
    """
    Get web search context for a query if needed.

    Returns formatted search results or None if search not needed/failed.
    """
    if not needs_web_search(query):
        return None

    if not TAVILY_API_KEY:
        print("Web search needed but TAVILY_API_KEY not configured")
        return None

    results = await search_web(query)
    if not results:
        return None

    return format_search_results(results)
