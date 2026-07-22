"""SearXNG search wrapper for the local Agent Suite assistant."""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - fallback path
    requests = None


def web_search(query: str, base_url: str = "http://localhost:8080") -> List[Dict[str, Any]]:
    if requests is None:
        return []
    try:
        response = requests.get(f"{base_url}/search", params={"q": query, "format": "json"}, timeout=3)
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])[:5]
    except Exception:
        return []


def summarize_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No search results were available locally."
    return f"Search results for {query}: " + "; ".join(result.get("title", "") for result in results if result.get("title"))
