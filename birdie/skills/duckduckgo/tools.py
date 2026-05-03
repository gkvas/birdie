"""
Entrypoints for the DuckDuckGo search skill.
"""

from ddgs import DDGS


def search(query: str, max_results: int = 5, **_) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
    return "\n\n".join(lines)
