---
name: DuckDuckGo
version: 1.0.0
description: Web search via DuckDuckGo - no API key required.
tags: [search, web]
enabled_by_default: false
triggers: [search, find, look up, google, web, online, current, latest, news]
---

## Tools

### duckduckgo_search
description: Search the web using DuckDuckGo and return the top results with title, URL, and summary. Use for current events, facts, or anything requiring up-to-date information. max_results defaults to 5; raise it up to 10 when broader coverage is needed.
entrypoint: python:birdie.skills.duckduckgo.tools.search
schema:
  type: object
  properties:
    query:
      type: string
      description: The search query
    max_results:
      type: integer
      description: Maximum number of results to return (default 5, max 10)
      default: 5
  required: [query]
