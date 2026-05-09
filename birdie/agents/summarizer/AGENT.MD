---
name: Summarizer
version: 1.0.0
description: Summarize a piece of text into concise bullet points. Enable with /agent enable Summarizer.
enabled_by_default: false
allowed_skills: []
---

## Input

### text
type: string
description: The text to summarize
required: true

### max_points
type: integer
description: Maximum number of bullet points (default 5)
required: false

## Output

### summary
type: string
description: Bullet-point summary of the text

## Prompt

Summarize the following text concisely. Return at most {{ max_points }} bullet points (default to 5 if not specified). Use plain bullet points starting with "- ".

Text to summarize:

{{ text }}
