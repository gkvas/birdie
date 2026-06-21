# Memory System

This document describes the memory system in Birdie, including both short-term and long-term memory mechanisms.

---

## Overview

Birdie's memory system consists of two primary components:

1. **Short-term memory**: Managed by LangGraph's checkpointer, this stores the conversation history for the duration of a session.
2. **Long-term memory (LTM)**: A persistent, user-scoped store for structured semantic memory created by the compaction pipeline.

---

## Short-term Memory

### LangGraph Checkpointer

Short-term memory is implemented using LangGraph's checkpointer, which stores the conversation history in a SQLite database (`checkpoints.db`). This allows the agent to maintain context across multiple turns in a session.

- **Location**: `~/.birdie/sessions/<user_id>/checkpoints.db`
- **Purpose**: Stores the conversation history for the current session.
- **Lifetime**: Persists for the duration of the session and can be reloaded if the session is resumed.

---

## Long-term Memory (LTM)

### LTM Store

The LTM store is a per-user, persistent store for structured semantic memory. It is implemented in `birdie/core/ltm.py` and consists of two parts:

1. **Manual Entries**: User-authored strings stored in `memory.json`. These are populated via the `/remember` command in the CLI.
2. **Automatic Entries**: Structured memory created by the compaction pipeline and stored in `ltm.json`.

### Manual Long-term Memory

- **Location**: `~/.birdie/sessions/<user_id>/memory.json`
- **Purpose**: Stores user-authored notes and facts.
- **Usage**: Populated via the `/remember` command in the CLI. These entries are injected into the system prompt as part of Tier 3 (Long-term memory).

### Automatic Long-term Memory

- **Location**: `~/.birdie/ltm/<user_id>.json`
- **Purpose**: Stores structured semantic memory created by the compaction pipeline.
- **Usage**: Automatically generated from conversation history and used to provide context in the system prompt.

---

## Memory in the System Prompt

Memory is integrated into the system prompt in Tier 3, which is divided into two parts:

1. **Manual Entries**: Strings from `memory.json`.
2. **Semantic Entries**: Retrieved from the LTM store based on semantic similarity to the current user message.

Example:

```
--- Long-term memory ---
User preferences:
- Prefers concise answers
- Avoids technical jargon

Relevant facts:
- The user's favorite programming language is Python.
- The user is working on a project named "Birdie".
```

---

## CLI Commands for Memory Management

### `/remember <text>`

Save a note to long-term memory. This appends the text to `memory.json`.

Example:

```
/remember Prefers concise answers
```

### `/compact`

Force-compact the current session's history into long-term memory immediately. This triggers the compaction pipeline to generate structured memory entries.

Example:

```
/compact
```

### `/session info`

Show session metadata, including memory usage and enabled skills/agents.

Example:

```
/session info
```

---

## Compaction Pipeline

The compaction pipeline is responsible for generating structured semantic memory from conversation history. It runs automatically when the conversation history exceeds a certain length or can be triggered manually using the `/compact` command.

---

## Session Management

### Session Files

Session files store metadata that is not part of the conversation history or memory:

- **Location**: `~/.birdie/sessions/<user_id>/session.json`
- **Contents**: Session metadata, such as enabled skills and agents, but not the conversation history or memory.

---

## Summary

- **Short-term memory**: Managed by LangGraph's checkpointer and stored in `checkpoints.db`.
- **Long-term memory**: Consists of manual entries in `memory.json` and automatic entries in `ltm.json`.
- **Memory in the system prompt**: Integrated into Tier 3 of the system prompt.
- **CLI commands**: `/remember`, `/compact`, and `/session info` for managing memory.
