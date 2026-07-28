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
- **Usage**: Populated via the `/remember` command in the CLI. These entries are injected into the model's context on every turn (as part of the ephemeral session-context message, under `--- Long-term memory ---`).

### Automatic Long-term Memory

- **Location**: `~/.birdie/ltm/<user_id>.json`
- **Purpose**: Stores structured semantic memory created by the compaction pipeline.
- **Usage**: Automatically generated from conversation history; the most relevant entries are retrieved each turn and injected into the model's context.

---

## Memory in the Model's Context

The system prompt itself carries only stable content (so provider prompt caches survive across turns). Memory is delivered through an ephemeral session-context message appended after the conversation on every turn, containing up to three blocks:

1. **Rolling summary**: The narrative summary of compacted-away history (`--- Earlier conversation (compacted) ---`). It stays with the session even when no LTM store is configured, and later compactions fold the previous summary in.
2. **Manual entries**: Strings from `memory.json`.
3. **Semantic entries**: Retrieved from the LTM store based on semantic similarity to the current user message (top 5 by default, subject to `ltm_min_score`).

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

The compaction pipeline is responsible for generating structured semantic memory from conversation history. It runs automatically as a background task when the stored history reaches 80 messages (`min_messages_auto` + `compression_window_size`) - or, when `compaction_token_threshold` is configured, when the last model call consumed at least that many input tokens - and can be triggered manually at any time with `/compact`. See [architecture.md](architecture.md) for the full algorithm and [cli.md](cli.md) for the configuration keys.

---

## Session Management

### Session Files

Session files store metadata that is not part of the conversation history or memory:

- **Location**: `~/.birdie/sessions/<user_id>/<session_id>.json` (e.g. `2026-04-29_1.json`)
- **Contents**: Session metadata - enabled/disabled skills and agents, permission approvals, turn count, and cumulative token totals - but not the conversation history or memory.

---

## Summary

- **Short-term memory**: Managed by LangGraph's checkpointer and stored in `checkpoints.db`, plus a rolling summary of compacted-away history kept in the graph state.
- **Long-term memory**: Consists of manual entries in `memory.json` and automatic entries in `~/.birdie/ltm/<user_id>.json`.
- **Memory in the model's context**: Delivered via the ephemeral session-context message appended each turn (the system prompt stays stable for prompt caching).
- **CLI commands**: `/remember`, `/compact`, and `/session info` for managing memory.
