"""
Session management for Birdie.

A session is the primary identity unit: it owns metadata and skill grants.
Conversation history is owned by LangGraph's SqliteSaver checkpointer
(``~/.birdie/sessions/<user_id>/checkpoints.db``); the session JSON files
store only lightweight metadata so they remain human-readable.

Long-term memory is user-scoped (``memory.json`` next to the session files)
so that notes added via ``/remember`` survive session switches.

Session IDs use the format ``YYYY-MM-DD_N`` where N increments from 1
for each new session created on that calendar day.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

SESSIONS_ROOT = Path.home() / ".birdie" / "sessions"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single long-term memory entry."""
    id: str
    timestamp: str
    content: str


@dataclass
class UserMemory:
    """
    User-scoped long-term memory.

    Stored at ``~/.birdie/sessions/<user_id>/memory.json`` and shared across
    all of a user's sessions so that ``/remember`` facts persist when the
    user starts a new session.
    """
    user_id: str
    entries: List[MemoryEntry] = field(default_factory=list)

    def add(self, content: str) -> MemoryEntry:
        """Append a new memory entry and return it."""
        entry = MemoryEntry(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            content=content,
        )
        self.entries.append(entry)
        return entry

    def as_strings(self) -> List[str]:
        """Return memory contents as a flat list of strings for prompt injection."""
        return [e.content for e in self.entries]


@dataclass
class Session:
    """
    Lightweight session metadata.

    Does **not** store conversation history — that is owned by LangGraph's
    SqliteSaver checkpointer.  The session file stores only what LangGraph
    cannot represent: skill grants and housekeeping metadata.
    """
    id: str
    user_id: str
    created_at: str
    updated_at: str
    turns: int
    enabled_skills: List[str]
    disabled_skills: List[str]

    def touch(self) -> None:
        """Increment turn counter and update the last-modified timestamp."""
        self.turns += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Creates, loads, saves, lists, and deletes sessions on disk.

    File layout::

        ~/.birdie/sessions/<user_id>/
            checkpoints.db          # LangGraph SqliteSaver checkpoint store
            memory.json             # user-scoped long-term memory
            2026-04-29_1.json       # session metadata (no messages)
            2026-04-29_2.json
    """

    def __init__(self, sessions_root: Optional[Path] = None) -> None:
        self.root = sessions_root or SESSIONS_ROOT

    def _user_dir(self, user_id: str) -> Path:
        return self.root / user_id

    def _session_path(self, user_id: str, session_id: str) -> Path:
        return self._user_dir(user_id) / f"{session_id}.json"

    def _memory_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "memory.json"

    def db_path(self, user_id: str) -> Path:
        """Return the path to the LangGraph SqliteSaver checkpoint database."""
        return self._user_dir(user_id) / "checkpoints.db"

    # -- session CRUD ---------------------------------------------------------

    def create(self, user_id: str) -> Session:
        """Create a new session with a date-incremented ID."""
        today = datetime.now().strftime("%Y-%m-%d")
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Find the highest existing increment for today
        max_n = 0
        for p in user_dir.glob(f"{today}_*.json"):
            suffix = p.stem[len(today) + 1:]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
        session_id = f"{today}_{max_n + 1}"

        now = datetime.now(timezone.utc).isoformat()
        session = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            turns=0,
            enabled_skills=[],
            disabled_skills=[],
        )
        self.save(session)
        return session

    def load(self, user_id: str, session_id: str) -> Session:
        """Load an existing session from disk.

        Raises:
            FileNotFoundError: if the session does not exist.
        """
        path = self._session_path(user_id, session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown session {session_id!r}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session(
            id=data["id"],
            user_id=data["user_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turns=data["turns"],
            enabled_skills=data.get("enabled_skills", []),
            disabled_skills=data.get("disabled_skills", []),
        )

    def save(self, session: Session) -> None:
        """Persist session metadata to disk (atomic write via temp file)."""
        path = self._session_path(session.user_id, session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "turns": session.turns,
            "enabled_skills": session.enabled_skills,
            "disabled_skills": session.disabled_skills,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def list_sessions(self, user_id: str) -> List[str]:
        """Return all session IDs for a user, sorted chronologically."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return []
        # Exclude the memory file and checkpoints DB
        return sorted(
            p.stem for p in user_dir.glob("*.json")
            if p.stem != "memory"
        )

    def delete(self, user_id: str, session_id: str) -> None:
        """Delete a session file from disk.

        Raises:
            FileNotFoundError: if the session does not exist.
        """
        path = self._session_path(user_id, session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown session {session_id!r}")
        path.unlink()

    # -- user memory ----------------------------------------------------------

    def load_user_memory(self, user_id: str) -> UserMemory:
        """Load user-scoped long-term memory from disk.

        Returns an empty ``UserMemory`` if no file exists yet.
        """
        path = self._memory_path(user_id)
        if not path.exists():
            return UserMemory(user_id=user_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserMemory(
            user_id=user_id,
            entries=[MemoryEntry(**m) for m in data.get("entries", [])],
        )

    def save_user_memory(self, memory: UserMemory) -> None:
        """Persist user-scoped long-term memory (atomic write via temp file)."""
        path = self._memory_path(memory.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "user_id": memory.user_id,
            "entries": [
                {"id": m.id, "timestamp": m.timestamp, "content": m.content}
                for m in memory.entries
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
