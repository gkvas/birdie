"""
Long-term memory (LTM) store for Birdie.

Each USER has one persistent LTM store backed by a JSON file at:
  ~/.birdie/ltm/<user_id>.json

Entries are created by the compaction pipeline and contain structured,
timeless semantic memory (summary, facts, preferences, world knowledge,
tool outcomes, open tasks).  Similarity search is delegated to
:mod:`birdie.core.retrieval`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from .retrieval import embed, cosine_similarity

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LTMEntry:
    id: str
    user_id: str
    summary: str
    extracted_facts: List[str]
    user_preferences: List[str]
    world_facts: List[str]
    tool_results: List[str]
    open_tasks: List[str]
    embedding: List[float]
    created_at: str


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LTMStore:
    """
    Persistent, user-scoped long-term memory store.

    Backed by ``<storage_dir>/<user_id>.json``.  Loaded lazily on first
    access; saved atomically (write-then-rename) after every ``add`` call.

    Eviction policy (applied on load and after each add):
    - Entries older than ``max_age_days`` are dropped.
    - If more than ``max_entries`` remain, the oldest are dropped.

    Retrieval threshold:
    - ``query()`` only returns entries whose cosine similarity to the query
      meets or exceeds ``min_score``.
    """

    DEFAULT_DIR = Path.home() / ".birdie" / "ltm"
    DEFAULT_MAX_AGE_DAYS: int = 90
    DEFAULT_MAX_ENTRIES: int = 100
    DEFAULT_MIN_SCORE: float = 0.05

    def __init__(
        self,
        user_id: str,
        storage_dir: Optional[Path] = None,
        max_age_days: Optional[int] = DEFAULT_MAX_AGE_DAYS,
        max_entries: Optional[int] = DEFAULT_MAX_ENTRIES,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.user_id = user_id
        self.storage_dir = Path(storage_dir) if storage_dir else self.DEFAULT_DIR
        self.max_age_days = max_age_days
        self.max_entries = max_entries
        self.min_score = min_score
        self._entries: List[LTMEntry] = []
        self._loaded = False

    # -- persistence -------------------------------------------------------

    def _path(self) -> Path:
        return self.storage_dir / f"{self.user_id}.json"

    def load(self) -> None:
        """Load entries from disk; no-op if the file does not exist yet."""
        path = self._path()
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._entries = [LTMEntry(**e) for e in raw.get("entries", [])]
            n_before = len(self._entries)
            self._evict()
            if len(self._entries) < n_before:
                self.save()
        self._loaded = True

    def save(self) -> None:
        """Persist all entries to disk via atomic write."""
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "user_id": self.user_id,
            "entries": [
                {
                    "id": e.id,
                    "user_id": e.user_id,
                    "summary": e.summary,
                    "extracted_facts": e.extracted_facts,
                    "user_preferences": e.user_preferences,
                    "world_facts": e.world_facts,
                    "tool_results": e.tool_results,
                    "open_tasks": e.open_tasks,
                    "embedding": e.embedding,
                    "created_at": e.created_at,
                }
                for e in self._entries
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _evict(self) -> None:
        """Drop entries that exceed the age limit or entry cap (in-place)."""
        if self.max_age_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
            self._entries = [
                e for e in self._entries
                if datetime.fromisoformat(e.created_at) >= cutoff
            ]
        if self.max_entries is not None and len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # -- operations --------------------------------------------------------

    def add(self, compaction_result: dict) -> LTMEntry:
        """Store a structured compaction result and persist to disk.

        ``compaction_result`` must have the keys produced by the compaction
        prompt: ``summary``, ``extracted_facts``, ``user_preferences``,
        ``world_facts``, ``tool_results``, ``open_tasks``.
        """
        self._ensure_loaded()

        retrieval_text = " ".join(filter(None, [
            compaction_result.get("summary", ""),
            " ".join(compaction_result.get("extracted_facts", [])),
            " ".join(compaction_result.get("user_preferences", [])),
            " ".join(compaction_result.get("world_facts", [])),
            " ".join(compaction_result.get("tool_results", [])),
            " ".join(compaction_result.get("open_tasks", [])),
        ]))

        entry = LTMEntry(
            id=str(uuid.uuid4())[:8],
            user_id=self.user_id,
            summary=compaction_result.get("summary", ""),
            extracted_facts=list(compaction_result.get("extracted_facts", [])),
            user_preferences=list(compaction_result.get("user_preferences", [])),
            world_facts=list(compaction_result.get("world_facts", [])),
            tool_results=list(compaction_result.get("tool_results", [])),
            open_tasks=list(compaction_result.get("open_tasks", [])),
            embedding=embed(retrieval_text),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._entries.append(entry)
        self._evict()
        self.save()
        return entry

    def query(self, text: str, k: int = 5) -> List[LTMEntry]:
        """Return the top-k entries whose similarity to *text* meets min_score."""
        self._ensure_loaded()
        if not self._entries:
            return []
        query_vec = embed(text)
        scored = sorted(
            ((e, cosine_similarity(query_vec, e.embedding)) for e in self._entries),
            key=lambda x: -x[1],
        )
        return [e for e, score in scored if score >= self.min_score][:k]

    def format_for_prompt(self, entries: List[LTMEntry]) -> str:
        """Render retrieved LTM entries as readable text for prompt injection."""
        if not entries:
            return ""
        parts: List[str] = []
        for e in entries:
            lines = [f"- {e.summary}"]
            if e.extracted_facts:
                lines.append("  Facts: " + "; ".join(e.extracted_facts))
            if e.user_preferences:
                lines.append("  Preferences: " + "; ".join(e.user_preferences))
            if e.world_facts:
                lines.append("  World knowledge: " + "; ".join(e.world_facts))
            if e.tool_results:
                lines.append("  Relevant outcomes: " + "; ".join(e.tool_results))
            if e.open_tasks:
                lines.append("  Open tasks: " + "; ".join(e.open_tasks))
            parts.append("\n".join(lines))
        return "\n".join(parts)

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._entries)
