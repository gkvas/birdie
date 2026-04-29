"""Tests for birdie.core.session."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from birdie.core.session import (
    MemoryEntry,
    Session,
    SessionManager,
    UserMemory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager(tmp_path):
    return SessionManager(sessions_root=tmp_path)


@pytest.fixture
def session(manager):
    return manager.create("alice")


# ---------------------------------------------------------------------------
# Session ID format
# ---------------------------------------------------------------------------

class TestSessionIdFormat:
    def test_first_session_is_date_1(self, manager):
        s = manager.create("alice")
        date_part = s.id.rsplit("_", 1)[0]
        assert s.id.endswith("_1")
        assert len(date_part) == 10  # YYYY-MM-DD

    def test_second_session_increments(self, manager):
        s1 = manager.create("alice")
        s2 = manager.create("alice")
        n1 = int(s1.id.rsplit("_", 1)[1])
        n2 = int(s2.id.rsplit("_", 1)[1])
        assert n2 == n1 + 1

    def test_different_users_independent(self, manager):
        sa = manager.create("alice")
        sb = manager.create("bob")
        assert sa.id.endswith("_1")
        assert sb.id.endswith("_1")

    def test_many_sessions_increment_correctly(self, manager):
        ids = [manager.create("alice").id for _ in range(5)]
        suffixes = [int(sid.rsplit("_", 1)[1]) for sid in ids]
        assert suffixes == list(range(1, 6))


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_empty_session(self, manager):
        s = manager.create("alice")
        loaded = manager.load("alice", s.id)
        assert loaded.id == s.id
        assert loaded.user_id == "alice"
        assert loaded.turns == 0

    def test_roundtrip_skill_grants(self, manager):
        s = manager.create("alice")
        s.enabled_skills = ["Filesystem"]
        s.disabled_skills = ["SSH"]
        manager.save(s)
        loaded = manager.load("alice", s.id)
        assert loaded.enabled_skills == ["Filesystem"]
        assert loaded.disabled_skills == ["SSH"]

    def test_load_unknown_session_raises(self, manager):
        with pytest.raises(FileNotFoundError, match="Unknown session"):
            manager.load("alice", "2000-01-01_99")

    def test_save_is_atomic(self, manager):
        """Save writes via .tmp then replaces — no partial-write corruption."""
        s = manager.create("alice")
        manager.save(s)
        path = manager._session_path("alice", s.id)
        tmp = path.with_suffix(".tmp")
        assert path.exists()
        assert not tmp.exists()

    def test_session_file_has_no_messages_field(self, manager):
        """Session JSON must not contain a messages key (owned by checkpointer)."""
        s = manager.create("alice")
        path = manager._session_path("alice", s.id)
        data = json.loads(path.read_text())
        assert "messages" not in data
        assert "memory" not in data

    def test_old_session_with_messages_loads_cleanly(self, manager, tmp_path):
        """Backward compat: old session files with messages/memory fields are ignored."""
        user_dir = tmp_path / "alice"
        user_dir.mkdir(parents=True, exist_ok=True)
        legacy = {
            "id": "2020-01-01_1",
            "user_id": "alice",
            "created_at": "2020-01-01T00:00:00+00:00",
            "updated_at": "2020-01-01T00:00:00+00:00",
            "turns": 3,
            "enabled_skills": [],
            "disabled_skills": [],
            "messages": [{"role": "user", "content": "hi"}],
            "memory": [{"id": "abc", "timestamp": "t", "content": "old fact"}],
        }
        (user_dir / "2020-01-01_1.json").write_text(json.dumps(legacy))
        mgr = SessionManager(sessions_root=tmp_path)
        s = mgr.load("alice", "2020-01-01_1")
        assert s.turns == 3  # metadata preserved
        assert not hasattr(s, "messages")
        assert not hasattr(s, "memory")


# ---------------------------------------------------------------------------
# list / delete
# ---------------------------------------------------------------------------

class TestListDelete:
    def test_list_empty(self, manager):
        assert manager.list_sessions("nobody") == []

    def test_list_returns_sorted_ids(self, manager):
        ids = [manager.create("alice").id for _ in range(3)]
        listed = manager.list_sessions("alice")
        assert listed == sorted(ids)

    def test_list_excludes_memory_file(self, manager):
        """memory.json must not appear in session list."""
        manager.create("alice")
        memory = manager.load_user_memory("alice")
        memory.add("a fact")
        manager.save_user_memory(memory)
        listed = manager.list_sessions("alice")
        assert "memory" not in listed

    def test_delete_removes_file(self, manager):
        s = manager.create("alice")
        manager.delete("alice", s.id)
        assert manager.list_sessions("alice") == []

    def test_delete_unknown_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.delete("alice", "2000-01-01_99")


# ---------------------------------------------------------------------------
# db_path
# ---------------------------------------------------------------------------

class TestDbPath:
    def test_db_path_is_under_user_dir(self, manager, tmp_path):
        path = manager.db_path("alice")
        assert path.parent == tmp_path / "alice"
        assert path.suffix == ".db"


# ---------------------------------------------------------------------------
# touch()
# ---------------------------------------------------------------------------

class TestTouch:
    def test_touch_increments_turns(self, session):
        assert session.turns == 0
        session.touch()
        assert session.turns == 1
        session.touch()
        assert session.turns == 2

    def test_touch_updates_timestamp(self, session):
        original = session.updated_at
        session.touch()
        assert session.updated_at >= original


# ---------------------------------------------------------------------------
# UserMemory
# ---------------------------------------------------------------------------

class TestUserMemory:
    def test_add_returns_entry(self):
        mem = UserMemory(user_id="alice")
        entry = mem.add("loves Python")
        assert isinstance(entry, MemoryEntry)
        assert entry.content == "loves Python"
        assert len(entry.id) == 8
        assert "T" in entry.timestamp  # ISO format

    def test_multiple_entries(self):
        mem = UserMemory(user_id="alice")
        mem.add("fact one")
        mem.add("fact two")
        assert len(mem.entries) == 2

    def test_as_strings(self):
        mem = UserMemory(user_id="alice")
        mem.add("a")
        mem.add("b")
        assert mem.as_strings() == ["a", "b"]

    def test_empty_as_strings(self):
        assert UserMemory(user_id="alice").as_strings() == []


# ---------------------------------------------------------------------------
# UserMemory persistence (via SessionManager)
# ---------------------------------------------------------------------------

class TestUserMemoryPersistence:
    def test_roundtrip_empty(self, manager):
        mem = manager.load_user_memory("alice")
        assert mem.user_id == "alice"
        assert mem.entries == []

    def test_roundtrip_with_entries(self, manager):
        mem = manager.load_user_memory("alice")
        mem.add("uses vim")
        mem.add("prefers dark mode")
        manager.save_user_memory(mem)

        loaded = manager.load_user_memory("alice")
        assert len(loaded.entries) == 2
        assert loaded.entries[0].content == "uses vim"
        assert loaded.entries[1].content == "prefers dark mode"

    def test_memory_is_user_scoped_not_session_scoped(self, manager):
        """Memory saved for alice is visible regardless of which session is active."""
        manager.create("alice")
        manager.create("alice")  # second session

        mem = manager.load_user_memory("alice")
        mem.add("shared fact")
        manager.save_user_memory(mem)

        # Loading from either session context gives the same user memory
        reloaded = manager.load_user_memory("alice")
        assert reloaded.entries[0].content == "shared fact"

    def test_save_is_atomic(self, manager, tmp_path):
        mem = manager.load_user_memory("alice")
        mem.add("something")
        manager.save_user_memory(mem)
        path = manager._memory_path("alice")
        tmp = path.with_suffix(".tmp")
        assert path.exists()
        assert not tmp.exists()

    def test_different_users_have_separate_memory(self, manager):
        ma = manager.load_user_memory("alice")
        ma.add("alice fact")
        manager.save_user_memory(ma)

        mb = manager.load_user_memory("bob")
        assert mb.entries == []
