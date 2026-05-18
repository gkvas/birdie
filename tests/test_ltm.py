"""
Tests for the long-term memory (LTM) store.
"""

import json

import pytest

from birdie.core.ltm import LTMStore, LTMEntry


# ---------------------------------------------------------------------------
# LTMStore construction and persistence
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    return LTMStore(user_id="testuser", storage_dir=tmp_path)


def _make_compaction_result(**overrides):
    base = {
        "summary": "Test summary.",
        "extracted_facts": ["fact A", "fact B"],
        "user_preferences": ["likes brevity"],
        "world_facts": [],
        "tool_results": ["tool returned 42"],
        "open_tasks": [],
    }
    base.update(overrides)
    return base


def test_ltm_store_initial_len_zero(tmp_store):
    assert len(tmp_store) == 0


def test_ltm_store_add_returns_entry(tmp_store):
    result = _make_compaction_result()
    entry = tmp_store.add(result)
    assert isinstance(entry, LTMEntry)
    assert entry.summary == "Test summary."
    assert entry.user_id == "testuser"


def test_ltm_store_len_increases(tmp_store):
    tmp_store.add(_make_compaction_result())
    assert len(tmp_store) == 1
    tmp_store.add(_make_compaction_result(summary="Second."))
    assert len(tmp_store) == 2


def test_ltm_store_persists_to_disk(tmp_path):
    store = LTMStore(user_id="alice", storage_dir=tmp_path)
    store.add(_make_compaction_result(summary="Persisted entry."))

    path = tmp_path / "alice.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["user_id"] == "alice"
    assert len(data["entries"]) == 1
    assert data["entries"][0]["summary"] == "Persisted entry."


def test_ltm_store_loads_from_disk(tmp_path):
    # Write, reload in fresh instance.
    s1 = LTMStore(user_id="bob", storage_dir=tmp_path)
    s1.add(_make_compaction_result(summary="Load test."))

    s2 = LTMStore(user_id="bob", storage_dir=tmp_path)
    assert len(s2) == 1
    assert s2.query("load test")[0].summary == "Load test."


def test_ltm_store_nonexistent_file_is_empty(tmp_path):
    store = LTMStore(user_id="nobody", storage_dir=tmp_path)
    assert len(store) == 0


def test_ltm_store_atomic_write(tmp_path):
    """The .tmp file should not exist after save()."""
    store = LTMStore(user_id="charlie", storage_dir=tmp_path)
    store.add(_make_compaction_result())
    tmp_file = tmp_path / "charlie.tmp"
    assert not tmp_file.exists()


def test_ltm_entry_has_id_and_timestamp(tmp_store):
    entry = tmp_store.add(_make_compaction_result())
    assert entry.id  # non-empty
    assert entry.created_at  # ISO timestamp string


def test_ltm_store_user_isolation(tmp_path):
    s_alice = LTMStore(user_id="alice", storage_dir=tmp_path)
    s_bob = LTMStore(user_id="bob", storage_dir=tmp_path)
    s_alice.add(_make_compaction_result(summary="Alice's memory."))
    # Bob's store should not see Alice's entries.
    assert len(s_bob) == 0


# ---------------------------------------------------------------------------
# Query / retrieval
# ---------------------------------------------------------------------------

def test_ltm_store_query_empty_store_returns_empty(tmp_store):
    assert tmp_store.query("anything") == []


def test_ltm_store_query_returns_at_most_k(tmp_store):
    for i in range(10):
        tmp_store.add(_make_compaction_result(summary=f"Entry {i}."))
    results = tmp_store.query("entry", k=3)
    assert len(results) <= 3


def test_ltm_store_query_relevance_order(tmp_store):
    """More relevant entries should rank higher than unrelated ones."""
    tmp_store.add(_make_compaction_result(
        summary="Discussion about Python async programming and asyncio.",
        extracted_facts=["used asyncio", "ran async tests"],
    ))
    tmp_store.add(_make_compaction_result(
        summary="Recipe for baking chocolate sourdough bread.",
        extracted_facts=["used flour", "added yeast"],
    ))
    results = tmp_store.query("asyncio async python", k=2)
    assert results[0].summary.startswith("Discussion about Python")


def test_ltm_store_query_default_k_five(tmp_store):
    for i in range(8):
        tmp_store.add(_make_compaction_result(summary=f"Entry {i}."))
    results = tmp_store.query("entry")
    assert len(results) <= 5


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_for_prompt_empty_returns_empty(tmp_store):
    assert tmp_store.format_for_prompt([]) == ""


def test_format_for_prompt_includes_summary(tmp_store):
    entry = tmp_store.add(_make_compaction_result(summary="The summary text."))
    text = tmp_store.format_for_prompt([entry])
    assert "The summary text." in text


def test_format_for_prompt_includes_facts(tmp_store):
    entry = tmp_store.add(_make_compaction_result(extracted_facts=["key fact X"]))
    text = tmp_store.format_for_prompt([entry])
    assert "key fact X" in text


def test_format_for_prompt_multiple_entries(tmp_store):
    e1 = tmp_store.add(_make_compaction_result(summary="First."))
    e2 = tmp_store.add(_make_compaction_result(summary="Second."))
    text = tmp_store.format_for_prompt([e1, e2])
    assert "First." in text
    assert "Second." in text


def test_format_for_prompt_omits_empty_sections(tmp_store):
    entry = tmp_store.add(_make_compaction_result(
        open_tasks=[],
        tool_results=[],
        world_facts=[],
        user_preferences=[],
    ))
    text = tmp_store.format_for_prompt([entry])
    assert "Open tasks" not in text
    assert "Relevant outcomes" not in text
