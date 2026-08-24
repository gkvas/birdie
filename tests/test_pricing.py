"""Tests for the model pricing table and session token accounting."""

import json

import pytest

from birdie.core.pricing import (
    estimate_cost,
    price_for_model,
    load_user_pricing_overrides,
)
import birdie.core.pricing as pricing_module


@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    """Every test starts with a clean user-overrides cache."""
    pricing_module._user_overrides_cache = None
    yield
    pricing_module._user_overrides_cache = None


class TestPriceLookup:
    def test_exact_match(self):
        assert price_for_model("claude-opus-5") == (5.00, 25.00)

    def test_prefix_match_for_dated_ids(self):
        assert price_for_model("claude-haiku-4-5-20251001") == (1.00, 5.00)

    def test_longest_prefix_wins(self):
        # gpt-4o-mini must not resolve to the shorter gpt-4o entry
        assert price_for_model("gpt-4o-mini") == (0.15, 0.60)
        assert price_for_model("gpt-4o-2024-08-06") == (2.50, 10.00)

    def test_unknown_model_returns_none(self):
        assert price_for_model("llama3") is None
        assert price_for_model("") is None


class TestEstimateCost:
    def test_known_model(self):
        # 1M input + 1M output on opus-5 = 5 + 25 USD
        assert estimate_cost("claude-opus-5", 1_000_000, 1_000_000) == 30.0

    def test_small_counts(self):
        cost = estimate_cost("claude-sonnet-5", 10_000, 2_000)
        assert abs(cost - (10_000 * 3 + 2_000 * 15) / 1_000_000) < 1e-9

    def test_unknown_model(self):
        assert estimate_cost("mystery-model", 1000, 1000) is None


class TestExtraOverrides:
    def test_extra_overrides_take_priority_over_builtin(self):
        assert price_for_model(
            "claude-opus-5", extra_overrides={"claude-opus-5": [1.0, 2.0]},
        ) == (1.0, 2.0)

    def test_extra_overrides_add_unknown_model(self):
        assert price_for_model(
            "my-custom-model", extra_overrides={"my-custom-model": [1.5, 6.0]},
        ) == (1.5, 6.0)

    def test_estimate_cost_uses_extra_overrides(self):
        cost = estimate_cost(
            "my-custom-model", 1_000_000, 1_000_000,
            extra_overrides={"my-custom-model": [1.0, 2.0]},
        )
        assert cost == 3.0

    def test_malformed_extra_override_entry_ignored(self):
        # A bad value must not crash the lookup, just be skipped.
        assert price_for_model(
            "claude-opus-5", extra_overrides={"claude-opus-5": "not-a-pair"},
        ) == (5.00, 25.00)


class TestUserPricingFile:
    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BIRDIE_PRICING_FILE", str(tmp_path / "nope.json"))
        assert load_user_pricing_overrides(force_reload=True) == {}

    def test_valid_file_overrides_builtin(self, monkeypatch, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text(json.dumps({"claude-opus-5": [1.0, 2.0], "my-model": [3.0, 4.0]}))
        monkeypatch.setenv("BIRDIE_PRICING_FILE", str(f))
        overrides = load_user_pricing_overrides(force_reload=True)
        assert overrides["claude-opus-5"] == (1.0, 2.0)
        assert overrides["my-model"] == (3.0, 4.0)
        assert price_for_model("claude-opus-5") == (1.0, 2.0)
        assert price_for_model("my-model") == (3.0, 4.0)

    def test_malformed_json_treated_as_empty(self, monkeypatch, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text("{not valid json")
        monkeypatch.setenv("BIRDIE_PRICING_FILE", str(f))
        assert load_user_pricing_overrides(force_reload=True) == {}
        # price_for_model still falls back to the built-in table
        assert price_for_model("claude-opus-5") == (5.00, 25.00)

    def test_malformed_entry_in_file_dropped(self, monkeypatch, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text(json.dumps({"good-model": [1.0, 2.0], "bad-model": "oops"}))
        monkeypatch.setenv("BIRDIE_PRICING_FILE", str(f))
        overrides = load_user_pricing_overrides(force_reload=True)
        assert overrides == {"good-model": (1.0, 2.0)}

    def test_cache_reused_without_force_reload(self, monkeypatch, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text(json.dumps({"model-a": [1.0, 2.0]}))
        monkeypatch.setenv("BIRDIE_PRICING_FILE", str(f))
        first = load_user_pricing_overrides()
        f.write_text(json.dumps({"model-a": [9.0, 9.0]}))
        second = load_user_pricing_overrides()  # cached, file change ignored
        assert first == second == {"model-a": (1.0, 2.0)}
        third = load_user_pricing_overrides(force_reload=True)
        assert third == {"model-a": (9.0, 9.0)}

    def test_extra_overrides_beat_user_file(self, monkeypatch, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text(json.dumps({"model-a": [1.0, 2.0]}))
        monkeypatch.setenv("BIRDIE_PRICING_FILE", str(f))
        load_user_pricing_overrides(force_reload=True)
        assert price_for_model(
            "model-a", extra_overrides={"model-a": [9.0, 9.0]},
        ) == (9.0, 9.0)


class TestSessionTokenAccounting:
    def test_token_totals_roundtrip(self, tmp_path):
        from birdie.core.session import SessionManager
        mgr = SessionManager(sessions_root=tmp_path)
        session = mgr.create("alice")
        session.total_input_tokens += 1234
        session.total_output_tokens += 567
        mgr.save(session)
        loaded = mgr.load("alice", session.id)
        assert loaded.total_input_tokens == 1234
        assert loaded.total_output_tokens == 567
