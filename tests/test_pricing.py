"""Tests for the model pricing table and session token accounting."""

from birdie.core.pricing import estimate_cost, price_for_model


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
