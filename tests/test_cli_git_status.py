"""Tests for the git status segment in the CLI bottom toolbar."""

import subprocess
from unittest.mock import MagicMock, patch

from birdie.cli import BirdieCLI, _format_git_segment, _read_git_status


HEADERS_CLEAN = (
    "# branch.oid 3f2c1ab9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3\n"
    "# branch.head main\n"
    "# branch.upstream origin/main\n"
    "# branch.ab +0 -0\n"
)


def _make_cli(tmp_path):
    from rich.console import Console
    from birdie.core.session import SessionManager

    mgr = SessionManager(sessions_root=tmp_path)
    session = mgr.create("alice")

    agent = MagicMock()
    agent._tool_output_cap = 0
    agent.provider.vendor_name = "anthropic"
    agent.provider.model_name = "claude-sonnet-4-6"

    return BirdieCLI(
        agent,
        session_manager=mgr,
        session=session,
        user_id="alice",
        user_memory=MagicMock(),
        console=Console(record=True, width=120),
    )


class TestFormatGitSegment:
    def test_clean_repo(self):
        assert _format_git_segment(HEADERS_CLEAN) == "⎇ main"

    def test_dirty_repo(self):
        out = HEADERS_CLEAN + "1 .M N... 100644 100644 100644 abc def cli.py\n"
        assert _format_git_segment(out) == "⎇ main*"

    def test_untracked_counts_as_dirty(self):
        out = HEADERS_CLEAN + "? scratch.txt\n"
        assert _format_git_segment(out) == "⎇ main*"

    def test_ahead_and_behind(self):
        out = HEADERS_CLEAN.replace("+0 -0", "+2 -1")
        assert _format_git_segment(out) == "⎇ main ↑2↓1"

    def test_ahead_only(self):
        out = HEADERS_CLEAN.replace("+0 -0", "+3 -0")
        assert _format_git_segment(out) == "⎇ main ↑3"

    def test_behind_only_and_dirty(self):
        out = HEADERS_CLEAN.replace("+0 -0", "+0 -4") + "? x\n"
        assert _format_git_segment(out) == "⎇ main* ↓4"

    def test_no_upstream(self):
        out = (
            "# branch.oid 3f2c1ab9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3\n"
            "# branch.head feature-x\n"
        )
        assert _format_git_segment(out) == "⎇ feature-x"

    def test_detached_head_shows_short_oid(self):
        out = (
            "# branch.oid 3f2c1ab9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3\n"
            "# branch.head (detached)\n"
        )
        assert _format_git_segment(out) == "⎇ (3f2c1ab)"

    def test_initial_commit(self):
        out = "# branch.oid (initial)\n# branch.head main\n"
        assert _format_git_segment(out) == "⎇ main"

    def test_empty_output(self):
        assert _format_git_segment("") == ""


class TestReadGitStatus:
    def test_not_a_repo_returns_empty(self):
        proc = MagicMock(returncode=128, stdout="")
        with patch("birdie.cli.subprocess.run", return_value=proc):
            assert _read_git_status("/somewhere") == ""

    def test_missing_git_binary_returns_empty(self):
        with patch("birdie.cli.subprocess.run", side_effect=FileNotFoundError):
            assert _read_git_status("/somewhere") == ""

    def test_timeout_returns_empty(self):
        err = subprocess.TimeoutExpired(cmd="git", timeout=1.0)
        with patch("birdie.cli.subprocess.run", side_effect=err):
            assert _read_git_status("/somewhere") == ""

    def test_success_formats_output(self):
        proc = MagicMock(returncode=0, stdout=HEADERS_CLEAN)
        with patch("birdie.cli.subprocess.run", return_value=proc) as run:
            assert _read_git_status("/somewhere") == "⎇ main"
        assert run.call_args.kwargs["cwd"] == "/somewhere"
        assert run.call_args.kwargs["timeout"] == 1.0


class TestGitSegmentCache:
    def test_cached_within_ttl_and_refreshed_after(self, tmp_path):
        cli = _make_cli(tmp_path)
        with patch(
            "birdie.cli._read_git_status", return_value="⎇ main"
        ) as read:
            assert cli._git_segment() == "⎇ main"
            assert cli._git_segment() == "⎇ main"
            assert read.call_count == 1

            # Expire the TTL: the next render re-reads git state
            cli._git_cache["at"] -= 10.0
            cli._git_segment()
            assert read.call_count == 2

            # A cwd change (e.g. /cd) also bypasses the cache
            cli._git_cache["cwd"] = "/elsewhere"
            cli._git_segment()
            assert read.call_count == 3


class TestToolbarGitSegment:
    def test_toolbar_shows_git_segment(self, tmp_path):
        cli = _make_cli(tmp_path)
        with patch("birdie.cli._read_git_status", return_value="⎇ main* ↑2"):
            toolbar = cli._get_toolbar()
        assert "⎇ main* ↑2" in toolbar.value
        assert "claude-sonnet-4-6" in toolbar.value

    def test_toolbar_without_repo_unchanged(self, tmp_path):
        cli = _make_cli(tmp_path)
        with patch("birdie.cli._read_git_status", return_value=""):
            toolbar = cli._get_toolbar()
        assert "⎇" not in toolbar.value
        assert "claude-sonnet-4-6" in toolbar.value

    def test_toolbar_right_aligns_when_wide(self, tmp_path):
        cli = _make_cli(tmp_path)
        fake_app = MagicMock()
        fake_app.output.get_size.return_value = MagicMock(columns=300)
        with patch("birdie.cli._read_git_status", return_value="⎇ main"), \
             patch("prompt_toolkit.application.get_app",
                   return_value=fake_app):
            toolbar = cli._get_toolbar()
        plain = toolbar.value.replace("<b>", "").replace("</b>", "")
        # Padded to full width; the trailing space leaves a 1-col margin
        assert plain.endswith("⎇ main ")
        assert len(plain) == 300

    def test_toolbar_falls_back_when_narrow(self, tmp_path):
        cli = _make_cli(tmp_path)
        fake_app = MagicMock()
        fake_app.output.get_size.return_value = MagicMock(columns=20)
        with patch("birdie.cli._read_git_status", return_value="⎇ main"), \
             patch("prompt_toolkit.application.get_app",
                   return_value=fake_app):
            toolbar = cli._get_toolbar()
        assert toolbar.value.endswith("   │   ⎇ main")

    def test_toolbar_escapes_branch_html(self, tmp_path):
        cli = _make_cli(tmp_path)
        with patch("birdie.cli._read_git_status",
                   return_value="⎇ fix<a&b>"):
            toolbar = cli._get_toolbar()
        assert "⎇ fix&lt;a&amp;b&gt;" in toolbar.value
        # Must stay parseable as prompt_toolkit HTML
        toolbar.formatted_text
