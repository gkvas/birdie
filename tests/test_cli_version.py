"""Tests for the ``--version`` CLI flag and the version helper."""

import importlib.metadata

import pytest

from birdie.cli import _package_version, main


class TestPackageVersion:
    def test_returns_installed_version(self):
        v = _package_version()
        assert isinstance(v, str) and v
        # Installed in the venv as birdie-agent; falls back to "dev" otherwise.
        assert v == importlib.metadata.version("birdie-agent")

    def test_falls_back_to_dev_when_not_installed(self, monkeypatch):
        def _raise(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _raise)
        assert _package_version() == "dev"


class TestVersionFlag:
    def test_version_flag_prints_and_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["birdie", "--version"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert out.strip() == "birdie " + _package_version()

    def test_version_flag_does_not_start_repl(self, monkeypatch, capsys):
        """--version must exit before any agent/session setup runs."""
        monkeypatch.setattr("sys.argv", ["birdie", "--version"])
        with pytest.raises(SystemExit):
            main()
        # Nothing but the version line was printed.
        assert capsys.readouterr().out.count("\n") == 1

    def test_version_in_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["birdie", "--help"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert "--version" in capsys.readouterr().out
