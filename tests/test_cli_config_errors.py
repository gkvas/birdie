"""Tests for graceful startup failures: bad --config path, invalid config JSON."""

import asyncio

import pytest

from birdie.cli import _async_main, main


@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    """Redirect session storage into tmp_path so tests never touch ~/.birdie."""
    monkeypatch.setattr("birdie.core.session.SESSIONS_ROOT", tmp_path / "sessions")
    monkeypatch.delenv("LLM_PROVIDER_CONFIG", raising=False)
    monkeypatch.delenv("LLM_VENDOR", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    return tmp_path


class TestConfigFlagValidation:
    def test_missing_config_file_exits_2_with_usage_error(self, monkeypatch, capsys, tmp_path):
        missing = tmp_path / "mistral.jsoan"
        monkeypatch.setattr("sys.argv", ["birdie", "--config", str(missing)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "config file not found" in err
        assert str(missing) in err
        assert "Traceback" not in err

    def test_config_path_that_is_a_directory(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr("sys.argv", ["birdie", "--config", str(tmp_path)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "not a file" in capsys.readouterr().err

    def test_tilde_in_config_path_is_expanded(self, monkeypatch, capsys, tmp_path, isolated_sessions):
        # A quoted "~" must still resolve to $HOME; the file exists, so we get
        # past the file check and fail later on the unknown vendor (exit 1).
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "prov.json").write_text('{"vendor": "definitely-unknown"}')
        monkeypatch.setattr("sys.argv", ["birdie", "--config", "~/prov.json", "-p", "hi"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "Unknown vendor" in out


class TestAsyncMainConfigErrors:
    def test_vanished_config_file_aborts_without_traceback(self, monkeypatch, capsys, isolated_sessions):
        missing = isolated_sessions / "gone.json"
        with pytest.raises(SystemExit) as exc:
            asyncio.run(_async_main(None, "tester", str(isolated_sessions), None, missing))
        assert exc.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "Config file not found" in out
        assert str(missing) in out
        assert "Traceback" not in out

    def test_malformed_config_json_aborts_gracefully(self, monkeypatch, capsys, isolated_sessions):
        bad = isolated_sessions / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(SystemExit) as exc:
            asyncio.run(_async_main(None, "tester", str(isolated_sessions), None, bad))
        assert exc.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "Configuration error" in out
        assert "Traceback" not in out
