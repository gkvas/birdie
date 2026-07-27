"""Tests for skill loading priority and additivity."""
import tempfile
from pathlib import Path
from unittest.mock import patch

from birdie.agent.run import DynamicAgent
from tests.test_integration import _write_skill, _NoopLLM


def test_additive_loading():
    """Test that skills from all sources are loaded additively."""
    with tempfile.TemporaryDirectory() as cli_dir, \
         tempfile.TemporaryDirectory() as fake_home:

        # Create unique skills in each location
        _write_skill(cli_dir, "CLISkill")
        
        # Mock Path.home() to use fake_home
        import birdie.agent.run as run_module
        original_path = run_module.Path
        
        class MockPath:
            @staticmethod
            def home():
                return Path(fake_home)
            
            def __new__(cls, *args, **kwargs):
                return original_path(*args, **kwargs)
        
        # Create user skills directory
        user_skills_dir = Path(fake_home) / ".birdie" / "skills"
        user_skills_dir.mkdir(parents=True)
        _write_skill(str(user_skills_dir), "UserSkill")
        
        run_module.Path = MockPath
        
        try:
            agent = DynamicAgent(_NoopLLM(), skills_dir=cli_dir)
            
            skills = {s.name for s in agent.registry.list_skills()}
            
            # Both CLI and user skills should be loaded
            assert "CLISkill" in skills
            assert "UserSkill" in skills
            
        finally:
            run_module.Path = original_path


def test_cli_dir_only():
    """Test that when only CLI dir is specified, it works correctly."""
    with tempfile.TemporaryDirectory() as cli_dir:
        
        _write_skill(cli_dir, "CLISkill")
        
        agent = DynamicAgent(_NoopLLM(), skills_dir=cli_dir)
        
        skills = {s.name for s in agent.registry.list_skills()}
        
        # CLI skill should be loaded
        assert "CLISkill" in skills


def test_cli_overrides_user():
    """Test that CLI skills override user skills with same name."""
    with tempfile.TemporaryDirectory() as cli_dir, \
         tempfile.TemporaryDirectory() as fake_home:

        # Create skills with same name but different versions in CLI and user directories
        _write_skill(cli_dir, "TestSkill", version="2.0.0")

        # Mock Path.home() to use fake_home
        import birdie.agent.run as run_module
        original_path = run_module.Path

        class MockPath:
            @staticmethod
            def home():
                return Path(fake_home)

            def __new__(cls, *args, **kwargs):
                return original_path(*args, **kwargs)

        # Create user skills directory
        user_skills_dir = Path(fake_home) / ".birdie" / "skills"
        user_skills_dir.mkdir(parents=True)
        _write_skill(str(user_skills_dir), "TestSkill", version="1.0.0")

        run_module.Path = MockPath

        try:
            agent = DynamicAgent(_NoopLLM(), skills_dir=cli_dir)

            skills = {s.name for s in agent.registry.list_skills()}
            assert "TestSkill" in skills

            # CLI version (2.0.0) should win over user version (1.0.0)
            test_skill = next(s for s in agent.registry.list_skills() if s.name == "TestSkill")
            assert test_skill.version == "2.0.0"

        finally:
            run_module.Path = original_path


class TestSkillEnabledByDefault:
    def test_skill_frontmatter_enabled_by_default_parsed(self):
        from birdie.core.loader import parse_skill_markdown
        content = (
            "---\n"
            "name: AlwaysOn\n"
            "description: default-granted skill\n"
            "enabled_by_default: true\n"
            "---\n\n"
            "Some instructions.\n"
        )
        skill = parse_skill_markdown(content)
        assert skill.enabled_by_default is True

    def test_skill_enabled_by_default_defaults_false(self):
        from birdie.core.loader import parse_skill_markdown
        content = (
            "---\n"
            "name: OptIn\n"
            "description: opt-in skill\n"
            "---\n\n"
            "Some instructions.\n"
        )
        skill = parse_skill_markdown(content)
        assert skill.enabled_by_default is False


class TestNoImplicitCwdLoading:
    def test_default_skills_dir_does_not_load_cwd_skills(self, tmp_path, monkeypatch):
        """DynamicAgent() without skills_dir must ignore ./skills in the CWD."""
        cwd_skills = tmp_path / "skills"
        cwd_skills.mkdir()
        _write_skill(str(cwd_skills), "CwdSkill")
        monkeypatch.chdir(tmp_path)
        # Point HOME somewhere empty so ~/.birdie/skills does not interfere.
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
        agent = DynamicAgent(_NoopLLM())
        names = {s.name for s in agent.registry.list_skills()}
        assert "CwdSkill" not in names

    def test_discover_missing_directory_returns_empty(self, tmp_path):
        from birdie.core.loader import discover_skills_from_directory
        import sys as _sys
        missing = tmp_path / "does-not-exist"
        assert discover_skills_from_directory(str(missing)) == []
        assert str(missing.resolve()) not in _sys.path
