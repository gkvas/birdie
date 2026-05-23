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

        # Create skills with same name in CLI and user directories
        _write_skill(cli_dir, "TestSkill", enabled_by_default=True)
        
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
        _write_skill(str(user_skills_dir), "TestSkill", enabled_by_default=False)
        
        run_module.Path = MockPath
        
        try:
            agent = DynamicAgent(_NoopLLM(), skills_dir=cli_dir)
            
            skills = {s.name for s in agent.registry.list_skills()}
            assert "TestSkill" in skills
            
            # Find the TestSkill and check its enabled_by_default value
            test_skill = next(s for s in agent.registry.list_skills() if s.name == "TestSkill")
            # It should have the CLI version's value (True), not the user version's (False)
            assert test_skill.enabled_by_default == True
            
        finally:
            run_module.Path = original_path
