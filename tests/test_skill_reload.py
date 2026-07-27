"""Tests for skill hot-reload and SKILL.MD scaffolding."""

import pytest

from birdie.agent.run import DynamicAgent
from birdie.core.loader import parse_skill_markdown, scaffold_skill
from tests.test_integration import _NoopLLM, _write_skill


class TestReloadSkills:
    def test_reload_picks_up_new_and_removed_skills(self, tmp_path, monkeypatch):
        from pathlib import Path
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(str(skills), "First")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

        agent = DynamicAgent(_NoopLLM(), skills_dir=str(skills),
                             skills_enabled=["First", "Second"])
        names = {s.name for s in agent.registry.list_skills()}
        assert "First" in names and "Second" not in names

        _write_skill(str(skills), "Second")
        import shutil
        shutil.rmtree(skills / "First")

        n = agent.reload_skills()
        names = {s.name for s in agent.registry.list_skills()}
        assert n == len(names)
        assert "Second" in names and "First" not in names

    def test_reload_reseeds_enabled_by_default(self, tmp_path, monkeypatch):
        from pathlib import Path
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(str(skills), "Plain")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

        agent = DynamicAgent(_NoopLLM(), skills_dir=str(skills))
        assert "Auto" not in agent.policy.get_allowed_skills()

        auto = skills / "Auto"
        auto.mkdir()
        (auto / "SKILL.MD").write_text(
            "---\nname: Auto\nversion: 1.0.0\ndescription: d\n"
            "enabled_by_default: true\n---\n\nbody\n"
        )
        agent.reload_skills()
        assert "Auto" in agent.policy.get_allowed_skills()


class TestScaffoldSkill:
    def test_scaffold_creates_parseable_freetext_skill(self, tmp_path):
        path = scaffold_skill(str(tmp_path), "MyNewSkill")
        assert path.exists()
        skill = parse_skill_markdown(path.read_text())
        assert skill.name == "MyNewSkill"
        assert skill.tools == []          # template must not register phantom tools
        assert skill.permissions == []    # nor phantom permissions
        assert skill.body                 # freetext body present
        assert skill.enabled_by_default is False

    def test_scaffold_refuses_to_overwrite(self, tmp_path):
        scaffold_skill(str(tmp_path), "Dup")
        with pytest.raises(FileExistsError):
            scaffold_skill(str(tmp_path), "Dup")
