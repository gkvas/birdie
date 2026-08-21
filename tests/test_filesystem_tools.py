"""
Tests for the Filesystem skill's python: entrypoints (edit_file).
"""

import pytest

from birdie.core.entrypoints import resolve_python
from birdie.skills.filesystem.tools import edit_file


@pytest.fixture
def sample(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text('def greet():\n    print("hello")\n    return "hello"\n')
    return f


class TestEditFile:
    def test_unique_replacement(self, sample):
        out = edit_file(str(sample), 'print("hello")', 'print("goodbye")')
        assert out == f"Edited {sample}: 1 replacement"
        assert 'print("goodbye")' in sample.read_text()
        assert 'print("hello")' not in sample.read_text()

    def test_content_with_quotes_backslashes_newlines(self, sample):
        """The failure mode that motivated this tool: content that shell
        templating would corrupt must round-trip verbatim."""
        gnarly = 'lines = "\\n".join(x)\nsay(\'it\\\'s "fine"\')\n'
        edit_file(str(sample), '    return "hello"\n', gnarly)
        assert gnarly in sample.read_text()

    def test_ambiguous_match_raises(self, sample):
        with pytest.raises(ValueError, match="occurs 2 times"):
            edit_file(str(sample), '"hello"', '"bye"')
        assert '"hello"' in sample.read_text()  # unchanged

    def test_replace_all(self, sample):
        out = edit_file(str(sample), '"hello"', '"bye"', replace_all=True)
        assert out == f"Edited {sample}: 2 replacements"
        assert '"hello"' not in sample.read_text()

    def test_not_found_raises(self, sample):
        with pytest.raises(ValueError, match="not found in"):
            edit_file(str(sample), "no such text", "x")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="File not found"):
            edit_file(str(tmp_path / "nope.txt"), "a", "b")

    def test_empty_old_string_raises(self, sample):
        with pytest.raises(ValueError, match="must not be empty"):
            edit_file(str(sample), "", "x")

    def test_identical_strings_raise(self, sample):
        with pytest.raises(ValueError, match="identical"):
            edit_file(str(sample), '"hello"', '"hello"')

    def test_callable_via_python_entrypoint(self, sample):
        """The SKILL.MD entrypoint path must resolve and execute."""
        out = resolve_python(
            "python:birdie.skills.filesystem.tools.edit_file",
            path=str(sample),
            old_string="def greet():",
            new_string="def salute():",
        )
        assert out == f"Edited {sample}: 1 replacement"
        assert "def salute():" in sample.read_text()
