"""
Python entrypoints for the Filesystem skill.

These functions are called via ``python:birdie.skills.filesystem.tools.<name>``
entrypoints and their return values appear as ToolMessage content in the CLI.

Unlike the ``bash:`` tools in this skill, arguments arrive as real Python
values - no shell templating, no quoting, no corruption of content that
contains quotes, backslashes, or newlines.
"""

from pathlib import Path


def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    **_,
) -> str:
    """Replace an exact string in a file with a new string.

    Args:
        path: Path to the file to edit.
        old_string: Exact text to find, including whitespace/indentation.
        new_string: Replacement text.
        replace_all: Replace every occurrence instead of a unique one.
        **_: Ignored extra kwargs from the StructuredTool wrapper.

    Returns:
        A one-line confirmation, e.g. ``Edited foo.py: 1 replacement``.

    Raises:
        ValueError: If the file is missing, ``old_string`` is empty or not
            found, equals ``new_string``, or matches more than once without
            ``replace_all``.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise ValueError(f"File not found: {path}")
    if not old_string:
        raise ValueError(
            "old_string must not be empty; use write_file to create content."
        )
    if old_string == new_string:
        raise ValueError(
            "old_string and new_string are identical - nothing to change."
        )

    content = target.read_text()
    count = content.count(old_string)
    if count == 0:
        raise ValueError(
            f"old_string not found in {path}. It must match the file "
            "exactly, including whitespace and indentation - re-read the "
            "file and retry."
        )
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {count} times in {path}. Add surrounding "
            "lines to make it unique, or pass replace_all=true to replace "
            "every occurrence."
        )

    target.write_text(content.replace(old_string, new_string))
    n = count if replace_all else 1
    return f"Edited {path}: {n} replacement{'s' if n != 1 else ''}"
