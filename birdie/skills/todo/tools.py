"""
Python entrypoints for the ToDo planning skill.

These functions are called via ``python:birdie.skills.todo.tools.<name>``
entrypoints and their return values appear as ToolMessage content in the CLI.
"""


# Last plan handed out by create_plan. Used to reject verbatim re-plans, a
# common failure mode where a model "recovers" from an error by re-issuing
# the same plan and restarting from step 1 instead of acting on the error.
_last_plan: list | None = None


def create_plan(steps: list, **_) -> str:
    """Format and return a numbered plan from a list of step descriptions.

    Args:
        steps: Ordered list of step strings.
        **_: Ignored extra kwargs from the StructuredTool wrapper.

    Returns:
        A multiline string with a header and one checkbox line per step, or a
        short refusal if ``steps`` is identical to the previous plan.
    """
    global _last_plan
    if not steps:
        return "No steps provided."
    steps = [str(s) for s in steps]
    if steps == _last_plan:
        return (
            f"Plan unchanged ({len(steps)} steps) - not re-created. Do not "
            "re-plan; continue with the next open step, or act on the last "
            "error (run a different diagnostic command or ask the user)."
        )
    _last_plan = steps
    n = len(steps)
    lines = [f"Plan ({n} step{'s' if n != 1 else ''}):"]
    for i, step in enumerate(steps, 1):
        lines.append(f"  [ ] {i}. {step}")
    return "\n".join(lines)


def complete_step(step_number: int, summary: str = "", **_) -> str:
    """Return a completion confirmation line for a single plan step.

    Args:
        step_number: 1-based index of the step being marked done.
        summary: Optional one-line description of what was found or done.
        **_: Ignored extra kwargs from the StructuredTool wrapper.

    Returns:
        A short confirmation string, e.g. ``[x] Step 2 done: listed 5 files``.
    """
    msg = f"[x] Step {step_number} done"
    if summary:
        msg += f": {summary}"
    return msg
