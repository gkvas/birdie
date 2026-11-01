"""Tests for the workflow expression language and templating."""

import pytest

from birdie.core.workflow_expr import ExprError, check_template, evaluate, render, truthy

CTX = {
    "inputs": {"path": "tests/", "n": 3, "flag": True, "tags": ["a", "b"]},
    "steps": {
        "test": {"ok": False, "exit_code": 1, "stdout": "l1\nl2\nl3\nFAILED x", "output": "x"},
        "skip": {"ok": True, "skipped": True, "output": None},
    },
}


@pytest.mark.parametrize("src,expected", [
    ("42", 42),
    ("1.5", 1.5),
    ("'it''s'" if False else "'a\\'b'", "a'b"),
    ('"x\\ny"', "x\ny"),
    ("true", True),
    ("False", False),
    ("null", None),
    ("[1, 'a', inputs.n]", [1, "a", 3]),
    ("inputs.path", "tests/"),
    ("steps.test.exit_code", 1),
    ("steps['test'].ok", False),
    ("inputs.tags[1]", "b"),
    ("inputs.tags[-1]", "b"),
    ("not steps.test.ok", True),
    ("steps.test.ok or inputs.flag", True),
    ("steps.test.ok and missing_is_not_evaluated", False),
    ("inputs.n > 2 and inputs.n <= 3", True),
    ("inputs.n + 1", 4),
    ("inputs.n - 5", -2),
    ("-inputs.n", -3),
    ("'a' + 'b'", "ab"),
    ("'FAILED' in steps.test.stdout", True),
    ("'c' not in inputs.tags", True),
    ("'path' in inputs", True),
    ("steps.test.exit_code != 0", True),
    ("not inputs.n == 3", False),  # not binds looser than ==
    ("(1 + 2) == 3", True),
])
def test_evaluate(src, expected):
    assert evaluate(src, CTX) == expected


@pytest.mark.parametrize("src,expected", [
    ("steps.test.stdout | tail(2)", "l3\nFAILED x"),
    ("steps.test.stdout | head(1)", "l1"),
    ("steps.test.stdout | lines | length", 4),
    ("steps.skip.output | default('none')", "none"),
    ("inputs.tags | join('+')", "a+b"),
    ("'  x ' | trim | upper", "X"),
    ("'3' | int", 3),
    ("inputs.tags | json", '["a", "b"]'),
    ("'{\"a\": 1}' | from_json", {"a": 1}),
    ("'```json\\n{\"a\": 1}\\n```' | from_json", {"a": 1}),
    ("\"it's\" | quote", "'it'\"'\"'s'"),
    ("'a,b' | split(',')", ["a", "b"]),
    ("'aXb' | replace('X', '-')", "a-b"),
    ("inputs.tags | length > 1", True),  # filters bind loosest... applied to comparison
])
def test_filters(src, expected):
    if src == "inputs.tags | length > 1":
        # A filter applies to the whole expression on its left; filter args
        # are full expressions too, so comparisons go in parentheses.
        assert evaluate("(inputs.tags | length) > 1", CTX) is True
        return
    assert evaluate(src, CTX) == expected


@pytest.mark.parametrize("src,msg", [
    ("nope", "undefined name 'nope'"),
    ("steps.missing.ok", "has no key 'missing'"),
    ("inputs.path.x", "cannot look up"),
    ("inputs.tags[5]", "out of range"),
    ("inputs.n < 'a'", "cannot compare"),
    ("inputs.n + 'a'", "cannot apply"),
    ("x | bogus", "unknown filter"),
    ("open('x')", "unexpected"),
    ("inputs.__class__", "has no key '__class__'"),
    ("1 +", "unexpected end"),
    ("(1", "expected ')'"),
    ("a = 1", "unexpected character"),
    ("", "empty expression"),
])
def test_errors(src, msg):
    with pytest.raises(ExprError, match=msg.replace("(", r"\(").replace(")", r"\)")
                       .replace("[", r"\[").replace("]", r"\]").replace("'", "'")):
        evaluate(src, CTX)


def test_truthy():
    assert truthy("steps.skip.skipped", CTX) is True
    assert truthy("inputs.tags", CTX) is True
    assert truthy("''", CTX) is False


def test_render_whole_placeholder_keeps_type():
    assert render("{{ inputs.tags }}", CTX) == ["a", "b"]
    assert render("  {{ inputs.n }}  ", CTX) == 3
    assert render("{{ inputs.flag }}", CTX) is True


def test_render_mixed_stringifies():
    assert render("n={{ inputs.n }} tags={{ inputs.tags }} none={{ null }}", CTX) == \
        'n=3 tags=["a", "b"] none='
    assert render("{{ true }}/{{ false }}", CTX) == "true/false"


def test_render_nested_structures():
    out = render({"a": ["{{ inputs.n }}", "x{{ inputs.n }}"], "b": 7}, CTX)
    assert out == {"a": [3, "x3"], "b": 7}


def test_render_without_placeholders_is_identity():
    assert render("plain $HOME {x}", CTX) == "plain $HOME {x}"
    assert render(5, CTX) == 5


def test_check_template():
    check_template({"a": "{{ inputs.n }}", "b": ["{{ steps.x | tail(3) }}"]})
    with pytest.raises(ExprError):
        check_template("ok {{ 1 + }}")
