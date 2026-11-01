"""
Tiny, safe expression language and ``{{ }}`` templating for workflows.

The language is deliberately small - just enough to steer a workflow
deterministically, with no access to Python objects or builtins:

* literals: ``42``, ``1.5``, ``'text'``, ``"text"``, ``true``, ``false``,
  ``null``, lists ``[1, 'a']``
* names and lookups: ``steps.test.ok``, ``inputs.path``, ``items[0]``,
  ``steps['my-step'].output``
* operators (lowest to highest precedence): ``|`` filters, ``or``, ``and``,
  ``not``, comparisons ``== != < <= > >= in``/``not in``, ``+ -``, unary ``-``
* filters: ``value | tail(20)``, ``value | default('x')`` - see ``FILTERS``

Lookups only descend into dicts and lists (JSON-shaped data).  A missing
name or key raises :class:`ExprError` with a precise message instead of
silently evaluating to ``null``, so typos fail loudly - use the ``default``
filter with an explicit ``null`` value where a value may be absent.

Templates replace each ``{{ expr }}`` in a string.  A string that consists
of exactly one ``{{ expr }}`` evaluates to the raw value (a list stays a
list, a number a number); otherwise values are stringified (``null`` -> "",
lists/dicts -> JSON).
"""

from __future__ import annotations

import json
import re
import shlex
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Tuple


class ExprError(ValueError):
    """Raised for syntax errors and evaluation failures."""


# -- tokenizer ----------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>\d+(?:\.\d+)?)
  | (?P<str>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<op>==|!=|<=|>=|[<>()\[\].,|+\-])
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not", "in", "true", "false", "null", "True", "False", "None"}
_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"'}


def _unescape(body: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), "\\" + m.group(1)), body)


def _tokenize(src: str) -> List[Tuple[str, Any]]:
    tokens: List[Tuple[str, Any]] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ExprError(f"unexpected character {src[pos]!r} at position {pos} in {src!r}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind == "ws":
            continue
        if kind == "num":
            tokens.append(("lit", float(text) if "." in text else int(text)))
        elif kind == "str":
            tokens.append(("lit", _unescape(text[1:-1])))
        elif kind == "name" and text in _KEYWORDS:
            if text in ("true", "True"):
                tokens.append(("lit", True))
            elif text in ("false", "False"):
                tokens.append(("lit", False))
            elif text in ("null", "None"):
                tokens.append(("lit", None))
            else:
                tokens.append(("kw", text))
        elif kind == "name":
            tokens.append(("name", text))
        else:
            tokens.append(("op", text))
    tokens.append(("eof", None))
    return tokens


# -- parser (recursive descent -> tuple AST) ---------------------------------

class _Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.toks = _tokenize(src)
        self.i = 0

    def peek(self, offset: int = 0) -> Tuple[str, Any]:
        return self.toks[min(self.i + offset, len(self.toks) - 1)]

    def next(self) -> Tuple[str, Any]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def accept(self, kind: str, value: Any = None) -> bool:
        k, v = self.peek()
        if k == kind and (value is None or v == value):
            self.i += 1
            return True
        return False

    def expect(self, kind: str, value: Any = None) -> Any:
        k, v = self.peek()
        if k != kind or (value is not None and v != value):
            want = value if value is not None else kind
            got = "end of expression" if k == "eof" else repr(v)
            raise ExprError(f"expected {want!r} but found {got} in {self.src!r}")
        self.i += 1
        return v

    def parse(self):
        node = self.pipe()
        if self.peek()[0] != "eof":
            raise ExprError(f"unexpected {self.peek()[1]!r} in {self.src!r}")
        return node

    def pipe(self):
        node = self.or_()
        while self.accept("op", "|"):
            name = self.expect("name")
            args: List[Any] = []
            if self.accept("op", "("):
                if not self.accept("op", ")"):
                    args.append(self.or_())
                    while self.accept("op", ","):
                        args.append(self.or_())
                    self.expect("op", ")")
            if name not in FILTERS:
                raise ExprError(f"unknown filter {name!r} (available: {', '.join(sorted(FILTERS))})")
            node = ("filter", name, node, args)
        return node

    def or_(self):
        node = self.and_()
        while self.accept("kw", "or"):
            node = ("or", node, self.and_())
        return node

    def and_(self):
        node = self.not_()
        while self.accept("kw", "and"):
            node = ("and", node, self.not_())
        return node

    def not_(self):
        if self.accept("kw", "not"):
            return ("not", self.not_())
        return self.cmp()

    def cmp(self):
        node = self.add()
        k, v = self.peek()
        if k == "op" and v in ("==", "!=", "<", "<=", ">", ">="):
            self.next()
            return ("cmp", v, node, self.add())
        if k == "kw" and v == "in":
            self.next()
            return ("cmp", "in", node, self.add())
        if k == "kw" and v == "not" and self.peek(1) == ("kw", "in"):
            self.i += 2
            return ("cmp", "not in", node, self.add())
        return node

    def add(self):
        node = self.unary()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            node = ("arith", op, node, self.unary())
        return node

    def unary(self):
        if self.accept("op", "-"):
            return ("neg", self.unary())
        return self.postfix()

    def postfix(self):
        node = self.atom()
        while True:
            if self.accept("op", "."):
                node = ("get", node, ("lit", self.expect("name")))
            elif self.accept("op", "["):
                key = self.pipe()
                self.expect("op", "]")
                node = ("get", node, key)
            else:
                return node

    def atom(self):
        k, v = self.next()
        if k == "lit":
            return ("lit", v)
        if k == "name":
            return ("name", v)
        if k == "op" and v == "(":
            node = self.pipe()
            self.expect("op", ")")
            return node
        if k == "op" and v == "[":
            items = []
            if not self.accept("op", "]"):
                items.append(self.pipe())
                while self.accept("op", ","):
                    items.append(self.pipe())
                self.expect("op", "]")
            return ("list", items)
        got = "end of expression" if k == "eof" else repr(v)
        raise ExprError(f"unexpected {got} in {self.src!r}")


@lru_cache(maxsize=512)
def compile_expr(src: str):
    """Parse *src* into an AST (cached).  Raises :class:`ExprError`."""
    if not isinstance(src, str) or not src.strip():
        raise ExprError("empty expression")
    return _Parser(src).parse()


# -- evaluation ---------------------------------------------------------------

def _describe(node) -> str:
    kind = node[0]
    if kind == "name":
        return node[1]
    if kind == "get":
        key = node[2]
        if key[0] == "lit" and isinstance(key[1], str) and key[1].isidentifier():
            return f"{_describe(node[1])}.{key[1]}"
        return f"{_describe(node[1])}[...]"
    return "value"


def _eval(node, ctx: Mapping[str, Any]) -> Any:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "name":
        if node[1] not in ctx:
            raise ExprError(f"undefined name {node[1]!r} (available: {', '.join(sorted(ctx))})")
        return ctx[node[1]]
    if kind == "get":
        base = _eval(node[1], ctx)
        key = _eval(node[2], ctx)
        if isinstance(base, dict):
            if key not in base:
                avail = ", ".join(sorted(map(str, base))) or "none"
                raise ExprError(f"{_describe(node[1])} has no key {key!r} (available: {avail})")
            return base[key]
        if isinstance(base, (list, str)):
            if isinstance(base, str) and not isinstance(key, int):
                raise ExprError(f"cannot look up {key!r} in {_describe(node[1])} (string)")
            if isinstance(key, bool) or not isinstance(key, int):
                raise ExprError(f"{_describe(node[1])} index must be an integer, got {key!r}")
            try:
                return base[key]
            except IndexError:
                raise ExprError(f"{_describe(node[1])} index {key} out of range") from None
        raise ExprError(f"cannot look up {key!r} in {_describe(node[1])} ({_type(base)})")
    if kind == "and":
        left = _eval(node[1], ctx)
        return _eval(node[2], ctx) if left else left
    if kind == "or":
        left = _eval(node[1], ctx)
        return left if left else _eval(node[2], ctx)
    if kind == "not":
        return not _eval(node[1], ctx)
    if kind == "neg":
        val = _eval(node[1], ctx)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ExprError(f"cannot negate {_type(val)}")
        return -val
    if kind == "list":
        return [_eval(item, ctx) for item in node[1]]
    if kind == "cmp":
        return _compare(node[1], _eval(node[2], ctx), _eval(node[3], ctx))
    if kind == "arith":
        return _arith(node[1], _eval(node[2], ctx), _eval(node[3], ctx))
    if kind == "filter":
        value = _eval(node[2], ctx)
        args = [_eval(a, ctx) for a in node[3]]
        try:
            return FILTERS[node[1]](value, *args)
        except ExprError:
            raise
        except Exception as exc:
            raise ExprError(f"filter {node[1]!r} failed: {exc}") from None
    raise ExprError(f"bad node {kind!r}")  # pragma: no cover


def _type(v: Any) -> str:
    if v is None:
        return "null"
    return {bool: "boolean", int: "number", float: "number", str: "string",
            list: "list", dict: "object"}.get(type(v), type(v).__name__)


def _compare(op: str, a: Any, b: Any) -> bool:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op in ("in", "not in"):
        if isinstance(b, str) and isinstance(a, str):
            res = a in b
        elif isinstance(b, (list, dict)):
            res = a in b
        else:
            raise ExprError(f"'in' needs a string, list or object on the right, got {_type(b)}")
        return res if op == "in" else not res
    numeric = (int, float)
    if (isinstance(a, numeric) and isinstance(b, numeric)
            and not isinstance(a, bool) and not isinstance(b, bool)) or (
            isinstance(a, str) and isinstance(b, str)):
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    raise ExprError(f"cannot compare {_type(a)} {op} {_type(b)}")


def _arith(op: str, a: Any, b: Any) -> Any:
    numeric = (int, float)
    if (isinstance(a, numeric) and isinstance(b, numeric)
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return a + b if op == "+" else a - b
    if op == "+" and isinstance(a, str) and isinstance(b, str):
        return a + b
    if op == "+" and isinstance(a, list) and isinstance(b, list):
        return a + b
    raise ExprError(f"cannot apply {op!r} to {_type(a)} and {_type(b)}")


def evaluate(src: str, ctx: Mapping[str, Any]) -> Any:
    """Evaluate expression *src* against *ctx*."""
    return _eval(compile_expr(src), ctx)


def truthy(src: str, ctx: Mapping[str, Any]) -> bool:
    """Evaluate *src* as a condition."""
    return bool(evaluate(src, ctx))


# -- filters ------------------------------------------------------------------

def _lines(v: Any) -> List[str]:
    return to_text(v).splitlines()


def _tail(v: Any, n: int = 20) -> str:
    return "\n".join(_lines(v)[-int(n):]) if int(n) > 0 else ""


def _head(v: Any, n: int = 20) -> str:
    return "\n".join(_lines(v)[: int(n)])


def _length(v: Any) -> int:
    if isinstance(v, (str, list, dict)):
        return len(v)
    if v is None:
        return 0
    raise ExprError(f"length of {_type(v)} is undefined")


def _default(v: Any, fallback: Any = "") -> Any:
    return fallback if v is None or v == "" else v


def _to_int(v: Any) -> int:
    return int(float(v.strip())) if isinstance(v, str) else int(v)


def _from_json(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    text = v.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    return json.loads(text)


FILTERS: Dict[str, Callable[..., Any]] = {
    "length": _length,
    "lines": _lines,
    "head": _head,
    "tail": _tail,
    "trim": lambda v: to_text(v).strip(),
    "lower": lambda v: to_text(v).lower(),
    "upper": lambda v: to_text(v).upper(),
    "default": _default,
    "int": _to_int,
    "string": lambda v: to_text(v),
    "json": lambda v: json.dumps(v, ensure_ascii=False),
    "from_json": _from_json,
    "join": lambda v, sep=", ": sep.join(to_text(x) for x in v),
    "split": lambda v, sep=None: to_text(v).split(sep),
    "replace": lambda v, old, new: to_text(v).replace(old, new),
    "quote": lambda v: shlex.quote(to_text(v)),
}


# -- templating ---------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


def to_text(v: Any) -> str:
    """Stringify a value the way templates do."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render(template: Any, ctx: Mapping[str, Any]) -> Any:
    """Render ``{{ expr }}`` placeholders in *template*.

    Strings are rendered; lists and dicts are rendered recursively; other
    values are returned unchanged.  A string that is exactly one
    placeholder yields the raw (un-stringified) value.
    """
    if isinstance(template, dict):
        return {k: render(v, ctx) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, ctx) for v in template]
    if not isinstance(template, str) or "{{" not in template:
        return template
    whole = _TEMPLATE_RE.fullmatch(template.strip())
    if whole and "{{" not in whole.group(1):
        return evaluate(whole.group(1).strip(), ctx)
    return _TEMPLATE_RE.sub(lambda m: to_text(evaluate(m.group(1).strip(), ctx)), template)


def check_template(template: Any) -> None:
    """Syntax-check every placeholder in *template* (raises ExprError)."""
    if isinstance(template, dict):
        for v in template.values():
            check_template(v)
    elif isinstance(template, list):
        for v in template:
            check_template(v)
    elif isinstance(template, str):
        for m in _TEMPLATE_RE.finditer(template):
            compile_expr(m.group(1).strip())
