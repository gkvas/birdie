"""Birdie - a LangGraph agent that discovers capabilities at runtime from SKILL.MD files."""

import warnings

# Newer langgraph versions construct a ``JsonPlusSerializer`` (e.g. at import of
# ``langgraph.cache.base``) and emit a ``LangChainPendingDeprecationWarning``
# about the future default of ``allowed_objects``.  We never build that
# serializer ourselves and the parameter does not exist on the langgraph floor
# this package declares, so there is nothing for us to pass through - suppress
# the message.
#
# Ordering matters: ``langchain_core/__init__.py`` calls
# ``surface_langchain_deprecation_warnings()`` at import time, which *prepends* a
# ``"default"`` filter for its deprecation-warning classes.  Because
# ``filterwarnings`` prepends, a plain filter registered here would be pushed
# behind langchain's once langchain is imported (which happens later, when the
# CLI loads), and the warning would resurface.  So we import ``langchain_core``
# first to let it install its filters, then register ours last so it stays in
# front.  We match on the message only (against the base ``Warning`` class)
# because the concrete subclass varies across langchain versions.
try:  # langchain_core is a hard dependency, but stay defensive
    import langchain_core  # noqa: F401
except Exception:  # pragma: no cover - only if langchain_core is unavailable
    pass

warnings.filterwarnings(
    "ignore",
    message=r".*`allowed_objects` will change.*",
)
