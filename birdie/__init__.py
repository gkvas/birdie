"""Birdie - a LangGraph agent that discovers capabilities at runtime from SKILL.MD files."""

import warnings

# Newer langgraph versions construct a ``JsonPlusSerializer`` (e.g. at import
# of ``langgraph.cache.base``) and emit a ``LangChainPendingDeprecationWarning``
# about the future default of ``allowed_objects``.  We never build that
# serializer ourselves and the parameter does not exist on the langgraph floor
# this package declares, so there is nothing for us to pass through - suppress
# the specific message.  We match on the message only (against the base
# ``Warning`` class) because the concrete warning subclass varies across
# langchain versions: it may descend from ``PendingDeprecationWarning`` or from
# ``DeprecationWarning``, and matching on the wrong base would miss it.
warnings.filterwarnings(
    "ignore",
    message=r".*`allowed_objects` will change.*",
)
