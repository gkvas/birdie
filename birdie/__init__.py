"""Birdie - a LangGraph agent that discovers capabilities at runtime from SKILL.MD files."""

import warnings

# Newer langgraph versions construct a ``JsonPlusSerializer`` inside every
# checkpointer (MemorySaver, AsyncSqliteSaver, ...) and emit a
# ``LangChainPendingDeprecationWarning`` about the future default of
# ``allowed_objects``.  We never build that serializer ourselves and the
# parameter does not exist on the langgraph version this package pins against,
# so there is nothing for us to pass through - suppress the specific message.
# ``LangChainPendingDeprecationWarning`` subclasses ``PendingDeprecationWarning``,
# so we filter on that base class to avoid importing langchain at import time.
warnings.filterwarnings(
    "ignore",
    message=r".*default value of `allowed_objects` will change.*",
    category=PendingDeprecationWarning,
)
