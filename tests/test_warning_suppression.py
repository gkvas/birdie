"""
Regression test: the langgraph `allowed_objects` pending-deprecation warning
must stay suppressed even though ``langchain_core`` re-surfaces its deprecation
warnings (prepending a ``"default"`` filter) when it is imported after birdie.

Runs in a clean subprocess so it is unaffected by pytest's per-test warning
filter handling and reproduces the real interpreter import order.
"""

import subprocess
import sys


def test_allowed_objects_warning_stays_suppressed():
    code = (
        "import warnings\n"
        "import birdie  # registers our ignore filter (after importing langchain_core)\n"
        "import langchain_core  # its surface_*() prepends a 'default' filter\n"
        "from langchain_core._api.deprecation import "
        "LangChainPendingDeprecationWarning as W\n"
        "with warnings.catch_warnings(record=True) as rec:\n"
        "    warnings.warn('The default value of `allowed_objects` will change in "
        "a future version.', W)\n"
        "    hits = [w for w in rec if 'allowed_objects' in str(w.message)]\n"
        "print('HITS', len(hits))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "HITS 0" in result.stdout, (
        f"warning not suppressed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
