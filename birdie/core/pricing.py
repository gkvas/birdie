"""
Model pricing table and cost estimation.

Prices are USD per million tokens (input, output). The built-in table is
cached from the vendors' published list prices (2026-07) and is meant as a
reasonable default, not a source of truth - vendors change prices and add
models more often than birdie ships releases. Three layers are consulted,
highest priority first:

1. extra_overrides passed directly to price_for_model() / estimate_cost()
   (wired up from the "pricing" field of a provider config - see
   ProviderConfig.pricing in llm_provider.py).
2. A user-editable JSON file, by default ~/.birdie/pricing.json (override
   the path with BIRDIE_PRICING_FILE). Format:

       {
         "my-custom-model": [1.50, 6.00],
         "claude-sonnet-4-6": [2.50, 12.00]
       }

   Each value is [input USD/MTok, output USD/MTok]. A missing or
   malformed file is silently treated as empty - it never raises.
3. The built-in table below.

Lookups fall back to the longest matching prefix so dated model IDs (e.g.
claude-haiku-4-5-20251001) resolve to their family entry. Unknown models
return None - callers should render that as "pricing unknown" rather than
assuming zero cost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# model id (or family prefix) -> (input USD/MTok, output USD/MTok)
_BUILTIN_PRICES: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Mistral
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    # AWS Bedrock (foundation-model list prices; same models hosted
    # elsewhere may be priced under their native vendor entry above)
    "anthropic.claude-sonnet-4-20250514-v1:0": (3.00, 15.00),
    "anthropic.claude-opus-4-20250514-v1:0": (15.00, 75.00),
    "anthropic.claude-3-5-sonnet": (3.00, 15.00),
    "anthropic.claude-3-5-haiku": (0.80, 4.00),
    "amazon.nova-pro-v1:0": (0.80, 3.20),
    "amazon.nova-lite-v1:0": (0.06, 0.24),
    "amazon.nova-micro-v1:0": (0.035, 0.14),
    "meta.llama3-1-70b-instruct-v1:0": (0.72, 0.72),
    "mistral.mistral-large-2407-v1:0": (2.00, 6.00),
}

DEFAULT_PRICING_FILE = Path.home() / ".birdie" / "pricing.json"

# Cache of the parsed user pricing file, populated lazily on first lookup.
_user_overrides_cache: Optional[Dict[str, Tuple[float, float]]] = None


def _pricing_file_path() -> Path:
    override = os.environ.get("BIRDIE_PRICING_FILE")
    return Path(override) if override else DEFAULT_PRICING_FILE


def _normalize_overrides(raw: Optional[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Coerce a JSON-ish {model: [input, output]} mapping into
    {model: (float, float)}, silently dropping malformed entries."""
    result: Dict[str, Tuple[float, float]] = {}
    for model, value in (raw or {}).items():
        try:
            inp, out = value
            result[str(model)] = (float(inp), float(out))
        except (TypeError, ValueError):
            continue
    return result


def load_user_pricing_overrides(force_reload: bool = False) -> Dict[str, Tuple[float, float]]:
    """
    Load user-supplied pricing overrides from ~/.birdie/pricing.json (or
    the path in BIRDIE_PRICING_FILE).

    Cached after the first successful load; pass force_reload=True to
    re-read from disk (e.g. after the file was edited in a long-running
    process). A missing file, unreadable file, or invalid JSON is treated
    as "no overrides" rather than raising.
    """
    global _user_overrides_cache
    if _user_overrides_cache is not None and not force_reload:
        return _user_overrides_cache

    path = _pricing_file_path()
    if not path.is_file():
        _user_overrides_cache = {}
        return _user_overrides_cache

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        _user_overrides_cache = {}
        return _user_overrides_cache

    _user_overrides_cache = _normalize_overrides(raw)
    return _user_overrides_cache


def _effective_prices(
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Merge built-in table < user pricing file < caller-supplied overrides."""
    merged = dict(_BUILTIN_PRICES)
    merged.update(load_user_pricing_overrides())
    if extra_overrides:
        merged.update(_normalize_overrides(extra_overrides))
    return merged


def price_for_model(
    model: str,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Return (input, output) USD per million tokens, or None if unknown to
    any of the three pricing layers (see module docstring).
    """
    if not model:
        return None
    prices = _effective_prices(extra_overrides)
    if model in prices:
        return prices[model]
    # Longest-prefix match so dated/suffixed IDs resolve to their family.
    best = None
    for prefix, price in prices.items():
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, price)
    return best[1] if best else None


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Estimated USD cost for the given token counts, or None if unknown.

    Cache reads/writes are not modelled - treat the result as an upper-bound
    estimate for cached workloads. extra_overrides takes priority over both
    the user pricing file and the built-in table (see module docstring).
    """
    price = price_for_model(model, extra_overrides)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
