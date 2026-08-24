"""
Model pricing table and cost estimation.

Prices are USD per million tokens (input, output), cached from the vendors'
published list prices (2026-07).  Lookups fall back to the longest matching
prefix so dated model IDs (e.g. ``claude-haiku-4-5-20251001``) resolve to
their family entry.  Unknown models return ``None`` - callers should render
that as "pricing unknown" rather than assuming zero cost.
"""

from __future__ import annotations

from typing import Optional, Tuple

# model id (or family prefix) -> (input USD/MTok, output USD/MTok)
_PRICES: dict[str, Tuple[float, float]] = {
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


def price_for_model(model: str) -> Optional[Tuple[float, float]]:
    """Return (input, output) USD per million tokens, or None if unknown."""
    if not model:
        return None
    if model in _PRICES:
        return _PRICES[model]
    # Longest-prefix match so dated/suffixed IDs resolve to their family.
    best = None
    for prefix, price in _PRICES.items():
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, price)
    return best[1] if best else None


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int,
) -> Optional[float]:
    """Estimated USD cost for the given token counts, or None if unknown.

    Cache reads/writes are not modelled - treat the result as an upper-bound
    estimate for cached workloads.
    """
    price = price_for_model(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
