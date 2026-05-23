"""
Weather skill tool implementations.

Reads the weatherapi.com API key from the WEATHERAPI_KEY environment variable.
Get a free key at https://www.weatherapi.com/signup.aspx
"""

import os
import requests
from typing import Any


def _api_key() -> str:
    key = os.environ.get("WEATHERAPI_KEY", "")
    if not key:
        raise RuntimeError(
            "WEATHERAPI_KEY environment variable is not set. "
            "Get a free key at https://www.weatherapi.com/signup.aspx"
        )
    return key


def get_weather(city: str) -> Any:
    """Get current weather conditions for a city."""
    response = requests.get(
        "https://api.weatherapi.com/v1/current.json",
        params={"key": _api_key(), "q": city},
    )
    response.raise_for_status()
    return response.json()


def get_forecast(city: str, days: int = 3) -> Any:
    """Get a multi-day weather forecast for a city."""
    response = requests.get(
        "https://api.weatherapi.com/v1/forecast.json",
        params={"key": _api_key(), "q": city, "days": days},
    )
    response.raise_for_status()
    return response.json()
