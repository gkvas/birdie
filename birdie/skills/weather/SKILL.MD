---
name: Weather
version: 1.0.0
description: Current weather and forecasts via weatherapi.com. Requires WEATHERAPI_KEY environment variable.
tags: [public, http, weather]
enabled_by_default: false
---

## Tools

### get_weather
description: Get current weather conditions for a city (temperature, humidity, wind, condition text). The API key is read from the WEATHERAPI_KEY environment variable - no key argument needed.
entrypoint: python:birdie.skills.weather.tools.get_weather
schema:
  type: object
  properties:
    city:
      type: string
      description: City name (e.g. "London", "New York", "Graz")
  required: [city]

### get_forecast
description: Get a multi-day weather forecast for a city (daily high/low, condition, precipitation chance). `days` must be 1–14, defaults to 3. The API key is read from the WEATHERAPI_KEY environment variable - no key argument needed.
entrypoint: python:birdie.skills.weather.tools.get_forecast
schema:
  type: object
  properties:
    city:
      type: string
      description: City name (e.g. "London", "New York", "Graz")
    days:
      type: integer
      description: Number of days to forecast (1–14, default 3)
      default: 3
  required: [city]
