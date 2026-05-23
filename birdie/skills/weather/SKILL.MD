---
name: Weather
version: 1.0.0
description: Current weather and forecasts via weatherapi.com.
tags: [public, http, weather]
enabled_by_default: false
---

## Tools

### get_weather
description: Get current weather conditions for a city (temperature, humidity, wind, condition). Requires a free API key from weatherapi.com passed as `key`; omitting it returns an authentication error.
entrypoint: http:get https://api.weatherapi.com/v1/current.json
schema:
  type: object
  properties:
    city:
      type: string
      description: City name (e.g. "London", "New York", "Graz")
    key:
      type: string
      description: weatherapi.com API key
  required: [city, key]

### get_forecast
description: Get a multi-day weather forecast for a city (daily high/low, condition, precipitation chance). `days` must be 1–14, defaults to 3. Requires a weatherapi.com API key passed as `key`.
entrypoint: http:get https://api.weatherapi.com/v1/forecast.json
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
    key:
      type: string
      description: weatherapi.com API key
  required: [city, key]
