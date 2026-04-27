---
name: Weather
version: 1.0.0
description: Provides weather lookup capabilities via an external API.
tags: [public, http, weather]
enabled_by_default: true
---

# Skill: Weather

## Tools

### get_weather
description: Get current weather for a given city.
entrypoint: http:get https://api.weatherapi.com/v1/current.json
schema:
  type: object
  properties:
    city:
      type: string
      description: The city name to get weather for
    key:
      type: string
      description: API key for weatherapi.com (optional)
  required: [city]

### get_forecast
description: Get weather forecast for a given city.
entrypoint: http:get https://api.weatherapi.com/v1/forecast.json
schema:
  type: object
  properties:
    city:
      type: string
      description: The city name to get forecast for
    days:
      type: integer
      description: Number of days to forecast (1-14)
      default: 3
    key:
      type: string
      description: API key for weatherapi.com (optional)
  required: [city]

## Permissions
- outbound_http

## Examples
User: "What's the weather in Graz?"
Tool: get_weather(city="Graz")

User: "What's the 5-day forecast for Berlin?"
Tool: get_forecast(city="Berlin", days=5)
