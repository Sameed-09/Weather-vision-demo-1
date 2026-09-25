from __future__ import annotations

import argparse
import os

import pandas as pd

from weather_service import fetch_forecast, get_location


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and save a 10-day Open-Meteo forecast."
    )
    parser.add_argument("city", nargs="?", default="Mumbai")
    parser.add_argument(
        "--output",
        default="live_weather_data.csv",
    )
    args = parser.parse_args()

    location = get_location(args.city)

    forecast = fetch_forecast(
        location["latitude"],
        location["longitude"],
        location["timezone"],
    )

    forecast.insert(0, "location", location["name"])
    forecast["latitude"] = location["latitude"]
    forecast["longitude"] = location["longitude"]

    forecast.to_csv(args.output, index=False)

    print(f"Location: {location['name']}")
    print(f"Coordinates: {location['latitude']}, {location['longitude']}")
    print(f"Rows: {len(forecast)}")
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
