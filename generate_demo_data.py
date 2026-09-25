from __future__ import annotations

import os

import numpy as np
import pandas as pd


RNG = np.random.default_rng(26079)

FEATURES = [
    "lead_day",
    "temperature_error",
    "rainfall_error",
    "wind_speed_error",
    "pressure_error",
    "humidity_error",
    "ensemble_spread",
]


def generate(n_samples: int = 6000) -> pd.DataFrame:
    lead = RNG.integers(1, 11, n_samples)

    # Correlated forecast-error distributions: uncertainty grows with lead time.
    temp = np.abs(
        RNG.normal(
            0.20 + 0.22 * lead,
            0.18 + 0.055 * lead,
            n_samples,
        )
    )

    rain = np.abs(
        RNG.gamma(
            shape=1.7,
            scale=0.75 + 0.45 * lead,
            size=n_samples,
        )
    )

    wind = np.abs(
        RNG.normal(
            0.8 + 0.42 * lead,
            0.45 + 0.10 * lead,
            n_samples,
        )
    )

    pressure = np.abs(
        RNG.normal(
            0.7 + 0.38 * lead,
            0.35 + 0.07 * lead,
            n_samples,
        )
    )

    humidity = np.abs(
        RNG.normal(
            1.5 + 0.85 * lead,
            0.7 + 0.16 * lead,
            n_samples,
        )
    )

    spread = np.abs(
        RNG.normal(
            0.55 + 0.48 * lead,
            0.25 + 0.08 * lead,
            n_samples,
        )
    )

    # Continuous bust severity gives a less brittle synthetic target.
    severity = (
        0.85 * (temp / 4.0)
        + 0.90 * (rain / 18.0)
        + 0.55 * (wind / 6.0)
        + 0.45 * (pressure / 6.0)
        + 0.45 * (humidity / 13.0)
        + 0.85 * (spread / 6.0)
        + 0.10 * lead
        + RNG.normal(0, 0.30, n_samples)
    )

    # Prototype operational definition; tune against real historical archives.
    bust = (severity > 2.55).astype(int)

    data = pd.DataFrame(
        {
            "lead_day": lead,
            "temperature_error": np.round(temp, 3),
            "rainfall_error": np.round(rain, 3),
            "wind_speed_error": np.round(wind, 3),
            "pressure_error": np.round(pressure, 3),
            "humidity_error": np.round(humidity, 3),
            "ensemble_spread": np.round(spread, 3),
            "bust": bust,
        }
    )

    return data


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    output = "data/processed_weather_data.csv"
    df.to_csv(output, index=False)

    print(f"Created {output}")
    print(f"Records: {len(df):,}")
    print(f"Bust rate: {df['bust'].mean() * 100:.1f}%")
