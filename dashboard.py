from __future__ import annotations

from functools import lru_cache
import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


from weather_service import (
    FEATURES,
    GEOCODING_URL,
    HTTP,
    build_live_features,
    fetch_forecast,
    geocode_location,
    search_indian_locations,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bust_model.pkl")
DEMO_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_weather_data.csv")

st.set_page_config(
    page_title="WEATHER-VISION | SIH26079",
    page_icon=os.path.join(BASE_DIR, "assets", "icon.png")
    if os.path.exists(os.path.join(BASE_DIR, "assets", "icon.png"))
    else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Theme
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 16px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: .02em;
    }
    .hero p {
        margin: .25rem 0 0;
        color: #6b7280;
    }
    .status {
        padding: .55rem .75rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,.20);
        font-size: .9rem;
    }
    .muted {
        color: #6b7280;
        font-size: .86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
FEATURES = [
    "lead_day",
    "temperature_error",
    "rainfall_error",
    "wind_speed_error",
    "pressure_error",
    "humidity_error",
    "ensemble_spread",
]


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_demo_data():
    if not os.path.exists(DEMO_DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DEMO_DATA_PATH)


def model_probability(model, feature_frame):
    if model is None:
        return None
    try:
        return float(model.predict_proba(feature_frame[FEATURES])[:, 1][0] * 100)
    except Exception:
        return None


def risk_message(probability):
    if probability >= 70:
        return "HIGH"
    if probability >= 40:
        return "MEDIUM"
    return "LOW"


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------


st.markdown(
    """
    <div class="hero">
        <h1>WEATHER-VISION</h1>
        <p>AI-Based Forecast Bust Detection and Early Warning</p>
        <div class="muted">SIH26079 | NCMRWF / Ministry of Earth Sciences</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Model status
# ------------------------------------------------------------
model = load_model()


if model is None:
    st.warning(
        "The trained model is not available. Run train_model.py before the final demo."
    )

# ------------------------------------------------------------
# Historical / uploaded forecast-risk analysis
# ------------------------------------------------------------
st.header("Forecast Risk Analysis")
st.caption(
    "Use a forecast-risk CSV containing the seven model features for historical "
    "or research-data analysis."
)

uploaded = st.file_uploader("Upload forecast-risk CSV", type=["csv"])
data = pd.read_csv(uploaded) if uploaded is not None else load_demo_data()

if not data.empty and model is not None and all(c in data.columns for c in FEATURES):
    analysis = data.copy()
    analysis["bust_probability"] = (
        model.predict_proba(analysis[FEATURES])[:, 1] * 100
    )
    analysis["risk"] = analysis["bust_probability"].apply(risk_message)

    highest = analysis.loc[analysis["bust_probability"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest Risk Day", f"D{int(highest['lead_day'])}")
    c2.metric("Maximum AI Risk", f"{highest['bust_probability']:.1f}%")
    c3.metric("Risk Band", highest["risk"])
  

    chart = px.bar(
        analysis,
        x="lead_day",
        y="bust_probability",
        labels={
            "lead_day": "Forecast Day",
            "bust_probability": "AI Risk (%)",
        },
        title="AI Forecast-Bust Risk by Lead Day",
    )
    chart.update_yaxes(range=[0, 100])
    st.plotly_chart(chart, use_container_width=True)

    if highest["risk"] == "HIGH":
        st.error("HIGH RISK: The model identifies a strong forecast-bust signal.")
    elif highest["risk"] == "MEDIUM":
        st.warning("MEDIUM RISK: The model identifies elevated forecast uncertainty.")
    else:
        st.success("LOW RISK: No high-risk signal was identified in the supplied data.")

    with st.expander("Model input details"):
        details = analysis[
            FEATURES + ["bust_probability", "risk"]
        ].copy()
        details.columns = [
            "Lead Day",
            "Temperature Error",
            "Rainfall Error",
            "Wind Error",
            "Pressure Error",
            "Humidity Error",
            "Ensemble Spread",
            "AI Risk (%)",
            "Risk",
        ]
        details["AI Risk (%)"] = details["AI Risk (%)"].round(1)
        st.dataframe(details, use_container_width=True, hide_index=True)

elif not data.empty:
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        st.info(
            "The supplied dataset is missing model fields: "
            + ", ".join(missing)
        )

# ------------------------------------------------------------
# Live weather
# ------------------------------------------------------------
st.divider()
st.header("Live Weather Monitoring")
st.caption(
    "Open-Meteo forecast data is cached for 10 minutes. A manual refresh is "
    "available when a fresh request is required."
)

@lru_cache(maxsize=128)
def search_indian_locations(query: str, count: int = 15):
    """
    Search Indian cities/towns using Open-Meteo geocoding.

    This allows the dashboard to find locations that are not in the
    curated city dictionary.
    """
    query = query.strip()

    if len(query) < 2:
        return []

    response = HTTP.get(
        GEOCODING_URL,
        params={
            "name": query,
            "count": count,
            "language": "en",
            "format": "json",
        },
        timeout=(2.0, 4.0),
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    locations = []

    for item in results:
        country_code = item.get("country_code", "")

        # Keep India only.
        if country_code != "IN":
            continue

        locations.append(
            {
                "name": item.get("name", query),
                "admin1": item.get("admin1", ""),
                "admin2": item.get("admin2", ""),
                "country": item.get("country", "India"),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "timezone": item.get("timezone", "auto"),
            }
        )

    return locations
from weather_service import (
    CITIES,
    build_live_features,
    fetch_forecast,
    get_location,
    risk_band,
    search_indian_locations,
)
st.subheader("Select Location")

search_query = st.text_input(
    "Search city or town",
    placeholder="Type a city, town or district headquarters...",
)

# ------------------------------------------------------------
# Search mode
# ------------------------------------------------------------
if search_query.strip():

    try:
        search_results = search_indian_locations(search_query)

    except Exception:
        search_results = []

    if search_results:

        location_labels = []

        for item in search_results:

            parts = [item["name"]]

            if item.get("admin2"):
                parts.append(item["admin2"])

            if item.get("admin1"):
                parts.append(item["admin1"])

            location_labels.append(", ".join(parts))

        selected_label = st.selectbox(
            "Matching locations",
            location_labels,
        )

        selected_index = location_labels.index(selected_label)

        selected_location = search_results[selected_index]

        selected_city = selected_location["name"]

    else:

        st.warning(
            "No matching Indian location found. "
            "Try another spelling or a nearby district headquarters."
        )

        selected_location = None
        selected_city = None

# ------------------------------------------------------------
# Browse mode
# ------------------------------------------------------------
else:

    city_options = sorted(CITIES.keys())

    selected_city = st.selectbox(
        "Browse major Indian locations",
        city_options,
        index=city_options.index("Mumbai"),
    )

    selected_location = get_location(selected_city)

if "live_forecast" not in st.session_state:
    st.session_state.live_forecast = None
    st.session_state.live_city = None
    st.session_state.live_error = None
    st.session_state.live_time = None

fetch_col, info_col = st.columns([1, 4])

with fetch_col:
    refresh = st.button("Refresh live weather", type="primary")

location = selected_location

if location:
    try:
        st.session_state.live_forecast = fetch_forecast(
            location["latitude"],
            location["longitude"],
            location["timezone"],
        )

        st.session_state.live_city = selected_city
        st.session_state.live_error = None

    except Exception as exc:
        st.session_state.live_error = str(exc)
        st.session_state.live_city = selected_city
        st.session_state.live_error = None
        st.session_state.live_time = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    except Exception as exc:
        st.session_state.live_error = str(exc)

with info_col:
    if st.session_state.live_error:
        st.error("Live weather request failed. No fabricated live values are shown.")
    elif st.session_state.live_forecast is not None:
        st.success(
            f"Live forecast loaded for {selected_city} | "
            f"Last request: {st.session_state.live_time}"
        )

forecast = st.session_state.live_forecast

if forecast is not None:
    current = forecast.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Location", selected_city)
    c2.metric("Max Temp", f"{current['temperature_max']:.1f} °C")
    c3.metric("Humidity", f"{current['humidity']:.0f}%")
    c4.metric("Rain Probability", f"{current['rain_probability']:.0f}%")
    c5.metric("Wind", f"{current['wind_speed']:.1f} km/h")

    display = forecast.copy()
    display["Date"] = display["date"].dt.strftime("%d %b")
    display.insert(0, "Day", [f"D{i}" for i in range(1, len(display) + 1)])
    display = display[
        [
            "Day",
            "Date",
            "temperature_max",
            "temperature_min",
            "rain_probability",
            "rainfall",
            "wind_speed",
            "humidity",
            "pressure",
        ]
    ]
    display.columns = [
        "Day",
        "Date",
        "Max Temp (°C)",
        "Min Temp (°C)",
        "Rain Probability (%)",
        "Rainfall (mm)",
        "Wind (km/h)",
        "Humidity (%)",
        "Pressure (hPa)",
    ]
    for col in display.columns[2:]:
        display[col] = display[col].round(1)

    st.subheader(f"{selected_city} | 10-Day Forecast")
    st.dataframe(display, use_container_width=True, hide_index=True)

    tfig = px.line(
        display,
        x="Day",
        y="Max Temp (°C)",
        markers=True,
        title=f"{selected_city} | Temperature Trend",
    )
    st.plotly_chart(tfig, use_container_width=True)

    rfig = px.bar(
        display,
        x="Day",
        y="Rain Probability (%)",
        title=f"{selected_city} | Rain Probability",
    )
    rfig.update_yaxes(range=[0, 100])
    st.plotly_chart(rfig, use_container_width=True)

    # --------------------------------------------------------
    # Live AI risk
    # --------------------------------------------------------
    st.subheader("AI Early-Warning Layer")
    live_features = build_live_features(forecast)

    selected_day = st.slider(
        "Forecast horizon",
        min_value=1,
        max_value=len(live_features),
        value=min(3, len(live_features)),
    )

    row = live_features.iloc[[selected_day - 1]]
    probability = model_probability(model, row)

    if probability is None:
        st.info(
            "AI inference is unavailable. Live weather monitoring is still active."
        )
    else:
        band = risk_band(probability)
        a1, a2, a3 = st.columns(3)
        a1.metric("AI Risk Score", f"{probability:.1f}%")
        a2.metric("Risk Band", band)
        a3.metric("Forecast Horizon", f"D{selected_day}")

        if band == "HIGH":
            st.error(
                "HIGH RISK: The prototype AI layer indicates elevated forecast-bust risk."
            )
        elif band == "MEDIUM":
            st.warning(
                "MEDIUM RISK: Forecast uncertainty is elevated; continue monitoring."
            )
        else:
            st.success(
                "LOW RISK: No elevated bust signal is detected by the prototype AI layer."
            )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()


st.caption(
    "WEATHER-VISION | SIH26079 |"
    "Predictions are research/prototype outputs and should not be treated as official forecasts."
)
