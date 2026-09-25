from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# API CONFIGURATION
# ============================================================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "lead_day",
    "temperature_error",
    "rainfall_error",
    "wind_speed_error",
    "pressure_error",
    "humidity_error",
    "ensemble_spread",
]


# ============================================================
# FAST HTTP SESSION
# ============================================================

def create_session() -> requests.Session:
    """
    Creates one reusable HTTP session.

    Connection pooling makes repeated API requests faster.
    Retries handle temporary API/network failures.
    """

    session = requests.Session()

    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.25,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "WEATHER-VISION-SIH26079/2.0"
            )
        }
    )

    return session


HTTP = create_session()


# ============================================================
# MAJOR INDIAN LOCATIONS
# ============================================================

CITIES = {

    # --------------------------------------------------------
    # Andhra Pradesh
    # --------------------------------------------------------

    "Amaravati": (16.574, 80.358),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.648),
    "Tirupati": (13.6288, 79.4192),
    "Guntur": (16.3067, 80.4365),
    "Nellore": (14.4426, 79.9865),
    "Kurnool": (15.8281, 78.0373),
    "Rajahmundry": (17.0005, 81.8040),
    "Kakinada": (16.9891, 82.2475),

    # --------------------------------------------------------
    # Arunachal Pradesh
    # --------------------------------------------------------

    "Itanagar": (27.0844, 93.6053),
    "Tawang": (27.5861, 91.8594),
    "Pasighat": (28.0660, 95.3260),
    "Bomdila": (27.2645, 92.4150),
    "Ziro": (27.5450, 93.8290),

    # --------------------------------------------------------
    # Assam
    # --------------------------------------------------------

    "Dispur": (26.1433, 91.7898),
    "Guwahati": (26.1445, 91.7362),
    "Dibrugarh": (27.4728, 94.9120),
    "Silchar": (24.8333, 92.7789),
    "Jorhat": (26.7500, 94.2167),
    "Tezpur": (26.6338, 92.8000),
    "Nagaon": (26.3500, 92.6833),

    # --------------------------------------------------------
    # Bihar
    # --------------------------------------------------------

    "Patna": (25.5941, 85.1376),
    "Gaya": (24.7914, 85.0002),
    "Bhagalpur": (25.2425, 86.9842),
    "Muzaffarpur": (26.1197, 85.3910),
    "Darbhanga": (26.1542, 85.8918),
    "Purnia": (25.7771, 87.4753),
    "Arrah": (25.5560, 84.6630),
    "Begusarai": (25.4182, 86.1272),

    # --------------------------------------------------------
    # Chhattisgarh
    # --------------------------------------------------------

    "Raipur": (21.2514, 81.6296),
    "Bilaspur": (22.0796, 82.1391),
    "Durg": (21.1904, 81.2849),
    "Jagdalpur": (19.0748, 82.0080),
    "Korba": (22.3595, 82.7501),
    "Rajnandgaon": (21.0971, 81.0284),
    "Ambikapur": (23.1189, 83.1954),

    # --------------------------------------------------------
    # Goa
    # --------------------------------------------------------

    "Panaji": (15.4909, 73.8278),
    "Margao": (15.2832, 73.9862),
    "Vasco da Gama": (15.3860, 73.8440),

    # --------------------------------------------------------
    # Gujarat
    # --------------------------------------------------------

    "Gandhinagar": (23.2156, 72.6369),
    "Ahmedabad": (23.0225, 72.5714),
    "Vadodara": (22.3072, 73.1812),
    "Surat": (21.1702, 72.8311),
    "Rajkot": (22.3039, 70.8022),
    "Bhavnagar": (21.7645, 72.1519),
    "Jamnagar": (22.4707, 70.0577),
    "Junagadh": (21.5222, 70.4579),
    "Bhuj": (23.2420, 69.6669),
    "Anand": (22.5645, 72.9289),
    "Bharuch": (21.7051, 72.9959),
    "Mehsana": (23.5880, 72.3693),

    # --------------------------------------------------------
    # Haryana
    # --------------------------------------------------------

    "Chandigarh": (30.7333, 76.7794),
    "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
    "Panipat": (29.3909, 76.9635),
    "Hisar": (29.1492, 75.7217),
    "Rohtak": (28.8955, 76.6066),
    "Ambala": (30.3782, 76.7767),
    "Karnal": (29.6857, 76.9905),
    "Sonipat": (28.9931, 77.0151),
    "Rewari": (28.1990, 76.6190),

    # --------------------------------------------------------
    # Himachal Pradesh
    # --------------------------------------------------------

    "Shimla": (31.1048, 77.1734),
    "Dharamshala": (32.2190, 76.3234),
    "Mandi": (31.5892, 76.9182),
    "Kullu": (31.9579, 77.1095),
    "Solan": (30.9045, 77.0967),
    "Chamba": (32.5553, 76.1265),
    "Nahan": (30.5600, 77.2940),

    # --------------------------------------------------------
    # Jharkhand
    # --------------------------------------------------------

    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    "Dhanbad": (23.7957, 86.4304),
    "Bokaro": (23.6693, 86.1511),
    "Deoghar": (24.4763, 86.6940),
    "Hazaribagh": (23.9925, 85.3637),
    "Giridih": (24.1850, 86.3000),

    # --------------------------------------------------------
    # Karnataka
    # --------------------------------------------------------

    "Bengaluru": (12.9716, 77.5946),
    "Mysuru": (12.2958, 76.6394),
    "Mangaluru": (12.9141, 74.8560),
    "Hubballi": (15.3647, 75.1240),
    "Dharwad": (15.4589, 75.0078),
    "Belagavi": (15.8497, 74.4977),
    "Kalaburagi": (17.3297, 76.8343),
    "Shivamogga": (13.9299, 75.5681),
    "Ballari": (15.1394, 76.9214),
    "Tumakuru": (13.3392, 77.1130),
    "Udupi": (13.3409, 74.7421),
    "Hassan": (13.0068, 76.1004),
    "Raichur": (16.2120, 77.3439),

    # --------------------------------------------------------
    # Kerala
    # --------------------------------------------------------

    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
    "Thrissur": (10.5276, 76.2144),
    "Kollam": (8.8932, 76.6141),
    "Kannur": (11.8745, 75.3704),
    "Alappuzha": (9.4981, 76.3388),
    "Kottayam": (9.5916, 76.5222),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0510, 76.0711),
    "Kasaragod": (12.4996, 74.9869),
    "Idukki": (9.8500, 76.9800),

    # --------------------------------------------------------
    # Madhya Pradesh
    # --------------------------------------------------------

    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Gwalior": (26.2183, 78.1828),
    "Jabalpur": (23.1815, 79.9864),
    "Ujjain": (23.1765, 75.7885),
    "Sagar": (23.8388, 78.7378),
    "Rewa": (24.5373, 81.3042),
    "Satna": (24.6005, 80.8322),
    "Ratlam": (23.3315, 75.0367),
    "Dewas": (22.9676, 76.0534),
    "Chhindwara": (22.0574, 78.9382),

    # --------------------------------------------------------
    # Maharashtra
    # --------------------------------------------------------

    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Kolhapur": (16.7050, 74.2433),
    "Nanded": (19.1383, 77.3210),
    "Amravati": (20.9374, 77.7796),
    "Solapur": (17.6599, 75.9064),
    "Akola": (20.7002, 77.0082),
    "Jalgaon": (21.0077, 75.5626),
    "Satara": (17.6805, 74.0183),
    "Latur": (18.4088, 76.5604),
    "Chandrapur": (19.9615, 79.2961),

    # --------------------------------------------------------
    # Manipur
    # --------------------------------------------------------

    "Imphal": (24.8170, 93.9368),
    "Churachandpur": (24.3333, 93.6833),
    "Thoubal": (24.6388, 94.0100),

    # --------------------------------------------------------
    # Meghalaya
    # --------------------------------------------------------

    "Shillong": (25.5788, 91.8933),
    "Tura": (25.5144, 90.2028),
    "Jowai": (25.4500, 92.2000),

    # --------------------------------------------------------
    # Mizoram
    # --------------------------------------------------------

    "Aizawl": (23.7271, 92.7176),
    "Lunglei": (22.8897, 92.7467),
    "Champhai": (23.4670, 93.3280),

    # --------------------------------------------------------
    # Nagaland
    # --------------------------------------------------------

    "Kohima": (25.6751, 94.1086),
    "Dimapur": (25.8620, 93.7533),
    "Mokokchung": (26.3250, 94.5167),

    # --------------------------------------------------------
    # Odisha
    # --------------------------------------------------------

    "Bhubaneswar": (20.2961, 85.8245),
    "Cuttack": (20.4625, 85.8830),
    "Rourkela": (22.2604, 84.8536),
    "Sambalpur": (21.4669, 83.9812),
    "Puri": (19.8135, 85.8312),
    "Balasore": (21.4942, 86.9316),
    "Berhampur": (19.3150, 84.7941),
    "Koraput": (18.8135, 82.7123),
    "Baripada": (21.9320, 86.7510),

    # --------------------------------------------------------
    # Punjab
    # --------------------------------------------------------

    "Ludhiana": (30.9010, 75.8573),
    "Amritsar": (31.6340, 74.8723),
    "Jalandhar": (31.3260, 75.5762),
    "Patiala": (30.3398, 76.3869),
    "Bathinda": (30.2110, 74.9455),
    "Pathankot": (32.2643, 75.6421),
    "Hoshiarpur": (31.5143, 75.9115),
    "Moga": (30.8160, 75.1720),

    # --------------------------------------------------------
    # Rajasthan
    # --------------------------------------------------------

    "Jaipur": (26.9124, 75.7873),
    "Jodhpur": (26.2389, 73.0243),
    "Udaipur": (24.5854, 73.7125),
    "Kota": (25.2138, 75.8648),
    "Ajmer": (26.4499, 74.6399),
    "Bikaner": (28.0229, 73.3119),
    "Alwar": (27.5530, 76.6346),
    "Bharatpur": (27.2152, 77.5030),
    "Jaisalmer": (26.9157, 70.9083),
    "Sikar": (27.6094, 75.1399),
    "Bhilwara": (25.3407, 74.6313),
    "Chittorgarh": (24.8887, 74.6269),
    "Barmer": (25.7521, 71.3967),

    # --------------------------------------------------------
    # Sikkim
    # --------------------------------------------------------

    "Gangtok": (27.3389, 88.6065),
    "Namchi": (27.1667, 88.3500),
    "Gyalshing": (27.2833, 88.2667),

    # --------------------------------------------------------
    # Tamil Nadu
    # --------------------------------------------------------

    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
    "Vellore": (12.9165, 79.1325),
    "Thanjavur": (10.7870, 79.1378),
    "Thoothukudi": (8.7642, 78.1348),
    "Dindigul": (10.3673, 77.9803),
    "Cuddalore": (11.7480, 79.7714),
    "Kanchipuram": (12.8342, 79.7036),
    "Ooty": (11.4102, 76.6950),

    # --------------------------------------------------------
    # Telangana
    # --------------------------------------------------------

    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9689, 79.5941),
    "Nizamabad": (18.6725, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Adilabad": (19.6641, 78.5320),
    "Mahbubnagar": (16.7488, 77.9850),
    "Siddipet": (18.1048, 78.8486),

    # --------------------------------------------------------
    # Tripura
    # --------------------------------------------------------

    "Agartala": (23.8315, 91.2868),
    "Udaipur Tripura": (23.5333, 91.4833),
    "Dharmanagar": (24.3667, 92.1667),

    # --------------------------------------------------------
    # Uttar Pradesh
    # --------------------------------------------------------

    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Prayagraj": (25.4358, 81.8463),
    "Meerut": (28.9845, 77.7064),
    "Bareilly": (28.3670, 79.4304),
    "Moradabad": (28.8386, 78.7733),
    "Gorakhpur": (26.7606, 83.3732),
    "Ayodhya": (26.7922, 82.1998),
    "Jhansi": (25.4484, 78.5685),
    "Aligarh": (27.8974, 78.0880),
    "Saharanpur": (29.9680, 77.5552),
    "Mathura": (27.4924, 77.6737),
    "Firozabad": (27.1591, 78.3957),
    "Noida": (28.5355, 77.3910),
    "Ghaziabad": (28.6692, 77.4538),
    "Mirzapur": (25.1337, 82.5644),
    "Bareilly": (28.3670, 79.4304),

    # --------------------------------------------------------
    # Uttarakhand
    # --------------------------------------------------------

    "Dehradun": (30.3165, 78.0322),
    "Haridwar": (29.9457, 78.1642),
    "Nainital": (29.3919, 79.4542),
    "Almora": (29.5971, 79.6591),
    "Haldwani": (29.2183, 79.5130),
    "Pauri": (30.1500, 78.7800),
    "Chamoli": (30.4030, 79.3290),
    "Uttarkashi": (30.7268, 78.4354),
    "Rudrapur": (28.9845, 79.4000),

    # --------------------------------------------------------
    # West Bengal
    # --------------------------------------------------------

    "Kolkata": (22.5726, 88.3639),
    "Siliguri": (26.7271, 88.3953),
    "Durgapur": (23.5204, 87.3119),
    "Asansol": (23.6739, 87.1705),
    "Darjeeling": (27.0410, 88.2663),
    "Howrah": (22.5958, 88.2636),
    "Malda": (25.0108, 88.1411),
    "Kharagpur": (22.3460, 87.2320),
    "Jalpaiguri": (26.5167, 88.7333),

    # ========================================================
    # UNION TERRITORIES
    # ========================================================

    "New Delhi": (28.6139, 77.2090),
    "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570),
    "Leh": (34.1526, 77.5771),
    "Kavaratti": (10.5669, 72.6420),
    "Puducherry": (11.9416, 79.8083),
    "Port Blair": (11.6234, 92.7265),
    "Daman": (20.3974, 72.8328),
    "Diu": (20.7144, 70.9874),
}


# ============================================================
# LOCATION LOOKUP
# ============================================================

def get_location(city: str) -> Dict:
    """
    Returns coordinates for a known major Indian location.

    If the location isn't in the curated list, it automatically
    falls back to the geocoding API.
    """

    city = city.strip()

    if not city:
        raise ValueError(
            "Location cannot be empty."
        )

    if city in CITIES:

        latitude, longitude = CITIES[city]

        return {
            "name": city,
            "country": "India",
            "country_code": "IN",
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
        }

    return geocode_location(city)


# ============================================================
# GEOCODING
# ============================================================

@lru_cache(maxsize=128)
def geocode_location(
    query: str,
) -> Dict:
    """
    Convert a city/town name into coordinates.

    Cached so repeated searches don't repeatedly call
    the geocoding service.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Location cannot be empty."
        )

    response = HTTP.get(
        GEOCODING_URL,
        params={
            "name": query,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=(2.0, 5.0),
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        [],
    )

    if not results:
        raise ValueError(
            f"Location not found: {query}"
        )

    # Prefer India.
    indian_results = [
        item
        for item in results
        if item.get("country_code") == "IN"
    ]

    item = (
        indian_results[0]
        if indian_results
        else results[0]
    )

    return {
        "name": item.get(
            "name",
            query,
        ),
        "country": item.get(
            "country",
            "India",
        ),
        "country_code": item.get(
            "country_code",
            "IN",
        ),
        "admin1": item.get(
            "admin1",
            "",
        ),
        "admin2": item.get(
            "admin2",
            "",
        ),
        "latitude": float(
            item["latitude"]
        ),
        "longitude": float(
            item["longitude"]
        ),
        "timezone": item.get(
            "timezone",
            "auto",
        ),
    }


# ============================================================
# SEARCH INDIAN CITIES / TOWNS
# ============================================================

@lru_cache(maxsize=256)
def search_indian_locations(
    query: str,
    count: int = 20,
) -> List[Dict]:
    """
    Search Indian cities, towns and district locations.

    This is the function dashboard.py imports.

    It uses Open-Meteo geocoding and filters the results
    to India.
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
        timeout=(2.0, 5.0),
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        [],
    )

    locations = []

    for result in results:

        if result.get("country_code") != "IN":
            continue

        if "latitude" not in result:
            continue

        if "longitude" not in result:
            continue

        locations.append(
            {
                "name": result.get(
                    "name",
                    query,
                ),
                "country": result.get(
                    "country",
                    "India",
                ),
                "country_code": "IN",
                "admin1": result.get(
                    "admin1",
                    "",
                ),
                "admin2": result.get(
                    "admin2",
                    "",
                ),
                "latitude": float(
                    result["latitude"]
                ),
                "longitude": float(
                    result["longitude"]
                ),
                "timezone": result.get(
                    "timezone",
                    "auto",
                ),
            }
        )

    # --------------------------------------------------------
    # Also check the curated list.
    # --------------------------------------------------------

    query_lower = query.lower()

    curated_matches = []

    for name, coordinates in CITIES.items():

        if query_lower in name.lower():

            latitude, longitude = coordinates

            curated_matches.append(
                {
                    "name": name,
                    "country": "India",
                    "country_code": "IN",
                    "admin1": "",
                    "admin2": "",
                    "latitude": latitude,
                    "longitude": longitude,
                    "timezone": "auto",
                }
            )

    # Curated locations first.
    combined = (
        curated_matches
        + locations
    )

    # Remove duplicate city/coordinate results.
    unique = []
    seen = set()

    for location in combined:

        key = (
            location["name"].lower(),
            round(
                location["latitude"],
                3,
            ),
            round(
                location["longitude"],
                3,
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(location)

    return unique[:count]


# ============================================================
# WEATHER FORECAST
# ============================================================

@lru_cache(maxsize=128)
def _fetch_forecast_cached(
    latitude: float,
    longitude: float,
    timezone: str,
) -> pd.DataFrame:

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "relative_humidity_2m_mean,"
            "precipitation_sum,"
            "precipitation_probability_max,"
            "wind_speed_10m_max,"
            "surface_pressure_mean"
        ),
        "forecast_days": 10,
        "timezone": timezone,
    }

    response = HTTP.get(
        FORECAST_URL,
        params=params,
        timeout=(2.0, 6.0),
    )

    response.raise_for_status()

    payload = response.json()

    daily = payload.get(
        "daily"
    )

    if not daily:
        raise ValueError(
            "Weather API returned no daily forecast."
        )

    frame = pd.DataFrame(
        {
            "date": daily.get(
                "time",
                [],
            ),
            "temperature_max": daily.get(
                "temperature_2m_max",
                [],
            ),
            "temperature_min": daily.get(
                "temperature_2m_min",
                [],
            ),
            "humidity": daily.get(
                "relative_humidity_2m_mean",
                [],
            ),
            "rainfall": daily.get(
                "precipitation_sum",
                [],
            ),
            "rain_probability": daily.get(
                "precipitation_probability_max",
                [],
            ),
            "wind_speed": daily.get(
                "wind_speed_10m_max",
                [],
            ),
            "pressure": daily.get(
                "surface_pressure_mean",
                [],
            ),
        }
    )

    if frame.empty:
        raise ValueError(
            "Weather API returned an empty forecast."
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    numeric_columns = [
        "temperature_max",
        "temperature_min",
        "humidity",
        "rainfall",
        "rain_probability",
        "wind_speed",
        "pressure",
    ]

    for column in numeric_columns:

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "date",
            "temperature_max",
            "temperature_min",
        ]
    ).reset_index(
        drop=True
    )

    return frame


def fetch_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Public weather function used by dashboard.py.
    """

    return _fetch_forecast_cached(
        round(
            float(latitude),
            4,
        ),
        round(
            float(longitude),
            4,
        ),
        timezone,
    ).copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _absolute_change(
    series: pd.Series,
) -> pd.Series:

    return (
        series.diff()
        .abs()
        .fillna(0.0)
    )


def build_live_features(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds uncertainty proxy features for live inference.

    Important:
    These are NOT actual future forecast errors.

    A production SIH system should replace these with
    forecast-vs-observation error archives and ensemble
    spread data.
    """

    frame = forecast.copy()

    lead = np.arange(
        1,
        len(frame) + 1,
        dtype=float,
    )

    temperature_range = (
        frame["temperature_max"]
        - frame["temperature_min"]
    ).abs()

    temperature_change = _absolute_change(
        frame["temperature_max"]
    )

    rainfall_change = _absolute_change(
        frame["rainfall"]
    )

    wind_change = _absolute_change(
        frame["wind_speed"]
    )

    pressure_change = _absolute_change(
        frame["pressure"]
    )

    humidity_change = _absolute_change(
        frame["humidity"]
    )

    rain_probability = (
        frame["rain_probability"]
        .fillna(0)
        .clip(
            0,
            100,
        )
    )

    wind_speed = (
        frame["wind_speed"]
        .fillna(0)
        .clip(
            lower=0
        )
    )

    # --------------------------------------------------------
    # Uncertainty proxies
    # --------------------------------------------------------

    temperature_error = (
        0.30
        + 0.11 * lead
        + 0.16 * temperature_range
        + 0.28 * temperature_change
    )

    rainfall_error = (
        0.80
        + 0.09 * lead
        + 0.045 * rain_probability
        + 0.65 * rainfall_change
    )

    wind_speed_error = (
        0.70
        + 0.09 * lead
        + 0.16 * wind_speed
        + 0.22 * wind_change
    )

    pressure_error = (
        0.60
        + 0.07 * lead
        + 0.30 * pressure_change
    )

    humidity_error = (
        1.50
        + 0.40 * lead
        + 0.07 * humidity_change
    )

    ensemble_spread = (
        0.45
        + 0.24 * lead
        + 0.13 * temperature_range
        + 0.18 * wind_change
        + 0.025 * rain_probability
    )

    features = pd.DataFrame(
        {
            "lead_day": lead,
            "temperature_error": temperature_error,
            "rainfall_error": rainfall_error,
            "wind_speed_error": wind_speed_error,
            "pressure_error": pressure_error,
            "humidity_error": humidity_error,
            "ensemble_spread": ensemble_spread,
        }
    )

    features = (
        features
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
        .clip(
            lower=0
        )
    )

    return features


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def risk_band(
    probability: float,
) -> str:

    probability = float(
        probability
    )

    if probability >= 70:
        return "HIGH"

    if probability >= 40:
        return "MEDIUM"

    return "LOW"