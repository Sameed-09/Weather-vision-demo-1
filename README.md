# WEATHER-VISION | SIH26079

Cleaned Round-2 prototype for AI-based forecast bust detection and early warning.

## What changed

- Separated dashboard, weather service, data generation and model training.
- Removed repeated subprocess/API calls from the dashboard.
- Added HTTP connection pooling, retries and short timeouts.
- Added cached geocoding and cached forecast retrieval.
- Replaced the original 10-row toy training set with a larger synthetic prototype dataset.
- Added a train/test split and evaluation metrics.
- Added model metadata.
- Removed dashboard emojis and development clutter.
- Added explicit prototype limitations around live uncertainty proxies.

## Project structure

```text
WEATHER_VISION_SIH_R2/
├── dashboard.py
├── weather_service.py
├── live_weather_api.py
├── generate_demo_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── assets/
│   ├── logo1.png
│   └── icon.png
└── data/
    └── processed_weather_data.csv
```

After training, these are also created:

```text
bust_model.pkl
model_metadata.json
```

## Windows setup

Open PowerShell in this folder:

```powershell
python -m pip install -r requirements.txt
python generate_demo_data.py
python train_model.py
python -m streamlit run dashboard.py
```

## Optional live API test

```powershell
python live_weather_api.py Mumbai
```

## Important scientific limitation

The prototype model is trained on synthetic forecast-error data. The live weather endpoint provides a forecast, not future observations, so the live dashboard converts forecast variability into uncertainty proxies.

For a research/production version, the core dataset should contain matched:

1. forecast issued time
2. forecast lead time
3. forecast value
4. observed value
5. forecast error
6. ensemble member/spread information where available

Then retrain the model against real forecast-bust events.

Do not present synthetic evaluation metrics as operational NCMRWF performance.
