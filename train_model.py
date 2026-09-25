from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURES = [
    "lead_day",
    "temperature_error",
    "rainfall_error",
    "wind_speed_error",
    "pressure_error",
    "humidity_error",
    "ensemble_spread",
]

DATA_PATH = "data/processed_weather_data.csv"
MODEL_PATH = "bust_model.pkl"
METADATA_PATH = "model_metadata.json"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run generate_demo_data.py first."
        )

    data = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + ["bust"] if c not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    if data["bust"].nunique() < 2:
        raise ValueError("Training data must contain both bust and non-bust classes.")

    X = data[FEATURES]
    y = data["bust"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=26079,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=26079,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "total_records": int(len(data)),
        "bust_rate": round(float(y.mean()), 4),
        "features": FEATURES,
        "model": "RandomForestClassifier",
        "prototype_note": (
            "Synthetic training data. Replace with matched NWP forecast-"
            "observation archives before operational deployment."
        ),
    }

    joblib.dump(model, MODEL_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nWEATHER-VISION MODEL TRAINING")
    print("=" * 36)
    print(f"Records:  {len(data):,}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1:       {metrics['f1']:.3f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.3f}")
    print(f"Saved:    {MODEL_PATH}")
    print(f"Saved:    {METADATA_PATH}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=3))


if __name__ == "__main__":
    main()
