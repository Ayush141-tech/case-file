"""
CASEFILE — Shared utilities for the Missing Person Location Prediction System.
================================================================================
This module holds the feature-engineering logic and column definitions shared
between the model builder (build_model.py) and the web UI (app.py), so both
apply exactly the same transformations to raw case records.

All data used by this project is 100% synthetic / fictional.
"""

import numpy as np
import pandas as pd

# Columns dropped from the feature matrix (identifiers / raw timestamp / target)
DROP_COLS = ["Case_ID", "Person_ID", "Last_Seen_Time", "Target_Area"]

# Categorical columns one-hot encoded by the model preprocessor
CAT_COLS = ["Age_Group", "Gender", "Day", "Weather", "Usual_Area", "Previous_Area"]


def engineer_features(input_df):
    """
    Transform raw case columns into meaningful predictors without leaking
    target information.

    - Converts Last_Seen_Time into a cyclical hour encoding (Hour_Sin/Cos) so
      midnight and 1 AM are treated as close together.
    - Computes movement ratios (Speed_Distance_Ratio, Movement_Intensity).
    - Flags whether the person's usual area matches their previous area.
    """
    df_clean = input_df.copy()

    # Parse timestamp (HH:MM) and extract the hour; default to 12:00 if missing
    df_clean["Last_Seen_Time"] = pd.to_datetime(df_clean["Last_Seen_Time"], errors="coerce")
    df_clean["Last_Seen_Hour"] = df_clean["Last_Seen_Time"].dt.hour.fillna(12)

    # Cyclical hour encoding (handles midnight wrap-around mathematically)
    df_clean["Hour_Sin"] = np.sin(2 * np.pi * df_clean["Last_Seen_Hour"] / 24.0)
    df_clean["Hour_Cos"] = np.cos(2 * np.pi * df_clean["Last_Seen_Hour"] / 24.0)

    # Movement & spatial interactions
    df_clean["Speed_Distance_Ratio"] = df_clean["Average_Speed"] / (df_clean["Average_Distance"] + 1e-5)
    df_clean["Movement_Intensity"] = df_clean["Average_Speed"] * df_clean["Time_Since_Last_Seen"]
    df_clean["Is_Usual_Prev_Match"] = (df_clean["Usual_Area"] == df_clean["Previous_Area"]).astype(int)

    return df_clean


def get_feature_cols(df_engineered):
    """Return the ordered list of model feature columns from an engineered frame."""
    return [c for c in df_engineered.columns if c not in DROP_COLS]


def get_num_cols(feature_cols):
    """Numeric feature columns (everything not in CAT_COLS)."""
    return [c for c in feature_cols if c not in CAT_COLS]