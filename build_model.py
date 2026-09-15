"""
CASEFILE — Model Builder & 5x Repeated Evaluation
==================================================
Builds the Missing Person Location Prediction model from the synthetic case
data, evaluates it 5 times (repeated stratified 80/20 train/test splits with
different random seeds), aggregates the results, and saves:

    models/location_model.pkl      -> best pipeline (preprocessor + XGBoost)
    models/label_encoder.pkl       -> target-area label encoder
    results/evaluation_runs.csv    -> per-run metrics for every repeat
    results/evaluation_summary.json-> mean +/- std per model (for the UI)

Run:
    python build_model.py [--repeats 5]

All data is 100% synthetic / fictional (academic experimentation only).
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from casefile_common import CAT_COLS, engineer_features, get_feature_cols, get_num_cols

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "files (1)", "synthetic_case_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DEFAULT_REPEAT_SEEDS = [42, 7, 21, 99, 123]

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=6,
        random_state=42, eval_metric="mlogloss",
    ),
}


def load_and_engineer():
    """Load the synthetic case data and apply feature engineering."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]:,} case records x {df.shape[1]} columns from:")
    print(f"  {DATA_PATH}")
    print(f"Missing values: {int(df.isnull().sum().sum())} | "
          f"Duplicate rows: {int(df.duplicated().sum())}")

    df_eng = engineer_features(df)
    feature_cols = get_feature_cols(df_eng)
    X = df_eng[feature_cols]
    y = df_eng["Target_Area"]
    return df_eng, X, y, feature_cols


def build_preprocessor(feature_cols):
    """ColumnTransformer: scale numerics, one-hot encode categoricals."""
    num_cols = get_num_cols(feature_cols)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def evaluate_top_k(proba_matrix, y_true, k_vals=(1, 3, 5)):
    """Top-1 / Top-3 / Top-5 accuracy from a probability matrix."""
    sorted_preds = np.argsort(proba_matrix, axis=1)[:, ::-1]
    results = {}
    for k in k_vals:
        correct = sum(1 for i, true_label in enumerate(y_true)
                      if true_label in sorted_preds[i, :k])
        results[k] = correct / len(y_true)
    return results


def run_repeated_evaluation(X, y_encoded, feature_cols, seeds):
    """Train every model on each repeat seed and collect per-run metrics."""
    preprocessor = build_preprocessor(feature_cols)
    records = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.20, random_state=seed, stratify=y_encoded
        )
        for name, model in MODELS.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
            pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)
            probs = pipe.predict_proba(X_test)
            top_k = evaluate_top_k(probs, y_test)

            macro_f1 = precision_recall_fscore_support(
                y_test, preds, average="macro", zero_division=0
            )[2]
            weighted_f1 = precision_recall_fscore_support(
                y_test, preds, average="weighted", zero_division=0
            )[2]

            records.append({
                "repeat": seed,
                "model": name,
                "top1_accuracy": round(accuracy_score(y_test, preds), 4),
                "top3_accuracy": round(top_k[3], 4),
                "top5_accuracy": round(top_k[5], 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
            })
            print(f"  [seed {seed:>3}] {name:<20} "
                  f"Top-1 {records[-1]['top1_accuracy']:.4f} | "
                  f"Top-3 {records[-1]['top3_accuracy']:.4f} | "
                  f"Top-5 {records[-1]['top5_accuracy']:.4f} | "
                  f"Macro-F1 {records[-1]['macro_f1']:.4f}")

    return pd.DataFrame(records)


def aggregate_results(runs_df):
    """Mean +/- std per model across all repeats."""
    summary = []
    for model_name, group in runs_df.groupby("model"):
        summary.append({
            "model": model_name,
            "top1_mean": round(group["top1_accuracy"].mean(), 4),
            "top1_std": round(group["top1_accuracy"].std(), 4),
            "top3_mean": round(group["top3_accuracy"].mean(), 4),
            "top3_std": round(group["top3_accuracy"].std(), 4),
            "top5_mean": round(group["top5_accuracy"].mean(), 4),
            "top5_std": round(group["top5_accuracy"].std(), 4),
            "macro_f1_mean": round(group["macro_f1"].mean(), 4),
            "macro_f1_std": round(group["macro_f1"].std(), 4),
            "weighted_f1_mean": round(group["weighted_f1"].mean(), 4),
            "weighted_f1_std": round(group["weighted_f1"].std(), 4),
        })
    return pd.DataFrame(summary)


def train_final_model(X, y_encoded, feature_cols, best_model_name):
    """Retrain the best-performing model on 100% of the data for deployment."""
    preprocessor = build_preprocessor(feature_cols)
    final_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", MODELS[best_model_name]),
    ])
    final_pipe.fit(X, y_encoded)
    return final_pipe


def main():
    parser = argparse.ArgumentParser(description="CASEFILE model builder + 5x evaluation")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of repeated evaluations (default: 5)")
    args = parser.parse_args()

    seeds = DEFAULT_REPEAT_SEEDS[:args.repeats]

    print("=" * 78)
    print(" CASEFILE: MISSING PERSON LOCATION PREDICTION SYSTEM")
    print(f" Model Builder + {len(seeds)}x Repeated Evaluation")
    print("=" * 78)

    # 1. Load + engineer
    print("\n--- STEP 1: LOAD & FEATURE ENGINEERING ---")
    df_eng, X, y, feature_cols = load_and_engineer()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # 2. Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Target areas ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    # 3. Repeated evaluation
    print(f"\n--- STEP 2: TRAIN + EVALUATE x{len(seeds)} (stratified 80/20, seeds {seeds}) ---")
    runs_df = run_repeated_evaluation(X, y_encoded, feature_cols, seeds)

    # 4. Aggregate
    print("\n--- STEP 3: AGGREGATED RESULTS (mean +/- std over repeats) ---")
    summary_df = aggregate_results(runs_df)
    print(summary_df.to_string(index=False))

    # 5. Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    runs_path = os.path.join(RESULTS_DIR, "evaluation_runs.csv")
    runs_df.to_csv(runs_path, index=False)

    # Pick the best model by mean Top-1 accuracy across the repeats
    best_model_name = summary_df.loc[summary_df["top1_mean"].idxmax(), "model"]
    summary_json = {
        "repeats": len(seeds),
        "seeds": seeds,
        "test_size": 0.20,
        "models": summary_df.to_dict(orient="records"),
        "best_model": best_model_name,
        "n_cases": int(len(df_eng)),
        "n_features": len(feature_cols),
        "areas": list(label_encoder.classes_),
    }
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    print("\n--- STEP 4: TRAIN FINAL MODEL ON 100% OF DATA & SAVE ---")
    print(f"Best model by mean Top-1 accuracy: {best_model_name}")
    final_pipe = train_final_model(X, y_encoded, feature_cols, best_model_name)
    model_path = os.path.join(MODELS_DIR, "location_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    joblib.dump(final_pipe, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved pipeline  -> {model_path}")
    print(f"Saved encoder   -> {encoder_path}")
    print(f"Saved runs CSV  -> {runs_path}")
    print(f"Saved summary   -> {summary_path}")

    print("\n" + "=" * 78)
    print(" DONE. The web UI (app.py) will load these artifacts automatically.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())